import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedding import format_document, validate_vectors
from .parsing import CHUNKER_VERSION, chunk_document
from .schemas import AppConfig, CmrError, EmbeddingInput, SourceDocument, identity_hash, source_key
from .sources import STATE_RULE_VERSION, build_corpus_manifest, git_state, read_allowed_file, scan_source, scan_source_steps
from .store import Store, atomic_json


@dataclass
class PreparedFile:
    document: SourceDocument
    chunks: list
    rows: list[dict]


class Indexer:
    def __init__(self, config: AppConfig, encoder):
        self.config, self.encoder = config, encoder
        self.runtime = config.storage.runtime_dir
        grammar = self.runtime / "grammar.lock.json"
        grammar_hash = hashlib.sha256(grammar.read_bytes()).hexdigest() if grammar.exists() else "fixture-or-unavailable"
        self.index_profile = identity_hash([encoder.profile_id, CHUNKER_VERSION, "identity-v1", STATE_RULE_VERSION, "icu-v1", grammar_hash])
        self.store = Store(self.runtime, encoder.profile_id, self.index_profile, allow_profile_upgrade=True)
        self.states = {}

    def sources(self, project):
        sources = [s for s in self.config.sources if s.project_id == project and s.enabled and s.allow_cloud_context]
        if not sources:
            raise CmrError("SOURCE_NOT_CONFIGURED")
        return sources

    def encode_cached(self, chunks):
        cache = self.runtime / "cache/embeddings"
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
        vectors = []
        missing, indices = [], []
        for i, chunk in enumerate(chunks):
            key = identity_hash(["embedding-cache-v1", self.encoder.profile_id, format_document(chunk.title, chunk.text)])
            path = cache / (key + ".npy")
            if path.exists():
                try:
                    vectors.append(validate_vectors(np.load(path, allow_pickle=False)[None, :])[0])
                except (ValueError, CmrError, OSError):
                    raise CmrError("EMBEDDING_CONTRACT_MISMATCH") from None
            else:
                vectors.append(None)
                missing.append(EmbeddingInput(title=chunk.title, content=chunk.text))
                indices.append((i, path))
        if missing:
            values = self.encoder.encode_documents(missing)
            values = validate_vectors(values)
            for (i, path), value in zip(indices, values, strict=True):
                temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
                try:
                    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "wb") as stream:
                        np.save(stream, value, allow_pickle=False)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                vectors[i] = value
        return vectors

    def prepare_file(self, source, path, encode=True) -> PreparedFile:
        document = read_allowed_file(source, path)
        chunks = chunk_document(document, self.encoder)
        job = PreparedFile(document, chunks, [])
        if encode:
            self.encode_job_batch(job, chunks)
        return job

    def encode_job_batch(self, job, chunks):
        vectors = self.encode_cached(chunks)
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding_profile_id = self.encoder.profile_id
            job.rows.append(chunk.model_dump() | {"vector": vector.tolist()})

    def commit_file(self, job: PreparedFile):
        source, path = job.document.source, job.document.relative_path
        current = read_allowed_file(source, path)
        if current.file_sha256 != job.document.file_sha256:
            raise CmrError("SOURCE_CHANGED")
        if current.git_state != job.document.git_state:
            raise CmrError("STALE_SOURCE")
        key = source_key(source.project_id, source.source_id, source.worktree_id, path)
        return self.store.replace_source(key, job.rows)

    def plan_source(self, source, force=False):
        steps = self.plan_source_steps(source, force)
        while True:
            try:
                next(steps)
            except StopIteration as completed:
                return completed.value

    def plan_source_steps(self, source, force=False):
        """Yield after each verified file so the worker can serve queued reads."""
        scan = yield from scan_source_steps(source)
        if not scan.accessible or scan.failed:
            self.states[source.source_id] = {"state": "unavailable", "error": "STALE_SOURCE"}
            raise CmrError("STALE_SOURCE")
        existing = {}
        if self.store.table is not None:
            rows = self.store.table.search().where(f"source_id = '{source.source_id}' AND project_id = '{source.project_id}'").select(["relative_path", "file_sha256", "source_epoch", "source_key"]).to_list()
            existing = {r["relative_path"]: r for r in rows}
        changed = []
        state = git_state(source) or {}
        for path in scan.allowed:
            document = read_allowed_file(source, path)
            old = existing.get(path)
            if force or not old or old["file_sha256"] != document.file_sha256 or old["source_epoch"] != identity_hash(state):
                changed.append(path)
            yield
        deleted = [r["source_key"] for path, r in existing.items() if path not in scan.allowed]
        self.states[source.source_id] = {"state": "pending" if changed or deleted else "ready", "allowed": len(scan.allowed), "rejected": len(scan.rejected), "failed": len(scan.failed)}
        return changed, deleted

    def sync(self, project):
        if self.store.table is None:
            raise CmrError("INDEX_NOT_READY")
        receipts = []
        for source in self.sources(project):
            changed, deleted = self.plan_source(source)
            for path in changed:
                receipts.append(self.commit_file(self.prepare_file(source, path)).__dict__)
            for key in deleted:
                receipts.append(self.store.delete_source(key).__dict__)
            self.states[source.source_id]["state"] = "ready"
        return {"project_id": project, "commits": receipts}

    def build(self, project):
        sources = self.sources(project)
        if self.store.active_path.exists() and not self.store.active_matches():
            sources = [s for s in self.config.sources if s.enabled and s.allow_cloud_context]
        for source in sources:
            scan = scan_source(source)
            if not scan.accessible or scan.failed:
                raise CmrError("STALE_SOURCE")
        corpus_before = build_corpus_manifest(sources).corpus_manifest_hash
        old_table, old_generation = self.store.table, self.store.generation
        generation = self.begin_rebuild(project)
        try:
            reports = [self.sync(p) for p in sorted({s.project_id for s in sources})]
            report = {"projects": reports}
            # Query the staging table before activation; profile/schema validation ran per row.
            self.store.table.search().limit(1).to_list()
            if build_corpus_manifest(sources).corpus_manifest_hash != corpus_before:
                raise CmrError("SOURCE_CHANGED")
            self.store.activate_generation(generation.generation_id)
            atomic_json(self.runtime / "data/manifests/last-build.json", report | {"generation": generation.generation_id})
            return report | {"generation": generation.generation_id}
        except Exception:
            self.store.table, self.store.generation = old_table, old_generation
            raise

    def begin_rebuild(self, project):
        # Generations are global to the profile; a project rebuild must retain its peers.
        previous = self.store.get_snapshot() if self.store.active_matches() else None
        generation = self.store.open_generation()
        if previous:
            self.sources(project)  # validates project before using it in the generated filter
            others = previous.table.search().where(f"project_id != '{project}'").to_arrow()
            if others.num_rows:
                self.store.table.add(others)
        return generation
