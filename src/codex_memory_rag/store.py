"""Versioned single LanceDB store, one writer and independent pinned readers."""
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS

from .embedding import validate_vectors
from .schemas import Chunk, EmbeddedChunk, CmrError


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Generation:
    generation_id: str
    embedding_profile_id: str
    index_profile_id: str


@dataclass(frozen=True)
class Snapshot:
    generation_id: str
    version: int
    table: object


@dataclass(frozen=True)
class CommitReceipt:
    version: int
    source_key: str
    inserted: int
    deleted: int


def schema():
    fields = []
    for name in Chunk.model_fields:
        dtype = pa.list_(pa.string()) if name in {"identifiers", "retrieval_reasons"} else pa.int64() if name in {"line_start", "line_end"} else pa.string()
        fields.append(pa.field(name, dtype))
    fields.append(pa.field("vector", pa.list_(pa.float32(), 768), nullable=False))
    return pa.schema(fields)


class Store:
    def __init__(self, runtime: Path, profile: str, index_profile: str, allow_profile_upgrade=False):
        self.runtime = runtime
        self.path = runtime / "data/lance"
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db = lancedb.connect(self.path)
        self.profile, self.index_profile = profile, index_profile
        self.active_path = runtime / "data/manifests/active.json"
        self.lock = threading.RLock()
        self.table = None
        self.generation = None
        if self.active_path.exists():
            data = json.loads(self.active_path.read_text())
            if data["embedding_profile_id"] != profile or data["index_profile_id"] != index_profile:
                if not allow_profile_upgrade:
                    raise CmrError("INDEX_PROFILE_MISMATCH")
                return
            self.generation = Generation(data["generation_id"], profile, index_profile)
            self.table = self.db.open_table(self.generation.generation_id)

    def active_matches(self):
        if not self.active_path.exists():
            return False
        record = json.loads(self.active_path.read_text())
        return record["embedding_profile_id"] == self.profile and record["index_profile_id"] == self.index_profile

    def open_generation(self) -> Generation:
        with self.lock:
            self.generation = Generation("g_" + uuid.uuid4().hex, self.profile, self.index_profile)
            self.table = self.db.create_table(self.generation.generation_id, schema=schema())
            self.table.create_index("lexical_text", config=FTS(base_tokenizer="icu", stem=False,
                                    remove_stop_words=False, ascii_folding=False, max_token_length=256))
            atomic_json(self.runtime / "data/manifests" / (self.generation.generation_id + ".json"),
                        self.generation.__dict__)
            return self.generation

    @staticmethod
    def _key(key):
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("unscoped source mutation denied")

    def replace_source(self, key: str, chunks: list[dict]) -> CommitReceipt:
        self._key(key)
        if not chunks:
            return self.delete_source(key)
        rows = []
        for item in chunks:
            data = dict(item)
            if data.get("source_key") != key or data.get("embedding_profile_id") != self.profile:
                raise CmrError("INDEX_PROFILE_MISMATCH")
            vector = validate_vectors(np.asarray(data.pop("vector"))[None, :])[0].tolist()
            rows.append(EmbeddedChunk.model_validate(data | {"vector": vector}).model_dump())
        with self.lock:
            previous = self.table.count_rows(f"source_key = '{key}'")
            self.table.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all().when_not_matched_by_source_delete(
                f"source_key = '{key}'").execute(pa.Table.from_pylist(rows, schema=schema()))
            return CommitReceipt(self.table.version, key, len(rows), previous)

    def delete_source(self, key: str) -> CommitReceipt:
        self._key(key)
        with self.lock:
            old = self.table.count_rows(f"source_key = '{key}'")
            self.table.delete(f"source_key = '{key}'")
            return CommitReceipt(self.table.version, key, 0, old)

    def activate_generation(self, generation_id: str):
        with self.lock:
            record = json.loads((self.runtime / "data/manifests" / (generation_id + ".json")).read_text())
            if record["embedding_profile_id"] != self.profile or record["index_profile_id"] != self.index_profile:
                raise CmrError("INDEX_PROFILE_MISMATCH")
            table = self.db.open_table(generation_id)
            atomic_json(self.active_path, record | {"activated_version": table.version})
            self.generation = Generation(generation_id, self.profile, self.index_profile)
            self.table = table

    def get_snapshot(self) -> Snapshot:
        with self.lock:
            if not self.active_path.exists():
                raise CmrError("INDEX_NOT_READY")
            active = json.loads(self.active_path.read_text())
            if active["embedding_profile_id"] != self.profile or active["index_profile_id"] != self.index_profile:
                raise CmrError("INDEX_PROFILE_MISMATCH")
            table = self.db.open_table(active["generation_id"])
            version = table.version
            table.checkout(version)
            return Snapshot(active["generation_id"], version, table)

    def checkout_snapshot(self, generation_id: str, version: int) -> Snapshot:
        record = json.loads((self.runtime / "data/manifests" / (generation_id + ".json")).read_text())
        if record["embedding_profile_id"] != self.profile or record["index_profile_id"] != self.index_profile:
            raise CmrError("INDEX_PROFILE_MISMATCH")
        table = self.db.open_table(generation_id)
        table.checkout(version)
        return Snapshot(generation_id, version, table)
