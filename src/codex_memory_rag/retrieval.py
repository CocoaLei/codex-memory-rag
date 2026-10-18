import ast
import json
import re
import time
from collections import defaultdict

from .schemas import AppConfig, Chunk, CmrError, ContextRequest, SearchRequest, SearchResponse, identity_hash
from .sources import derive_retrieval_state, explicit_replacements, git_state, read_allowed_file

MAX_RESPONSE = 32 * 1024


def _python_definition_spans(text):
    """Return complete named function bounds; malformed sources get no preference."""
    spans = {}
    try:
        tree = ast.parse(text)

        def visit(node, parents=()):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = ".".join(parents + (child.name,))
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        first = min([child.lineno] + [d.lineno for d in child.decorator_list])
                        spans[name] = (first, child.end_lineno)
                    visit(child, parents + (child.name,))
                else:
                    visit(child, parents)

        visit(tree)
    except (SyntaxError, ValueError, RecursionError):
        return {}
    return spans


def rrf(lists: list[list[str]]) -> dict[str, float]:
    scores = defaultdict(float)
    for ranking in lists:
        seen = set()
        rank = 0
        for candidate in ranking:
            if candidate in seen:
                continue
            seen.add(candidate)
            rank += 1
            scores[candidate] += 1 / (60 + rank)
    return dict(scores)


def _size(value):
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


class Retriever:
    def __init__(self, config: AppConfig, store, encoder, reranker=None):
        self.config, self.store, self.encoder = config, store, encoder
        self.reranker = reranker

    def _sources(self, project: str, kind: str | None = None, source_id=None, shared=False):
        primary = [s for s in self.config.sources if s.project_id == project and s.enabled and not s.shared]
        if not primary:
            raise CmrError("SOURCE_NOT_CONFIGURED")
        sources = primary + ([s for s in self.config.sources if s.shared and s.enabled] if shared else [])
        sources = [s for s in sources if s.allow_cloud_context and (kind is None or (s.kind == "code" if kind == "code" else s.kind != "code"))]
        if source_id is not None:
            sources = [s for s in sources if s.source_id == source_id]
            if not sources:
                raise CmrError("SCOPE_DENIED")
        if kind == "code" and len(sources) > 1:
            raise CmrError("AMBIGUOUS_SOURCE")
        if not sources:
            raise CmrError("SCOPE_DENIED")
        return sources

    @staticmethod
    def _scope(sources, history):
        # IDs come from validated registry, never from arbitrary SQL request fields.
        ranges = " OR ".join(f"(source_id = '{s.source_id}' AND project_id = '{s.project_id}')" for s in sources)
        states = "'eligible', 'source_current'" + (", 'historical'" if history else "")
        return f"({ranges}) AND retrieval_state IN ({states})"

    def _verify(self, row, sources, history, candidate_cache=None):
        source = next((s for s in sources if s.source_id == row["source_id"] and s.project_id == row["project_id"]), None)
        if not source or not source.allow_cloud_context:
            raise CmrError("SCOPE_DENIED")
        key = (source.source_id, source.project_id, row["relative_path"])
        cached = candidate_cache.get(key) if candidate_cache is not None else None
        if cached is None:
            document = read_allowed_file(source, row["relative_path"])
            state, reasons = derive_retrieval_state(document)
            if explicit_replacements(document, sources, candidate_cache):
                state, reasons = "historical", reasons + ["explicit_replacement"]
            if source.kind == "code":
                epoch_key = (None, source.source_id, source.project_id)
                epoch = candidate_cache.get(epoch_key) if candidate_cache is not None else None
                if epoch is None:
                    # The secure read already checked Git before and after the
                    # bytes. Reuse that epoch; publication still checks afresh.
                    epoch = document.git_state
                    if candidate_cache is not None:
                        candidate_cache[epoch_key] = epoch
                if document.git_state != epoch:
                    raise CmrError("STALE_SOURCE")
            cached = document, state, reasons
            if candidate_cache is not None:
                candidate_cache[key] = cached
        document, state, reasons = cached
        if document.file_sha256 != row["file_sha256"]:
            raise CmrError("SOURCE_CHANGED")
        if state == "unverified" or (state == "historical" and not history):
            raise CmrError("SCOPE_DENIED")
        if source.kind == "code" and identity_hash(document.git_state or {}) != row["source_epoch"]:
            raise CmrError("STALE_SOURCE")
        lines = document.text.splitlines(keepends=True)
        actual = "".join(lines[row["line_start"] - 1:row["line_end"]])
        if actual != row["text"]:
            raise CmrError("SOURCE_CHANGED")
        if not actual.strip():
            raise CmrError("NO_STRONG_MATCH")
        return actual, state, reasons

    @staticmethod
    def _public(row, state, reasons, *, candidate_pilot=False):
        keys = ["chunk_id", "project_id", "source_id", "worktree_id", "relative_path", "line_start",
                "line_end", "file_sha256", "status", "indexed_at", "symbol", "git_commit", "branch"]
        from .candidate_pilot import CANDIDATE_FIELDS
        return {k: row.get(k) for k in keys} | {"retrieval_state": state, "retrieval_reasons": reasons,
                                              "current_evidence": state == "source_current",
                                              "evidence_basis": "verified_source_bytes" if state == "source_current" else "dated_memory_assertion"} | (CANDIDATE_FIELDS if candidate_pilot else {})

    @staticmethod
    def _verify_publication_epochs(sources, rows):
        for source in sources:
            if source.kind == "code":
                selected = [r for r in rows if r["source_id"] == source.source_id]
                if selected:
                    current_epoch = identity_hash(git_state(source) or {})
                    if any(r["source_epoch"] != current_epoch for r in selected):
                        raise CmrError("STALE_SOURCE")

    def search(self, request: SearchRequest, snapshot=None) -> SearchResponse:
        started = time.perf_counter()
        sources = self._sources(request.project_id, request.kind, request.source_id, request.include_shared)
        snapshot = snapshot or self.store.get_snapshot()
        scope = self._scope(sources, request.include_archived)
        rerank_config = self.config.retrieval.reranker
        use_reranker = (request.mode == "hybrid" and not request.literal and rerank_config.enabled
                       and self.reranker is not None and self.reranker.profile_id == rerank_config.profile_id)
        pair_query = request.query
        file_scoped = False
        if use_reranker and request.kind == "code":
            paths = {r["relative_path"] for r in snapshot.table.search().where(scope).select(["relative_path"]).to_list()}
            mentioned = [path for path in paths if re.search(r"(?<![\w./-])" + re.escape(path) + r"(?![\w./-])", request.query)]
            if len(mentioned) == 1:
                file_scoped = True
                scope += " AND relative_path = '" + mentioned[0].replace("'", "''") + "'"
                semantic_query = re.sub(r"(?<![\w./-])" + re.escape(mentioned[0]) + r"(?![\w./-])", "", request.query).strip()
                if re.search(r"\w", semantic_query):
                    pair_query = semantic_query
        rows, lists, warnings, pair_scores = {}, [], [], {}
        actual_mode = request.mode
        exact = []
        if request.literal:
            # Scoped projection avoids materializing vectors for exact identifiers.
            candidates = snapshot.table.search().where(scope).select(list(Chunk.model_fields)).to_list()
            exact = [r for r in candidates if request.literal in r["identifiers"] or request.literal == r["symbol"] or request.literal == r["relative_path"]]
            exact.sort(key=lambda r: (not (r["symbol"].split(".")[-1] == request.literal or r["relative_path"] == request.literal), r["relative_path"], r["line_start"]))
            rows.update({r["chunk_id"]: r for r in exact})
            lists.append([r["chunk_id"] for r in exact])
            actual_mode = "literal"
        else:
            if request.mode in {"hybrid", "lexical"}:
                candidates = snapshot.table.search(request.query, query_type="fts", fts_columns="lexical_text").where(scope, prefilter=True).limit(self.config.retrieval.candidate_k).to_list()
                lexical_threshold = self.config.retrieval.code_min_bm25 if request.kind == "code" else self.config.retrieval.memory_min_bm25
                if not use_reranker:
                    candidates = [r for r in candidates if r["_score"] >= lexical_threshold]
                rows.update({r["chunk_id"]: r for r in candidates})
                lists.append([r["chunk_id"] for r in candidates])
            if request.mode in {"hybrid", "vector"}:
                if self.encoder is None:
                    warnings.append("MODEL_UNAVAILABLE")
                    actual_mode = "lexical_only" if request.mode == "hybrid" else "unavailable"
                else:
                    try:
                        vector = self.encoder.encode_queries([request.query], request.kind)[0]
                        threshold = self.config.retrieval.code_min_cosine if request.kind == "code" else self.config.retrieval.memory_min_cosine
                        if threshold is None:
                            warnings.append("CALIBRATION_REQUIRED")
                            actual_mode = "lexical_only" if request.mode == "hybrid" else "uncalibrated"
                        else:
                            candidates = snapshot.table.search(vector).distance_type("cosine").where(scope, prefilter=True).limit(self.config.retrieval.candidate_k).to_list()
                            if not use_reranker:
                                candidates = [r for r in candidates if 1 - r["_distance"] >= threshold]
                            rows.update({r["chunk_id"]: r for r in candidates})
                            lists.append([r["chunk_id"] for r in candidates])
                    except CmrError as exc:
                        if exc.code != "MODEL_UNAVAILABLE":
                            raise
                        warnings.append(exc.code)
                        actual_mode = "lexical_only" if request.mode == "hybrid" else "unavailable"
        scores = rrf(lists)
        ranking = sorted(scores, key=lambda k: (-scores[k], k))
        response = SearchResponse(generation=snapshot.generation_id, table_version=snapshot.version,
                                  mode=actual_mode, warnings=warnings)
        from .candidate_pilot import candidate_warnings
        response.warnings.extend(candidate_warnings(self.config))
        if request.mode == "hybrid" and not request.literal and rerank_config.enabled:
            if not use_reranker:
                response.warnings.append("RERANKER_UNAVAILABLE")
                if actual_mode == "hybrid":
                    response.mode = "hybrid_without_reranker"
            else:
                from .reranking import format_passage, validate_scores
                qualified = []
                candidate_cache = {}
                candidate_limit = rerank_config.max_candidates
                if rerank_config.scope_aware_budget and (request.kind == "memory" or file_scoped):
                    candidate_limit = min(candidate_limit, 40)
                for key in ranking[:candidate_limit]:
                    try:
                        self._verify(rows[key], sources, request.include_archived, candidate_cache)
                    except CmrError as exc:
                        if exc.code != "NO_STRONG_MATCH" and exc.code not in response.warnings:
                            response.warnings.append(exc.code)
                        continue
                    qualified.append(key)
                try:
                    relevance = validate_scores(self.reranker.score(pair_query, [format_passage(rows[k]) for k in qualified]), len(qualified)) if qualified else []
                    pair_scores = dict(zip(qualified, relevance, strict=True))
                    threshold = rerank_config.code_min_score if request.kind == "code" else rerank_config.memory_min_score
                    definitions = {}

                    def definition_tier(key):
                        if not (rerank_config.definition_priority and actual_mode == "hybrid"
                                and request.kind == "code" and not file_scoped):
                            return 0
                        row = rows[key]
                        if not row["relative_path"].endswith(".py") or row["relative_path"].startswith("tests/"):
                            return 1
                        cache_key = (row["source_id"], row["project_id"], row["relative_path"])
                        if cache_key not in definitions:
                            document = candidate_cache[cache_key][0]
                            definitions[cache_key] = _python_definition_spans(document.text)
                        span = definitions[cache_key].get(row["symbol"])
                        return 0 if span and row["line_start"] <= span[0] and row["line_end"] >= span[1] else 1

                    ranking = sorted([k for k in qualified if pair_scores[k] >= threshold],
                                     key=lambda k: (definition_tier(k), -pair_scores[k], -scores[k], k))
                    response.reranker_profile = self.reranker.profile_id
                    response.mode = "hybrid_reranked" if actual_mode == "hybrid" else actual_mode
                except CmrError as exc:
                    if exc.code not in {"RERANKER_UNAVAILABLE", "RERANKER_QUERY_TOO_LONG"}:
                        raise
                    response.warnings.append(exc.code)
                    # Re-enter the original gated path, retaining its observable
                    # embedding fallback and current-source verification.
                    disabled = self.config.model_copy(deep=True)
                    disabled.retrieval.reranker.enabled = False
                    fallback = Retriever(disabled, self.store, self.encoder).search(request, snapshot)
                    fallback.warnings.extend(v for v in response.warnings if v not in fallback.warnings)
                    if fallback.mode == "hybrid":
                        fallback.mode = "hybrid_without_reranker"
                    fallback.elapsed_ms = (time.perf_counter() - started) * 1000
                    return fallback
        counts = defaultdict(int)
        per_file_limit = (8 if rerank_config.scope_aware_budget and response.mode == "hybrid_reranked"
                          and (request.kind == "memory" or file_scoped) else 2)
        for key in ranking:
            row = rows[key]
            if counts[row["source_key"]] >= per_file_limit:
                continue
            try:
                text, state, reasons = self._verify(row, sources, request.include_archived)
            except CmrError as exc:
                if exc.code != "NO_STRONG_MATCH" and exc.code not in response.warnings:
                    response.warnings.append(exc.code)
                continue
            item = self._public(row, state, reasons, candidate_pilot=self.config.candidate_pilot.enabled) | {"summary": text[:350], "rrf_score": scores[key], "literal_match": bool(request.literal)}
            if key in pair_scores:
                item["rerank_score"] = pair_scores[key]
            proposed = response.model_dump() | {"results": response.results + [item]}
            if _size(proposed) > MAX_RESPONSE:
                response.truncated = True
                break
            response.results.append(item)
            counts[row["source_key"]] += 1
            if len(response.results) == request.top_k:
                break
        # A branch switch after checking the first candidate invalidates the whole response.
        self._verify_publication_epochs(sources, [rows[x["chunk_id"]] for x in response.results])
        if not response.results:
            response.warnings.append("NO_STRONG_MATCH")
        response.elapsed_ms = (time.perf_counter() - started) * 1000
        return response

    def read_context(self, request: ContextRequest, snapshot=None) -> dict:
        sources = self._sources(request.project_id)
        snapshot = snapshot or self.store.get_snapshot()
        response = {"items": [], "generation": snapshot.generation_id, "table_version": snapshot.version, "truncated": False}
        if self.config.candidate_pilot.enabled:
            from .candidate_pilot import candidate_warnings
            response["warnings"] = candidate_warnings(self.config)
        budget = request.max_chars
        verified_rows = []
        for key in request.chunk_ids:
            if not __import__("re").fullmatch(r"[0-9a-f]{64}", key):
                raise CmrError("SCOPE_DENIED")
            rows = snapshot.table.search().where(f"chunk_id = '{key}' AND ({self._scope(sources, request.include_archived)})").limit(1).to_list()
            if not rows:
                raise CmrError("SCOPE_DENIED")
            row = rows[0]
            text, state, reasons = self._verify(row, sources, request.include_archived)
            item = self._public(row, state, reasons, candidate_pilot=self.config.candidate_pilot.enabled) | {"text": text[:budget]}
            if len(text) > budget:
                response["truncated"] = True
            low, high = 0, len(item["text"])
            while low < high:
                middle = (low + high + 1) // 2
                trial = item | {"text": item["text"][:middle]}
                if _size(response | {"items": response["items"] + [trial]}) <= MAX_RESPONSE:
                    low = middle
                else:
                    high = middle - 1
            if low < len(item["text"]):
                response["truncated"] = True
                item["text"] = item["text"][:low]
            response["items"].append(item)
            verified_rows.append(row)
            budget -= len(item["text"])
            if budget <= 0:
                response["truncated"] |= len(response["items"]) < len(request.chunk_ids)
                break
        self._verify_publication_epochs(sources, verified_rows)
        return response
