"""Version-locked evaluation, never silently scoring changed source material."""
from collections import defaultdict

from .schemas import CmrError, ContextRequest, EvaluationLock, SearchRequest, identity_hash
from .sources import build_corpus_manifest


def validate_evaluation_split(suite):
    """Enforce declared fact identities; semantic equivalence still needs review."""
    facts = {}
    for question in suite["questions"]:
        fact = question.get("fact_id")
        if fact is None:
            continue
        if not isinstance(fact, str) or not fact.strip():
            raise CmrError("EVAL_SPLIT_LEAK")
        split = question.get("split")
        if fact in facts and facts[fact] != split:
            raise CmrError("EVAL_SPLIT_LEAK")
        facts[fact] = split


def validate_evaluation_corpus(config, lock: EvaluationLock):
    try:
        current = build_corpus_manifest([s for s in config.sources if s.enabled and s.allow_cloud_context])
    except CmrError:
        raise CmrError("EVAL_CORPUS_CHANGED") from None
    if current.corpus_manifest_hash != lock.corpus.corpus_manifest_hash:
        raise CmrError("EVAL_CORPUS_CHANGED")
    if identity_hash(config.retrieval.model_dump(mode="json")) != lock.retrieval_config_hash:
        raise CmrError("EVAL_CORPUS_CHANGED")
    expected_reranker = config.retrieval.reranker.profile_id if config.retrieval.reranker.enabled else None
    if lock.reranker_profile != expected_reranker:
        raise CmrError("EVAL_CORPUS_CHANGED")


def evidence_hit(result, expected):
    return (result["source_id"] == expected["source_id"] and result["relative_path"] == expected["relative_path"]
            and result["line_start"] <= expected["line_start"] and result["line_end"] >= expected["line_end"]
            and ("file_sha256" not in expected or result["file_sha256"] == expected["file_sha256"]))


def metrics(records):
    positive = [r for r in records if r["positive"]]
    negative = [r for r in records if not r["positive"]]
    return {"positive_count": len(positive), "negative_count": len(negative),
            "recall_at_8": sum(r["rank"] is not None and r["rank"] <= 8 for r in positive) / len(positive) if positive else None,
            "mrr_at_8": sum(1 / r["rank"] if r["rank"] and r["rank"] <= 8 else 0 for r in positive) / len(positive) if positive else None,
            "negative_no_strong_match": sum(not r["returned"] for r in negative),
            "failed_ids": [r["id"] for r in records if (r["positive"] and r["rank"] is None) or (not r["positive"] and r["returned"])]}


class Evaluator:
    def __init__(self, config, retriever):
        if config.candidate_pilot.enabled:
            raise CmrError("SCOPE_DENIED")
        self.config, self.retriever = config, retriever

    def freeze(self, suite) -> EvaluationLock:
        reranker = self.retriever.reranker
        rc = self.config.retrieval.reranker
        if rc.enabled and (reranker is None or reranker.profile_id != rc.profile_id):
            raise CmrError("RERANKER_UNAVAILABLE")
        if not suite.get("questions"):
            raise ValueError("empty evaluation suite")
        validate_evaluation_split(suite)
        from .sources import derive_retrieval_state, explicit_replacements, read_allowed_file
        registry = {s.source_id: s for s in self.config.sources if s.enabled and s.allow_cloud_context}
        for question in suite["questions"]:
            if not question.get("positive"):
                continue
            if not question.get("evidence"):
                raise CmrError("LABEL_INELIGIBLE")
            for evidence in question["evidence"]:
                source = registry.get(evidence["source_id"])
                kind = question.get("kind", "memory")
                if not source or source.project_id != question["project_id"] or (source.kind == "code") != (kind == "code"):
                    raise CmrError("LABEL_INELIGIBLE")
                document = read_allowed_file(source, evidence["relative_path"])
                state, _ = derive_retrieval_state(document)
                if explicit_replacements(document, list(registry.values())):
                    state = "historical"
                if state == "unverified" or state == "historical" and not question.get("include_archived"):
                    raise CmrError("LABEL_INELIGIBLE")
                if not 1 <= evidence["line_start"] <= evidence["line_end"] <= len(document.text.splitlines()):
                    raise CmrError("LABEL_INELIGIBLE")
                if evidence.get("file_sha256", document.file_sha256) != document.file_sha256:
                    raise CmrError("EVAL_CORPUS_CHANGED")
        snapshot = self.retriever.store.get_snapshot()
        corpus = build_corpus_manifest([s for s in self.config.sources if s.enabled and s.allow_cloud_context])
        return EvaluationLock(corpus=corpus, generation=snapshot.generation_id, table_version=snapshot.version,
                              embedding_profile_id=self.retriever.store.profile,
                              index_profile_id=self.retriever.store.index_profile,
                              retrieval_config_hash=identity_hash(self.config.retrieval.model_dump(mode="json")), suite_hash=identity_hash(suite),
                              reranker_profile=rc.profile_id if rc.enabled else None)

    def _calibration_item(self, item, snapshot):
        if item.get("literal") or item.get("expected_error") or item.get("operation"):
            return None
        request = SearchRequest(**{k: item[k] for k in ["query", "project_id", "kind", "source_id"] if k in item})
        sources = self.retriever._sources(request.project_id, request.kind, request.source_id)
        vector = self.retriever.encoder.encode_queries([request.query], request.kind)[0]
        rows = snapshot.table.search(vector).distance_type("cosine").where(self.retriever._scope(sources, False), prefilter=True).limit(40).to_list()
        correct = [1-r["_distance"] for r in rows if any(evidence_hit(r, e) for e in item.get("evidence", []))]
        lexical = snapshot.table.search(request.query, query_type="fts", fts_columns="lexical_text").where(self.retriever._scope(sources, False), prefilter=True).limit(40).to_list()
        correct_lexical = [r["_score"] for r in lexical if any(evidence_hit(r, e) for e in item.get("evidence", []))]
        return {"id": item["id"], "kind": request.kind, "positive": item["positive"],
                        "top_cosine": max((1-r["_distance"] for r in rows), default=None),
                        "correct_cosine": max(correct, default=None),
                        "top_bm25": max((r["_score"] for r in lexical), default=None),
                        "correct_bm25": max(correct_lexical, default=None)}

    def calibration_samples(self, suite, lock):
        validate_evaluation_split(suite)
        if identity_hash(suite) != lock.suite_hash or any(q.get("split") != "dev" for q in suite["questions"]):
            raise CmrError("SCOPE_DENIED")
        validate_evaluation_corpus(self.config, lock)
        snapshot = self.retriever.store.checkout_snapshot(lock.generation, lock.table_version)
        samples = [v for item in suite["questions"] if (v := self._calibration_item(item, snapshot)) is not None]
        validate_evaluation_corpus(self.config, lock)
        return {"lock": lock.model_dump(), "samples": samples}

    async def calibration_samples_async(self, suite, lock, submit):
        validate_evaluation_split(suite)
        if identity_hash(suite) != lock.suite_hash or any(q.get("split") != "dev" for q in suite["questions"]):
            raise CmrError("SCOPE_DENIED")
        await submit(lambda: validate_evaluation_corpus(self.config, lock), 10)
        snapshot = await submit(lambda: self.retriever.store.checkout_snapshot(lock.generation, lock.table_version), 10)
        samples = []
        for item in suite["questions"]:
            value = await submit(lambda i=item: self._calibration_item(i, snapshot), 10)
            if value is not None:
                samples.append(value)
        await submit(lambda: validate_evaluation_corpus(self.config, lock), 10)
        return {"lock": lock.model_dump(), "samples": samples}

    def evaluate(self, suite, lock: EvaluationLock):
        validate_evaluation_split(suite)
        if identity_hash(suite) != lock.suite_hash:
            raise CmrError("EVAL_CORPUS_CHANGED")
        snapshot = self.retriever.store.checkout_snapshot(lock.generation, lock.table_version)
        report = {"lock": lock.model_dump(), "modes": {}, "basic_memory_baseline": {
            "status": "BASELINE_NOT_COMPARABLE", "reason": "No byte-hash and synchronized-index proof supplied; original database untouched."}}
        for mode in ["lexical", "vector", "hybrid"]:
            validate_evaluation_corpus(self.config, lock)
            records = []
            for item in suite["questions"]:
                records.append(self._question(item, mode, snapshot))
            validate_evaluation_corpus(self.config, lock)
            by_group, dev_groups = defaultdict(list), defaultdict(list)
            for record in records:
                (by_group if record["split"] == "test" else dev_groups)[record["group"]].append(record)
            report["modes"][mode] = {"table_version": snapshot.version, "records": records,
                                    "dev": metrics([r for r in records if r["split"] == "dev"]),
                                    "test": metrics([r for r in records if r["split"] == "test"]),
                                    "groups": {g: metrics(v) for g, v in by_group.items()},
                                    "dev_groups": {g: metrics(v) for g, v in dev_groups.items()}}
        validate_evaluation_corpus(self.config, lock)
        report["corpus_comparable"] = True
        return report

    def _question(self, item, mode, snapshot):
        data = {key: item[key] for key in ["query", "project_id", "kind", "source_id", "literal", "include_archived"] if key in item}
        try:
            if item.get("operation") == "context":
                value = self.retriever.read_context(ContextRequest(project_id=item["project_id"], chunk_ids=item["chunk_ids"]), snapshot=snapshot)
                results, warnings, elapsed = value["items"], [], 0
            else:
                result = self.retriever.search(SearchRequest(**data, mode=mode), snapshot=snapshot)
                if (mode == "hybrid" and not item.get("literal") and self.config.retrieval.reranker.enabled
                        and (result.reranker_profile != self.config.retrieval.reranker.profile_id
                             or result.mode != "hybrid_reranked")):
                    raise CmrError("RERANKER_QUERY_TOO_LONG" if "RERANKER_QUERY_TOO_LONG" in result.warnings else "RERANKER_UNAVAILABLE")
                results, warnings, elapsed = result.results, result.warnings, result.elapsed_ms
        except CmrError as exc:
            if item["positive"] or item.get("expected_error") != exc.code:
                raise
            results, warnings, elapsed = [], [exc.code], 0
        rank = next((i for i, hit in enumerate(results, 1) if any(evidence_hit(hit, expected) for expected in item.get("evidence", []))), None)
        return {"id": item["id"], "group": item["group"], "split": item["split"],
                        "positive": item["positive"], "rank": rank, "returned": len(results),
                        "warnings": warnings, "elapsed_ms": elapsed}


    async def evaluate_async(self, suite, lock, submit):
        validate_evaluation_split(suite)
        if identity_hash(suite) != lock.suite_hash:
            raise CmrError("EVAL_CORPUS_CHANGED")
        snapshot = await submit(lambda: self.retriever.store.checkout_snapshot(lock.generation, lock.table_version), 10)
        report = {"lock": lock.model_dump(), "modes": {}, "basic_memory_baseline": {
            "status": "BASELINE_NOT_COMPARABLE", "reason": "No byte-hash and synchronized-index proof supplied; original database untouched."}}
        for mode in ["lexical", "vector", "hybrid"]:
            await submit(lambda: validate_evaluation_corpus(self.config, lock), 10)
            records = []
            for item in suite["questions"]:
                records.append(await submit(lambda i=item, m=mode: self._question(i, m, snapshot), 10))
            await submit(lambda: validate_evaluation_corpus(self.config, lock), 10)
            by_group, dev_groups = defaultdict(list), defaultdict(list)
            for record in records:
                (by_group if record["split"] == "test" else dev_groups)[record["group"]].append(record)
            report["modes"][mode] = {"table_version": snapshot.version, "records": records,
                                    "dev": metrics([r for r in records if r["split"] == "dev"]),
                                    "test": metrics([r for r in records if r["split"] == "test"]),
                                    "groups": {g: metrics(v) for g, v in by_group.items()},
                                    "dev_groups": {g: metrics(v) for g, v in dev_groups.items()}}
        await submit(lambda: validate_evaluation_corpus(self.config, lock), 10)
        report["corpus_comparable"] = True
        return report
