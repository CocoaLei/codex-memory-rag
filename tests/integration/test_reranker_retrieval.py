import pytest

from codex_memory_rag.retrieval import Retriever
from codex_memory_rag.schemas import CmrError, ContextRequest, RerankerConfig, SearchRequest
from test_watcher import engine, Encoder


def enable(e):
    e.config.retrieval.memory_min_cosine = 0.0
    e.config.retrieval.code_min_cosine = 0.0
    e.config.retrieval.reranker = RerankerConfig(
        enabled=True, model_dir=e.runtime / "model", lock_file=e.runtime / "reranker.lock.json",
        profile_id="fixture-reranker", memory_min_score=0.5, code_min_score=0.5)


class Scores:
    profile_id = "fixture-reranker"

    def __init__(self, score):
        self.value = score

    def score(self, query, passages):
        return [self.value for _ in passages]


def test_strong_lexical_but_low_pair_score_is_rejected(engine):
    e, s = engine
    (s.root / "a.md").write_text("退款恢复")
    e.build("p")
    enable(e)
    r = Retriever(e.config, e.store, Encoder(), Scores(0.1))
    out = r.search(SearchRequest(project_id="p", query="退款"))
    assert not out.results and "NO_STRONG_MATCH" in out.warnings
    assert out.reranker_profile == "fixture-reranker"
    assert r.search(SearchRequest(project_id="p", query="退款", mode="lexical")).results
    hit = Retriever(e.config, e.store, Encoder(), Scores(0.9)).search(
        SearchRequest(project_id="p", query="退款")).results[0]
    assert hit["rerank_score"] == 0.9


def test_legacy_blank_span_is_not_a_search_or_context_evidence(engine):
    # Existing indexes can contain an exact original blank span from r4.
    import hashlib
    e, s = engine
    text = '\n\n# Evidence\n退款恢复\n'
    (s.root / 'a.md').write_text(text)
    e.build('p')
    enable(e)
    row = e.store.get_snapshot().table.search().to_list()[0]
    row.update(line_start=1, line_end=2, text='\n\n', text_sha256=hashlib.sha256(b'\n\n').hexdigest())
    r = Retriever(e.config, e.store, Encoder(), Scores(1))
    with pytest.raises(CmrError, match='NO_STRONG_MATCH'):
        r._verify(row, [s], False)


def test_skipping_legacy_blank_row_does_not_mark_valid_results_no_strong_match(engine):
    from codex_memory_rag.schemas import identity_hash
    import hashlib
    e, s = engine
    (s.root / 'a.md').write_text('\n\n# Evidence\n退款恢复\n')
    e.build('p')
    enable(e)
    row = e.store.get_snapshot().table.search().to_list()[0]
    row.update(chunk_id=identity_hash('legacy-blank-fixture'), line_start=1, line_end=2,
               text='\n\n', text_sha256=hashlib.sha256(b'\n\n').hexdigest())
    e.store.table.add([row])
    r = Retriever(e.config, e.store, Encoder(), Scores(1))
    out = r.search(SearchRequest(project_id='p', query='退款'))
    assert out.results
    assert all(result['summary'].strip() for result in out.results)
    assert 'NO_STRONG_MATCH' not in out.warnings


def test_registered_code_filename_is_scope_not_pair_answer_evidence(engine):
    import subprocess
    from codex_memory_rag.schemas import SourceConfig
    e, old = engine
    source = SourceConfig(source_id='s', project_id='p', root=old.root, kind='code',
                          worktree_id='w', include=['*.py'], enabled=True, allow_cloud_context=True)
    (source.root / 'tool.py').write_text('def count_items(items):\n    return len(items)\n')
    for args in [('init', '-q'), ('config', 'user.email', 'fixture@example.invalid'),
                 ('config', 'user.name', 'Fixture'), ('add', 'tool.py'), ('commit', '-qm', 'fixture')]:
        subprocess.run(['git', '-C', str(source.root), *args], check=True, capture_output=True)
    e.config.sources = [source]
    e.build('p')
    enable(e)
    query = 'tool.py 中二维码纠错等级的实现在哪里？'
    class OriginalQuery(Encoder):
        def encode_queries(self, texts, kind):
            assert texts == [query]
            return super().encode_queries(texts, kind)
    class SemanticOnly(Scores):
        def score(self, text, passages):
            assert text == '中二维码纠错等级的实现在哪里？'
            return [0.1 for _ in passages]
    out = Retriever(e.config, e.store, OriginalQuery(), SemanticOnly(0)).search(
        SearchRequest(project_id='p', kind='code', query=query))
    assert not out.results and out.warnings == ['NO_STRONG_MATCH']


def test_missing_reranker_has_visible_fallback_and_cannot_freeze_acceptance(engine):
    from codex_memory_rag.evaluation import Evaluator
    e, s = engine
    (s.root / "a.md").write_text("退款恢复")
    e.build("p")
    enable(e)
    r = Retriever(e.config, e.store, Encoder())
    out = r.search(SearchRequest(project_id="p", query="退款"))
    assert out.results and "RERANKER_UNAVAILABLE" in out.warnings
    assert out.mode == "hybrid_without_reranker"
    with pytest.raises(CmrError, match="RERANKER_UNAVAILABLE"):
        Evaluator(e.config, r).freeze({"questions": [{"project_id": "p", "positive": False}]})


def test_wrong_length_or_nonfinite_scorer_cannot_publish(engine):
    e, s = engine
    (s.root / "a.md").write_text("退款恢复")
    e.build("p")
    enable(e)
    for values in [[], [float("nan")]]:
        class Invalid(Scores):
            def score(self, *args):
                return values
        r = Retriever(e.config, e.store, Encoder(), Invalid(0))
        with pytest.raises(CmrError, match="RERANKER_CONTRACT_MISMATCH"):
            r.search(SearchRequest(project_id="p", query="退款"))


def test_relevance_precedes_per_file_limit_and_sources_are_verified_before_scoring(engine):
    e, s = engine
    (s.root / "a.md").write_text("# A\n退款恢复 第一\n# B\n退款恢复 第二\n# C\n退款恢复 正确答案\n")
    e.build("p")
    enable(e)
    class Ranked(Scores):
        def score(self, query, passages):
            assert all("退款恢复" in text for text in passages)
            return [0.9 if "正确答案" in text else 0.6 for text in passages]
    r = Retriever(e.config, e.store, Encoder(), Ranked(0))
    out = r.search(SearchRequest(project_id="p", query="退款恢复"))
    assert len(out.results) == 2
    hit = out.results[0]
    assert "正确答案" in r.read_context(ContextRequest(project_id="p", chunk_ids=[hit["chunk_id"]]))["items"][0]["text"]
    (s.root / "a.md").write_text("changed")
    class NeverCalled(Scores):
        def score(self, *args):
            pytest.fail("stale source passed into reranker")
    assert not Retriever(e.config, e.store, Encoder(), NeverCalled(0)).search(
        SearchRequest(project_id="p", query="退款恢复")).results


def test_literal_does_not_load_or_call_reranker(engine):
    e, s = engine
    (s.root / "a.md").write_text("退款恢复 RDGClientTransport")
    e.build("p")
    enable(e)
    class NeverCalled(Scores):
        def score(self, *args):
            pytest.fail("exact identifiers do not need pair scoring")
    out = Retriever(e.config, e.store, Encoder(), NeverCalled(0)).search(
        SearchRequest(project_id="p", query="退款", literal="RDGClientTransport"))
    assert out.results and out.mode == "literal"


def test_pair_runtime_failure_uses_original_gates_and_is_not_formal_acceptance(engine):
    from codex_memory_rag.evaluation import Evaluator
    e, s = engine
    (s.root / "a.md").write_text("退款恢复")
    e.build("p")
    enable(e)
    class Unavailable(Scores):
        def score(self, *args):
            raise CmrError("RERANKER_UNAVAILABLE")
    r = Retriever(e.config, e.store, Encoder(), Unavailable(0))
    out = r.search(SearchRequest(project_id="p", query="退款"))
    assert out.results and "RERANKER_UNAVAILABLE" in out.warnings
    e.config.retrieval.memory_min_bm25 = 1e9
    e.config.retrieval.memory_min_cosine = 1.01
    assert not r.search(SearchRequest(project_id="p", query="退款")).results
    q = {"id": "n", "project_id": "p", "kind": "memory", "query": "退款", "positive": False, "split": "test", "group": "negative"}
    with pytest.raises(CmrError, match="RERANKER_UNAVAILABLE"):
        ev = Evaluator(e.config, r)
        ev.evaluate({"questions": [q]}, ev.freeze({"questions": [q]}))


def test_candidate_verification_reads_each_file_once_but_publication_reads_again(engine, monkeypatch):
    import codex_memory_rag.retrieval as module
    e, s = engine
    text = "# A\n退款恢复 第一\n# B\n退款恢复 第二\n# C\n退款恢复 第三\n"
    (s.root / "a.md").write_text(text)
    e.build("p")
    enable(e)
    reads = []
    original = module.read_allowed_file
    def read(*args):
        reads.append(args[1])
        return original(*args)
    monkeypatch.setattr(module, "read_allowed_file", read)
    r = Retriever(e.config, e.store, Encoder(), Scores(0.9))
    out = r.search(SearchRequest(project_id="p", query="退款恢复"))
    assert len(out.results) == 2
    assert reads == ["a.md", "a.md", "a.md"]
    class MutatesDuringScoring(Scores):
        def score(self, query, passages):
            (s.root / "a.md").write_text("changed during model inference")
            return super().score(query, passages)
    out = Retriever(e.config, e.store, Encoder(), MutatesDuringScoring(0.9)).search(
        SearchRequest(project_id="p", query="退款恢复"))
    assert not out.results and "SOURCE_CHANGED" in out.warnings


def test_replacement_scan_is_shared_only_for_candidates_and_new_replacement_is_rechecked(engine, monkeypatch):
    import codex_memory_rag.sources as module
    e, source = engine
    s = source.model_copy(update={"kind": "memory", "code_project": "p"})
    e.config.sources = [s]
    for name in ["a.md", "b.md", "c.md"]:
        (s.root / name).write_text("---\nstatus: active\ncode_project: p\n---\n退款恢复 " + name)
    e.build("p")
    enable(e)
    scans = []
    original = module.scan_source
    def scan(source):
        scans.append(source.source_id)
        return original(source)
    monkeypatch.setattr(module, "scan_source", scan)
    r = Retriever(e.config, e.store, Encoder(), Scores(0.9))
    out = r.search(SearchRequest(project_id="p", query="退款恢复"))
    assert len(out.results) == 3
    assert scans == ["s"] * 4
    class NewReplacement(Scores):
        def score(self, *args):
            (s.root / "new.md").write_text("---\nstatus: active\ncode_project: p\nsupersedes: a.md\n---\n新决定")
            return super().score(*args)
    out = Retriever(e.config, e.store, Encoder(), NewReplacement(0.9)).search(
        SearchRequest(project_id="p", query="退款恢复"))
    assert "a.md" not in [v["relative_path"] for v in out.results]


def test_long_legal_query_falls_back_without_truncation_or_false_acceptance(engine):
    from codex_memory_rag.evaluation import Evaluator
    e, s = engine
    (s.root / "a.md").write_text("退款恢复")
    e.build("p")
    enable(e)
    query = "退款 " + "synthetic context " * 80
    class PairWindow(Scores):
        def score(self, text, passages):
            assert text == query
            raise CmrError("RERANKER_QUERY_TOO_LONG")
    r = Retriever(e.config, e.store, Encoder(), PairWindow(0))
    out = r.search(SearchRequest(project_id="p", query=query))
    assert out.results and out.mode == "hybrid_without_reranker"
    assert "RERANKER_QUERY_TOO_LONG" in out.warnings
    q = {"id": "n", "project_id": "p", "kind": "memory", "query": query,
         "positive": False, "split": "test", "group": "negative"}
    with pytest.raises(CmrError, match="RERANKER_QUERY_TOO_LONG"):
        ev = Evaluator(e.config, r)
        ev.evaluate({"questions": [q]}, ev.freeze({"questions": [q]}))
