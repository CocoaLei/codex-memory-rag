from types import SimpleNamespace

from codex_memory_rag.reranking import profile_id
from codex_memory_rag.retrieval import Retriever, rrf
from codex_memory_rag.schemas import RerankerConfig, SearchRequest, identity_hash
from test_reranker_retrieval import Scores, enable
from test_watcher import Encoder, engine


def scoped(e):
    # Trusted test setup also expresses desired behavior before the schema exists.
    e.config.retrieval.reranker = e.config.retrieval.reranker.model_copy(
        update={'scope_aware_budget': True, 'max_candidates': 64})


def test_scoped_text_results_can_retain_third_complete_evidence(engine):
    e, source = engine
    (source.root / 'a.md').write_text('# A\n退款恢复 主题 ScopedFixture\n# B\n退款恢复 相关 ScopedFixture\n# C\n退款恢复 完整答案 ScopedFixture\n')
    e.build('p')
    enable(e)
    scoped(e)
    class Ranked(Scores):
        def score(self, query, passages):
            return [0.7 if '完整答案' in p else 0.9 for p in passages]
    ret = Retriever(e.config, e.store, Encoder(), Ranked(0))
    out = ret.search(SearchRequest(project_id='p', query='退款恢复', top_k=3))
    assert len(out.results) == 3
    assert '完整答案' in out.results[2]['summary']
    # Exact/fallback paths keep the old file diversity contract.
    assert len(Retriever(e.config, e.store, Encoder()).search(
        SearchRequest(project_id='p', query='退款恢复')).results) == 2
    assert len(ret.search(SearchRequest(project_id='p', query='退款恢复', literal='ScopedFixture')).results) == 2
    class Changed(Ranked):
        def score(self, query, passages):
            (source.root / 'a.md').write_text('changed during scoring')
            return super().score(query, passages)
    stale = Retriever(e.config, e.store, Encoder(), Changed(0)).search(
        SearchRequest(project_id='p', query='退款恢复'))
    assert not stale.results and 'SOURCE_CHANGED' in stale.warnings


def channel_snapshot(e):
    actual = e.store.get_snapshot()
    rows = sorted(actual.table.search().to_list(), key=lambda r: r['line_start'])
    assert len(rows) >= 60
    rows = rows[:60]
    class Query:
        def __init__(self, selected):
            self.selected = selected
            self.count = len(selected)
        def where(self, scope, **kwargs):
            assert "source_id = 's'" in scope and "project_id = 'p'" in scope
            return self
        def distance_type(self, *args):
            return self
        def limit(self, count):
            self.count = count
            return self
        def select(self, *args):
            return self
        def to_list(self):
            return self.selected[:self.count]
    class Table:
        def search(self, value=None, **kwargs):
            if value is None:
                return Query(rows)
            if kwargs.get('query_type') == 'fts':
                return Query([r | {'_score': 20.0} for r in rows[:40]])
            return Query([r | {'_distance': 0.01} for r in rows[20:60]])
    return SimpleNamespace(table=Table(), generation_id=actual.generation_id, version=actual.version), rows


def test_text_pool_preserves_window_budget_when_code_ceiling_is_64(engine):
    e, source = engine
    (source.root / 'a.md').write_text(''.join(f'# S{i}\n退款恢复 segment{i}\n' for i in range(60)))
    e.build('p')
    enable(e)
    scoped(e)
    snapshot, _ = channel_snapshot(e)
    class Forty(Scores):
        def score(self, query, passages):
            assert len(passages) == 40
            return [0.9] * len(passages)
    out = Retriever(e.config, e.store, Encoder(), Forty(0)).search(
        SearchRequest(project_id='p', query='退款恢复'), snapshot)
    assert len(out.results) == 8


def test_broad_code_reaches_channel_tail_but_known_file_keeps_40(engine):
    import subprocess
    from codex_memory_rag.schemas import SourceConfig
    e, original = engine
    source = SourceConfig(source_id='s', project_id='p', root=original.root,
                          kind='code', worktree_id='fixture', include=['*.py'], enabled=True, allow_cloud_context=True)
    (source.root / 'tool.py').write_text(''.join(f'def item_{i}():\n    return "refund{i}"\n\n' for i in range(60)))
    for args in [('init', '-q'), ('config', 'user.email', 'fixture@example.invalid'),
                 ('config', 'user.name', 'Fixture'), ('add', 'tool.py'), ('commit', '-qm', 'fixture')]:
        subprocess.run(['git', '-C', str(source.root), *args], check=True, capture_output=True)
    e.config.sources = [source]
    e.build('p')
    enable(e)
    snapshot, rows = channel_snapshot(e)
    fusion = rrf([[r['chunk_id'] for r in rows[:40]], [r['chunk_id'] for r in rows[20:60]]])
    tail = sorted(fusion, key=lambda k: (-fusion[k], k))[50]
    text = next(r['text'] for r in rows if r['chunk_id'] == tail)
    class Tail(Scores):
        def score(self, query, passages):
            return [0.9 if text in passage else 0.1 for passage in passages]
    assert not Retriever(e.config, e.store, Encoder(), Tail(0)).search(
        SearchRequest(project_id='p', kind='code', query='退款函数'), snapshot).results
    scoped(e)
    out = Retriever(e.config, e.store, Encoder(), Tail(0)).search(
        SearchRequest(project_id='p', kind='code', query='退款函数'), snapshot)
    assert out.results[0]['chunk_id'] == tail
    class Protected(Scores):
        def score(self, query, passages):
            assert 'tool.py' not in query
            assert len(passages) == 40
            return [0.9] * len(passages)
    assert len(Retriever(e.config, e.store, Encoder(), Protected(0)).search(
        SearchRequest(project_id='p', kind='code', query='tool.py 中退款函数'), snapshot).results) == 8


def test_scoped_profile_is_distinct_and_default_keeps_old_identity():
    lock = {'fixture': True}
    legacy = RerankerConfig(max_tokens=256, supplementary_windows=True)
    expected = identity_hash(['reranker-v1', lock, legacy.backend, 'fp32-sdpa', 256, 4,
                              'bilingual-symbol-v2-scoped-query', 'prefix-supplement-idf-v1-64pairs'])
    assert profile_id(lock, legacy) == expected
    trial = legacy.model_copy(update={'scope_aware_budget': True, 'max_candidates': 64})
    assert profile_id(lock, trial) != expected
