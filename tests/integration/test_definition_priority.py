import subprocess

from codex_memory_rag.retrieval import Retriever
from codex_memory_rag.schemas import CmrError, SourceConfig, SearchRequest, RerankerConfig
from test_reranker_retrieval import Scores, enable
from test_watcher import Encoder, engine


def test_complete_runtime_definition_precedes_fragments_and_tests_only_in_broad_trial(engine):
    e, original = engine
    root = original.root
    (root / 'tests').mkdir()
    giant = 'def giant():\n' + '    value = "refund background background"\n' * 220 + '    return value\n'
    (root / 'impl.py').write_text(giant + '\ndef answer():\n    return "refund complete"\n')
    (root / 'tests/test_impl.py').write_text('def test_refund():\n    return "refund test"\n')
    for args in [('init', '-q'), ('config', 'user.email', 'fixture@example.invalid'),
                 ('config', 'user.name', 'Fixture'), ('add', '.'), ('commit', '-qm', 'fixture')]:
        subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)
    source = SourceConfig(source_id='s', project_id='p', root=root, kind='code',
                          worktree_id='fixture', include=['*.py'], enabled=True, allow_cloud_context=True)
    e.config.sources = [source]
    e.build('p')
    enable(e)
    e.config.retrieval.reranker = e.config.retrieval.reranker.model_copy(
        update={'scope_aware_budget': True, 'supplementary_windows': True, 'max_candidates': 64})
    class Ranked(Scores):
        def score(self, query, passages):
            return [0.7 if 'def answer()' in p else 0.95 for p in passages]
    request = SearchRequest(project_id='p', kind='code', query='refund')
    plain = Retriever(e.config, e.store, Encoder(), Ranked(0)).search(request)
    assert plain.results[0]['symbol'] != 'answer'
    e.config.retrieval.reranker = e.config.retrieval.reranker.model_copy(update={'definition_priority': True})
    trial = Retriever(e.config, e.store, Encoder(), Ranked(0))
    assert trial.search(request).results[0]['symbol'] == 'answer'
    assert trial.search(SearchRequest(project_id='p', kind='code', query='impl.py refund')).results[0]['symbol'] != 'answer'
    assert trial.search(SearchRequest(project_id='p', kind='code', query='refund', literal='answer')).results[0]['symbol'] == 'answer'
    class Unavailable(Encoder):
        def encode_queries(self, texts, kind):
            raise CmrError('MODEL_UNAVAILABLE')
    fallback = Retriever(e.config, e.store, Unavailable(), Ranked(0)).search(request)
    assert fallback.mode == 'lexical_only' and fallback.results[0]['symbol'] != 'answer'
    class Changed(Ranked):
        def score(self, query, passages):
            (root / 'impl.py').write_text('changed during scoring')
            return super().score(query, passages)
    changed = Retriever(e.config, e.store, Encoder(), Changed(0)).search(request)
    assert all(r['relative_path'] != 'impl.py' for r in changed.results)
    assert 'SOURCE_CHANGED' in changed.warnings


def test_definition_priority_has_its_own_profile_and_requires_scoped_windows():
    import pytest
    from pydantic import ValidationError
    from codex_memory_rag.reranking import profile_id
    original = RerankerConfig(scope_aware_budget=True, supplementary_windows=True, max_tokens=256, max_candidates=64)
    trial = original.model_copy(update={'definition_priority': True})
    assert profile_id({'fixture': True}, original) != profile_id({'fixture': True}, trial)
    with pytest.raises(ValidationError):
        RerankerConfig(definition_priority=True)
