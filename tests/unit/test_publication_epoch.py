from types import SimpleNamespace

import pytest

from codex_memory_rag import retrieval
from codex_memory_rag.schemas import CmrError, identity_hash


def test_final_epoch_snapshot_checks_all_rows_without_repeated_git_processes(monkeypatch):
    source = SimpleNamespace(kind='code', source_id='code')
    calls = []
    def state(s):
        calls.append(s.source_id)
        return {'head': 'current', 'branch': 'main'}
    monkeypatch.setattr(retrieval, 'git_state', state)
    epoch = identity_hash({'head': 'current', 'branch': 'main'})
    rows = [{'source_id': 'code', 'source_epoch': epoch} for _ in range(8)]
    retrieval.Retriever._verify_publication_epochs([source], rows)
    assert calls == ['code']
    rows[-1]['source_epoch'] = identity_hash({'head': 'older', 'branch': 'main'})
    with pytest.raises(CmrError, match='STALE_SOURCE'):
        retrieval.Retriever._verify_publication_epochs([source], rows)


def test_final_epoch_only_reads_represented_code_sources(monkeypatch):
    sources = [SimpleNamespace(kind=k, source_id=s) for k,s in [('memory','m'),('code','a'),('code','b')]]
    calls = []
    def state(source):
        calls.append(source.source_id)
        return {'head': source.source_id}
    monkeypatch.setattr(retrieval, 'git_state', state)
    retrieval.Retriever._verify_publication_epochs(sources, [])
    assert calls == []
    rows = [{'source_id':'a','source_epoch':identity_hash({'head':'a'})}] * 2
    retrieval.Retriever._verify_publication_epochs(sources, rows)
    assert calls == ['a']
