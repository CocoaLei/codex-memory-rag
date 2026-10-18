"""Reuse a verified read epoch without weakening source/publication checks."""
import hashlib

import pytest

from codex_memory_rag import retrieval, sources
from codex_memory_rag.schemas import AppConfig, CmrError, SourceConfig, identity_hash


@pytest.fixture
def reader(tmp_path, monkeypatch):
    source = SourceConfig(source_id='c', project_id='p', kind='code', root=tmp_path,
                          worktree_id='w', include=['*.py'], enabled=True,
                          allow_cloud_context=True)
    text = 'def answer():\n    return 42\n'
    (tmp_path / 'a.py').write_text(text)
    (tmp_path / 'b.py').write_text(text)
    epoch = {'head': 'A', 'branch': 'main'}
    calls = []
    def state(s):
        calls.append(s.source_id)
        return dict(epoch)
    monkeypatch.setattr(sources, 'git_state', state)
    extra = []
    def publication_state(s):
        extra.append(s.source_id)
        return dict(epoch)
    monkeypatch.setattr(retrieval, 'git_state', publication_state)
    config = AppConfig(sources=[source], storage={'runtime_dir': tmp_path / 'runtime'})
    engine = retrieval.Retriever(config, None, None)
    row = dict(source_id='c', project_id='p', relative_path='a.py',
               file_sha256=hashlib.sha256(text.encode()).hexdigest(),
               source_epoch=identity_hash(epoch), line_start=1, line_end=2, text=text)
    return engine, source, row, epoch, calls, extra


@pytest.mark.parametrize('candidate', [False, True])
def test_verified_read_supplies_epoch_but_publication_still_rechecks(reader, candidate):
    engine, source, row, epoch, calls, extra = reader
    cache = {} if candidate else None
    assert engine._verify(row, [source], False, cache)[0] == row['text']
    assert calls == ['c', 'c']  # Secure read still brackets bytes with Git checks.
    assert extra == []  # No third immediate Git read of the same source.
    epoch['head'] = 'B'
    with pytest.raises(CmrError, match='STALE_SOURCE'):
        engine._verify_publication_epochs([source], [row])
    assert extra == ['c']


def test_candidate_cache_rejects_different_epoch_even_with_matching_new_row(reader):
    engine, source, row, epoch, _, _ = reader
    cache = {}
    engine._verify(row, [source], False, cache)
    epoch['head'] = 'B'
    changed = row | {'relative_path': 'b.py', 'source_epoch': identity_hash(epoch)}
    with pytest.raises(CmrError, match='STALE_SOURCE'):
        engine._verify(changed, [source], False, cache)


@pytest.mark.parametrize('change,error', [
    ({'file_sha256': '0' * 64}, 'SOURCE_CHANGED'),
    ({'text': 'unverified bytes'}, 'SOURCE_CHANGED'),
    ({'source_epoch': '0' * 64}, 'STALE_SOURCE'),
    ({'project_id': 'other'}, 'SCOPE_DENIED'),
])
def test_reusing_read_epoch_preserves_row_checks(reader, change, error):
    engine, source, row, _, _, _ = reader
    with pytest.raises(CmrError, match=error):
        engine._verify(row | change, [source], False)


def test_head_switch_during_secure_read_still_fails(reader, monkeypatch):
    engine, source, row, _, _, _ = reader
    states = iter([{'head': 'A'}, {'head': 'B'}])
    monkeypatch.setattr(sources, 'git_state', lambda s: next(states))
    with pytest.raises(CmrError, match='STALE_SOURCE'):
        engine._verify(row, [source], False)
