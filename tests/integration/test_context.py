import pytest
from codex_memory_rag.schemas import SearchRequest,ContextRequest,CmrError
from test_retrieval import retrieval

def test_context_rechecks_hash_and_state(retrieval):
    r,s,store=retrieval
    result=r.search(SearchRequest(project_id='p',query='退款',mode='lexical',include_archived=True))
    current=next(x['chunk_id'] for x in result.results if x['relative_path']=='a.md')
    old=next(x['chunk_id'] for x in result.results if x['relative_path']=='old.md')
    assert '退款恢复' in r.read_context(ContextRequest(project_id='p',chunk_ids=[current]))['items'][0]['text']
    with pytest.raises(CmrError):r.read_context(ContextRequest(project_id='p',chunk_ids=[old]))
    assert r.read_context(ContextRequest(project_id='p',chunk_ids=[old],include_archived=True))['items'][0]['current_evidence'] is False
    (s.root/'a.md').write_text('---\ncode_project: p\nstatus: active\n---\nchanged\n')
    with pytest.raises(CmrError,match='SOURCE_CHANGED'):r.read_context(ContextRequest(project_id='p',chunk_ids=[current]))

def test_context_budget_keeps_provenance(retrieval):
    r,_,_=retrieval
    x=r.search(SearchRequest(project_id='p',query='退款',mode='lexical')).results[0]
    value=r.read_context(ContextRequest(project_id='p',chunk_ids=[x['chunk_id']],max_chars=2))
    assert value['truncated'] and len(value['items'][0]['text'])==2 and value['items'][0]['file_sha256']

def test_context_rechecks_head_after_verifying_last_item(retrieval,monkeypatch):
    from codex_memory_rag.schemas import CmrError
    from codex_memory_rag import retrieval as module
    r,s,store=retrieval
    code=s.model_copy(update={'kind':'code','worktree_id':'w'})
    r.config.sources=[code]
    rows=store.get_snapshot().table.search().where("relative_path='a.md'").to_list();row=rows[0]
    row['source_epoch']=module.identity_hash({'head':'A'})
    store.table.update(where="chunk_id='"+row['chunk_id']+"'",values={'source_epoch':row['source_epoch']})
    monkeypatch.setattr(r,'_verify',lambda *a:('x','source_current',[]))
    monkeypatch.setattr(module,'git_state',lambda source:{'head':'B'})
    with pytest.raises(CmrError,match='STALE_SOURCE'):r.read_context(ContextRequest(project_id='p',chunk_ids=[row['chunk_id']]))
