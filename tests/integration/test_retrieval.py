import numpy as np
import pytest
from pathlib import Path
from codex_memory_rag.store import Store
from codex_memory_rag.sources import read_allowed_file
from codex_memory_rag.parsing import chunk_document
from codex_memory_rag.retrieval import Retriever
from codex_memory_rag.schemas import AppConfig,SourceConfig,SearchRequest,CmrError
class Encoder:
    profile_id='fixture'
    def token_count(self,text):return len(text)//3+1
    def encode_queries(self,texts,kind):return np.ones((len(texts),768),dtype=np.float32)/np.sqrt(768)

@pytest.fixture
def retrieval(tmp_path):
    root=tmp_path/'notes';root.mkdir()
    (root/'a.md').write_text('---\ncode_project: p\nstatus: active\n---\n退款恢复 RDGClientTransport\n')
    (root/'old.md').write_text('---\ncode_project: p\nstatus: deprecated\n---\n退款旧流程\n')
    (root/'draft.md').write_text('---\ncode_project: p\nstatus: draft\n---\n退款草稿\n')
    src=SourceConfig(source_id='s',project_id='p',code_project='p',root=root,kind='memory',include=['*.md'],enabled=True,allow_cloud_context=True)
    cfg=AppConfig(sources=[src],storage={'runtime_dir':tmp_path/'runtime'})
    store=Store(cfg.storage.runtime_dir,'fixture','i');g=store.open_generation()
    for name in ['a.md','old.md','draft.md']:
        chunks=chunk_document(read_allowed_file(src,name),Encoder())
        for c in chunks:c.embedding_profile_id='fixture'
        rows=[c.model_dump()|{'vector':(np.ones(768)/np.sqrt(768)).tolist()} for c in chunks]
        store.replace_source(chunks[0].source_key,rows)
    store.activate_generation(g.generation_id)
    return Retriever(cfg,store,Encoder()),src,store

def test_scope_and_history_do_not_leak(retrieval):
    r,_,_=retrieval
    result=r.search(SearchRequest(project_id='p',query='退款',mode='lexical'))
    assert [x['relative_path'] for x in result.results]==['a.md']
    result=r.search(SearchRequest(project_id='p',query='退款',mode='lexical',include_archived=True))
    assert set(x['relative_path'] for x in result.results)=={'a.md','old.md'}
    assert next(x for x in result.results if x['relative_path']=='old.md')['current_evidence'] is False
    with pytest.raises(CmrError):r.search(SearchRequest(project_id='other',query='退款'))

def test_literal_is_exact_and_model_failure_is_observable(retrieval):
    r,_,_=retrieval
    assert r.search(SearchRequest(project_id='p',query='code',literal='RDGClientTransport')).results
    assert not r.search(SearchRequest(project_id='p',query='none',literal='RDGClientTransports',mode='lexical')).results
    r.encoder=None
    result=r.search(SearchRequest(project_id='p',query='退款'))
    assert result.mode=='lexical_only' and 'MODEL_UNAVAILABLE' in result.warnings

def test_changed_and_deleted_files_do_not_appear_in_summary(retrieval):
    r,s,_=retrieval
    (s.root/'a.md').write_text('changed')
    assert not r.search(SearchRequest(project_id='p',query='RDGClientTransport',mode='lexical')).results
    (s.root/'a.md').unlink()
    assert not r.search(SearchRequest(project_id='p',query='RDGClientTransport',mode='lexical')).results

from test_watcher import engine

def test_active_replacement_excludes_old_default_and_marks_history(engine):
    from test_watcher import Encoder
    from codex_memory_rag.retrieval import Retriever
    from codex_memory_rag.schemas import SearchRequest
    e,s=engine
    s=s.model_copy(update={'kind':'memory','code_project':'p'})
    e.config.sources=[s]
    (s.root/'old.md').write_text('---\nstatus: active\ncode_project: p\n---\n退款旧决策')
    (s.root/'new.md').write_text('---\nstatus: active\ncode_project: p\nsupersedes: old.md\n---\n退款新决策')
    e.build('p');r=Retriever(e.config,e.store,Encoder())
    out=r.search(SearchRequest(project_id='p',kind='memory',query='退款',mode='lexical'))
    assert 'old.md' not in [x['relative_path'] for x in out.results]
    hist=r.search(SearchRequest(project_id='p',kind='memory',query='退款',mode='lexical',include_archived=True))
    old=next(x for x in hist.results if x['relative_path']=='old.md')
    assert old['current_evidence'] is False
    assert 'explicit_replacement' in old['retrieval_reasons']
