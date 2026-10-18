import numpy as np
import pytest
from codex_memory_rag.store import Store
from codex_memory_rag.schemas import CmrError,Chunk,source_key,chunk_id

def row(key,content="old",file_hash="h",profile="fixture"):
    return dict(chunk_id=chunk_id(key,file_hash,"v1",0),source_key=key,source_id="s",project_id="p",worktree_id=None,relative_path="a.md",title="a",source_kind="docs",retrieval_state="source_current",text=content,lexical_text=content,line_start=1,line_end=1,file_sha256=file_hash,text_sha256=file_hash,embedding_profile_id=profile,vector=(np.ones(768)/np.sqrt(768)).tolist())

def test_upsert_delete_and_snapshot_are_isolated(tmp_path):
    s=Store(tmp_path,"fixture","index-v1");g=s.open_generation();a=source_key("p","a","bc","d.py");b=source_key("p","ab","c","d.py")
    s.replace_source(a,[row(a)]);s.replace_source(b,[row(b,"keep")]);s.activate_generation(g.generation_id)
    old=s.get_snapshot();s.replace_source(a,[row(a,"new","h2")]);s.replace_source(a,[row(a,"new","h2")])
    assert s.get_snapshot().table.count_rows()==2
    assert old.table.search().where("source_key = '"+a+"'").to_list()[0]["text"]=="old"
    s.delete_source(a);assert s.get_snapshot().table.count_rows()==1
    assert s.get_snapshot().table.search().to_list()[0]["text"]=="keep"

@pytest.mark.parametrize("vector",[[1.]*512,[float('nan')]*768,[0.]*768])
def test_bad_vector_never_commits(tmp_path,vector):
    s=Store(tmp_path,"fixture","index-v1");s.open_generation();k=source_key("p","a",None,"a.md");r=row(k);r["vector"]=vector
    with pytest.raises(CmrError):s.replace_source(k,[r])
    assert s.table.count_rows()==0

def test_wrong_profile_and_unscoped_delete_rejected(tmp_path):
    s=Store(tmp_path,"fixture","index-v1");s.open_generation();k=source_key("p","a",None,"a.md")
    with pytest.raises(CmrError):s.replace_source(k,[row(k,profile="other")])
    with pytest.raises(ValueError):s.delete_source("")

def test_profile_upgrade_preserves_previous_generation_until_activation(tmp_path):
    old=Store(tmp_path,'fixture','old');g=old.open_generation();k=source_key('p','s',None,'a.md');old.replace_source(k,[row(k)]);old.activate_generation(g.generation_id)
    with pytest.raises(CmrError):Store(tmp_path,'fixture','new')
    stage=Store(tmp_path,'fixture','new',allow_profile_upgrade=True)
    with pytest.raises(CmrError):stage.get_snapshot()
    stage.open_generation();assert old.get_snapshot().table.count_rows()==1
