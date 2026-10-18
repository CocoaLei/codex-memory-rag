import pytest
from codex_memory_rag.schemas import SourceConfig,SourceDocument
from codex_memory_rag.sources import derive_retrieval_state

@pytest.mark.parametrize("meta,path,want",[({"status":"active"},"a.md","eligible"),({"status":"deprecated"},"a.md","historical"),({"status":"superseded"},"a.md","historical"),({"status":"rejected"},"a.md","historical"),({"status":"archived"},"a.md","historical"),({"status":"active","type":"archive"},"a.md","historical"),({"status":"active"},"90-archive/a.md","historical"),({"status":"draft"},"a.md","unverified"),({},"a.md","unverified"),({"status":"unrecognized"},"a.md","unverified"),({"status":"active","valid_until":"2000-01-01T00:00:00Z"},"a.md","historical")])
def test_memory_state_never_promotes_unverified(tmp_path,meta,path,want):
    s=SourceConfig(source_id="s",project_id="p",root=tmp_path,kind="memory")
    d=SourceDocument(source=s,relative_path=path,text="x",file_sha256="h",metadata=meta)
    original=meta.copy();state,reasons=derive_retrieval_state(d)
    assert state==want and d.metadata==original

def test_code_without_frontmatter_is_current(tmp_path):
    s=SourceConfig(source_id="s",project_id="p",root=tmp_path,kind="code",worktree_id="w")
    assert derive_retrieval_state(SourceDocument(source=s,relative_path="a.py",text="",file_sha256="h"))[0]=="source_current"

@pytest.mark.parametrize('until',['2000-01-01','2000-01-01T00:00:00','not-a-date'])
def test_unresolved_expiry_is_reported_and_cannot_be_current_evidence(tmp_path,until):
    s=SourceConfig(source_id='s',project_id='p',root=tmp_path,kind='memory')
    d=SourceDocument(source=s,relative_path='a.md',text='x',file_sha256='h',metadata={'status':'active','valid_until':until})
    state,reasons=derive_retrieval_state(d)
    assert state=='unverified'
    assert 'invalid_or_timezone_unspecified_expiry' in reasons
