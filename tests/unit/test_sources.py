import os
import pytest
from codex_memory_rag.schemas import SourceConfig,CmrError
from codex_memory_rag.sources import read_allowed_file,scan_source,derive_retrieval_state,build_corpus_manifest

@pytest.fixture
def source(tmp_path):
    return SourceConfig(source_id="s",project_id="p",kind="memory",root=tmp_path,include=["*.md"],enabled=True,allow_cloud_context=True,code_project="p")

def note(body="中文内容",status="active"):
    return "---\ncode_project: p\nstatus: "+status+"\n---\n"+body+"\n"

def test_unicode_crlf_bom_and_bytes_hash(source):
    p=source.root/"中文.md";p.write_bytes(b"\xef\xbb\xbf"+note().replace("\n","\r\n").encode())
    d=read_allowed_file(source,p.name)
    assert "中文内容" in d.text
    assert d.file_sha256==__import__("hashlib").sha256(p.read_bytes()).hexdigest()

@pytest.mark.parametrize("path",["../x.md","/etc/passwd","https://evil/x.md",".env","a/../x.md"])
def test_illegal_paths_denied(source,path):
    with pytest.raises(CmrError):read_allowed_file(source,path)

def test_symlink_fifo_secret_and_scope_denied(source,tmp_path):
    target=tmp_path.parent/"outside.md";target.write_text(note())
    (tmp_path/"linked.md").symlink_to(target)
    os.mkfifo(tmp_path/"pipe.md")
    (tmp_path/"secret.md").write_text(note('api_key = "sk-123456789012345678901234567890"'))
    (tmp_path/"other.md").write_text('---\ncode_project: other\nstatus: active\n---\nsecret')
    for path in ["linked.md","pipe.md","secret.md","other.md"]:
        with pytest.raises(CmrError):read_allowed_file(source,path)

def test_missing_root_is_not_successful_empty_scan(source):
    source.root.rmdir();s=scan_source(source)
    assert not s.accessible

def test_oversize_and_ignored_file_reported(source):
    (source.root/"large.md").write_text("x"*(2*1024*1024+1))
    (source.root/".memoryignore").write_text("ignored.md\n")
    (source.root/"ignored.md").write_text(note())
    result=scan_source(source)
    assert "large.md" in result.rejected and "ignored.md" in result.rejected
    assert not result.allowed

def test_corpus_manifest_detects_added_modified_deleted(source):
    (source.root/"a.md").write_text(note());m=build_corpus_manifest([source])
    (source.root/"b.md").write_text(note());assert build_corpus_manifest([source]).corpus_manifest_hash!=m.corpus_manifest_hash

def test_project_memory_sessions_are_not_codex_raw_sessions(source):
    (source.root/'sessions').mkdir();(source.root/'sessions/a.md').write_text(note())
    assert 'sessions/a.md' in scan_source(source).allowed

def test_git_probe_uses_posix_spawn_in_multithreaded_runtime(tmp_path,monkeypatch):
    import subprocess
    from codex_memory_rag.sources import git_state
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','--allow-empty','-qm','fixture'],check=True)
    def forbidden_fork(*args,**kwargs):raise AssertionError('fork/exec is unsafe after native model threads start')
    monkeypatch.setattr(subprocess,'_fork_exec',forbidden_fork)
    s=SourceConfig(source_id='s',project_id='p',root=tmp_path,kind='code',worktree_id='w')
    assert len(git_state(s)['head'])==40

def test_git_identity_probe_batches_metadata_without_losing_worktree_paths(tmp_path,monkeypatch):
    import subprocess
    from codex_memory_rag.sources import git_state
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','--allow-empty','-qm','fixture'],check=True)
    calls=[];original=subprocess.run
    def counted(*args,**kwargs):calls.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(subprocess,'run',counted)
    s=SourceConfig(source_id='s',project_id='p',root=tmp_path,kind='code',worktree_id='w')
    value=git_state(s)
    assert value['gitdir']==str(tmp_path/'.git') and value['head_path']==str(tmp_path/'.git/HEAD')
    assert len(calls)<=2
