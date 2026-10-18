import json
from pathlib import Path
import pytest,tomllib
from codex_memory_rag.installer import Installer
from codex_memory_rag.schemas import CmrError

@pytest.fixture
def installer(tmp_path):
    home=tmp_path/'codex';home.mkdir();(home/'config.toml').write_bytes(b'# original\nmodel = "keep"\n[mcp_servers.original]\ncommand = "old"\n')
    (home/'AGENTS.md').write_bytes(b'# original instructions\n')
    runtime=tmp_path/'runtime';runtime.mkdir()
    return Installer(home,runtime,Path('/synthetic/venv/bin/python'),runtime/'config.toml')

def test_install_is_idempotent_and_rollback_preserves_later_tables(installer):
    i=installer;original=(i.home/'config.toml').read_bytes();agents=(i.home/'AGENTS.md').read_bytes()
    plan=i.preview();assert not (i.home/'config.toml').read_bytes()!=original
    i.apply(plan);once=(i.home/'config.toml').read_bytes();i.apply(i.preview());assert (i.home/'config.toml').read_bytes()==once
    assert once.startswith(original) and (i.home/'AGENTS.md').read_bytes().startswith(agents)
    with (i.home/'config.toml').open('ab') as f:f.write(b'\n[unrelated]\nx = 2\n')
    i.rollback(apply=True);restored=(i.home/'config.toml').read_bytes()
    assert 'codex_memory_rag' not in tomllib.loads(restored.decode())['mcp_servers']
    assert tomllib.loads(restored.decode())['unrelated']['x']==2 and restored.startswith(original)

def test_foreign_service_and_modified_owned_block_are_conflicts(installer):
    i=installer
    with (i.home/'config.toml').open('ab') as f:f.write(b'\n[mcp_servers.codex_memory_rag]\ncommand="foreign"\n')
    with pytest.raises(CmrError,match='CONFIG_CONFLICT'):i.preview()

def test_concurrent_edit_and_rollback_edit_do_not_overwrite(installer):
    i=installer;plan=i.preview()
    with (i.home/'AGENTS.md').open('ab') as f:f.write(b'user added\n')
    before=(i.home/'config.toml').read_bytes()
    with pytest.raises(CmrError):i.apply(plan)
    assert (i.home/'config.toml').read_bytes()==before
    i.apply(i.preview());p=i.home/'AGENTS.md';p.write_bytes(p.read_bytes().replace(b'NO_STRONG_MATCH',b'USER_CHANGED'))
    with pytest.raises(CmrError):i.rollback(apply=True)
    assert 'codex_memory_rag' in (i.home/'config.toml').read_text()

def test_pending_install_journal_recovers_after_manifest_failure(installer,monkeypatch):
    import codex_memory_rag.installer as module
    i=installer;original=(i.home/'config.toml').read_bytes();real=module.atomic_json;calls=0
    def fail_second(path,value):
        nonlocal calls
        calls+=1
        if calls==2:raise OSError('injected journal finalization failure')
        return real(path,value)
    monkeypatch.setattr(module,'atomic_json',fail_second)
    with pytest.raises(OSError):i.apply(i.preview())
    monkeypatch.setattr(module,'atomic_json',real)
    i.rollback(apply=True)
    assert (i.home/'config.toml').read_bytes()==original
    assert (i.home/'AGENTS.md').read_bytes()==b'# original instructions\n'

def test_unrelated_temporary_file_is_preserved(installer):
    from codex_memory_rag.installer import digest
    target=installer.home/'AGENTS.md';collision=target.with_name(target.name+'.cmr-install.tmp');collision.write_bytes(b'foreign pending work')
    installer._replace(target,b'new',digest(target.read_bytes()))
    assert collision.read_bytes()==b'foreign pending work'

def test_rollback_can_resume_after_ownership_finalization_failure(installer,monkeypatch):
    import codex_memory_rag.installer as module
    i=installer;original=(i.home/'config.toml').read_bytes();i.apply(i.preview());real=module.atomic_json;calls=0
    def fail_second(path,value):
        nonlocal calls
        calls+=1
        if calls==2:raise OSError('injected rollback finalization failure')
        return real(path,value)
    monkeypatch.setattr(module,'atomic_json',fail_second)
    with pytest.raises(OSError):i.rollback(apply=True)
    monkeypatch.setattr(module,'atomic_json',real);i.rollback(apply=True)
    assert (i.home/'config.toml').read_bytes()==original
    assert (i.home/'AGENTS.md').read_bytes()==b'# original instructions\n'
