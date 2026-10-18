"""Real local files; no real project config, daemon, socket or cloud calls."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
DATA = b'# SYNTHETIC CONFIG\n[mcp_servers.synthetic]\nenabled=true\n'


def module():
    spec = importlib.util.spec_from_file_location('isolated_transaction', ROOT / 'config_transaction.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert hasattr(m, 'Transaction'), 'durable transaction/recovery API is missing'
    return m


def create(tmp_path):
    m = module()
    root = tmp_path / 'operation'
    root.mkdir(mode=0o700)
    parent = tmp_path / 'project'
    parent.mkdir()
    target = parent / 'config.toml'
    tx = m.Transaction.prepare(root, target, DATA)
    return m, tx, target


def interrupt_after(tx, event):
    original = tx.event
    def event_write(name, **fields):
        if name == event:
            raise OSError('SYNTHETIC JOURNAL FAILURE')
        return original(name, **fields)
    tx.event = event_write


def test_install_rollback_keeps_owned_original_as_evidence(tmp_path):
    m, tx, target = create(tmp_path)
    assert tx.apply()['state'] == 'installed'
    assert target.read_bytes() == DATA
    result = tx.rollback()
    assert result['state'] == 'rolled_back' and not target.exists()
    assert Path(result['retained_path']).read_bytes() == DATA
    assert tx.rollback()['state'] == 'rolled_back'


def test_concurrent_replace_during_rollback_intent_is_preserved(tmp_path):
    m, tx, target = create(tmp_path)
    tx.apply()
    original = tx.event
    def replace(name, **fields):
        result = original(name, **fields)
        if name == 'rollback_intent':
            temporary = target.with_name('editor.tmp')
            temporary.write_bytes(b'CONCURRENT EDIT')
            temporary.replace(target)
        return result
    tx.event = replace
    with pytest.raises(m.TransactionError):
        tx.rollback()
    assert target.read_bytes() == b'CONCURRENT EDIT'


@pytest.mark.parametrize('event', ['published', 'install_complete'])
def test_published_config_recovers_without_final_receipt(tmp_path, event):
    m, tx, target = create(tmp_path)
    interrupt_after(tx, event)
    with pytest.raises(OSError):
        tx.apply()
    assert target.read_bytes() == DATA
    reopened = m.Transaction.open(tx.root)
    assert reopened.recover()['state'] == 'rolled_back'
    assert not target.exists()
    assert reopened.recover()['state'] == 'rolled_back'


def test_journal_failure_before_publication_never_installs(tmp_path):
    m, tx, target = create(tmp_path)
    interrupt_after(tx, 'publish_intent')
    with pytest.raises(OSError):
        tx.apply()
    assert not target.exists()
    assert m.Transaction.open(tx.root).recover()['state'] == 'not_published'


def test_recovery_refuses_changed_published_config(tmp_path):
    m, tx, target = create(tmp_path)
    interrupt_after(tx, 'published')
    with pytest.raises(OSError):
        tx.apply()
    target.write_bytes(b'CHANGED AFTER FAILURE')
    with pytest.raises(m.TransactionError):
        m.Transaction.open(tx.root).recover()
    assert target.read_bytes() == b'CHANGED AFTER FAILURE'


def test_rollback_after_capture_journal_failure_is_reentrant(tmp_path):
    m, tx, target = create(tmp_path)
    tx.apply()
    interrupt_after(tx, 'withdrawn')
    with pytest.raises(OSError):
        tx.rollback()
    assert not target.exists()
    reopened = m.Transaction.open(tx.root)
    result = reopened.recover()
    assert result['state'] == 'rolled_back'
    assert Path(result['retained_path']).read_bytes() == DATA


def test_target_created_by_other_writer_is_never_overwritten(tmp_path):
    m, tx, target = create(tmp_path)
    target.write_bytes(b'OTHER WRITER')
    with pytest.raises(m.TransactionError):
        tx.apply()
    assert target.read_bytes() == b'OTHER WRITER'


def test_replacement_at_atomic_capture_is_restored_without_losing_bytes(tmp_path):
    m, tx, target = create(tmp_path)
    tx.apply()
    capture = tx.capture
    def competing_capture():
        temporary = target.with_name('editor.tmp')
        temporary.write_bytes(b'CAPTURE RACE EDIT')
        temporary.replace(target)
        return capture()
    tx.capture = competing_capture
    with pytest.raises(m.TransactionError, match='COMPETING_FILE_RESTORED'):
        tx.rollback()
    assert target.read_bytes() == b'CAPTURE RACE EDIT'
    assert (tx.root / 'withdrawn.toml').read_bytes() == b'CAPTURE RACE EDIT'
    assert not (tx.root / 'rollback-receipt.json').exists()


def test_inplace_write_at_capture_is_restored_and_never_destroyed(tmp_path):
    m, tx, target = create(tmp_path)
    tx.apply()
    capture = tx.capture
    def competing_capture():
        target.write_bytes(b'INPLACE RACE EDIT')
        return capture()
    tx.capture = competing_capture
    with pytest.raises(m.TransactionError, match='COMPETING_FILE_RESTORED'):
        tx.rollback()
    assert target.read_bytes() == b'INPLACE RACE EDIT'
    assert (tx.root / 'withdrawn.toml').read_bytes() == b'INPLACE RACE EDIT'


def test_restoration_never_replaces_another_new_config(tmp_path):
    m, tx, target = create(tmp_path)
    tx.apply()
    capture = tx.capture
    def competing_capture():
        temporary = target.with_name('editor.tmp')
        temporary.write_bytes(b'CAPTURED EDIT')
        temporary.replace(target)
        capture()
        target.write_bytes(b'NEWER EDIT')
    tx.capture = competing_capture
    with pytest.raises(m.TransactionError, match='CONFLICT_RETAINED'):
        tx.rollback()
    assert target.read_bytes() == b'NEWER EDIT'
    assert (tx.root / 'withdrawn.toml').read_bytes() == b'CAPTURED EDIT'


def test_final_journal_cannot_make_reappeared_target_look_rolled_back(tmp_path):
    m, tx, target = create(tmp_path)
    tx.apply()
    original = tx.event
    def reappear(name, **fields):
        result = original(name, **fields)
        if name == 'rollback_complete':
            target.write_bytes(b'NEW TARGET')
        return result
    tx.event = reappear
    with pytest.raises(m.TransactionError, match='ROLLBACK_FINAL_CHANGED'):
        tx.rollback()
    with pytest.raises(m.TransactionError):
        m.Transaction.open(tx.root).recover()
    assert target.read_bytes() == b'NEW TARGET'


@pytest.mark.parametrize('name', ['binding.json', 'staged-config.toml'])
def test_binding_or_stage_drift_is_rejected(tmp_path, name):
    m, tx, target = create(tmp_path)
    (tx.root / name).write_bytes(b'CHANGED')
    with pytest.raises(m.TransactionError):
        tx.apply()
    assert not target.exists()


def test_unknown_capture_file_is_never_replaced(tmp_path):
    m, tx, target = create(tmp_path)
    tx.apply()
    (tx.root / 'withdrawn.toml').write_bytes(b'UNKNOWN CAPTURE')
    with pytest.raises(m.TransactionError):
        tx.rollback()
    assert target.read_bytes() == DATA
    assert (tx.root / 'withdrawn.toml').read_bytes() == b'UNKNOWN CAPTURE'


def test_operation_lock_prevents_second_writer(tmp_path):
    m, tx, target = create(tmp_path)
    with tx.locked():
        with pytest.raises(m.TransactionError, match='OPERATION_BUSY'):
            m.Transaction.open(tx.root).apply()
    assert not target.exists()


def test_existing_config_and_symlink_target_are_not_adopted(tmp_path):
    m = module()
    root = tmp_path / 'op'
    root.mkdir(mode=0o700)
    target = tmp_path / 'config.toml'
    target.write_bytes(b'EXISTING')
    with pytest.raises(m.TransactionError):
        m.Transaction.prepare(root, target, DATA)
    assert target.read_bytes() == b'EXISTING'
    target.unlink()
    target.symlink_to(tmp_path / 'elsewhere')
    with pytest.raises(m.TransactionError):
        m.Transaction.prepare(root, target, DATA)


def test_parent_replacement_is_rejected_without_touching_new_directory(tmp_path):
    m, tx, target = create(tmp_path)
    target.parent.rename(target.parent.with_name('old-project'))
    target.parent.mkdir()
    target.write_bytes(b'NEW DIRECTORY CONFIG')
    with pytest.raises(m.TransactionError, match='TARGET_PARENT_CHANGED'):
        tx.apply()
    assert target.read_bytes() == b'NEW DIRECTORY CONFIG'


@pytest.mark.parametrize('point', ['publish_intent', 'published', 'install_complete', 'withdrawn'])
def test_abrupt_owned_process_exit_recovers_from_durable_records(tmp_path, point):
    m, tx, target = create(tmp_path)
    if point == 'withdrawn':
        tx.apply()
    script = '''import os,sys
sys.path.insert(0,sys.argv[1])
from config_transaction import Transaction
t=Transaction.open(sys.argv[2]);original=t.event
def event(name,**fields):
 if name==sys.argv[3]:os._exit(77)
 return original(name,**fields)
t.event=event
t.rollback() if sys.argv[3]=='withdrawn' else t.apply()
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(ROOT), str(tx.root), point],
                            env={'PATH': '/usr/bin:/bin', 'PYTHONUTF8': '0'},
                            capture_output=True, timeout=5)
    assert result.returncode == 77 and not result.stdout and not result.stderr
    result = m.Transaction.open(tx.root).recover()
    assert result['state'] in {'rolled_back', 'restored_absent', 'not_published'} and not target.exists()
    assert Path(result['retained_path']).read_bytes() == DATA
    assert m.Transaction.open(tx.root).recover()['state'] == result['state']


def test_parent_swap_at_publication_never_writes_new_directory(tmp_path, monkeypatch):
    m, tx, target = create(tmp_path)
    original = m.os.link
    def replace_parent(source, destination, *args, **kwargs):
        if str(source).endswith('staged-config.toml'):
            target.parent.rename(target.parent.with_name('moved-project'))
            target.parent.mkdir()
            (target.parent / 'unrelated.txt').write_bytes(b'FOREIGN NEW DIRECTORY')
        return original(source, destination, *args, **kwargs)
    monkeypatch.setattr(m.os, 'link', replace_parent)
    with pytest.raises(m.TransactionError):
        tx.apply()
    assert not target.exists()
    assert (target.parent / 'unrelated.txt').read_bytes() == b'FOREIGN NEW DIRECTORY'


def test_final_install_journal_cannot_confirm_changed_target(tmp_path):
    m, tx, target = create(tmp_path)
    original = tx.event
    def replace(name, **fields):
        result = original(name, **fields)
        if name == 'install_complete':
            temporary = target.with_name('new-config')
            temporary.write_bytes(b'REPLACED BEFORE ACK')
            temporary.replace(target)
        return result
    tx.event = replace
    with pytest.raises(m.TransactionError, match='INSTALL_FINAL_CHANGED'):
        tx.apply()
    assert target.read_bytes() == b'REPLACED BEFORE ACK'


def test_receipt_write_failure_after_publish_can_be_recovered(tmp_path, monkeypatch):
    m, tx, target = create(tmp_path)
    original = m.write_once
    def fail_receipt(path, data):
        if Path(path).name == 'install-receipt.json':
            raise OSError('SYNTHETIC RECEIPT FAILURE')
        return original(path, data)
    monkeypatch.setattr(m, 'write_once', fail_receipt)
    with pytest.raises(OSError):
        tx.apply()
    assert target.read_bytes() == DATA
    assert m.Transaction.open(tx.root).recover()['state'] == 'rolled_back'
    assert not target.exists()


@pytest.mark.parametrize('point', ['publish_intent', 'published', 'withdrawn'])
def test_recovery_response_matches_durable_terminal_receipt(tmp_path, point):
    m, tx, target = create(tmp_path)
    if point == 'withdrawn':
        tx.apply()
    interrupt_after(tx, point)
    with pytest.raises(OSError):
        tx.rollback() if point == 'withdrawn' else tx.apply()
    result = m.Transaction.open(tx.root).recover()
    receipt = json.loads((tx.root / 'rollback-receipt.json').read_text())
    assert result == receipt
    assert m.Transaction.open(tx.root).recover() == receipt


def test_unknown_retained_file_is_not_installed_into_absent_target(tmp_path):
    m, tx, target = create(tmp_path)
    (tx.root / 'withdrawn.toml').write_bytes(b'UNKNOWN NOT CAPTURED BY THIS CALL')
    with pytest.raises(m.TransactionError, match='UNKNOWN_CAPTURE_RETAINED'):
        tx.recover()
    assert not target.exists()
    assert (tx.root / 'withdrawn.toml').read_bytes() == b'UNKNOWN NOT CAPTURED BY THIS CALL'
