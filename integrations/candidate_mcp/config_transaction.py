"""Initially-absent configuration transaction; standalone CLI is disabled.

No unlinked configuration bytes: rollback captures to this private operation,
verifies the captured inode/content, and restores competing content without
replacement. Durable stage and binding precede any target publication.
"""
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import uuid


class TransactionError(RuntimeError):
    pass


def require(value, code):
    if not value:
        raise TransactionError(code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'UNSAFE_PATH')
    require(not any(p.is_symlink() for p in [path, *path.parents]), 'UNSAFE_PATH')
    return path


def identity(path):
    s = safe(path).stat()
    require(stat.S_ISDIR(s.st_mode) and s.st_uid == os.getuid(), 'DIRECTORY_UNVERIFIED')
    return {'dev': s.st_dev, 'ino': s.st_ino, 'uid': s.st_uid, 'mode': stat.S_IMODE(s.st_mode)}


def state(path):
    path = safe(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {'exists': False}
    with os.fdopen(fd, 'rb') as stream:
        a = os.fstat(stream.fileno())
        require(stat.S_ISREG(a.st_mode) and a.st_uid == os.getuid() and a.st_size <= 131072, 'FILE_UNVERIFIED')
        data = stream.read(131073)
        b = os.fstat(stream.fileno())
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(len(data) <= 131072 and all(getattr(a, k) == getattr(b, k) for k in fields), 'FILE_CHANGED')
    current = path.lstat()
    require(all(getattr(b, k) == getattr(current, k) for k in fields), 'FILE_CHANGED')
    return {'exists': True, 'dev': a.st_dev, 'ino': a.st_ino, 'uid': a.st_uid,
            'mode': stat.S_IMODE(a.st_mode), 'size': a.st_size, 'mtime_ns': a.st_mtime_ns,
            'ctime_ns': a.st_ctime_ns, 'sha256': sha(data)}


def stable_equal(a, b):
    # Our own hard-link publication and rename change ctime. All other identity,
    # byte and modification fields remain binding; ordinary pre-CAS is exact.
    return {k: v for k, v in a.items() if k != 'ctime_ns'} == {k: v for k, v in b.items() if k != 'ctime_ns'}


def sync(path):
    fd = os.open(safe(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path, data):
    """Durably publish an immutable record without replacing an existing file."""
    path = safe(path)
    if path.exists():
        require(state(path)['mode'] == 0o600 and path.read_bytes() == data, 'RECORD_CONFLICT')
        return
    temporary = path.with_name('.pending-' + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    temporary.unlink()
    sync(path.parent)


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()


def read_json(path):
    observed = state(path)
    require(observed.get('mode') == 0o600, 'RECORD_UNVERIFIED')
    payload = Path(path).read_bytes()
    require(sha(payload) == observed['sha256'], 'RECORD_CHANGED')
    return json.loads(payload)


def rename_exclusive(source_fd, source, destination_fd, destination):
    """Darwin's atomic no-replace rename; refuse an unsupported platform."""
    require(sys.platform == 'darwin', 'ATOMIC_CAPTURE_UNAVAILABLE')
    libc = ctypes.CDLL(None, use_errno=True)
    fn = libc.renameatx_np
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    if fn(source_fd, os.fsencode(source), destination_fd, os.fsencode(destination), 0x00000004):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


class Transaction:
    @classmethod
    def prepare(cls, root, target, data):
        root, target = safe(root), safe(target)
        require(identity(root)['mode'] == 0o700, 'OPERATION_DIRECTORY_UNVERIFIED')
        require(not list(root.iterdir()), 'OPERATION_NOT_EMPTY')
        require(type(data) is bytes and 0 < len(data) <= 131072, 'INVALID_CONFIG_BYTES')
        require(state(target) == {'exists': False}, 'CAS_TARGET_EXISTS')
        write_once(root / 'staged-config.toml', data)
        binding = {'schema': 'absent-config-transaction-v1', 'target': str(target),
                   'operation_identity': identity(root), 'parent_identity': identity(target.parent),
                   'before': {'exists': False}, 'desired_sha256': sha(data),
                   'stage_state': state(root / 'staged-config.toml')}
        write_once(root / 'binding.json', encoded(binding))
        (root / 'journal').mkdir(mode=0o700)
        sync(root)
        tx = cls.open(root)
        tx.event('prepared')
        return tx

    @classmethod
    def open(cls, root):
        self = cls()
        self.root = safe(root)
        self.binding = read_json(self.root / 'binding.json')
        self.binding_hash = state(self.root / 'binding.json')['sha256']
        require(self.binding_hash == sha(encoded(self.binding)), 'BINDING_CHANGED')
        b = self.binding
        require(b['schema'] == 'absent-config-transaction-v1' and b['before'] == {'exists': False}, 'BINDING_INVALID')
        self.target = safe(b['target'])
        require(not self.target.is_relative_to(self.root), 'BINDING_INVALID')
        require(identity(self.root) == b['operation_identity'] and identity(self.root)['mode'] == 0o700, 'OPERATION_CHANGED')
        require(b['stage_state']['sha256'] == b['desired_sha256'] and b['stage_state']['mode'] == 0o600, 'BINDING_INVALID')
        return self

    def guard(self):
        require(state(self.root / 'binding.json')['sha256'] == self.binding_hash, 'BINDING_CHANGED')
        require(identity(self.root) == self.binding['operation_identity'], 'OPERATION_CHANGED')
        require(identity(self.target.parent) == self.binding['parent_identity'], 'TARGET_PARENT_CHANGED')

    @contextlib.contextmanager
    def locked(self):
        self.guard()
        fd = os.open(self.root / 'operation.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            s = os.fstat(fd)
            require(stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and stat.S_IMODE(s.st_mode) == 0o600, 'LOCK_UNVERIFIED')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise TransactionError('OPERATION_BUSY') from None
            self.guard()
            yield
        finally:
            os.close(fd)

    def event(self, name, **fields):
        directory = self.root / 'journal'
        require(identity(directory)['mode'] == 0o700, 'JOURNAL_UNVERIFIED')
        write_once(directory / (str(time.time_ns()) + '-' + uuid.uuid4().hex + '.json'),
                   encoded({'event': name, 'binding_sha256': self.binding_hash, **fields}))

    def owned(self, observed):
        return stable_equal(observed, self.binding['stage_state'])

    def stage_guard(self):
        require(self.owned(state(self.root / 'staged-config.toml')), 'STAGE_CHANGED')

    @contextlib.contextmanager
    def directories(self):
        self.guard()
        parent_fd = os.open(self.target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                for fd, wanted in [(parent_fd, self.binding['parent_identity']), (root_fd, self.binding['operation_identity'])]:
                    s = os.fstat(fd)
                    require((s.st_dev, s.st_ino) == (wanted['dev'], wanted['ino']), 'DIRECTORY_CHANGED')
                self.guard()
                yield parent_fd, root_fd
                self.guard()
            finally:
                os.close(root_fd)
        finally:
            os.close(parent_fd)

    def apply(self):
        with self.locked():
            require(not (self.root / 'rollback-receipt.json').exists(), 'ALREADY_ROLLED_BACK')
            self.stage_guard()
            require(state(self.target) == self.binding['before'], 'CAS_TARGET_EXISTS')
            self.event('publish_intent', target=str(self.target))
            self.guard()
            self.stage_guard()
            require(state(self.target) == self.binding['before'], 'CAS_TARGET_EXISTS')
            with self.directories() as (parent_fd, root_fd):
                sync(self.root)
                os.link('staged-config.toml', self.target.name, src_dir_fd=root_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
                os.fsync(parent_fd)
            self.guard()
            observed = state(self.target)
            require(self.owned(observed), 'PUBLICATION_UNVERIFIED')
            self.event('published', observed=observed)
            receipt = {'state': 'installed', 'binding_sha256': self.binding_hash,
                       'target': str(self.target), 'owned_state': observed}
            write_once(self.root / 'install-receipt.json', encoded(receipt))
            self.event('install_complete', receipt_sha256=state(self.root / 'install-receipt.json')['sha256'])
            self.guard()
            require(state(self.target) == observed and read_json(self.root / 'install-receipt.json') == receipt, 'INSTALL_FINAL_CHANGED')
            return receipt

    def capture(self):
        with self.directories() as (parent_fd, root_fd):
            rename_exclusive(parent_fd, self.target.name, root_fd, 'withdrawn.toml')
            os.fsync(parent_fd)
            os.fsync(root_fd)

    def restore_competing(self):
        captured = self.root / 'withdrawn.toml'
        self.guard()
        try:
            with self.directories() as (parent_fd, root_fd):
                os.link(captured.name, self.target.name, src_dir_fd=root_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
                os.fsync(parent_fd)
        except FileExistsError:
            raise TransactionError('ROLLBACK_CONFLICT_RETAINED') from None
        raise TransactionError('ROLLBACK_COMPETING_FILE_RESTORED')

    def finish_rollback(self, captured):
        self.guard()
        require(state(self.target) == {'exists': False}, 'ROLLBACK_TARGET_REAPPEARED')
        if captured:
            require(self.owned(state(self.root / 'withdrawn.toml')), 'RETAINED_CHANGED')
        self.stage_guard()
        result_state = 'rolled_back'
        if not captured:
            events = [read_json(p)['event'] for p in (self.root / 'journal').glob('*.json')]
            result_state = 'restored_absent' if 'publish_intent' in events else 'not_published'
        receipt = {'state': result_state,
                   'binding_sha256': self.binding_hash, 'target': str(self.target),
                   'retained_path': str(self.root / ('withdrawn.toml' if captured else 'staged-config.toml'))}
        write_once(self.root / 'rollback-receipt.json', encoded(receipt))
        self.event('rollback_complete', receipt_sha256=state(self.root / 'rollback-receipt.json')['sha256'])
        self.guard()
        require(state(self.target) == {'exists': False} and read_json(self.root / 'rollback-receipt.json') == receipt, 'ROLLBACK_FINAL_CHANGED')
        require(self.owned(state(Path(receipt['retained_path']))), 'RETAINED_CHANGED')
        return receipt

    def rollback_locked(self):
        captured = self.root / 'withdrawn.toml'
        if captured.exists() or captured.is_symlink():
            require(self.owned(state(captured)), 'UNKNOWN_CAPTURE_RETAINED')
            return self.finish_rollback(True)
        observed = state(self.target)
        if observed == {'exists': False}:
            return self.finish_rollback(False)
        self.stage_guard()
        require(self.owned(observed), 'ROLLBACK_CAS_CHANGED')
        installed = self.root / 'install-receipt.json'
        if installed.exists():
            receipt = read_json(installed)
            require(receipt['binding_sha256'] == self.binding_hash and observed == receipt['owned_state'], 'ROLLBACK_CAS_CHANGED')
        self.event('rollback_intent', observed=observed)
        self.guard()
        require(state(self.target) == observed, 'ROLLBACK_CAS_CHANGED')
        self.capture()
        if not stable_equal(state(captured), observed):
            self.restore_competing()
        self.event('withdrawn', observed=state(captured))
        return self.finish_rollback(True)

    def rollback(self):
        with self.locked():
            return self.rollback_locked()

    def recover(self):
        # Recovery always restores the absent before-state; never resumes install.
        with self.locked():
            return self.rollback_locked()


if __name__ == '__main__':
    print('ISOLATED_TRANSACTION_NOT_ENABLED', file=sys.stderr)
    raise SystemExit(2)
