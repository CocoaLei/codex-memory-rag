"""Owned local deployment, write-ahead recovery and shared entry guards.

No command mutates deployment targets without an exact separate authorization.
The old MCP/AGENTS installer is intentionally never called.
"""
import argparse
import contextlib
import copy
import datetime
import fcntl
import hashlib
import json
import os
import plistlib
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import tomllib
import uuid
from pathlib import Path


class DeployError(RuntimeError):
    pass


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def fail(condition, code='CONFIG_CONFLICT'):
    if not condition:
        raise DeployError(code)


def safe_path(path):
    path = Path(path)
    fail(path.is_absolute() and '..' not in path.parts)
    for current in [path, *path.parents]:
        fail(not current.is_symlink())
    return path


def path_present(path):
    return path.exists() or path.is_symlink()


def file_state(path):
    path = safe_path(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {'path': str(path), 'exists': False}
    fail(stat.S_ISREG(info.st_mode))
    return {'path': str(path), 'exists': True, 'uid': info.st_uid,
            'mode': oct(stat.S_IMODE(info.st_mode)), 'dev': info.st_dev,
            'ino': info.st_ino, 'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
            'ctime_ns': info.st_ctime_ns, 'sha256': sha_file(path), 'symlink': False}


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def cas_write(path, payload, expected, mode=0o600):
    """Cooperative CAS; caller holds deployment lock and a confirmed quiet window."""
    path = safe_path(path)
    fail(file_state(path) == expected)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        fail(file_state(path) == expected)
        if expected['exists']:
            os.replace(temporary, path)
        else:
            # link is an atomic create-if-absent, unlike replace(empty hash).
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                raise DeployError('CONFIG_CONFLICT') from None
            temporary.unlink()
        fsync_directory(path.parent)
        actual = file_state(path)
        fail(actual['sha256'] == sha_bytes(payload) and actual['mode'] == oct(mode)
             and actual['uid'] == os.getuid())
        return actual
    finally:
        temporary.unlink(missing_ok=True)


def json_payload(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def private_json(path, value):
    path = safe_path(path)
    payload = json_payload(value)
    return cas_write(path, payload, file_state(path))


def field_value(document, name):
    value = document
    for key in name.split('.'):
        if not isinstance(value, dict) or key not in value:
            return False, None
        value = value[key]
    return True, value


def patch_fields(payload, changes, reverse=False):
    # tomlkit is already installed and covered by the existing v2 environment.
    import tomlkit
    original = tomllib.loads(payload.decode())
    document = tomlkit.parse(payload.decode())
    for change in changes:
        present, value = field_value(original, change['field'])
        expected_present = True if reverse else change['before_exists']
        expected = change['after_value'] if reverse else change['before_value']
        fail(present == expected_present and (not present or (type(value) is type(expected) and value == expected)))
        keys = change['field'].split('.')
        target = document
        for key in keys[:-1]:
            if key not in target:
                target[key] = tomlkit.table()
            fail(hasattr(target[key], 'keys'))
            target = target[key]
        if reverse and not change['before_exists']:
            comment = target.item(keys[-1]).trivia.comment
            del target[keys[-1]]
            if comment:
                target.add(tomlkit.comment(comment.lstrip('#').lstrip()))
            # Remove only newly introduced empty tables, never user siblings.
            for length in range(len(keys) - 1, 0, -1):
                parent = document
                for key in keys[:length - 1]:
                    parent = parent[key]
                key = keys[length - 1]
                if key in parent and not list(parent[key].keys()):
                    # A table containing only our deleted leaves was not present
                    # in before, as recorded by all its owned missing fields.
                    prefix = '.'.join(keys[:length]) + '.'
                    if all(not c['before_exists'] for c in changes if c['field'].startswith(prefix)):
                        from tomlkit.items import Comment
                        comments = [copy.deepcopy(item) for _, item in parent[key].value.body if isinstance(item, Comment)]
                        del parent[key]
                        # Preserve later user comments, without leaving a new
                        # empty table that an older schema may reject.
                        for item in comments:
                            parent.add(item)
        else:
            target[keys[-1]] = change['before_value'] if reverse else change['after_value']
    encoded = tomlkit.dumps(document).encode()
    parsed = tomllib.loads(encoded.decode())
    for change in changes:
        present, value = field_value(parsed, change['field'])
        expected_present = change['before_exists'] if reverse else True
        expected = change['before_value'] if reverse else change['after_value']
        fail(present == expected_present and (not present or value == expected))
    return encoded


class Deadline:
    def __init__(self, seconds, clock=time.monotonic):
        self.clock = clock
        self.end = clock() + seconds

    def remaining(self, maximum=None):
        value = self.end - self.clock()
        fail(value > 0, 'DEADLINE_EXCEEDED')
        return min(value, maximum) if maximum is not None else value

    def check(self):
        self.remaining()


def kill_command_group(process):
    # Popen has not reaped this child, so its PID is still reserved by the
    # kernel, even if the leader has exited and descendants retain the pipes.
    # Do not poll()/wait() before signalling: those would release that anchor.
    # A previously reaped leader supplies no authority over a numeric PGID.
    fail(process.returncode is None and type(process.pid) is int and process.pid > 1,
         'PROCESS_IDENTITY_UNVERIFIED')
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=2)


def run_command(arguments, deadline, maximum=15, **kwargs):
    timeout = deadline.remaining(maximum)
    # These standalone control entries own all child waits. SIG_IGN or an
    # external SIGCHLD reaper would invalidate the unreaped-child reservation.
    fail(signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL, 'PROCESS_IDENTITY_UNVERIFIED')
    process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True, **kwargs)
    completed = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        completed = True
        deadline.check()
        fail(len(stdout) + len(stderr) <= 8 * 1024 * 1024, 'OUTPUT_BUDGET_EXCEEDED')
        return subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
    except BaseException:
        # communicate() has already waited/reaped after successful pipe EOF.
        # A later deadline/output rejection must never signal its former group.
        if not completed:
            kill_command_group(process)
        raise


class Journal:
    def __init__(self, directory, release_id, fault=None, deadline=None):
        self.directory = safe_path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fail(self.directory.stat().st_uid == os.getuid() and stat.S_IMODE(self.directory.stat().st_mode) == 0o700)
        self.path = self.directory / 'events.jsonl'
        self.release_id, self.fault, self.deadline = release_id, fault, deadline
        self.events = []
        self.tail = b''
        if self.path.exists():
            state = file_state(self.path)
            fail(state['uid'] == os.getuid() and state['mode'] == '0o600', 'JOURNAL_INVALID')
            data = self.path.read_bytes()
            lines = data.splitlines(keepends=True)
            for index, raw in enumerate(lines):
                if not raw.endswith(b'\n') and index == len(lines) - 1:
                    self.tail = raw
                    break
                try:
                    event = json.loads(raw)
                    checksum = event.pop('event_sha256')
                    fail(checksum == sha_bytes(json.dumps(event, sort_keys=True, separators=(',', ':')).encode()), 'JOURNAL_INVALID')
                    fail(event['release_id'] == release_id and event['sequence'] == len(self.events) + 1
                         and event['previous_sha256'] == (self.events[-1]['event_sha256'] if self.events else None), 'JOURNAL_INVALID')
                    event['event_sha256'] = checksum
                    self.events.append(event)
                except (ValueError, KeyError, TypeError):
                    raise DeployError('JOURNAL_INVALID') from None

    def latest(self, step):
        return next((event for event in reversed(self.events) if event['step'] == step), None)

    def view(self):
        names = ['revalidate_and_lock','record_prestate_and_backups','publish_release','clone_private_runtime_and_bind',
                 'relocated_environment_and_input_lock','ownership_and_manual_instance_check','config_field_cas',
                 'install_unique_plist','stop_owned_manual_if_any','bootstrap_owned_job','readiness_and_readback',
                 'client_reconnect_and_controlled_restart','confirm_takeover','login_cycle']
        names += sorted({e['step'] for e in self.events} - set(names))
        terminal = self.latest('terminal')
        return {'release_id':self.release_id,'sequence':len(self.events),
                'state':'unconfirmed' if terminal else 'applying' if self.events else 'planned',
                'steps':[{'step':name,'state':(self.latest(name) or {}).get('state','planned')} for name in names]}

    def record(self, step, state, **values):
        try:
            if self.deadline:
                self.deadline.check()
            if self.fault:
                self.fault(state, step)
            if self.tail:
                tail_path = self.directory / ('tail-' + uuid.uuid4().hex + '.bin')
                cas_write(tail_path, self.tail, file_state(tail_path))
                current = self.path.read_bytes()
                fail(current.endswith(self.tail), 'JOURNAL_INVALID')
                cas_write(self.path, current[:-len(self.tail)], file_state(self.path))
                self.tail = b''
            previous = self.latest(step) or {}
            event = {'sequence': len(self.events) + 1,
                     'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'release_id': self.release_id, 'attempt_id': values.pop('attempt_id', previous.get('attempt_id', uuid.uuid4().hex)),
                     'step': step, 'state': state, 'previous_sha256': self.events[-1]['event_sha256'] if self.events else None}
            for key in ['before', 'cas_expected', 'observed_after', 'error', 'recovery']:
                event[key] = values.pop(key, previous.get(key))
            fail(not values, 'JOURNAL_INVALID')
            event['event_sha256'] = sha_bytes(json.dumps(event, sort_keys=True, separators=(',', ':')).encode())
            payload = (json.dumps(event, sort_keys=True, separators=(',', ':')) + '\n').encode()
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'ab') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            fsync_directory(self.directory)
            self.events.append(event)
            private_json(self.directory / 'state.json', self.view())
            if self.deadline:
                self.deadline.check()
            return event
        except DeployError:
            raise
        except Exception:
            raise DeployError('JOURNAL_WRITE_FAILED') from None


def verify_package(package):
    package = safe_path(package)
    seal_path = package / 'package-seal.json'
    checksum = sha_file(seal_path)
    fail(checksum == (package / 'package-seal.sha256').read_text().strip(), 'INPUT_LOCK_CHANGED')
    seal = json.loads(seal_path.read_text())
    for name, expected in seal['files'].items():
        relative = Path(name)
        fail(not relative.is_absolute() and '..' not in relative.parts, 'INPUT_LOCK_CHANGED')
        path = safe_path(package / relative)
        fail(path.is_file() and sha_file(path) == expected, 'INPUT_LOCK_CHANGED')
    return checksum


def copy_owned_tree(source, target, files, release_id, deadline):
    source, target = safe_path(source), safe_path(target)
    marker = target / '.deployment-owner.json'
    if target.exists():
        fail(marker.is_file() and json.loads(marker.read_text()) == {'release_id': release_id})
    else:
        target.mkdir(mode=0o700, parents=True, exist_ok=False)
        private_json(marker, {'release_id': release_id})
    expected_members = set(files) | {'.deployment-owner.json'}
    actual_members = {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file() or p.is_symlink()}
    fail(actual_members <= expected_members)
    for relative, expected in files.items():
        deadline.check()
        rel = Path(relative)
        fail(not rel.is_absolute() and '..' not in rel.parts)
        src, dst = safe_path(source / rel), safe_path(target / rel)
        checksum = expected if isinstance(expected, str) else expected['sha256']
        fail(src.is_file() and sha_file(src) == checksum, 'INPUT_LOCK_CHANGED')
        dst.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if dst.exists():
            fail(sha_file(dst) == checksum)
            continue
        temporary = dst.with_name('.copy-' + uuid.uuid4().hex)
        try:
            copy_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with src.open('rb') as incoming, os.fdopen(copy_fd, 'wb') as outgoing:
                while block := incoming.read(1024 * 1024):
                    deadline.check()
                    outgoing.write(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            fail(sha_file(temporary) == checksum, 'INPUT_LOCK_CHANGED')
            os.link(temporary, dst, follow_symlinks=False)
            temporary.unlink()
            fsync_directory(dst.parent)
        finally:
            temporary.unlink(missing_ok=True)
    return {'release_id': release_id, 'files': len(files)}


class Deployment:
    def __init__(self, package, backend=None, fault=None, seconds=None):
        self.package = safe_path(package)
        self.manifest = json.loads((self.package / 'release-manifest.json').read_text())
        self.owner = json.loads((self.package / 'ownership.json').read_text())
        self.release_id = self.manifest['release_id']
        self.config = safe_path(self.manifest['config'])
        self.plist = safe_path(self.manifest['plist'])
        self.release = safe_path(self.manifest['target_release'])
        self.runtime = safe_path(self.manifest['target_runtime'])
        self.journal_dir = safe_path(self.manifest['journal_directory'])
        self.plist_bytes = (self.package / 'local.codex-memory-rag.plist').read_bytes()
        self.plist_value = plistlib.loads(self.plist_bytes)
        self.backend = backend or RealBackend()
        self.fault, self.seconds, self.journal = fault, seconds, None
        self.deadline = None

    def authorize(self, path, operation):
        fail(path is not None, 'AUTHORIZATION_REQUIRED')
        state = file_state(Path(path))
        fail(state['exists'] and state['uid'] == os.getuid() and state['mode'] == '0o600', 'AUTHORIZATION_REQUIRED')
        approval = json.loads(Path(path).read_text())
        checksum = verify_package(self.package)
        expected_paths = {key: self.manifest[key] for key in ['config', 'plist', 'target_release', 'target_runtime', 'journal_directory']}
        fail(approval.get('approved') is True and approval.get('release_id') == self.release_id
             and approval.get('package_seal_sha256') == checksum and approval.get('config_edit_quiet') is True
             and operation in approval.get('operations', []) and approval.get('paths') == expected_paths, 'AUTHORIZATION_REQUIRED')
        self.approval = approval

    @contextlib.contextmanager
    def locked(self, operation, authorization):
        self.journal = None
        self._locked = False
        self._commit_started = False
        self.authorize(authorization, operation)
        self.deadline = Deadline(self.seconds if self.seconds is not None else self.manifest['limits_seconds']['rollback' if operation == 'rollback' else 'full_takeover'])
        # A sibling lock serializes all versions that share the same journal root.
        parent = self.journal_dir.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = parent / '.deployment.lock'
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise DeployError('DEPLOYMENT_BUSY') from None
            self._locked = True
            self.journal = Journal(self.journal_dir, self.release_id, self.fault, self.deadline)
            try:
                yield
            except BaseException as exc:
                self.fail_receipt(exc, operation)
                raise
        finally:
            self._locked = False
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def step(self, name, observe, action, matches):
        self.deadline.check()
        prior = self.journal.latest(name)
        actual = observe()
        if prior and prior['state'] == 'verified':
            fail(matches(actual), 'CONFIG_CONFLICT')
            return actual
        if prior and prior['state'] in {'pending', 'applied', 'failed'}:
            if matches(actual):
                self.journal.record(name, 'applied', observed_after=actual)
                self.journal.record(name, 'verified', observed_after=actual)
                return actual
            partial_copy = name in {'publish_release','clone_private_runtime_and_bind'} and isinstance(actual, dict) and actual.get('owned') is True
            fail(actual == prior['before'] or partial_copy, 'CONFIG_CONFLICT')
        else:
            self.journal.record(name, 'pending', before=actual, cas_expected={'desired': name}, recovery={'completed': [], 'remaining': [name]})
        action(actual)
        self.deadline.check()
        actual = observe()
        fail(matches(actual), 'READBACK_FAILED')
        self.journal.record(name, 'applied', observed_after=actual)
        self.journal.record(name, 'verified', observed_after=actual, recovery={'completed': [name], 'remaining': []})
        return actual

    def check_step(self, name, callback):
        self.journal.record(name, 'pending')
        result = callback()
        self.deadline.check()
        self.journal.record(name, 'applied', observed_after=result)
        self.journal.record(name, 'verified', observed_after=result)
        return result

    def assert_owned_job(self, job):
        fail(job is not None and job.get('path') == str(self.plist)
             and job.get('arguments') == self.plist_value['ProgramArguments'])
        if job.get('pid') is not None:
            fail(type(job['pid']) is int and job['pid'] > 1 and isinstance(job.get('identity'), dict) and job['identity']['pid'] == job['pid'])
        state = file_state(self.plist)
        fail(state['exists'] and state['uid'] == os.getuid() and state['mode'] == '0o600'
             and state['sha256'] == sha_bytes(self.plist_bytes))
        return True

    def check_config(self, reverse=False):
        state = file_state(self.config)
        fail(state['exists'] and state['uid'] == os.getuid())
        if reverse:
            fail(state['mode'] == '0o600')
            patch_fields(self.config.read_bytes(), self.owner['fields'], reverse=True)
        else:
            expected = self.owner['config_before']
            fail(all(state.get(key) == expected.get(key) for key in ['exists','uid','mode','dev','ino','mtime_ns','ctime_ns','sha256']))
            patch_fields(self.config.read_bytes(), self.owner['fields'])
        return state

    def config_is_after(self, state):
        return state.get('sha256') == self.owner['config_proposed_sha256'] and state.get('mode') == '0o600'

    def old_binding(self):
        """Read old artifacts only; absence, identity and bytes are all bound."""
        try:
            root = safe_path(tomllib.loads((self.package / 'config.before.toml').read_text())['storage']['runtime_dir'])
            names = ['data/manifests/active.json', 'model.lock.json', 'grammar.lock.json', 'reranker.lock.json']
            before = {name:file_state(root / name) for name in names}
            fail(all(before[name]['exists'] for name in names[:3]), 'OLD_BINDING_CHANGED')
            fail(all(not item['exists'] or item['uid'] == os.getuid() for item in before.values()), 'OLD_BINDING_CHANGED')
            active = json.loads((root / names[0]).read_text())
            after = {name:file_state(root / name) for name in names}
            fail(before == after, 'OLD_BINDING_CHANGED')
            actual = {'runtime':str(root), 'files':after, 'active':active}
            fail(actual == self.owner['old_binding_expected'] and active == self.owner['old_active'], 'OLD_BINDING_CHANGED')
            return actual
        except (OSError, ValueError, KeyError, TypeError, DeployError):
            raise DeployError('OLD_BINDING_CHANGED') from None

    def check_old_binding(self):
        actual = self.old_binding()
        path = self.journal_dir / 'execution-prestate.json'
        record = self.journal.latest('record_prestate_and_backups')
        if path.exists():
            try:
                identity = file_state(path)
                fail(identity['uid'] == os.getuid() and identity['mode'] == '0o600', 'PRESTATE_INVALID')
                value = json.loads(path.read_text())
                expected = {'config':self.owner['config_before'], 'plist':{'exists':False}, 'job':None,
                            'old_active':actual['active'], 'old_binding':actual}
                fail(value == expected, 'PRESTATE_INVALID')
                digest = (record or {}).get('observed_after') or {}
                if digest.get('prestate_sha256'):
                    fail(sha_file(path) == digest['prestate_sha256'], 'PRESTATE_INVALID')
            except (OSError, ValueError, KeyError, TypeError):
                raise DeployError('PRESTATE_INVALID') from None
        else:
            # An interrupted backup may be retried, but a missing receipt after
            # publication must not be reconstructed from the frozen proposal.
            fail(not self.journal.latest('publish_release'), 'PRESTATE_INVALID')
        return actual

    def directory_observation(self, path):
        marker = path / '.deployment-owner.json'
        if not path.exists():
            return {'exists': False}
        fail(marker.is_file() and json.loads(marker.read_text()) == {'release_id': self.release_id})
        return {'exists': True, 'owned': True}

    def terminal(self, state):
        self.deadline.check()
        event = self.journal.record('terminal', 'verified', observed_after={'state': state, 'login_startup': 'NOT_RUN'})
        result = {'state': state, 'release_id': self.release_id, 'sequence': event['sequence'],
                  'event_sha256': event['event_sha256'], 'login_startup': 'NOT_RUN',
                  'commit_rule':'deadline_checked_before_atomic_publication'}
        staged = self.journal_dir / ('terminal.candidate-' + uuid.uuid4().hex + '.json')
        private_json(staged, result)
        fail(json.loads(staged.read_text()) == result, 'READBACK_FAILED')
        self.deadline.check()
        # The atomic publication is the single decision point. All potentially
        # slow preparation/fsync/readback precedes deadline admission. Never
        # reverse a published success solely because the clock advanced during
        # a kernel rename/fsync; a publication I/O error is explicitly unknown.
        self._commit_started = True
        try:
            os.replace(staged, self.journal_dir / 'terminal.json')
            fsync_directory(self.journal_dir)
        except Exception:
            raise DeployError('COMMIT_CONFIRM_UNKNOWN') from None
        return result

    def fail_receipt(self, exc, operation):
        if not self.journal or not getattr(self, '_locked', False):
            return
        if getattr(self, '_commit_started', False):
            with contextlib.suppress(Exception):
                private_json(self.journal_dir / 'commit-observation-error.json', {'state':'commit_unknown','release_id':self.release_id})
            return
        # Never overwrite the WAL with a guessed rollback. A separate failure
        # receipt invalidates an earlier success acknowledgement.
        with contextlib.suppress(Exception):
            private_json(self.journal_dir / 'terminal.json', {'state': 'partial_rollback' if operation == 'rollback' else 'failed',
                         'release_id': self.release_id, 'error': str(exc) if isinstance(exc, DeployError) else type(exc).__name__,
                         'last_sequence': len(self.journal.events), 'recovery_required': True})

    def apply(self, authorization, resume=False):
        operation = 'resume' if resume else 'apply'
        try:
            with self.locked(operation, authorization):
                old_binding = self.check_old_binding()
                job = self.backend.job(self)
                prior_boot = self.journal.latest('bootstrap_owned_job')
                if job:
                    fail(resume and prior_boot is not None)
                    self.assert_owned_job(job)
                elif self.plist.exists():
                    fail(resume and self.journal.latest('install_unique_plist') is not None)
                    fail(file_state(self.plist)['sha256'] == sha_bytes(self.plist_bytes))
                if not self.journal.latest('config_field_cas'):
                    self.check_config()
                else:
                    state = file_state(self.config)
                    fail(self.config_is_after(state) or state['sha256'] == self.owner['config_before']['sha256'])
                self.check_step('revalidate_and_lock', lambda: {'package_hash':verify_package(self.package),'owned_job':job,'old_binding':old_binding})
                self.check_step('ownership_and_manual_instance_check', lambda: self.backend.preflight(self))
                self.check_step('stop_owned_manual_if_any', lambda: {'applicable':False,'reason':'this release requires no manual instance; changed baseline conflicts'})
                def backup_before():
                    current_binding = self.check_old_binding()
                    backup = self.journal_dir / 'config.before.toml'
                    expected = (self.package / 'config.before.toml').read_bytes()
                    if not backup.exists():
                        cas_write(backup, expected, file_state(backup))
                    fail(sha_file(backup) == sha_bytes(expected))
                    path = self.journal_dir / 'execution-prestate.json'
                    if not path.exists():
                        private_json(path, {'config':self.owner['config_before'],'plist':{'exists':False},'job':None,
                                             'old_active':current_binding['active'],'old_binding':current_binding})
                    return {'backup_sha256':sha_file(backup),'prestate_sha256':sha_file(path)}
                self.check_step('record_prestate_and_backups', backup_before)
                fail(not self.release.exists() or self.journal.latest('publish_release') is not None)
                fail(not self.runtime.exists() or self.journal.latest('clone_private_runtime_and_bind') is not None)
                self.step('publish_release', lambda: self.directory_observation(self.release),
                          lambda before: copy_owned_tree(self.package / 'release', self.release, self.manifest['payload'], self.release_id, self.deadline),
                          lambda actual: actual.get('owned') is True and all((self.release / f).is_file() and sha_file(self.release / f) == h for f, h in self.manifest['payload'].items()))
                files = json.loads((self.package / self.manifest['seed']['copy_manifest']).read_text())['files']
                def copy_seed(before):
                    manager = self.backend.seed_lease(self) if hasattr(self.backend, 'seed_lease') else contextlib.nullcontext()
                    with manager:
                        return copy_owned_tree(Path(self.manifest['seed']['runtime']), self.runtime, files, self.release_id, self.deadline)
                def clone_matches(actual):
                    if actual.get('owned') is not True:
                        return False
                    copied = self.journal.latest('clone_private_runtime_and_bind')
                    if copied and copied['state'] == 'verified' and self.journal.latest('bootstrap_owned_job'):
                        # The owned live writer may legitimately advance data.
                        return all(sha_file(self.runtime / name) == expected for name, expected in self.manifest['seed'].get('resources', {}).items())
                    return all((self.runtime / f).is_file() and sha_file(self.runtime / f) == (h if isinstance(h, str) else h['sha256']) for f, h in files.items())
                self.step('clone_private_runtime_and_bind', lambda: self.directory_observation(self.runtime),
                          copy_seed, clone_matches)
                binding_path = self.journal_dir / 'binding.json'
                self.step('relocated_environment_and_input_lock', lambda: json.loads(binding_path.read_text()) if binding_path.exists() else None,
                          lambda before: private_json(binding_path, self.backend.bind(self)), lambda actual: isinstance(actual, dict) and bool(actual.get('input_hash')) and actual.get('source_verified') is True)
                def apply_config(before):
                    self.check_config()
                    return cas_write(self.config, patch_fields(self.config.read_bytes(), self.owner['fields']), before)
                self.step('config_field_cas', lambda: file_state(self.config), apply_config, self.config_is_after)
                self.plist.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                self.step('install_unique_plist', lambda: file_state(self.plist),
                          lambda before: cas_write(self.plist, self.plist_bytes, before),
                          lambda actual: actual.get('sha256') == sha_bytes(self.plist_bytes) and actual.get('mode') == '0o600')
                if not self.backend.isolated:
                    checked = self.backend.command(self, [self.manifest['python'], '-B', str(self.release / 'control/deploy_transaction.py'), '--verify-parent'], maximum=120, cwd=self.release)
                    fail(checked.returncode == 0, 'ENVIRONMENT_CHANGED')
                self.check_old_binding()
                self.step('bootstrap_owned_job', lambda: self.backend.job(self), lambda before: self.backend.bootstrap(self),
                          lambda actual: bool(actual) and self.assert_owned_job(actual))
                last_ready_job = self.backend.job(self)
                ready = self.check_step('readiness_and_readback', lambda: self.backend.ready(self, last_ready_job))
                fail(ready.get('guard_verified') is True and ready.get('sources_ready') is True and ready.get('readbacks') == 2, 'NOT_READY')
                previous = self.journal.latest('client_reconnect_and_controlled_restart')
                if not previous or previous['state'] != 'verified':
                    before = previous['before'] if previous else self.backend.job(self)
                    if not previous:
                        self.journal.record('client_reconnect_and_controlled_restart', 'pending', before=before)
                    current = self.backend.job(self)
                    self.assert_owned_job(current)
                    if current['identity'] == before['identity']:
                        self.backend.restart(self, before)
                    after = self.backend.job(self)
                    fail(after and after['identity'] != before['identity'], 'RESTART_NOT_CONFIRMED')
                    ready = self.backend.ready(self, after)
                    fail(ready.get('guard_verified') is True and ready.get('readbacks') == 2, 'NOT_READY')
                    self.journal.record('client_reconnect_and_controlled_restart', 'verified', observed_after=after)
                    last_ready_job = after
                def confirm():
                    self.check_old_binding()
                    if hasattr(self.backend, 'final_verify'):
                        self.backend.final_verify(self, last_ready_job)
                    current = self.backend.job(self)
                    fail(current is not None and current.get('pid') is not None and current.get('identity') == last_ready_job.get('identity')
                         and ready.get('pid') == current.get('pid'), 'FINAL_INSTANCE_CHANGED')
                    self.assert_owned_job(current)
                    return {'job':current,'login_startup':'NOT_RUN'}
                self.check_step('confirm_takeover', confirm)
                return self.terminal('ready')
        except BaseException as exc:
            self.fail_receipt(exc, operation)
            raise

    def resume(self, authorization):
        return self.apply(authorization, resume=True)

    def rollback(self, authorization):
        try:
            with self.locked('rollback', authorization):
                fail(bool(self.journal.events), 'NO_DEPLOYMENT_RECORD')
                self.check_old_binding()
                config_state = file_state(self.config)
                config_record = self.journal.latest('config_field_cas')
                restored_record = self.journal.latest('rollback_config')
                parsed_now = tomllib.loads(self.config.read_text())
                is_before = all(field_value(parsed_now, c['field']) == (c['before_exists'], c['before_value']) for c in self.owner['fields'])
                can_be_before = restored_record is not None or config_record is None or config_record['state'] == 'pending'
                config_changed = not (is_before and can_be_before and config_state['mode'] == self.owner['config_before']['mode'])
                if config_changed:
                    self.check_config(reverse=True)
                elif restored_record and restored_record['state'] != 'verified':
                    self.journal.record('rollback_config', 'verified', observed_after=config_state)
                plist_state = file_state(self.plist)
                if plist_state['exists']:
                    fail(plist_state['sha256'] == sha_bytes(self.plist_bytes) and plist_state['uid'] == os.getuid() and plist_state['mode'] == '0o600')
                job = self.backend.job(self)
                if job:
                    self.assert_owned_job(job)
                    self.journal.record('rollback_stop', 'pending', before=job)
                    self.backend.stop(self, job)
                    fail(self.backend.job(self) is None and self.backend.clean(self, job), 'CLEANUP_INCOMPLETE')
                    self.journal.record('rollback_stop', 'verified', observed_after={'job':None,'clean':True})
                else:
                    prior = self.journal.latest('rollback_stop') or self.journal.latest('bootstrap_owned_job')
                    known = prior.get('before') if prior and prior['step'] == 'rollback_stop' else prior.get('observed_after') if prior else None
                    fail(self.backend.clean(self, known), 'CLEANUP_INCOMPLETE')
                    if prior and prior['step'] == 'rollback_stop' and prior['state'] != 'verified':
                        self.journal.record('rollback_stop', 'verified', observed_after={'job':None,'clean':True})
                if plist_state['exists']:
                    self.journal.record('rollback_plist', 'pending', before=plist_state)
                    fail(file_state(self.plist) == plist_state)
                    self.plist.unlink()
                    fsync_directory(self.plist.parent)
                    fail(not self.plist.exists())
                    self.journal.record('rollback_plist', 'verified', observed_after={'exists':False})
                elif self.journal.latest('rollback_plist') and self.journal.latest('rollback_plist')['state'] != 'verified':
                    self.journal.record('rollback_plist', 'verified', observed_after={'exists':False})
                if config_changed:
                    before = file_state(self.config)
                    payload = patch_fields(self.config.read_bytes(), self.owner['fields'], reverse=True)
                    self.journal.record('rollback_config', 'pending', before=before, cas_expected={'sha256':sha_bytes(payload)})
                    cas_write(self.config, payload, before, int(self.owner['config_before']['mode'], 8))
                    self.journal.record('rollback_config', 'verified', observed_after=file_state(self.config))
                parsed = tomllib.loads(self.config.read_text())
                for change in self.owner['fields']:
                    present, value = field_value(parsed, change['field'])
                    fail(present == change['before_exists'] and (not present or value == change['before_value']))
                fail(self.backend.job(self) is None and not self.plist.exists())
                observed = self.check_old_binding()
                self.check_step('rollback_old_binding', lambda: {'old_binding':observed})
                return self.terminal('rolled_back')
        except BaseException as exc:
            self.fail_receipt(exc, 'rollback')
            raise


# RealBackend and guarded entry support follow below. No fake backend is exposed
# through the command line; tests supply one by constructor dependency injection.

def terminal_status(directory, release_id):
    try:
        fail(Path(directory).is_dir(), 'TERMINAL_UNCONFIRMED')
        journal = Journal(directory, release_id)
        receipt = json.loads((Path(directory) / 'terminal.json').read_text())
        event = journal.latest('terminal')
        fail(event is not None and not journal.tail and event['sequence'] == len(journal.events)
             and receipt.get('sequence') == event['sequence'] and receipt.get('event_sha256') == event['event_sha256']
             and receipt.get('state') == event['observed_after']['state'], 'TERMINAL_UNCONFIRMED')
        return receipt
    except (OSError, ValueError, KeyError, DeployError):
        return {'state':'unconfirmed','release_id':release_id}

def process_identity(pid, deadline=None):
    if type(pid) is not int or pid <= 1:
        return None
    result = run_command(['/bin/ps', '-p', str(pid), '-o', 'uid=', '-o', 'pgid=', '-o', 'lstart=', '-o', 'command='], deadline or Deadline(5), maximum=3)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parts = result.stdout.decode().strip().split(maxsplit=7)
    fail(len(parts) == 8 and int(parts[0]) == os.getuid(), 'PROCESS_IDENTITY_UNVERIFIED')
    return {'pid': pid, 'uid': int(parts[0]), 'pgid': int(parts[1]), 'start': ' '.join(parts[2:7]),
            'command_sha256': sha_bytes(parts[7].encode())}


def group_members(pgid, deadline):
    result = run_command(['/bin/ps','-axo','pid=,pgid=,uid=,stat=,lstart='], deadline, maximum=3)
    fail(result.returncode == 0, 'PROCESS_IDENTITY_UNVERIFIED')
    members = []
    for line in result.stdout.decode().splitlines():
        parts = line.split()
        if len(parts) >= 9 and int(parts[1]) == pgid and not parts[3].startswith('Z'):
            fail(int(parts[2]) == os.getuid(), 'PROCESS_IDENTITY_UNVERIFIED')
            members.append({'pid':int(parts[0]),'start':' '.join(parts[4:9])})
    return members


def cleanup_owned_group(identity, deadline, known_members=None):
    fail(isinstance(identity, dict) and identity.get('pgid') == identity.get('pid') and identity['pid'] > 1,
         'PROCESS_IDENTITY_UNVERIFIED')
    current = process_identity(identity['pid'], deadline)
    fail(current is None or current == identity, 'PROCESS_IDENTITY_UNVERIFIED')
    members = group_members(identity['pgid'], deadline)
    for member in members:
        if member['pid'] == identity['pid']:
            fail(member['start'] == identity['start'], 'PROCESS_IDENTITY_UNVERIFIED')
    if not members:
        return True
    if current is None:
        known = {(item['pid'], item['start']) for item in known_members or []}
        fail(all((item['pid'], item['start']) in known for item in members), 'PROCESS_IDENTITY_UNVERIFIED')
        # A stale numeric PGID is never sufficient authority. Signal only
        # individually rechecked descendants recorded while the leader lived.
        for member in members:
            child = process_identity(member['pid'], deadline)
            if child is not None:
                fail(child['start'] == member['start'] and child['pgid'] == identity['pgid'], 'PROCESS_IDENTITY_UNVERIFIED')
                with contextlib.suppress(ProcessLookupError):
                    os.kill(member['pid'], signal.SIGTERM)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(identity['pgid'], signal.SIGTERM)
    while deadline.remaining() > .05:
        if not group_members(identity['pgid'], deadline):
            return True
        time.sleep(.02)
    raise DeployError('CLEANUP_INCOMPLETE')


def install_offline():
    originals = {}
    for name in ['connect', 'connect_ex', 'bind', 'sendto', 'sendmsg']:
        if not hasattr(socket.socket, name):
            continue
        original = getattr(socket.socket, name)
        originals[name] = original
        def blocked(sock, *args, _original=original, **kwargs):
            fail(sock.family not in (socket.AF_INET, socket.AF_INET6), 'NETWORK_FORBIDDEN')
            return _original(sock, *args, **kwargs)
        setattr(socket.socket, name, blocked)
    original_dns = socket.getaddrinfo
    def blocked_dns(*args, **kwargs):
        raise DeployError('NETWORK_FORBIDDEN')
    socket.getaddrinfo = blocked_dns
    def restore():
        for name, original in originals.items():
            setattr(socket.socket, name, original)
        socket.getaddrinfo = original_dns
    return restore


def validate_startup_environment(deployment):
    fail(deployment.get('startup_environment') == {'PYTHONUTF8':'0'}
         and os.environ.get('PYTHONUTF8') == '0'
         and list(sys.flags) == deployment.get('startup_python_flags'), 'STARTUP_FLAGS_CHANGED')


def entry_setup(root, validate_startup=True):
    os.umask(0o077)
    root = safe_path(root)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    src = str(root / 'src')
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)
    import importlib.util
    spec = importlib.util.find_spec('codex_memory_rag')
    fail(spec is not None and spec.origin == str(root / 'src/codex_memory_rag/__init__.py'), 'INPUT_LOCK_CHANGED')
    deployment = json.loads((root / 'deployment.json').read_text())
    if validate_startup:
        validate_startup_environment(deployment)
    return deployment


def environment_check(root, role, record, full=False):
    from environment_lock import verify_environment, verify_metadata, EnvironmentDrift
    expected = json.loads((root / record['environment']).read_text())
    metadata = json.loads((root / record['metadata']).read_text())
    try:
        verify_metadata(metadata)
        if full:
            verify_environment(expected)
            verify_metadata(metadata)
    except EnvironmentDrift as exc:
        raise DeployError(str(exc)) from None
    return {'verified': True}


def guard_context(root, config, role, full=False):
    root, config = safe_path(root), safe_path(config)
    fail(role in {'parent', 'cli', 'daemon'}, 'INPUT_LOCK_CHANGED')
    try:
        deployment = json.loads((root / 'deployment.json').read_text())
        validate_startup_environment(deployment)
        lock_path = root / 'input-lock.json'
        lock = json.loads(lock_path.read_text())
        binding = json.loads((Path(deployment['journal_directory']) / 'binding.json').read_text())
        fail(sha_file(lock_path) == binding['input_hash'], 'INPUT_LOCK_CHANGED')
        fail(lock['release_id'] == deployment['release_id'] and str(config) == lock['config'] == deployment['config']
             and sha_file(config) == lock['config_sha256'] == deployment['config_sha256'], 'INPUT_LOCK_CHANGED')
        for name, expected in lock['files'].items():
            rel = Path(name)
            fail(not rel.is_absolute() and '..' not in rel.parts, 'INPUT_LOCK_CHANGED')
            fail(sha_file(safe_path(root / rel)) == expected, 'INPUT_LOCK_CHANGED')
        for name, expected in lock.get('resources', {}).items():
            fail(sha_file(safe_path(Path(deployment['target_runtime']) / name)) == expected, 'INPUT_LOCK_CHANGED')
        record = lock['roles'][role]
        for kind in ['environment', 'metadata']:
            rel = Path(record[kind])
            fail(not rel.is_absolute() and '..' not in rel.parts, 'INPUT_LOCK_CHANGED')
            fail(sha_file(safe_path(root / rel)) == record[kind + '_sha256'], 'INPUT_LOCK_CHANGED')
        environment_check(root, role, record, full=full)
        return {'role': role, 'input_hash': binding['input_hash'], 'metadata_verified': True,
                'content_verified': full, 'verified_at': time.time()}
    except (OSError, KeyError, ValueError, TypeError):
        raise DeployError('INPUT_LOCK_CHANGED') from None


def guarded_call(root, config, role, callback, full=False):
    before = guard_context(root, config, role, full=full)
    try:
        value = callback()
    finally:
        after = guard_context(root, config, role, full=full)
    return value, {'before': before, 'after': after}


def capture_role(root, role):
    deployment = entry_setup(root)
    fail(role in {'parent', 'cli', 'daemon'}, 'INPUT_LOCK_CHANGED')
    from environment_lock import capture_environment, capture_metadata, digest, EnvironmentDrift
    try:
        before = capture_metadata()
        environment = capture_environment()
        after = capture_metadata()
        fail(digest(before) == digest(after), 'ENVIRONMENT_CHANGED')
        reference = json.loads((root / 'reference-environment.json').read_text())
        # Relocation changes application paths; the installed/runtime bytes must
        # still equal the proven environment rather than accepting a new install.
        for key in ['interpreter', 'pyvenv_cfg', 'interpreter_related', 'stdlib', 'site_packages', 'distributions']:
            fail(environment[key] == reference[key], 'ENVIRONMENT_CHANGED')
        for key in ['version', 'implementation', 'cache_tag', 'architecture', 'base_prefix', 'prefix', 'os_release']:
            fail(environment['python_runtime'][key] == reference['python_runtime'][key], 'ENVIRONMENT_CHANGED')
        fail(environment['python_runtime']['application_origin'] == str(root / 'src/codex_memory_rag/__init__.py'), 'INPUT_LOCK_CHANGED')
        directory = root / 'guards'
        directory.mkdir(mode=0o700, exist_ok=True)
        record = {}
        for kind, value in [('environment', environment), ('metadata', after)]:
            path = directory / (role + '.' + kind + '.json')
            if path.exists():
                fail(json.loads(path.read_text()) == value, 'ENVIRONMENT_CHANGED')
            else:
                private_json(path, value)
            record[kind] = str(path.relative_to(root))
            record[kind + '_sha256'] = sha_file(path)
        return {'release_id': deployment['release_id'], 'role': role, 'record': record}
    except EnvironmentDrift as exc:
        raise DeployError(str(exc)) from None


def check_daemon_attestation(root, config, now=None):
    try:
        deployment = json.loads((root / 'deployment.json').read_text())
        path = Path(deployment['target_runtime']) / 'run/daemon-current.json'
        receipt = json.loads(path.read_text())
        fail(receipt['release_id'] == deployment['release_id'] and receipt['state'] == 'running'
             and receipt['input_hash'] == sha_file(root / 'input-lock.json') and receipt['content_verified'] is True
             and process_identity(receipt['identity']['pid']) == receipt['identity'], 'DAEMON_UNVERIFIED')
        now = time.time() if now is None else now
        fail(receipt['verified_at'] <= now <= receipt['expires_at'], 'GROUP_VERIFICATION_EXPIRED')
        fail(str(config) == deployment['config'], 'INPUT_LOCK_CHANGED')
        return receipt
    except (OSError, ValueError, KeyError, TypeError):
        raise DeployError('DAEMON_UNVERIFIED') from None


def allowed_cli(arguments):
    if not arguments or arguments[0] not in {'status', 'search', 'context'} or '--config' in arguments:
        return False
    # The product parser validates all remaining flags and values. Pin project
    # here as a second boundary; forbid abbreviation/duplicate override flags.
    if arguments.count('--project') != 1:
        return False
    flags = {'status':{'--project'},'search':{'--project','--kind','--query','--literal','--source','--mode','--history'},
             'context':{'--project','--chunk-id','--max-chars','--history'}}[arguments[0]]
    if any(arg.startswith('--') and arg not in flags for arg in arguments[1:]):
        return False
    index = arguments.index('--project')
    return index + 1 < len(arguments) and arguments[index + 1] == 'example-project' and not any(
        arg.startswith(('--proj=', '--project=', '--conf')) for arg in arguments)


def load_startup_intent(root):
    """Read only a controller-committed, exact-attempt launch intent."""
    try:
        deployment = json.loads((root / 'deployment.json').read_text())
        path = Path(deployment['target_runtime']) / 'run/startup-intent.json'
        state = file_state(path)
        fail(state['exists'] and state['uid'] == os.getuid() and state['mode'] == '0o600', 'STARTUP_PROOF_INVALID')
        intent = json.loads(path.read_text())
        journal = Journal(deployment['journal_directory'], deployment['release_id'])
        event = journal.latest('launch_attempt')
        observed = (event or {}).get('observed_after') or {}
        fail(event is not None and event['state'] == 'verified' and observed.get('intent') == intent
             and observed.get('intent_sha256') == state['sha256'], 'STARTUP_PROOF_INVALID')
        binding = json.loads((Path(deployment['journal_directory']) / 'binding.json').read_text())
        fail(intent['schema'] == 'startup-intent-v1' and intent['release_id'] == deployment['release_id']
             and re.fullmatch('[0-9a-f]{32}', intent['attempt_id']) is not None
             and intent['input_hash'] == binding['input_hash'] == sha_file(root / 'input-lock.json')
             and intent['config_sha256'] == deployment['config_sha256'], 'STARTUP_PROOF_INVALID')
        return intent
    except (OSError, ValueError, KeyError, TypeError):
        raise DeployError('STARTUP_PROOF_INVALID') from None


def record_prelease_failure(root, config, intent, identity, error):
    deployment = json.loads((root / 'deployment.json').read_text())
    fail(error in {'ENVIRONMENT_CHANGED','INPUT_LOCK_CHANGED','STARTUP_FLAGS_CHANGED'}, 'STARTUP_PROOF_INVALID')
    fail(load_startup_intent(root) == intent and str(config) == deployment['config']
         and sha_file(config) == intent['config_sha256'], 'STARTUP_PROOF_INVALID')
    runtime = safe_path(deployment['target_runtime'])
    fail(not path_present(runtime / 'run/daemon.lock') and not path_present(runtime / 'run/daemon.sock'), 'STARTUP_PROOF_INVALID')
    fail(type(identity.get('pid')) is int and identity['pid'] > 1 and identity.get('pgid') == identity['pid']
         and identity.get('uid') == os.getuid(), 'STARTUP_PROOF_INVALID')
    receipt = {'schema':'prelease-failure-v1','release_id':intent['release_id'],'attempt_id':intent['attempt_id'],
               'input_hash':intent['input_hash'],'config_sha256':intent['config_sha256'],
               'phase':'pre_lease_guard','state':'failed','error':error,'identity':identity,
               'instances':{'encoder':0,'reranker':0,'worker':0}}
    path = runtime / 'run' / ('startup-failure-' + intent['attempt_id'] + '.json')
    # Never replace evidence from an earlier start, even under the same intent.
    cas_write(path, json_payload(receipt), {'path':str(path),'exists':False})
    return receipt


def prelease_failure_output(exc):
    value = {'error':str(exc) if isinstance(exc, DeployError) else type(exc).__name__,
             'state':'startup_or_runtime_failed'}
    if isinstance(exc, DeployError) and hasattr(exc, 'prelease_proof'):
        value['prelease_proof'] = exc.prelease_proof
    return value


class RealBackend:
    isolated = False

    def command(self, tx, args, maximum=None, **kwargs):
        env = dict(kwargs.pop('env', os.environ))
        env.update(tx.manifest.get('startup_environment', {'PYTHONUTF8':'0'}))
        env.pop('PYTHONPATH', None)
        kwargs['env'] = env
        return run_command(args, tx.deadline, maximum or tx.manifest['limits_seconds']['launchctl_command'], **kwargs)

    def job(self, tx):
        result = self.command(tx, ['/bin/launchctl', 'print', tx.manifest['job']])
        output = result.stdout.decode()
        if result.returncode:
            fail('Could not find service' in result.stderr.decode(), 'JOB_STATE_UNKNOWN')
            return None
        path = re.search(r'^\s*path = (.+)$', output, re.M)
        pid = re.search(r'^\s*pid = (\d+)$', output, re.M)
        arguments = re.search(r'^\s*arguments = \{\n(.*?)^\s*\}', output, re.M | re.S)
        fail(path is not None and arguments is not None, 'JOB_STATE_UNKNOWN')
        state = re.search(r'^\s*state = (.+)$', output, re.M)
        runs = re.search(r'^\s*runs = (\d+)$', output, re.M)
        exited = re.search(r'^\s*last exit code = (\d+)$', output, re.M)
        details = {'state':state[1].strip() if state else None, 'runs':int(runs[1]) if runs else None,
                   'last_exit_code':int(exited[1]) if exited else None}
        if not pid:
            # A loaded but exited job is still owned state, not absence.
            return {'path':path[1].strip(), 'arguments':[x.strip() for x in arguments[1].splitlines() if x.strip()],
                    'pid':None, 'identity':None, **details}
        value = int(pid[1])
        identity = process_identity(value, tx.deadline)
        return {'path':path[1].strip(), 'arguments':[x.strip() for x in arguments[1].splitlines() if x.strip()],
                'pid':value if identity else None, 'observed_pid':value, 'identity':identity, **details,
                'members':group_members(value, tx.deadline) if identity and identity.get('pgid') == value else []}

    def preflight(self, tx):
        # This release's original state has no owned manual instance. A new busy
        # lease is a changed baseline, not permission to stop an arbitrary PID.
        before = tomllib.loads((tx.package / 'config.before.toml').read_text())
        for directory in {Path(before['storage']['runtime_dir']), Path(tx.manifest['seed']['runtime'])}:
            self.assert_idle(directory)
        # Static bindings only; unknown wrappers must be explicitly reviewed in
        # the eventual deployment approval, not silently edited here.
        fail(tx.approval.get('opaque_mcp_routes_reviewed') is True, 'MCP_ROUTE_REVIEW_REQUIRED')
        config = Path.home() / ".codex/config.toml"
        entries = tomllib.loads(config.read_text()).get('mcp_servers', {})
        for entry in entries.values():
            serialized = json.dumps(entry)
            fail(not any(token in serialized for token in ['codex_memory_rag', str(tx.runtime), str(tx.config)]), 'MCP_BINDING_CONFLICT')
        return {'manual_instance':None}

    @staticmethod
    def assert_idle(runtime):
        runtime = safe_path(runtime)
        fail(not path_present(runtime / 'run/daemon.sock'), 'MANUAL_INSTANCE_UNVERIFIED')
        lock = runtime / 'run/daemon.lock'
        if path_present(lock):
            fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise DeployError('MANUAL_INSTANCE_UNVERIFIED') from None
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @contextlib.contextmanager
    def seed_lease(self, tx):
        runtime = Path(tx.manifest['seed']['runtime'])
        self.assert_idle(runtime)
        fd = os.open(runtime / 'run/daemon.lock', os.O_RDONLY | os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise DeployError('SEED_BUSY') from None
            fail(not (runtime / 'run/daemon.sock').exists(), 'SEED_BUSY')
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def bind(self, tx):
        self.assert_idle(Path(tx.manifest['seed']['runtime']))
        # All resource copies were in the signed seed copy manifest. Open only
        # the new instance's database, never the old user's database for writes.
        (tx.runtime / 'logs').mkdir(mode=0o700, exist_ok=True)
        (tx.runtime / 'run').mkdir(mode=0o700, exist_ok=True)
        active = json.loads((tx.runtime / 'data/manifests/active.json').read_text())
        fail(active['generation_id'] == tx.manifest['seed']['generation'], 'SNAPSHOT_CHANGED')
        fail(active['embedding_profile_id'] == tx.manifest['profiles']['embedding_profile']
             and active['index_profile_id'] == tx.manifest['profiles']['index_profile'], 'INDEX_PROFILE_MISMATCH')
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        records = {}
        for role in ['parent', 'cli', 'daemon']:
            entry = 'deploy_transaction.py' if role == 'parent' else role + '_entry.py'
            result = self.command(tx, [tx.manifest['python'], '-B', str(tx.release / 'control' / entry), '--capture-environment'],
                                  maximum=tx.manifest['limits_seconds']['environment_full_scan'], cwd=tx.release, env=env)
            fail(result.returncode == 0, 'ENVIRONMENT_BIND_FAILED')
            record = json.loads(result.stdout)
            fail(record['role'] == role and record['release_id'] == tx.release_id, 'ENVIRONMENT_BIND_FAILED')
            records[role] = record['record']
        lock = {'release_id':tx.release_id, 'config':str(tx.config), 'config_sha256':tx.owner['config_proposed_sha256'],
                'files':tx.manifest['payload'], 'roles':records, 'resources':tx.manifest['seed']['resources']}
        path = tx.release / 'input-lock.json'
        if path.exists():
            fail(json.loads(path.read_text()) == lock, 'INPUT_LOCK_CHANGED')
        else:
            private_json(path, lock)
        # Current source qualification, corpus membership and actual table are
        # verified in a local no-model helper using the published application.
        result = self.command(tx, [tx.manifest['python'], '-B', str(tx.release / 'control/deploy_transaction.py'), '--verify-seed'],
                              maximum=120, cwd=tx.release, env=env)
        fail(result.returncode == 0, 'SEED_OR_CORPUS_CHANGED')
        verified = json.loads(result.stdout)
        return {'input_hash':sha_file(path), 'source_verified':verified['source_verified'],
                'actual_table_version':verified['actual_table_version'], 'roles':records, 'isolated':False}

    def bootstrap(self, tx):
        self.assert_idle(tx.runtime)
        self.prepare_launch_attempt(tx, 'bootstrap')
        result = self.command(tx, ['/bin/launchctl', 'bootstrap', tx.manifest['job'].rsplit('/',1)[0], str(tx.plist)])
        fail(result.returncode == 0, 'BOOTSTRAP_RESULT_UNKNOWN')

    def stop(self, tx, job):
        tx.assert_owned_job(job)
        result = self.command(tx, ['/bin/launchctl', 'bootout', tx.manifest['job']])
        fail(result.returncode == 0 or self.job(tx) is None, 'BOOTOUT_RESULT_UNKNOWN')

    def prepare_launch_attempt(self, tx, reason):
        binding = json.loads((tx.journal_dir / 'binding.json').read_text())
        fail(binding['input_hash'] == sha_file(tx.release / 'input-lock.json'), 'INPUT_LOCK_CHANGED')
        path = tx.runtime / 'run/startup-intent.json'
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        before = file_state(path)
        prior = tx.journal.latest('launch_attempt')
        if before['exists']:
            fail(before['uid'] == os.getuid() and before['mode'] == '0o600' and prior is not None, 'CONFIG_CONFLICT')
        if prior and prior['state'] == 'pending':
            intent = prior['recovery']['intent']
            fail(intent['reason'] == reason and intent['input_hash'] == binding['input_hash']
                 and intent['config_sha256'] == tx.owner['config_proposed_sha256'], 'INPUT_LOCK_CHANGED')
            if before.get('sha256') != prior['cas_expected']['sha256']:
                fail(before == prior['before'], 'CONFIG_CONFLICT')
                cas_write(path, json_payload(intent), before)
            tx.journal.record('launch_attempt','verified',observed_after={'intent':intent,'intent_sha256':sha_file(path)})
            return intent
        if prior:
            fail(prior['state'] == 'verified' and before.get('sha256') == prior['observed_after']['intent_sha256'], 'CONFIG_CONFLICT')
        intent = {'schema':'startup-intent-v1','release_id':tx.release_id,'attempt_id':uuid.uuid4().hex,
                  'reason':reason,'input_hash':binding['input_hash'],'config_sha256':tx.owner['config_proposed_sha256']}
        tx.journal.record('launch_attempt','pending',before=before,cas_expected={'sha256':sha_bytes(json_payload(intent))},
                          observed_after=None,recovery={'intent':intent})
        cas_write(path,json_payload(intent),before)
        tx.journal.record('launch_attempt','verified',observed_after={'intent':intent,'intent_sha256':sha_file(path)})
        return intent

    def prove_prelease_exit(self, tx, job):
        try:
            fail(job is not None and job.get('path') == str(tx.plist) and job.get('arguments') == tx.plist_value['ProgramArguments']
                 and job.get('pid') is None and job.get('identity') is None and not job.get('observed_pid')
                 and job.get('state') == 'not running' and type(job.get('runs')) is int and job['runs'] == 1
                 and type(job.get('last_exit_code')) is int and job['last_exit_code'] == 0, 'PROCESS_IDENTITY_UNVERIFIED')
            fail(self.job(tx) is None, 'PROCESS_IDENTITY_UNVERIFIED')
            intent = load_startup_intent(tx.release)
            fail(intent['reason'] == 'bootstrap', 'PROCESS_IDENTITY_UNVERIFIED')
            # This exception applies only when there has never been a running
            # instance. A recorded live instance always uses normal owned cleanup.
            for event in tx.journal.events:
                for value in [event.get('before'), event.get('observed_after')]:
                    if isinstance(value, dict):
                        value = value.get('job', value)
                        fail(not (isinstance(value, dict) and value.get('path') == str(tx.plist)
                                  and value.get('identity')), 'PROCESS_IDENTITY_UNVERIFIED')
            path = tx.runtime / 'run' / ('startup-failure-' + intent['attempt_id'] + '.json')
            state = file_state(path)
            fail(state['exists'] and state['uid'] == os.getuid() and state['mode'] == '0o600', 'PROCESS_IDENTITY_UNVERIFIED')
            receipt = json.loads(path.read_text())
            fail(receipt['schema'] == 'prelease-failure-v1' and receipt['state'] == 'failed'
                 and all(receipt[k] == intent[k] for k in ['release_id','attempt_id','input_hash','config_sha256'])
                 and receipt['phase'] == 'pre_lease_guard' and receipt['error'] in {'ENVIRONMENT_CHANGED','INPUT_LOCK_CHANGED','STARTUP_FLAGS_CHANGED'}
                 and receipt['instances'] == {'encoder':0,'reranker':0,'worker':0}
                 and all(type(v) is int for v in receipt['instances'].values()), 'PROCESS_IDENTITY_UNVERIFIED')
            # A complete residual receipt after publication/sync/readback failure
            # is not a committed proof. Require the daemon's final acknowledgement,
            # emitted only after the writer returned successfully. First bootstrap
            # has exactly one small output record; partial/extra output is unknown.
            output_path = tx.runtime / 'logs/launchagent.stdout.log'
            output_state = file_state(output_path)
            fail(output_state['exists'] and output_state['uid'] == os.getuid()
                 and output_state['mode'] == '0o600' and output_state['size'] <= 4096,
                 'PROCESS_IDENTITY_UNVERIFIED')
            output = json.loads(output_path.read_text())
            fail(output == {'error':receipt['error'],'state':'startup_or_runtime_failed',
                            'prelease_proof':{'attempt_id':intent['attempt_id'],'receipt_sha256':state['sha256']}},
                 'PROCESS_IDENTITY_UNVERIFIED')
            identity = receipt['identity']
            fail(type(identity['pid']) is int and identity['pid'] > 1 and identity['pgid'] == identity['pid']
                 and identity['uid'] == os.getuid() and isinstance(identity['start'], str)
                 and re.fullmatch('[0-9a-f]{64}', identity['command_sha256']) is not None, 'PROCESS_IDENTITY_UNVERIFIED')
            fail(not path_present(tx.runtime / 'run/daemon.lock') and not path_present(tx.runtime / 'run/daemon.sock'), 'PROCESS_IDENTITY_UNVERIFIED')
            fail(process_identity(identity['pid'], tx.deadline) is None and not group_members(identity['pgid'], tx.deadline), 'PROCESS_IDENTITY_UNVERIFIED')
            self.assert_idle(tx.runtime)
            fail(all(sha_file(tx.release / f) == h for f,h in tx.manifest['payload'].items()), 'INPUT_LOCK_CHANGED')
            previous = tx.journal.latest('prelease_cleanup')
            if previous:
                fail(previous['observed_after']['receipt_sha256'] == state['sha256'], 'PROCESS_IDENTITY_UNVERIFIED')
            fail(file_state(path) == state and file_state(output_path) == output_state
                 and load_startup_intent(tx.release) == intent, 'PROCESS_IDENTITY_UNVERIFIED')
            proof = {'attempt_id':intent['attempt_id'],'input_hash':intent['input_hash'],'receipt_sha256':state['sha256'],
                     'acknowledgement_sha256':output_state['sha256'],'identity':identity,'signals_sent':0,'group_empty':True,'lease_socket_absent':True}
        except (OSError, ValueError, KeyError, TypeError, DeployError):
            raise DeployError('PROCESS_IDENTITY_UNVERIFIED') from None
        tx.journal.record('prelease_cleanup','verified',observed_after=proof)
        return proof

    def clean(self, tx, job):
        # Absence of both a job and its recorded exit observation is not proof
        # that a pending bootstrap never ran. Preserve the unknown boundary.
        fail(job is not None or not tx.journal or tx.journal.latest('bootstrap_owned_job') is None,
             'PROCESS_IDENTITY_UNVERIFIED')
        if job and not job.get('identity') and tx.journal:
            # An exited launchd job can retain no PID. Recover only a recorded
            # instance of this exact owned job, never a guessed current PGID.
            candidates = []
            for event in reversed(tx.journal.events):
                for value in [event.get('observed_after'), event.get('before')]:
                    if isinstance(value, dict):
                        value = value.get('job', value)
                        if isinstance(value, dict) and value.get('path') == str(tx.plist) and value.get('arguments') == tx.plist_value['ProgramArguments'] and value.get('identity'):
                            candidates.append(value)
            if not candidates:
                self.prove_prelease_exit(tx, job)
                return True
            job = candidates[0]
        if job and job.get('identity') and job['identity'].get('pgid') == job['pid']:
            group_deadline = Deadline(min(tx.deadline.remaining(), tx.manifest['limits_seconds']['stop_process']))
            cleanup_owned_group(job['identity'], group_deadline, job.get('members'))
        stop = min(tx.deadline.end, time.monotonic() + tx.manifest['limits_seconds']['stop_process'])
        while time.monotonic() < stop:
            identity = process_identity(job['pid'], tx.deadline) if job and job.get('pid') else None
            if identity:
                fail(identity == job['identity'], 'PROCESS_IDENTITY_UNVERIFIED')
            try:
                self.assert_idle(tx.runtime)
                if identity is None:
                    return True
            except DeployError:
                pass
            time.sleep(min(.05, tx.deadline.remaining()))
        raise DeployError('CLEANUP_INCOMPLETE')

    def cli(self, tx, args, transient=False):
        result = self.command(tx, [tx.manifest['python'], '-B', str(tx.release / 'control/cli_entry.py'), '--config', str(tx.config), *args],
                              maximum=tx.manifest['limits_seconds']['one_cli_call'], cwd=tx.release)
        value = json.loads(result.stdout)
        code = value.get('error') or value.get('body', {}).get('error')
        if transient and code in {'DAEMON_UNVERIFIED','MODEL_UNAVAILABLE','INDEX_NOT_READY'}:
            return None
        fail(result.returncode == 0 and value.get('exit_code') == 0, 'CLI_READBACK_FAILED')
        return value['body']

    def ready(self, tx, job):
        stop = min(tx.deadline.end, time.monotonic() + tx.manifest['limits_seconds']['startup_ready'])
        while time.monotonic() < stop:
            job = self.job(tx)
            if job and job.get('pid'):
                tx.assert_owned_job(job)
                status = self.cli(tx, ['status','--project','example-project'], transient=True)
                if status is None:
                    time.sleep(min(.1, tx.deadline.remaining()))
                    continue
                if not status.get('loading') and not status.get('syncing'):
                    fail(status.get('error') is None and status['pid'] == job['pid'], 'NOT_READY')
                    fail(int((tx.runtime / 'run/daemon.lock').read_text()) == job['pid'], 'PROCESS_IDENTITY_UNVERIFIED')
                    fail(status['embedding_profile'] == tx.manifest['profiles']['embedding_profile'] and status['dimension'] == 768, 'NOT_READY')
                    ranker = status['reranker']
                    fail(ranker['enabled'] and ranker['ready'] and ranker['error'] is None and ranker['backend'] == 'mps'
                         and ranker['profile'] == tx.manifest['profiles']['reranker_profile'], 'NOT_READY')
                    fail(set(status['sources']) == {'example-memory','example-docs','example-code'}
                         and all(x['state'] == 'ready' for x in status['sources'].values()), 'NOT_READY')
                    receipt = check_daemon_attestation(tx.release, tx.config)
                    counts = receipt.get('instances', {})
                    fail(counts == {'encoder':1,'reranker':1,'worker':1}, 'MODEL_INSTANCE_LIMIT')
                    readbacks = self.smoke(tx)
                    snapshot = self.cli(tx, ['status','--project','example-project'])
                    observed = getattr(self, 'last_smoke_snapshots', [])
                    return {'pid':job['pid'],'sources_ready':True,'guard_verified':True,'readbacks':readbacks,
                            'profiles':tx.manifest['profiles'],'generation':snapshot['generation'],
                            'readback_snapshots':observed,
                            'actual_table_version':observed[-1]['table_version'] if observed else None}
            time.sleep(min(.1, tx.deadline.remaining()))
        raise DeployError('STARTUP_TIMEOUT')

    def smoke(self, tx):
        cases = json.loads((tx.release / 'smoke.json').read_text())
        fail(len(cases) == 2, 'CLI_READBACK_FAILED')
        self.last_smoke_snapshots = []
        for case in cases:
            args = ['search','--project','example-project','--kind',case['kind'],'--query',case['query'],'--source',case['source_id']]
            search = self.cli(tx, args)
            hits = search.get('results', search.get('hits', []))
            fail(bool(hits), 'CLI_READBACK_FAILED')
            hit = hits[0]
            fail(hit.get('retrieval_role') == 'related_candidate' and hit.get('support_status') == 'not_checked', 'CLI_READBACK_FAILED')
            value = self.cli(tx, ['context','--project','example-project','--chunk-id',hit['chunk_id'],'--max-chars','12000'])
            items = value.get('items', [])
            fail(value.get('readback_status') == 'complete' and not value.get('truncated') and len(items) == 1
                 and value['generation'] == search['generation'] and value['table_version'] == search['table_version'], 'CLI_READBACK_FAILED')
            item = items[0]
            fail(item['chunk_id'] == hit['chunk_id'] and item.get('retrieval_role') == 'related_candidate'
                 and item.get('support_status') == 'not_checked', 'CLI_READBACK_FAILED')
            path = Path(item['path'])
            fail(sha_file(path) == item['file_sha256'], 'SOURCE_CHANGED')
            expected = ''.join(path.read_text().splitlines(keepends=True)[item['line_start']-1:item['line_end']])
            fail(item['text'] == expected, 'CLI_READBACK_FAILED')
            self.last_smoke_snapshots.append({'kind':case['kind'],'source_id':case['source_id'],
                 'generation':value['generation'],'table_version':value['table_version'],
                 'chunk_id':item['chunk_id'],'file_sha256':item['file_sha256'],
                 'line_start':item['line_start'],'line_end':item['line_end']})
        return 2

    def restart(self, tx, job):
        tx.assert_owned_job(job)
        self.prepare_launch_attempt(tx, 'restart')
        result = self.command(tx, ['/bin/launchctl', 'kickstart', '-k', tx.manifest['job']])
        fail(result.returncode == 0, 'RESTART_RESULT_UNKNOWN')
        end = min(tx.deadline.end, time.monotonic() + tx.manifest['limits_seconds']['startup_ready'])
        while time.monotonic() < end:
            current = self.job(tx)
            if current and current.get('identity') and current['identity'] != job['identity']:
                return
            time.sleep(min(.1, tx.deadline.remaining()))
        raise DeployError('RESTART_NOT_CONFIRMED')

    def final_verify(self, tx, expected):
        result = self.command(tx, [tx.manifest['python'], '-B', str(tx.release / 'control/deploy_transaction.py'), '--verify-parent'], maximum=120, cwd=tx.release)
        fail(result.returncode == 0, 'ENVIRONMENT_CHANGED')
        current = self.job(tx)
        fail(current and current.get('identity') == expected.get('identity'), 'FINAL_INSTANCE_CHANGED')
        receipt = check_daemon_attestation(tx.release, tx.config)
        fail(receipt['identity'] == current['identity'], 'FINAL_INSTANCE_CHANGED')
        fail(int((tx.runtime / 'run/daemon.lock').read_text()) == current['pid'], 'FINAL_INSTANCE_CHANGED')
        status = self.cli(tx, ['status','--project','example-project'])
        fail(status['pid'] == current['pid'] and status['error'] is None and not status['loading'] and not status['syncing']
             and status['embedding_profile'] == tx.manifest['profiles']['embedding_profile']
             and status['reranker']['ready'] and status['reranker']['error'] is None
             and status['reranker']['profile'] == tx.manifest['profiles']['reranker_profile']
             and all(item['state'] == 'ready' for item in status['sources'].values()), 'NOT_READY')


def verify_seed(root):
    deployment = entry_setup(root)
    from codex_memory_rag.config import load_config
    from codex_memory_rag.evaluation import build_corpus_manifest
    from codex_memory_rag.store import Store
    config = load_config(root / 'config.proposed.toml')
    current = build_corpus_manifest(config.sources)
    fail(current.corpus_manifest_hash == deployment['corpus_manifest_hash'], 'EVAL_CORPUS_CHANGED')
    store = Store(config.storage.runtime_dir, deployment['profiles']['embedding_profile'], deployment['profiles']['index_profile'])
    snapshot = store.get_snapshot()
    fail(snapshot.generation_id == deployment['seed_generation'] and snapshot.version == deployment['seed_table_version'], 'SNAPSHOT_CHANGED')
    fail(snapshot.table.count_rows() == deployment['seed_rows'], 'SNAPSHOT_CHANGED')
    return {'source_verified':True,'actual_table_version':snapshot.version,'rows':snapshot.table.count_rows()}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--capture-environment', action='store_true')
    parser.add_argument('--verify-seed', action='store_true')
    parser.add_argument('--verify-parent', action='store_true')
    parser.add_argument('operation', nargs='?', default='plan', choices=['plan','apply','resume','rollback'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.capture_environment:
        result = capture_role(root, 'parent')
    elif args.verify_seed:
        restore = install_offline()
        try:
            result = verify_seed(root)
        finally:
            restore()
    elif args.verify_parent:
        deployment = entry_setup(root)
        result = guard_context(root, Path(deployment['config']), 'parent', full=True)
    else:
        fail(args.package is not None, 'PACKAGE_REQUIRED')
        validate_startup_environment(json.loads((args.package / 'release-manifest.json').read_text()))
        if args.operation == 'plan':
            result = {'state':'NOT_DEPLOYED','package_seal_sha256':verify_package(args.package),
                      'activation_authorized':False,'executable':True}
        else:
            tx = Deployment(args.package)
            result = getattr(tx, args.operation)(args.authorization)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'error':str(exc) if isinstance(exc, DeployError) else type(exc).__name__}))
        sys.exit(2)
