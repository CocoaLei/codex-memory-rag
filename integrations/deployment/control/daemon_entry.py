"""Exact-release offline daemon with bounded environment attestation groups."""
import argparse
import contextlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from deploy_transaction import (DeployError, capture_role, entry_setup, fail,
                                guard_context, install_offline, private_json,
                                process_identity, sha_file, load_startup_intent,
                                record_prelease_failure, validate_startup_environment,
                                prelease_failure_output, sha_bytes, json_payload)


class DaemonGuard:
    def __init__(self, root, config, interval=240, ttl=300):
        self.root, self.config = root, config
        self.deployment = json.loads((root / 'deployment.json').read_text())
        self.runtime = Path(self.deployment['target_runtime'])
        self.interval, self.ttl = interval, ttl
        self.instances = {'encoder':0,'reranker':0,'worker':0}
        self.identity = process_identity(os.getpid())
        self.stop_event = threading.Event()
        self.owned = False
        self.mutex = threading.Lock()
        self.thread = None

    def receipt(self, state='running', error=None):
        return {'release_id':self.deployment['release_id'], 'identity':self.identity,
                'input_hash':sha_file(self.root / 'input-lock.json'), 'state':state,
                'content_verified':error is None, 'verified_at':time.time(),
                'expires_at':time.time() + self.ttl, 'instances':dict(self.instances),
                'error':error, 'process_group':os.getpgrp()}

    def attest(self):
        with self.mutex:
            guard_context(self.root, self.config, 'daemon', full=True)
            fail(self.owned and process_identity(os.getpid()) == self.identity, 'PROCESS_IDENTITY_UNVERIFIED')
            private_json(self.runtime / 'run/daemon-current.json', self.receipt())

    def own(self):
        fail(int((self.runtime / 'run/daemon.lock').read_text()) == os.getpid(), 'PROCESS_IDENTITY_UNVERIFIED')
        self.owned = True
        self.attest()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True, name='cmr-environment-attestation')
        self.thread.start()

    def count(self, name):
        self.instances[name] += 1
        fail(self.instances[name] == 1, 'MODEL_INSTANCE_LIMIT')
        if self.owned:
            with self.mutex:
                current = json.loads((self.runtime / 'run/daemon-current.json').read_text())
                # Loading a model is not a fresh full environment verification.
                current['instances'] = dict(self.instances)
                private_json(self.runtime / 'run/daemon-current.json', current)

    def heartbeat(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.attest()
            except Exception as exc:
                with contextlib.suppress(Exception):
                    private_json(self.runtime / 'run/daemon-current.json', self.receipt('failed', str(exc) if isinstance(exc, DeployError) else type(exc).__name__))
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.owned:
            error = None
            try:
                guard_context(self.root, self.config, 'daemon', full=True)
            except Exception as exc:
                error = str(exc) if isinstance(exc, DeployError) else type(exc).__name__
            private_json(self.runtime / 'run/daemon-current.json', self.receipt('stopped', error))


def run(root, config):
    deployment = entry_setup(root, validate_startup=False)
    fail(str(config) == deployment['config'], 'INPUT_LOCK_CHANGED')
    intent = load_startup_intent(root)
    if os.getpgrp() != os.getpid():
        os.setpgid(0, 0)
    identity = process_identity(os.getpid())
    fail(identity is not None, 'PROCESS_IDENTITY_UNVERIFIED')
    try:
        validate_startup_environment(deployment)
        guard_context(root, config, 'daemon', full=True)
    except DeployError as exc:
        if str(exc) in {'ENVIRONMENT_CHANGED','INPUT_LOCK_CHANGED','STARTUP_FLAGS_CHANGED'}:
            try:
                receipt = record_prelease_failure(root, config, intent, identity, str(exc))
                # Publish acknowledgement only after CAS, sync and readback all returned.
                exc.prelease_proof = {'attempt_id':intent['attempt_id'],
                                      'receipt_sha256':sha_bytes(json_payload(receipt))}
            except Exception:
                raise DeployError('STARTUP_PROOF_WRITE_FAILED') from None
        raise
    restore = install_offline()
    guard = None
    try:
        from codex_memory_rag import daemon, embedding, reranking
        from codex_memory_rag.config import load_config
        value = load_config(config)
        # Establish a dedicated owned group so launchd termination can include
        # descendants without touching another app's process group.
        if os.getpgrp() != os.getpid():
            os.setpgid(0, 0)
        guard = DaemonGuard(root, config)
        originals = []
        for klass, name in [(embedding.Encoder,'encoder'),(reranking.Reranker,'reranker'),(daemon.PriorityWorker,'worker')]:
            original = klass.__init__
            def counted(self, *args, _original=original, _name=name, **kwargs):
                guard.count(_name)
                _original(self, *args, **kwargs)
            originals.append((klass, original))
            klass.__init__ = counted
        acquire = daemon.DaemonLease.acquire
        def acquired(lease):
            result = acquire(lease)
            fail(result, 'LEASE_CONFLICT')
            guard.own()
            return result
        daemon.DaemonLease.acquire = acquired
        try:
            daemon.serve(value)
        finally:
            daemon.DaemonLease.acquire = acquire
            for klass, original in originals:
                klass.__init__ = original
    finally:
        if guard:
            guard.close()
        restore()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--capture-environment', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.capture_environment:
        fail(args.config is None, 'SCOPE_DENIED')
        print(json.dumps(capture_role(root, 'daemon')))
        return 0
    fail(args.config is not None, 'CONFIG_REQUIRED')
    run(root, args.config)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        code = str(exc) if isinstance(exc, DeployError) else type(exc).__name__
        print(json.dumps(prelease_failure_output(exc)), flush=True)
        # Configuration/integrity denial is permanent for this exact release.
        # Avoid KeepAlive retrying an input that cannot become valid by waiting.
        permanent = isinstance(exc, DeployError) and code in {'INPUT_LOCK_CHANGED','ENVIRONMENT_CHANGED','STARTUP_FLAGS_CHANGED','STARTUP_PROOF_INVALID','STARTUP_PROOF_WRITE_FAILED','LEASE_CONFLICT','CONFIG_REQUIRED','SCOPE_DENIED'}
        sys.exit(0 if permanent else 2)
