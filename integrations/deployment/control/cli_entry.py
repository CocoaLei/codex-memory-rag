"""Read-only CLI adapter. No result body escapes before the post guard."""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

from deploy_transaction import (DeployError, allowed_cli, capture_role,
                                check_daemon_attestation, entry_setup, fail,
                                guarded_call, install_offline)


def execute(root, config, arguments):
    fail(allowed_cli(arguments), 'SCOPE_DENIED')
    entry_setup(root)
    restore = install_offline()
    started = time.perf_counter()
    try:
        check_daemon_attestation(root, config)
        def invoke():
            from codex_memory_rag import cli
            original = sys.argv
            stdout, stderr = io.StringIO(), io.StringIO()
            code = 0
            try:
                sys.argv = ['cmr','--config',str(config),*arguments]
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    try:
                        cli.main()
                    except SystemExit as exc:
                        code = exc.code
                return {'exit_code':code,'body':json.loads(stdout.getvalue())}
            finally:
                sys.argv = original
        value, receipt = guarded_call(root, config, 'cli', invoke)
        check_daemon_attestation(root, config)
        value['environment'] = receipt
        value['process_wall_ms'] = (time.perf_counter() - started) * 1000
        return value
    finally:
        restore()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--capture-environment', action='store_true')
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parent.parent
    if args.capture_environment:
        fail(not remaining and args.config is None, 'SCOPE_DENIED')
        print(json.dumps(capture_role(root, 'cli')))
        return 0
    fail(args.config is not None, 'CONFIG_REQUIRED')
    value = execute(root, args.config, remaining)
    print(json.dumps(value, ensure_ascii=False))
    return value['exit_code']


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({'exit_code':2,'error':str(exc) if isinstance(exc, DeployError) else type(exc).__name__}))
        sys.exit(2)
