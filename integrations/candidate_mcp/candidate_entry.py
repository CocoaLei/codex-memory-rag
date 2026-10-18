"""Pinned STDIO entry. No configuration mutation, model or business tool at startup."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tomllib

from candidate_adapter import Adapter, AdapterError, PolicyFile, require, SAFE_ERRORS
from candidate_transport import ProcessRunner, create_server

EXCLUDED={'package-seal.json','package-seal.sha256','package-seal-verification.json','checkpoint-verification.json'}


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_package(package):
    try:
        package=Path(package)
        require(not any(x.is_symlink() for x in [package,*package.parents]),'ADAPTER_PACKAGE_CHANGED')
        seal=json.loads((package/'package-seal.json').read_text());checksum=sha_file(package/'package-seal.json')
        require(checksum==(package/'package-seal.sha256').read_text().strip(),'ADAPTER_PACKAGE_CHANGED')
        actual={str(p.relative_to(package)) for p in package.rglob('*') if p.is_file() and p.name not in EXCLUDED}
        require(actual==set(seal['files']),'ADAPTER_PACKAGE_CHANGED')
        for name,expected in seal['files'].items():
            rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts,'ADAPTER_PACKAGE_CHANGED')
            p=package/rel
            require(not any(x.is_symlink() for x in [p,*p.parents]) and p.is_file() and sha_file(p)==expected,'ADAPTER_PACKAGE_CHANGED')
        return checksum
    except AdapterError:raise
    except Exception:raise AdapterError('ADAPTER_PACKAGE_CHANGED') from None


class RuntimeBinding:
    def __init__(self,package,verifier=verify_package):
        self.package=Path(package);self.seal=verifier(self.package)
        try:
            ref=json.loads((self.package/'runtime-reference.json').read_text())
            require(os.environ.get('PYTHONUTF8')=='0' and list(sys.flags)==ref['python_flags']
                    and Path(sys.executable).resolve()==Path(ref['python_realpath']),'STARTUP_FLAGS_CHANGED')
            require(not os.environ.get('PYTHONPATH'),'STARTUP_FLAGS_CHANGED')
            release=Path(ref['r4_release']);self.release=release;self.python=ref['python'];self.config=ref['config']
            parent=Path(ref['r4_package'])
            # Verify the control code before importing its read-only guards.
            seal=json.loads((parent/'package-seal.json').read_text())
            require(sha_file(parent/'package-seal.json')==ref['r4_seal_sha256'],'INPUT_LOCK_CHANGED')
            control=release/'control/deploy_transaction.py'
            require(sha_file(control)==seal['files']['release/control/deploy_transaction.py'],'INPUT_LOCK_CHANGED')
            spec=importlib.util.spec_from_file_location('_cmr_candidate_r4_control',control)
            shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared);self.shared=shared
            require(shared.verify_package(parent)==ref['r4_seal_sha256'],'INPUT_LOCK_CHANGED')
            payload=json.loads((parent/'release-manifest.json').read_text())['payload']
            require(all(sha_file(release/name)==h for name,h in payload.items()),'INPUT_LOCK_CHANGED')
            self.input_hash=sha_file(release/'input-lock.json');require(self.input_hash==ref['r4_input_hash'],'INPUT_LOCK_CHANGED')
            require(sha_file(self.config)==ref['config_sha256'],'INPUT_LOCK_CHANGED')
            sys.path.insert(0,str(release/'control'))
            shared.entry_setup(release)
            lock=json.loads((release/'input-lock.json').read_text())
            record=lock['roles']['cli']
            for kind in ['environment','metadata']:
                require(sha_file(shared.safe_path(release/record[kind]))==record[kind+'_sha256'],'INPUT_LOCK_CHANGED')
            # Compare the already locked installed/runtime scope. The adapter's
            # added import providers are independently bound below, never by
            # rewriting or pretending to be the CLI role's original lock.
            from environment_lock import verify_metadata
            expected=json.loads((release/record['metadata']).read_text())
            verify_metadata(expected,spec=expected['spec'])
            config=tomllib.loads(Path(self.config).read_text())
            require(config['candidate_pilot']['enabled'] is True,'SCOPE_DENIED')
            self.sources={s['source_id']:{'root':s['root'],'worktree_id':s.get('worktree_id')}
                          for s in config['sources'] if s['project_id']=='composing-agent' and s['enabled'] and s['allow_cloud_context']}
            require(set(self.sources)=={'ca-memory','ca-docs','ca-code'},'SCOPE_DENIED')
        except AdapterError:raise
        except Exception as exc:
            code=str(exc)
            raise AdapterError(code if code in SAFE_ERRORS else 'R4_BINDING_UNVERIFIED') from None

    def command(self,args):
        require(self.shared.allowed_cli(args),'SCOPE_DENIED')
        return [self.python,'-B',str(self.release/'control/cli_entry.py'),'--config',self.config,*args]


def adapter_environment(binding,capture=False):
    from environment_lock import capture_environment, capture_metadata, verify_metadata, digest
    package=binding.package
    if capture:
        before=capture_metadata();environment=capture_environment();after=capture_metadata()
        require(digest(before)==digest(after),'ENVIRONMENT_CHANGED')
        reference=json.loads((binding.release/'reference-environment.json').read_text())
        for key in ['interpreter','pyvenv_cfg','interpreter_related','stdlib','site_packages','distributions']:
            require(environment[key]==reference[key],'ENVIRONMENT_CHANGED')
        for name,value in [('adapter-environment.json',environment),('adapter-metadata.json',after)]:
            path=package/name
            require(not path.exists(),'ADAPTER_LOCK_ALREADY_EXISTS')
            binding.shared.private_json(path,value)
        return {'environment_sha256':sha_file(package/'adapter-environment.json'),
                'metadata_sha256':sha_file(package/'adapter-metadata.json')}
    try:verify_metadata(json.loads((package/'adapter-metadata.json').read_text()))
    except Exception:raise AdapterError('ENVIRONMENT_CHANGED') from None


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--policy',type=Path)
    args=parser.parse_args();os.umask(0o077)
    package=Path(__file__).resolve().parent
    binding=RuntimeBinding(package)
    policy=PolicyFile(args.policy or package/'policy.template.json',binding.seal,binding.input_hash)
    adapter_environment(binding)
    snap=policy.read()
    require(not snap.policy.approved or not policy.path.resolve().is_relative_to(package),'CLOUD_POLICY_INVALID')
    # Metadata was checked before the SDK server is created; no CLI status or
    # business call is performed at startup. Every call rechecks its policy.
    restore=binding.shared.install_offline()
    try:
        server=create_server(Adapter(policy,ProcessRunner(binding.command,binding.input_hash),binding.sources))
        server.run(transport='stdio')
    finally:restore()


if __name__=='__main__':
    try:main()
    except AdapterError as exc:
        print(str(exc),file=sys.stderr);raise SystemExit(2)
    except Exception:
        print('ADAPTER_STARTUP_FAILED',file=sys.stderr);raise SystemExit(2)
