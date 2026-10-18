"""Generate only this unsealed adapter's new environment locks; no daemon call."""
from pathlib import Path
import json
from candidate_entry import RuntimeBinding,adapter_environment,require

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    require(not (root/'package-seal.json').exists(),'ADAPTER_PACKAGE_ALREADY_SEALED')
    require(not any((root/n).exists() for n in ['adapter-environment.json','adapter-metadata.json']),'ADAPTER_LOCK_ALREADY_EXISTS')
    # Staging bootstrap has no self-seal yet. All r4/hash/installed-runtime
    # checks remain enabled; only the new package checksum is not yet defined.
    b=RuntimeBinding(root,verifier=lambda p:'0'*64)
    print(json.dumps(adapter_environment(b,capture=True)))
