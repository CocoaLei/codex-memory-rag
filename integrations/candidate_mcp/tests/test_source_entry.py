"""A source checkout is not a sealed release and must stay closed."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]


def test_checkout_without_machine_seal_refuses_entry(tmp_path):
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONUTF8": "0"}
    result = subprocess.run([sys.executable, "-B", str(ROOT / "candidate_entry.py")],
                            cwd=tmp_path, env=env, capture_output=True, timeout=15)
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr.strip() == b"ADAPTER_PACKAGE_CHANGED"
