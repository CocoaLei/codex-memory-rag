"""The deployment CLI guard keeps the example project scope exact."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def allowed_cli():
    path = Path(__file__).parents[2] / "integrations/deployment/control/deploy_transaction.py"
    spec = importlib.util.spec_from_file_location("candidate_cli_scope", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.allowed_cli


@pytest.mark.parametrize("operation", ["status", "search", "context"])
def test_guard_accepts_example_project(allowed_cli, operation):
    assert allowed_cli([operation, "--project", "example-project"])


@pytest.mark.parametrize("arguments", [
    ["status", "--project", "other-project"],
    ["status"],
    ["status", "--project", "example-project", "--project", "other-project"],
    ["status", "--project=example-project"],
    ["status", "--project", "example-project", "--config", "/synthetic/config.toml"],
    ["index", "--project", "example-project"],
])
def test_guard_rejects_scope_overrides(allowed_cli, arguments):
    assert not allowed_cli(arguments)
