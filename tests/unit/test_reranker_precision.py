"""Keep legacy profile/config identity while binding opt-in MPS precision."""
import json
from pathlib import Path

import pytest

from codex_memory_rag.reranking import profile_id
from codex_memory_rag.schemas import AppConfig, RerankerConfig


def test_default_config_bytes_contract_is_legacy_compatible():
    expected = json.loads((Path(__file__).parents[1] / "legacy-default-config.json").read_text())
    assert AppConfig(storage={"runtime_dir": "/synthetic/runtime"}).model_dump(mode="json") == expected


def test_explicit_mps_fp16_is_supported_and_included_in_config_identity():
    cfg = RerankerConfig(backend="mps", dtype="float16")
    assert cfg.dtype == "float16"
    assert cfg.model_dump(mode="json")["dtype"] == "float16"


def test_fp16_is_not_silently_used_on_cpu():
    with pytest.raises(ValueError, match="float16 requires mps"):
        RerankerConfig(backend="cpu", dtype="float16")


def test_default_fp32_profile_inputs_preserve_existing_profile():
    resource = {"model_id": "BAAI/bge-reranker-v2-m3", "revision": "a" * 40}
    cfg = RerankerConfig()
    expected = __import__("hashlib").sha256(json.dumps(
        ["reranker-v1", resource, "cpu", "fp32-sdpa", 512, 4,
         "bilingual-symbol-v2-scoped-query"],
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert profile_id(resource, cfg) == expected


def test_fp16_profile_cannot_alias_same_resources_fp32_profile():
    resource = {"model_id": "BAAI/bge-reranker-v2-m3", "revision": "a" * 40}
    base = RerankerConfig(backend="mps")
    reduced = RerankerConfig(backend="mps", dtype="float16")
    assert profile_id(resource, base) != profile_id(resource, reduced)
