import hashlib
import json
from pathlib import Path

import pytest

from codex_memory_rag.schemas import AppConfig, CmrError


def test_reranker_requires_explicit_resources_and_finite_thresholds(tmp_path):
    assert AppConfig(retrieval={"reranker": {"enabled": False}}).retrieval.reranker.enabled is False
    with pytest.raises(ValueError):
        AppConfig(retrieval={"reranker": {"enabled": True}})
    with pytest.raises(ValueError):
        AppConfig(retrieval={"reranker": {"memory_min_score": float("nan")}})


def test_corrupt_resource_is_rejected_before_model_loading(tmp_path, monkeypatch):
    from codex_memory_rag.reranking import verify_resources
    root = tmp_path / "model"
    root.mkdir()
    names = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"]
    for name in names:
        (root / name).write_bytes(b"initial")
    lock = {"model_id": "BAAI/bge-reranker-v2-m3", "revision": "a" * 40,
            "files": [{"path": name, "bytes": 7,
                       "sha256": hashlib.sha256(b"initial").hexdigest()} for name in names]}
    verify_resources(root, lock)
    (root / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(CmrError, match="RERANKER_UNAVAILABLE"):
        verify_resources(root, lock)


def test_resource_paths_cannot_escape_the_locked_directory(tmp_path):
    from codex_memory_rag.reranking import verify_resources
    lock = {"model_id": "BAAI/bge-reranker-v2-m3", "revision": "a" * 40,
            "files": [{"path": "../outside", "bytes": 0, "sha256": "a" * 64}]}
    with pytest.raises(CmrError, match="RERANKER_UNAVAILABLE"):
        verify_resources(tmp_path, lock)


def test_score_boundary_rejects_nonfinite_or_wrong_length():
    from codex_memory_rag.reranking import validate_scores
    assert validate_scores([1.0, -2.0], 2) == [1.0, -2.0]
    for values in [[float("nan")], [float("inf")], [1.0, 2.0], []]:
        with pytest.raises(CmrError, match="RERANKER_CONTRACT_MISMATCH"):
            validate_scores(values, 1)


def test_passage_adds_symbol_components_without_altering_source():
    from codex_memory_rag.reranking import format_passage
    row = {"relative_path": "module.py", "symbol": "normalize_size", "text": "def normalize_size(x):\n    return x\n"}
    text = format_passage(row)
    assert "规范化" in text and "normalize size" in text
    assert text.endswith(row["text"])
    assert row["text"] == "def normalize_size(x):\n    return x\n"


def test_pair_window_limit_is_distinct_from_public_query_limit():
    from codex_memory_rag.reranking import Reranker
    from codex_memory_rag.schemas import RerankerConfig
    model = Reranker.__new__(Reranker)
    model.config = RerankerConfig(max_tokens=256)
    class Tokenizer:
        def encode(self, *args, **kwargs):
            return list(range(194))
    model.tokenizer = Tokenizer()
    # The independent real-tokenizer repro fits the original query contract.
    # No model/forward is present: this is the pre-inference boundary only.
    with pytest.raises(CmrError, match="RERANKER_QUERY_TOO_LONG"):
        model.score("synthetic valid original query", ["passage"])
