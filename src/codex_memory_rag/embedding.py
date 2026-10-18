"""Official encoder. Download is an explicit prepare operation, never a query."""
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path

import numpy as np

from .schemas import CmrError, EmbeddingInput, EmbeddingProfile, identity_hash

MODEL_ID = "google/embeddinggemma-2"


def unavailable_encoder(runtime: Path, device: str):
    """Only existing, identically profiled lexical evidence survives cold failure."""
    lock = json.loads((runtime / "model.lock.json").read_text())
    class UnavailableEncoder:
        profile_id = identity_hash(["embedding-v1", lock, device, "fp32-eager-text"])
        def token_count(self, *args):
            raise CmrError("MODEL_UNAVAILABLE")
        def encode_queries(self, *args):
            raise CmrError("MODEL_UNAVAILABLE")
        def encode_documents(self, *args):
            raise CmrError("MODEL_UNAVAILABLE")
    return UnavailableEncoder()


def format_query(query: str, kind: str) -> str:
    task = "code retrieval" if kind == "code" else "search result"
    return f"task: {task} | query: {query}"


def format_document(title: str | None, content: str) -> str:
    return f"title: {title or 'none'} | text: {content}"


def validate_vectors(vectors) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 768 or not np.isfinite(x).all():
        raise CmrError("EMBEDDING_CONTRACT_MISMATCH")
    norm = np.linalg.norm(x.astype(np.float64), axis=1, keepdims=True)
    if (norm == 0).any() or not np.isfinite(norm).all():
        raise CmrError("EMBEDDING_CONTRACT_MISMATCH")
    result = (x / norm).astype(np.float32)
    if not np.all(np.abs(np.linalg.norm(result, axis=1) - 1) <= 1e-4):
        raise CmrError("EMBEDDING_CONTRACT_MISMATCH")
    return result


def prepare_model(runtime: Path, dependency_lock: Path) -> dict:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = str(runtime / "cache/huggingface")
    os.environ["HF_XET_CACHE"] = str(runtime / "cache/xet")
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(MODEL_ID, token=False)
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise CmrError("MODEL_UNAVAILABLE")
    snapshot = Path(snapshot_download(MODEL_ID, revision=info.sha, token=False,
                                     cache_dir=runtime / "cache/models",
                                     allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]))
    files = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in snapshot.rglob("*") if p.is_file()}
    lock = {"model_id": MODEL_ID, "revision": info.sha, "snapshot": str(snapshot),
            "files": files, "dependency_lock_sha256": hashlib.sha256(dependency_lock.read_bytes()).hexdigest(),
            "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "sentence-transformers"]},
            "dimension": 768, "modalities": ["text"], "prefix_version": "prefix-v1",
            "normalization": "l2", "storage_dtype": "float32", "inference_dtype": "float32"}
    path = runtime / "model.lock.json"
    path.write_text(json.dumps(lock, indent=2) + "\n")
    path.chmod(0o600)
    return lock


class Encoder:
    def __init__(self, runtime: Path, device: str = "cpu", dependency_lock: Path | None = None):
        path = runtime / "model.lock.json"
        if not path.is_file():
            raise CmrError("MODEL_UNAVAILABLE")
        lock = json.loads(path.read_text())
        if lock.get("model_id") != MODEL_ID or lock.get("dimension") != 768:
            raise CmrError("EMBEDDING_CONTRACT_MISMATCH")
        if dependency_lock and hashlib.sha256(dependency_lock.read_bytes()).hexdigest() != lock["dependency_lock_sha256"]:
            raise CmrError("INDEX_PROFILE_MISMATCH")
        snapshot = Path(lock["snapshot"])
        for name, digest in lock["files"].items():
            p = snapshot / name
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                raise CmrError("MODEL_UNAVAILABLE")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from sentence_transformers import SentenceTransformer
        if device == "mps" and not torch.backends.mps.is_available():
            raise CmrError("MODEL_UNAVAILABLE")
        self.model = SentenceTransformer(str(snapshot), device=device, local_files_only=True,
                                        trust_remote_code=False, truncate_dim=None,
                                        config_kwargs={"vision_config": None, "audio_config": None},
                                        model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"})
        self.model.default_prompt_name = None
        if self.model.get_embedding_dimension() != 768:
            raise CmrError("EMBEDDING_CONTRACT_MISMATCH")
        self.max_tokens = 8192
        self.batch_size = 8
        self.profile_id = identity_hash(["embedding-v1", lock, device, "fp32-eager-text"])
        self.profile = EmbeddingProfile(model_lock=lock, backend=device, profile_id=self.profile_id)
        self.device = device

    @classmethod
    def from_verified_model(cls, model, profile_id: str, max_tokens: int = 8192):
        obj = cls.__new__(cls)
        obj.model, obj.profile_id, obj.max_tokens, obj.batch_size = model, profile_id, max_tokens, 8
        obj.device = "test"
        return obj

    def token_count(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=True))

    def _encode(self, texts: list[str], query: bool):
        if any(self.token_count(t) > self.max_tokens for t in texts):
            raise CmrError("QUERY_TOO_LONG" if query else "EMBEDDING_CONTRACT_MISMATCH")
        if not texts:
            return np.empty((0, 768), dtype=np.float32)
        try:
            values = self.model.encode(texts, prompt="", truncate_dim=None, batch_size=self.batch_size,
                                       normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False)
        except (RuntimeError, OSError):
            raise CmrError("MODEL_UNAVAILABLE") from None
        return validate_vectors(values)

    def encode_queries(self, queries: list[str], kind: str):
        return self._encode([format_query(q, kind) for q in queries], True)

    def encode_documents(self, documents: list[EmbeddingInput]):
        return self._encode([format_document(d.title, d.content) for d in documents], False)
