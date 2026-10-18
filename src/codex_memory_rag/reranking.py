"""Locked offline pair scoring, owned only by the daemon's model worker."""
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from .parsing import BILINGUAL_TERMS
from .schemas import CmrError, RerankerConfig, identity_hash

RESOURCE_NAMES = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "sentencepiece.bpe.model"}


def verify_resources(root: Path, lock: dict):
    try:
        files = lock["files"]
        if (root.is_symlink() or lock["model_id"] != "BAAI/bge-reranker-v2-m3"
                or not re.fullmatch(r"[0-9a-f]{40}", lock["revision"])
                or len(files) != 6 or {v["path"] for v in files} != RESOURCE_NAMES):
            raise ValueError("invalid resource manifest")
        for entry in files:
            path = root / entry["path"]
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]
                    or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]):
                raise ValueError("invalid resource")
    except (OSError, ValueError, KeyError, TypeError):
        raise CmrError("RERANKER_UNAVAILABLE") from None


def profile_id(lock: dict, config: RerankerConfig) -> str:
    precision = "fp16-sdpa" if getattr(config, "dtype", "float32") == "float16" else "fp32-sdpa"
    inputs = ["reranker-v1", lock, config.backend, precision,
              config.max_tokens, config.batch_size, "bilingual-symbol-v2-scoped-query"]
    if config.supplementary_windows:
        inputs.append("prefix-supplement-idf-v1-64pairs")
    if config.scope_aware_budget:
        inputs.append(["scope-aware-budget-v1", config.max_candidates, 40, 8, 2])
    if config.definition_priority:
        inputs.append(["complete-python-definition-first-v1", "single-query-window-above40-v1"])
    if config.length_sorted_batches:
        inputs.append("length-sorted-pair-batches-v1")
    return identity_hash(inputs)


def validate_scores(values, count: int) -> list[float]:
    try:
        scores = [float(v) for v in values]
        if len(scores) != count or not all(math.isfinite(v) for v in scores):
            raise ValueError("invalid scores")
    except (ValueError, TypeError, OverflowError):
        raise CmrError("RERANKER_CONTRACT_MISMATCH") from None
    return scores


def format_passage(row: dict) -> str:
    symbol = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", row["symbol"]).replace("_", " ")
    terms = " ".join(sorted({BILINGUAL_TERMS[w] for w in re.findall(r"[a-zA-Z]+", symbol.lower())
                             if w in BILINGUAL_TERMS}))
    header = row["relative_path"] + " " + symbol
    if symbol:
        header += " 术语: " + terms
    return header + "\n" + row["text"]


def query_windows(tokenizer, query, passages, max_tokens):
    """Select contiguous source windows without assigning a scoring budget."""
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    encoded = [tokenizer(p, add_special_tokens=False, return_offsets_mapping=True) for p in passages]
    frequencies = Counter(token for item in encoded for token in set(item["input_ids"]))
    weights = {token: math.log((len(encoded) + 1) / (frequencies[token] + 1)) + 1
               for token in set(query_ids)}
    selected = []
    for index, (passage, item) in enumerate(zip(passages, encoded, strict=True)):
        ids, offsets = item["input_ids"], item["offset_mapping"]
        if len(ids) + len(query_ids) + 4 <= max_tokens or "\n" not in passage:
            continue
        boundary = passage.index("\n") + 1
        header = passage[:boundary]
        budget = max_tokens - len(query_ids) - len(tokenizer.encode(header, add_special_tokens=False)) - 6
        first = next((i for i, span in enumerate(offsets) if span[0] >= boundary), None)
        if budget < 32 or first is None:
            continue
        last = max(first, len(ids) - budget)
        starts = sorted(set(range(first, last + 1, max(1, budget // 3))) | {last})
        start = max(starts, key=lambda i: (sum(weights.get(token, 0) for token in
                                             set(ids[i:i + budget])), -i))
        if start == first:
            continue
        stop = min(len(ids), start + budget)
        selected.append((index, header + passage[offsets[start][0]:offsets[stop - 1][1]]))
    return selected


def supplementary_passages(tokenizer, query, passages, max_tokens):
    """Preserve the original prefix plus supplementary 64-pair contract."""
    available = max(0, 64 - len(passages))
    return query_windows(tokenizer, query, passages, max_tokens)[:available] if available else []


class Reranker:
    def __init__(self, config: RerankerConfig, dependency_lock: Path):
        self.config = config
        try:
            if not config.enabled or not config.lock_file or not config.model_dir:
                raise CmrError("RERANKER_UNAVAILABLE")
            if config.lock_file.is_symlink():
                raise CmrError("RERANKER_UNAVAILABLE")
            lock = json.loads(config.lock_file.read_text())
            verify_resources(config.model_dir, lock)
            if hashlib.sha256(dependency_lock.read_bytes()).hexdigest() != lock["dependency_lock_sha256"]:
                raise CmrError("RERANKER_UNAVAILABLE")
            self.profile_id = profile_id(lock, config)
            if self.profile_id != config.profile_id:
                raise CmrError("RERANKER_UNAVAILABLE")
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            if config.backend == "mps" and not torch.backends.mps.is_available():
                raise CmrError("RERANKER_UNAVAILABLE")
            self.tokenizer = AutoTokenizer.from_pretrained(str(config.model_dir), local_files_only=True,
                                                          trust_remote_code=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(config.model_dir), local_files_only=True, trust_remote_code=False,
                dtype=getattr(torch, getattr(config, "dtype", "float32")),
                attn_implementation="sdpa").to(config.backend).eval()
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            raise CmrError("RERANKER_UNAVAILABLE") from None

    def score(self, query: str, passages: list[str]) -> list[float]:
        import torch
        if len(passages) > self.config.max_candidates:
            raise CmrError("RERANKER_CONTRACT_MISMATCH")
        if len(self.tokenizer.encode(query, add_special_tokens=True)) > self.config.max_tokens // 2:
            raise CmrError("RERANKER_QUERY_TOO_LONG")
        scores = []
        try:
            replace_prefix = getattr(self.config, "definition_priority", False) and len(passages) > 40
            if replace_prefix:
                pair_passages = list(passages)
                for index, text in query_windows(self.tokenizer, query, passages, self.config.max_tokens):
                    pair_passages[index] = text
                extra = []
            else:
                extra = supplementary_passages(self.tokenizer, query, passages, self.config.max_tokens) if self.config.supplementary_windows else []
                pair_passages = passages + [text for _, text in extra]
            order = list(range(len(pair_passages)))
            encoded = None
            if getattr(self.config, "length_sorted_batches", False) and pair_passages:
                encoded = self.tokenizer([[query, text] for text in pair_passages], padding=False,
                                         truncation="only_second", max_length=self.config.max_tokens)
                order.sort(key=lambda j: (len(encoded["input_ids"][j]), j))
            scores = [0.0] * len(pair_passages)
            with torch.inference_mode():
                for i in range(0, len(order), self.config.batch_size):
                    indices = order[i:i + self.config.batch_size]
                    if encoded is None:
                        pairs = [[query, pair_passages[j]] for j in indices]
                        tokens = self.tokenizer(pairs, padding=True, truncation="only_second",
                                                max_length=self.config.max_tokens, return_tensors="pt")
                    else:
                        tokens = self.tokenizer.pad([{k: v[j] for k, v in encoded.items()} for j in indices],
                                                    padding=True, return_tensors="pt")
                    tokens = {k: v.to(self.config.backend) for k, v in tokens.items()}
                    values = validate_scores(self.model(**tokens).logits.reshape(-1).float().tolist(), len(indices))
                    for j, value in zip(indices, values, strict=True):
                        scores[j] = value
        except (RuntimeError, OSError, ValueError):
            raise CmrError("RERANKER_UNAVAILABLE") from None
        scores = validate_scores(scores, len(pair_passages))
        result = scores[:len(passages)]
        for (index, _), score in zip(extra, scores[len(passages):], strict=True):
            result[index] = max(result[index], score)
        return result
