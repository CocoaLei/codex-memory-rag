"""Public validation contracts; this module never imports an inference runtime."""
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CmrError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def identity_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                         sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_key(project: str, source: str, worktree: str | None, path: str) -> str:
    return identity_hash(["source-v1", project, source, worktree, path])


def chunk_id(key: str, file_hash: str, version: str, ordinal: int) -> str:
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("invalid ordinal")
    return identity_hash(["chunk-v1", key, file_hash, version, ordinal])


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EmbeddingProfile(Strict):
    model_lock: dict
    backend: Literal["cpu", "mps"]
    dimension: Literal[768] = 768
    precision: Literal["fp32-eager-text"] = "fp32-eager-text"
    prefix_contract: Literal["embedding-v1"] = "embedding-v1"
    profile_id: str


class EmbeddingConfig(Strict):
    model_id: Literal["google/embeddinggemma-2"] = "google/embeddinggemma-2"
    dimension: Literal[768] = 768
    normalization: Literal["l2"] = "l2"
    storage_dtype: Literal["float32"] = "float32"
    inference_dtype: Literal["float32"] = "float32"
    modalities: list[Literal["text"]] = ["text"]
    device_policy: Literal["mps_fp32_after_cpu_validation"] = "mps_fp32_after_cpu_validation"
    batch_size: int = Field(default=8, ge=1, le=32)
    max_input_tokens: Literal[8192] = 8192
    allow_dimension_truncation: Literal[False] = False
    allow_remote_code: Literal[False] = False
    runtime_offline: Literal[True] = True


class SourceConfig(Strict):
    source_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$")
    project_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$")
    code_project: str | None = None
    codex_project_id: str | None = None
    kind: Literal["memory", "docs", "code"]
    root: Path
    include: list[str] = []
    exclude: list[str] = []
    enabled: bool = False
    allow_cloud_context: bool = False
    worktree_id: str | None = None
    shared: bool = False
    archive_dirs: list[str] = ["90-archive"]

    @field_validator("root")
    @classmethod
    def absolute_root(cls, v):
        if not v.is_absolute() or ".." in v.parts:
            raise ValueError("root must be absolute")
        return v

    @model_validator(mode="after")
    def validate_worktree(self):
        if self.kind == "code" and not self.worktree_id:
            raise ValueError("code source requires worktree_id")
        if self.kind != "code" and self.worktree_id is not None:
            raise ValueError("non-code worktree_id must be null")
        if self.enabled and not self.include:
            raise ValueError("enabled source needs explicit includes")
        return self


class StorageConfig(Strict):
    runtime_dir: Path = Path.home() / ".local/share/codex-memory-rag"

    @field_validator("runtime_dir")
    @classmethod
    def absolute(cls, v):
        if not v.is_absolute():
            raise ValueError("absolute runtime required")
        return v


class RerankerConfig(Strict):
    enabled: bool = False
    supplementary_windows: bool = False
    scope_aware_budget: bool = False
    definition_priority: bool = False
    length_sorted_batches: bool = False
    model_dir: Path | None = None
    lock_file: Path | None = None
    profile_id: str | None = None
    backend: Literal["cpu", "mps"] = "cpu"
    dtype: Literal["float32", "float16"] = Field(default="float32", exclude_if=lambda value: value == "float32")
    batch_size: int = Field(default=4, ge=1, le=8)
    max_tokens: Literal[256, 384, 512] = 512
    max_candidates: int = Field(default=40, ge=8, le=64)
    memory_min_score: float | None = Field(default=None, allow_inf_nan=False)
    code_min_score: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def explicit_resources(self):
        if self.dtype == "float16" and self.backend != "mps":
            raise ValueError("float16 requires mps")
        if self.definition_priority and (not self.scope_aware_budget or not self.supplementary_windows):
            raise ValueError("definition priority requires scoped supplementary windows")
        for path in [self.model_dir, self.lock_file]:
            if path is not None and (not path.is_absolute() or ".." in path.parts):
                raise ValueError("reranker resources must be absolute")
        if self.enabled and (not self.model_dir or not self.lock_file or not self.profile_id
                             or self.memory_min_score is None or self.code_min_score is None):
            raise ValueError("enabled reranker requires locked resources and calibrated thresholds")
        return self


class RetrievalConfig(Strict):
    memory_min_cosine: float | None = None
    code_min_cosine: float | None = None
    memory_min_bm25: float = Field(default=0, ge=0)
    code_min_bm25: float = Field(default=0, ge=0)
    candidate_k: int = Field(default=40, ge=8, le=100)
    rrf_k: Literal[60] = 60
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class RuntimeConfig(Strict):
    backend: Literal["cpu", "mps"] = "cpu"
    debounce_seconds: float = Field(default=2, ge=0.1)
    reconcile_seconds: float = Field(default=300, ge=1)
    force_polling: bool = True


class CandidatePilotConfig(Strict):
    enabled: bool = Field(default=False, strict=True)


class AppConfig(Strict):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    sources: list[SourceConfig] = []
    storage: StorageConfig = Field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    candidate_pilot: CandidatePilotConfig = Field(default_factory=CandidatePilotConfig)

    @model_validator(mode="after")
    def unique_sources(self):
        ids = [s.source_id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate source_id")
        for path in [self.retrieval.reranker.model_dir, self.retrieval.reranker.lock_file]:
            if path is not None and not path.is_relative_to(self.storage.runtime_dir):
                raise ValueError("reranker resources must remain inside private runtime")
        return self


class SearchRequest(Strict):
    query: str = Field(min_length=1, max_length=32768)
    project_id: str
    kind: Literal["memory", "code"] = "memory"
    source_id: str | None = None
    literal: str | None = Field(default=None, max_length=256)
    top_k: int = Field(default=8, ge=1, le=20)
    include_shared: bool = False
    include_archived: bool = False
    mode: Literal["hybrid", "vector", "lexical"] = "hybrid"


class ContextRequest(Strict):
    project_id: str
    chunk_ids: list[str] = Field(min_length=1, max_length=5)
    max_chars: int = Field(default=12000, ge=1, le=32000)
    include_archived: bool = False


class SourceDocument(Strict):
    source: SourceConfig
    relative_path: str
    text: str
    file_sha256: str
    metadata: dict = {}
    git_state: dict | None = None


class ScanResult(Strict):
    allowed: list[str] = []
    rejected: dict[str, str] = {}
    failed: dict[str, str] = {}
    accessible: bool = True


class Chunk(Strict):
    chunk_id: str
    source_key: str
    source_id: str
    project_id: str
    worktree_id: str | None = None
    relative_path: str
    title: str
    heading: str = ""
    symbol: str = ""
    language: str = "text"
    source_kind: str
    memory_type: str | None = None
    status: str | None = None
    retrieval_state: str
    retrieval_reasons: list[str] = []
    valid_until: str | None = None
    supersedes: str | None = None
    text: str
    lexical_text: str
    identifiers: list[str] = []
    line_start: int
    line_end: int
    file_sha256: str
    text_sha256: str
    git_commit: str | None = None
    branch: str | None = None
    source_epoch: str = ""
    observed_mtime: str = ""
    indexed_at: str = ""
    parse_quality: str = "parsed"
    embedding_profile_id: str = ""


class EmbeddedChunk(Chunk):
    vector: list[float] = Field(min_length=768, max_length=768)

    @field_validator("vector")
    @classmethod
    def normalized_vector(cls, value):
        import math
        if not all(math.isfinite(x) for x in value) or abs(math.sqrt(sum(x*x for x in value)) - 1) > 1e-4:
            raise ValueError("invalid finite normalized 768D vector")
        return value


class EmbeddingInput(Strict):
    title: str | None
    content: str


class SearchResponse(Strict):
    results: list[dict] = []
    generation: str | None = None
    table_version: int | None = None
    mode: str
    warnings: list[str] = []
    elapsed_ms: float = 0
    truncated: bool = False
    reranker_profile: str | None = None


class CorpusManifest(Strict):
    files: list[dict]
    policy_hash: str
    git_states: dict
    coverage: dict
    corpus_manifest_hash: str


class EvaluationLock(Strict):
    corpus: CorpusManifest
    generation: str
    table_version: int
    embedding_profile_id: str
    index_profile_id: str
    retrieval_config_hash: str
    suite_hash: str
    reranker_profile: str | None = None
