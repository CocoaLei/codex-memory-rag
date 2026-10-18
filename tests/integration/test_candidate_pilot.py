"""Local candidate presentation with synthetic sources and deterministic doubles."""
import json

import httpx
import pytest
from pydantic import ValidationError

from codex_memory_rag.daemon import create_app
from codex_memory_rag.evaluation import Evaluator
from codex_memory_rag.retrieval import Retriever
from codex_memory_rag.schemas import AppConfig, CmrError, ContextRequest, SearchRequest
from test_retrieval import retrieval as retrieval
from test_reranker_retrieval import Scores, enable
from test_watcher import Encoder, engine as engine


def pilot_config(config):
    try:
        return AppConfig.model_validate(
            config.model_dump() | {"candidate_pilot": {"enabled": True}}
        )
    except ValidationError as exc:
        pytest.fail("The explicit local candidate configuration is missing: " + str(exc))


def test_candidate_configuration_is_off_by_default_and_rejects_truthy_strings():
    assert AppConfig().model_dump().get("candidate_pilot") == {"enabled": False}
    for value in ["true", 1, None]:
        with pytest.raises(ValidationError):
            AppConfig(candidate_pilot={"enabled": value})


def test_off_mode_preserves_legacy_public_payload_and_retrieval_hash(retrieval):
    from codex_memory_rag.schemas import identity_hash
    r, _, _ = retrieval
    before = identity_hash(r.config.retrieval.model_dump(mode="json"))
    explicit = AppConfig.model_validate(
        r.config.model_dump() | {"candidate_pilot": {"enabled": False}}
    )
    default = r.search(SearchRequest(project_id="p", query="退款", mode="lexical"))
    off = Retriever(explicit, r.store, r.encoder).search(
        SearchRequest(project_id="p", query="退款", mode="lexical")
    )
    assert off.results == default.results
    assert off.warnings == default.warnings == []
    assert "retrieval_role" not in off.results[0]
    assert "support_status" not in off.results[0]
    assert identity_hash(explicit.retrieval.model_dump(mode="json")) == before
    context = Retriever(explicit, r.store, r.encoder).read_context(
        ContextRequest(project_id="p", chunk_ids=[off.results[0]["chunk_id"]])
    )
    assert set(context) == {"items", "generation", "table_version", "truncated"}
    assert "support_status" not in context["items"][0]


@pytest.mark.parametrize("mode,literal", [
    ("lexical", None), ("hybrid", None), ("vector", None),
    ("hybrid", "RDGClientTransport"),
])
def test_candidate_search_roles_do_not_change_legal_candidates(retrieval, mode, literal):
    r, _, _ = retrieval
    r.config.retrieval.memory_min_cosine = 0
    query = SearchRequest(project_id="p", query="退款", mode=mode, literal=literal)
    before = r.search(query)
    r.config = pilot_config(r.config)
    after = r.search(query)
    assert [x["chunk_id"] for x in after.results] == [x["chunk_id"] for x in before.results]
    assert after.results[0]["retrieval_role"] == "related_candidate"
    assert after.results[0]["support_status"] == "not_checked"
    assert after.results[0]["current_evidence"] is before.results[0]["current_evidence"]
    assert "完整支持未检查" in after.warnings
    context = r.read_context(ContextRequest(
        project_id="p", chunk_ids=[after.results[0]["chunk_id"]]
    ))
    assert context["items"][0]["support_status"] == "not_checked"
    assert context["items"][0]["text"] == "退款恢复 RDGClientTransport\n"
    assert "完整支持未检查" in context["warnings"]


def test_candidate_empty_history_and_model_failure_remain_visible(retrieval):
    r, _, _ = retrieval
    r.config = pilot_config(r.config)
    empty = r.search(SearchRequest(project_id="p", query="missing", literal="unknown"))
    assert not empty.results
    assert {"NO_STRONG_MATCH", "完整支持未检查"} <= set(empty.warnings)
    historical = r.search(SearchRequest(
        project_id="p", query="退款", mode="lexical", include_archived=True
    ))
    old = next(x for x in historical.results if x["relative_path"] == "old.md")
    assert old["current_evidence"] is False and old["support_status"] == "not_checked"
    with pytest.raises(CmrError, match="SCOPE_DENIED"):
        r.read_context(ContextRequest(project_id="p", chunk_ids=[old["chunk_id"]]))
    context = r.read_context(ContextRequest(
        project_id="p", chunk_ids=[old["chunk_id"]], include_archived=True
    ))
    assert context["items"][0]["current_evidence"] is False
    r.encoder = None
    degraded = r.search(SearchRequest(project_id="p", query="退款"))
    assert "MODEL_UNAVAILABLE" in degraded.warnings and degraded.mode == "lexical_only"
    assert all(x["support_status"] == "not_checked" for x in degraded.results)


def test_reranker_success_and_recursive_failure_preserve_candidate_role(engine):
    e, s = engine
    (s.root / "a.md").write_text("退款恢复")
    e.build("p")
    enable(e)
    e.config = pilot_config(e.config)
    good = Retriever(e.config, e.store, Encoder(), Scores(0.9)).search(
        SearchRequest(project_id="p", query="退款")
    )
    assert good.mode == "hybrid_reranked"
    assert good.results[0]["support_status"] == "not_checked"

    class Unavailable(Scores):
        def score(self, query, passages):
            raise CmrError("RERANKER_UNAVAILABLE")

    fallback = Retriever(e.config, e.store, Encoder(), Unavailable(0)).search(
        SearchRequest(project_id="p", query="退款")
    )
    assert "RERANKER_UNAVAILABLE" in fallback.warnings
    assert fallback.results and fallback.results[0]["retrieval_role"] == "related_candidate"
    assert "完整支持未检查" in fallback.warnings


def test_candidate_context_truncation_includes_markers_in_byte_budget(retrieval):
    import hashlib
    r, s, store = retrieval
    hit = r.search(SearchRequest(project_id="p", query="退款", mode="lexical")).results[0]
    body = "---\ncode_project: p\nstatus: active\n---\n" + "合成原文" * 8000
    (s.root / "a.md").write_text(body)
    store.table.update(where="chunk_id='" + hit["chunk_id"] + "'", values={
        "file_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "line_start": 1, "line_end": 5, "text": body,
    })
    r.config = pilot_config(r.config)
    value = r.read_context(ContextRequest(
        project_id="p", chunk_ids=[hit["chunk_id"]], max_chars=32000
    ))
    assert value["truncated"] is True
    assert len(json.dumps(value, ensure_ascii=False).encode()) <= 32768
    assert value["items"][0]["support_status"] == "not_checked"
    assert value["items"][0]["file_sha256"] == hashlib.sha256(body.encode()).hexdigest()


def test_candidate_multi_id_failure_does_not_publish_partial_success(retrieval):
    r, s, _ = retrieval
    r.config = pilot_config(r.config)
    result = r.search(SearchRequest(
        project_id="p", query="退款", mode="lexical", include_archived=True
    ))
    ids = [x["chunk_id"] for x in result.results]
    (s.root / "old.md").write_text(
        "---\ncode_project: p\nstatus: deprecated\n---\nchanged synthetic source\n"
    )
    with pytest.raises(CmrError, match="SOURCE_CHANGED"):
        r.read_context(ContextRequest(project_id="p", chunk_ids=ids, include_archived=True))
    with pytest.raises(CmrError):
        r.read_context(ContextRequest(project_id="other", chunk_ids=[ids[0]]))


def test_same_bytes_after_git_epoch_sync_are_current_not_original_snapshot(tmp_path, monkeypatch):
    from codex_memory_rag import retrieval as retrieval_module, sources
    from codex_memory_rag.parsing import chunk_document
    from codex_memory_rag.schemas import SourceConfig
    from codex_memory_rag.store import Store
    import numpy as np
    state = {"head": "a" * 40, "branch": "branch-a"}
    monkeypatch.setattr(sources, "git_state", lambda source: dict(state))
    monkeypatch.setattr(retrieval_module, "git_state", lambda source: dict(state))
    root = tmp_path / "code"
    root.mkdir()
    (root / "sample.py").write_text("def local_symbol():\n    return 1\n")
    source = SourceConfig(
        source_id="code", project_id="p", kind="code", worktree_id="w",
        root=root, include=["*.py"], enabled=True, allow_cloud_context=True
    )
    config = pilot_config(AppConfig(sources=[source], storage={"runtime_dir": tmp_path / "r"}))
    store = Store(config.storage.runtime_dir, "fixture", "fixture-index")
    generation = store.open_generation()
    def synchronize():
        chunks = chunk_document(sources.read_allowed_file(source, "sample.py"), Encoder())
        rows = [c.model_dump() | {
            "embedding_profile_id": "fixture",
            "vector": (np.ones(768) / np.sqrt(768)).tolist(),
        } for c in chunks]
        store.replace_source(chunks[0].source_key, rows)
        return chunks[0].chunk_id
    key = synchronize()
    store.activate_generation(generation.generation_id)
    reader = Retriever(config, store, Encoder())
    before = reader.read_context(ContextRequest(project_id="p", chunk_ids=[key]))
    state.update(head="b" * 40, branch="branch-b")
    with pytest.raises(CmrError, match="STALE_SOURCE"):
        reader.read_context(ContextRequest(project_id="p", chunk_ids=[key]))
    assert synchronize() == key
    after = reader.read_context(ContextRequest(project_id="p", chunk_ids=[key]))
    assert after["table_version"] > before["table_version"]
    assert after["items"][0]["branch"] == "branch-b"
    assert after["items"][0]["git_commit"] == "b" * 40
    assert after["items"][0]["current_evidence"] is True
    assert after["items"][0]["support_status"] == "not_checked"


def test_candidate_mode_cannot_use_original_strong_evaluator(retrieval):
    r, _, _ = retrieval
    r.config = pilot_config(r.config)
    with pytest.raises(CmrError, match="SCOPE_DENIED"):
        Evaluator(r.config, r).freeze({"questions": []})


@pytest.mark.asyncio
async def test_candidate_daemon_status_and_eval_gate_are_local_contract(engine):
    e, _ = engine
    e.config = pilot_config(e.config)
    app = create_app(e.config, e)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
        status = await client.get("/v1/status", params={"project_id": "p"})
        assert "CANDIDATE_PILOT" in status.json()["warnings"]
        result = await client.post("/admin/eval", json={"project_id": "p", "suite": {"questions": []}})
        assert result.status_code == 409 and result.json() == {"error": "SCOPE_DENIED"}
