"""CLI contracts over a real temporary UDS, using no real model or business source."""
import contextlib
import json
import socketserver
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest
import tomlkit

from codex_memory_rag import cli
from codex_memory_rag.schemas import CmrError, ContextRequest, SearchRequest
from test_candidate_pilot import pilot_config
from test_retrieval import retrieval as retrieval


def invoke(monkeypatch, capsys, config_path, *args):
    monkeypatch.setattr(sys, "argv", ["cmr", "--config", str(config_path), *args])
    code = 0
    try:
        cli.main()
    except SystemExit as exc:
        code = exc.code
    output = capsys.readouterr()
    return code, output.out, output.err


def write_config(config, path):
    path.write_text(tomlkit.dumps(config.model_dump(mode="json", exclude_none=True)))
    return path


@contextlib.contextmanager
def local_uds(reader):
    with tempfile.TemporaryDirectory(prefix="cmr-r2-wire-", dir="/private/tmp") as directory:
        reader.config.storage.runtime_dir = Path(directory)
        run = Path(directory) / "run"
        run.mkdir(mode=0o700)
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                try:
                    if self.path == "/v1/search":
                        result = reader.search(SearchRequest(**payload)).model_dump()
                    elif self.path == "/v1/context":
                        result = reader.read_context(ContextRequest(**payload))
                    else:
                        raise AssertionError("Unexpected route " + self.path)
                    status = 200
                except CmrError as exc:
                    result, status = {"error": exc.code}, 409
                raw = json.dumps(result, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass
        with socketserver.UnixStreamServer(str(run / "daemon.sock"), Handler) as server:
            (run / "daemon.sock").chmod(0o600)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield Path(directory)
            finally:
                server.shutdown()
                thread.join(timeout=5)
                assert not thread.is_alive()


def test_real_cli_search_context_partial_and_source_changed(retrieval):
    r, source, _ = retrieval
    r.config = pilot_config(r.config)
    with local_uds(r) as directory:
        config = write_config(r.config, directory / "config.toml")
        def run(*args):
            return subprocess.run(
                [sys.executable, "-B", "-m", "codex_memory_rag.cli", "--config", str(config), *args],
                capture_output=True, text=True, timeout=15
            )
        search = run("search", "--project", "p", "--query", "退款", "--mode", "lexical")
        assert search.returncode == 0, search.stderr
        found = json.loads(search.stdout)
        assert "完整支持未检查" in found["warnings"]
        assert found["results"][0]["path"] == str(source.root / "a.md")
        key = found["results"][0]["chunk_id"]
        full = run("context", "--project", "p", "--chunk-id", key)
        assert full.returncode == 0, full.stderr
        value = json.loads(full.stdout)
        assert value["readback_status"] == "complete" and value["truncated"] is False
        assert "退款恢复" in value["items"][0]["text"]
        assert value["items"][0]["path"] == str(source.root / "a.md")
        assert value["items"][0]["support_status"] == "not_checked"
        partial = run("context", "--project", "p", "--chunk-id", key, "--max-chars", "2")
        assert partial.returncode == 3, partial.stderr
        value = json.loads(partial.stdout)
        assert value["readback_status"] == "partial" and "原文回读未完成" in value["warnings"]
        assert value["items"][0]["file_sha256"] == found["results"][0]["file_sha256"]
        (source.root / "a.md").write_text(
            "---\ncode_project: p\nstatus: active\n---\nsynthetic source changed\n"
        )
        changed = run("context", "--project", "p", "--chunk-id", key)
        assert changed.returncode == 2
        error = json.loads(changed.stdout)
        assert error["error"] == "SOURCE_CHANGED" and error["readback_status"] == "error"
        assert set(error) == {"error", "readback_status", "client_elapsed_ms"}
        assert error["client_elapsed_ms"] >= 0
        assert "synthetic source changed" not in changed.stdout
        (source.root / "a.md").write_text("synthetic metadata qualification removed")
        denied = run("context", "--project", "p", "--chunk-id", key)
        assert denied.returncode == 2
        error = json.loads(denied.stdout)
        assert error["error"] == "SCOPE_DENIED" and error["readback_status"] == "error"
        assert error["client_elapsed_ms"] >= 0


@pytest.mark.parametrize("flag,value", [("--max-chars", "0"), ("--max-chars", "32001"), ("--chunk-id", "bad-id")])
def test_cli_context_bad_input_is_a_structured_failure(retrieval, monkeypatch, capsys, tmp_path, flag, value):
    r, _, _ = retrieval
    path = write_config(r.config, tmp_path / "cfg.toml")
    def unavailable(*args, **kwargs):
        raise CmrError("MODEL_UNAVAILABLE")
    monkeypatch.setattr("codex_memory_rag.client.request", unavailable)
    code, out, _ = invoke(monkeypatch, capsys, path, "context", "--project", "p",
                          "--chunk-id", "a" * 64, flag, value)
    assert code == 2 and out, "context must report invalid input without a traceback"
    data = json.loads(out)
    assert data["readback_status"] == "error"
    assert data["error"] in {"INVALID_REQUEST", "SCOPE_DENIED"}


def test_cli_context_missing_ids_is_partial_even_without_backend_truncated_flag(retrieval, monkeypatch, capsys, tmp_path):
    r, _, _ = retrieval
    hit = r.search(SearchRequest(project_id="p", query="退款", mode="lexical")).results[0]
    data = r.read_context(ContextRequest(project_id="p", chunk_ids=[hit["chunk_id"]]))
    path = write_config(r.config, tmp_path / "cfg.toml")
    monkeypatch.setattr("codex_memory_rag.client.request", lambda *a, **k: data)
    code, out, _ = invoke(monkeypatch, capsys, path, "context", "--project", "p",
                          "--chunk-id", hit["chunk_id"], "--chunk-id", "b" * 64)
    assert code == 3 and out
    value = json.loads(out)
    assert value["readback_status"] == "partial" and "原文回读未完成" in value["warnings"]


@pytest.mark.parametrize("change", [
    {"project_id": "other"}, {"relative_path": "../private.md"},
    {"file_sha256": "bad"}, {"worktree_id": "foreign-worktree"},
])
def test_cli_context_rejects_mismatched_provenance_without_echoing_text(retrieval, monkeypatch, capsys, tmp_path, change):
    r, _, _ = retrieval
    hit = r.search(SearchRequest(project_id="p", query="退款", mode="lexical")).results[0]
    data = r.read_context(ContextRequest(project_id="p", chunk_ids=[hit["chunk_id"]]))
    data["items"][0].update(change | {"text": "SYNTHETIC_PRIVATE_TEXT"})
    path = write_config(r.config, tmp_path / "cfg.toml")
    monkeypatch.setattr("codex_memory_rag.client.request", lambda *a, **k: data)
    code, out, _ = invoke(monkeypatch, capsys, path, "context", "--project", "p",
                          "--chunk-id", hit["chunk_id"])
    assert code == 2 and out
    assert json.loads(out)["readback_status"] == "error"
    assert "SYNTHETIC_PRIVATE_TEXT" not in out


def test_candidate_cli_rejects_legacy_daemon_response(retrieval, monkeypatch, capsys, tmp_path):
    r, _, _ = retrieval
    response = r.search(SearchRequest(project_id="p", query="退款", mode="lexical")).model_dump()
    config = pilot_config(r.config)
    path = write_config(config, tmp_path / "cfg.toml")
    monkeypatch.setattr("codex_memory_rag.client.request", lambda *a, **k: response)
    code, out, _ = invoke(monkeypatch, capsys, path, "search", "--project", "p", "--query", "退款")
    assert code == 2
    error = json.loads(out)
    assert error["error"] == "CANDIDATE_CONTRACT_MISMATCH"
    assert set(error) == {"error", "client_elapsed_ms"}


@pytest.mark.parametrize("command", ["install", "rollback", "eval"])
def test_candidate_cli_cannot_install_mcp_or_run_original_quality(retrieval, monkeypatch, capsys, tmp_path, command):
    r, _, _ = retrieval
    path = write_config(pilot_config(r.config), tmp_path / "cfg.toml")
    args = ["--suite", str(tmp_path / "absent.json"), "--output", str(tmp_path / "absent-output.json")] if command == "eval" else ["--apply"]
    code, out, _ = invoke(monkeypatch, capsys, path, command, *args)
    assert code == 2 and out
    assert json.loads(out) == {"error": "SCOPE_DENIED"}
    assert not (r.config.storage.runtime_dir / "backups/install-manifest.json").exists()


def test_legacy_cli_search_output_is_unchanged(retrieval, monkeypatch, capsys, tmp_path):
    r, _, _ = retrieval
    data = r.search(SearchRequest(project_id="p", query="退款", mode="lexical")).model_dump()
    path = write_config(r.config, tmp_path / "cfg.toml")
    monkeypatch.setattr("codex_memory_rag.client.request", lambda *a, **k: data)
    code, out, _ = invoke(monkeypatch, capsys, path, "search", "--project", "p", "--query", "退款")
    assert code == 0 and json.loads(out) == data


@pytest.mark.parametrize("change", [
    {"warnings": None}, {"items": "invalid"}, {"truncated": "false"},
    {"generation": None}, {"table_version": "7"},
])
def test_cli_context_malformed_frame_is_structured_error(retrieval, monkeypatch, capsys, tmp_path, change):
    r, _, _ = retrieval
    hit = r.search(SearchRequest(project_id="p", query="退款", mode="lexical")).results[0]
    data = r.read_context(ContextRequest(project_id="p", chunk_ids=[hit["chunk_id"]]))
    data.update(change)
    path = write_config(r.config, tmp_path / "cfg.toml")
    monkeypatch.setattr("codex_memory_rag.client.request", lambda *a, **k: data)
    code, out, _ = invoke(monkeypatch, capsys, path, "context", "--project", "p",
                          "--chunk-id", hit["chunk_id"])
    assert code == 2 and out
    assert json.loads(out)["readback_status"] == "error"


def test_candidate_client_time_includes_transport_wait_and_field_validation(retrieval, monkeypatch, capsys, tmp_path):
    r, _, _ = retrieval
    r.config = pilot_config(r.config)
    response = r.search(SearchRequest(project_id="p", query="退款", mode="lexical")).model_dump()
    response["elapsed_ms"] = 2.0
    path = write_config(r.config, tmp_path / "cfg.toml")
    clock = {"seconds": 10.0}
    monkeypatch.setattr(cli.time, "perf_counter", lambda: clock["seconds"])
    def waiting_transport(*a, **k):
        clock["seconds"] += 1.0
        return response
    monkeypatch.setattr("codex_memory_rag.client.request", waiting_transport)
    from codex_memory_rag import candidate_pilot
    real_check = candidate_pilot._item_with_path
    def checking(*a, **k):
        clock["seconds"] += .25
        return real_check(*a, **k)
    monkeypatch.setattr(candidate_pilot, "_item_with_path", checking)
    code, out, _ = invoke(monkeypatch, capsys, path, "search", "--project", "p", "--query", "退款")
    assert code == 0
    result = json.loads(out)
    assert result.get("client_elapsed_ms") == 1250.0
    assert result["elapsed_ms"] == 2.0


def test_candidate_transport_error_keeps_wait_duration(retrieval, monkeypatch, capsys, tmp_path):
    r, _, _ = retrieval
    path = write_config(pilot_config(r.config), tmp_path / "cfg.toml")
    clock = {"seconds": 10.0}
    monkeypatch.setattr(cli.time, "perf_counter", lambda: clock["seconds"])
    def failed_transport(*a, **k):
        clock["seconds"] += 1.0
        raise CmrError("MODEL_UNAVAILABLE")
    monkeypatch.setattr("codex_memory_rag.client.request", failed_transport)
    code, out, _ = invoke(monkeypatch, capsys, path, "context", "--project", "p",
                          "--chunk-id", "a" * 64)
    assert code == 2 and out
    result = json.loads(out)
    assert result["error"] == "MODEL_UNAVAILABLE"
    assert result["readback_status"] == "error"
    assert result.get("client_elapsed_ms") == 1000.0


@pytest.mark.parametrize("field,value,missing", [
    ("generation", None, True), ("table_version", None, True),
    ("mode", None, True), ("truncated", None, True),
    ("generation", None, False), ("generation", "", False),
    ("table_version", "7", False), ("table_version", True, False),
    ("table_version", -1, False), ("mode", None, False),
    ("mode", "", False), ("truncated", "false", False),
    ("warnings", ["CANDIDATE_PILOT", "SUPPORT_NOT_CHECKED", "完整支持未检查", 1], False),
])
def test_candidate_search_rejects_incomplete_or_invalid_frame(
    retrieval, monkeypatch, capsys, tmp_path, field, value, missing
):
    r, _, _ = retrieval
    r.config = pilot_config(r.config)
    response = r.search(SearchRequest(project_id="p", query="无匹配", mode="lexical")).model_dump()
    assert response["results"] == []
    if missing:
        del response[field]
    else:
        response[field] = value
    path = write_config(r.config, tmp_path / "cfg.toml")
    monkeypatch.setattr("codex_memory_rag.client.request", lambda *a, **k: response)
    code, out, _ = invoke(monkeypatch, capsys, path, "search", "--project", "p", "--query", "无匹配")
    assert code == 2, "incomplete candidate frames must not be reported as a successful search"
    error = json.loads(out)
    assert error["error"] == "RETRIEVAL_RESPONSE_INVALID"
    assert set(error) == {"error", "client_elapsed_ms"}
    assert error["client_elapsed_ms"] >= 0


@pytest.mark.parametrize("mode", [
    "hybrid", "vector", "lexical", "literal", "lexical_only", "unavailable",
    "uncalibrated", "hybrid_without_reranker", "hybrid_reranked",
])
def test_candidate_search_preserves_actual_mode_diagnostics(
    retrieval, monkeypatch, capsys, tmp_path, mode
):
    r, _, _ = retrieval
    r.config = pilot_config(r.config)
    response = r.search(SearchRequest(project_id="p", query="无匹配", mode="lexical")).model_dump()
    response["mode"] = mode
    path = write_config(r.config, tmp_path / "cfg.toml")
    monkeypatch.setattr("codex_memory_rag.client.request", lambda *a, **k: response)
    code, out, _ = invoke(monkeypatch, capsys, path, "search", "--project", "p", "--query", "无匹配")
    assert code == 0
    result = json.loads(out)
    assert result["mode"] == mode
    assert result["generation"] == response["generation"]
    assert result["table_version"] == response["table_version"]
