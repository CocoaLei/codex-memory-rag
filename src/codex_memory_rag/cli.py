"""CLI administration; queries will go through the single daemon."""
import argparse
import importlib.metadata
import json
import platform
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import load_config
from .schemas import AppConfig, CmrError
from pydantic import ValidationError


def doctor(config: AppConfig) -> dict:
    return {"model_id": config.embedding.model_id, "dimension": 768, "architecture": platform.machine(),
            "python": platform.python_version(), "sources": len(config.sources),
            "runtime": str(config.storage.runtime_dir),
            "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "sentence-transformers", "lancedb", "mcp"]}}


def main():
    parser = argparse.ArgumentParser(prog="cmr")
    parser.add_argument("--config", type=Path, default=Path.home() / ".config/codex-memory-rag/config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("doctor")
    d.add_argument("--report", type=Path)
    m = commands.add_parser("model")
    m.add_argument("action", choices=["prepare", "smoke"])
    m.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    s = commands.add_parser("source")
    s.add_argument("action", choices=["audit"])
    commands.add_parser("serve")
    status = commands.add_parser("status")
    status.add_argument("--project", required=True)
    search = commands.add_parser("search")
    search.add_argument("--project", required=True)
    search.add_argument("--kind", choices=["memory", "code"], default="memory")
    search.add_argument("--query", required=True)
    search.add_argument("--literal")
    search.add_argument("--source")
    search.add_argument("--mode", choices=["hybrid", "vector", "lexical"], default="hybrid")
    search.add_argument("--history", action="store_true")
    context = commands.add_parser("context")
    context.add_argument("--project", required=True)
    context.add_argument("--chunk-id", action="append", required=True)
    context.add_argument("--max-chars", type=int, default=12000)
    context.add_argument("--history", action="store_true")
    index = commands.add_parser("index")
    index.add_argument("action", choices=["build", "sync"])
    index.add_argument("--project", required=True)
    install = commands.add_parser("install")
    install.add_argument("--apply", action="store_true")
    install.add_argument("--dry-run", action="store_true")
    rollback = commands.add_parser("rollback")
    rollback.add_argument("--apply", action="store_true")
    rollback.add_argument("--dry-run", action="store_true")
    service = commands.add_parser("service")
    service.add_argument("action", choices=["start", "stop", "status"])
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--suite", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config) if args.config.exists() else AppConfig()
    exit_code = 0
    client_started = None
    try:
        if config.candidate_pilot.enabled and args.command in {"install", "rollback", "eval"}:
            raise CmrError("SCOPE_DENIED")
        if args.command == "doctor":
            data = doctor(config)
            if args.report:
                args.report.write_text(json.dumps(data, indent=2) + "\n")
        elif args.command == "model":
            from .embedding import Encoder, prepare_model
            if args.action == "prepare":
                lock = prepare_model(config.storage.runtime_dir, Path(__file__).parents[2] / "uv.lock")
                data = {"revision": lock["revision"], "dimension": lock["dimension"], "files": len(lock["files"])}
            else:
                encoder = Encoder(config.storage.runtime_dir, args.device, Path(__file__).parents[2] / "uv.lock")
                vector = encoder.encode_queries(["本地检索状态"], "memory")
                data = {"profile": encoder.profile_id, "dimension": vector.shape[1], "dtype": str(vector.dtype), "device": args.device}
        elif args.command == "source":
            from .sources import scan_source
            data = {s.source_id: scan_source(s).model_dump() for s in config.sources if s.enabled and s.allow_cloud_context}
        elif args.command == "serve":
            from .daemon import serve
            serve(config)
            return
        elif args.command in {"install", "rollback"}:
            from .installer import Installer
            installer = Installer(Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))), config.storage.runtime_dir, Path(sys.executable), args.config)
            if args.command == "install":
                plan = installer.preview()
                installer.write_preview(plan)
                data = installer.apply(plan) if args.apply else {"files": [{"path": c["path"], "before_sha256": c["before_sha256"]} for c in plan["files"]], "launchagent_pending": True}
            else:
                data = installer.rollback(apply=args.apply)
        else:
            from .client import request
            from .schemas import ContextRequest, SearchRequest
            from .candidate_pilot import context_cli_output, is_candidate_payload, require_candidate_response, search_cli_output
            if args.command in {"search", "context"}:
                client_started = time.perf_counter()
            if args.command == "status":
                data = request(config, "GET", "/v1/status", params={"project_id": args.project})
                require_candidate_response(config, data)
            elif args.command == "search":
                query = SearchRequest(project_id=args.project, kind=args.kind, query=args.query,
                                      literal=args.literal, source_id=args.source, mode=args.mode, include_archived=args.history)
                data = request(config, "POST", "/v1/search", query.model_dump())
                data = search_cli_output(config, query, data)
            elif args.command == "context":
                try:
                    query = ContextRequest(project_id=args.project, chunk_ids=args.chunk_id,
                                           max_chars=args.max_chars, include_archived=args.history)
                except ValidationError:
                    raise CmrError("INVALID_REQUEST") from None
                import re
                if any(not re.fullmatch(r"[0-9a-f]{64}", key) for key in query.chunk_ids):
                    raise CmrError("SCOPE_DENIED")
                if not any(s.project_id == query.project_id and s.enabled and s.allow_cloud_context for s in config.sources):
                    raise CmrError("SOURCE_NOT_CONFIGURED")
                data = request(config, "POST", "/v1/context", query.model_dump())
                data, exit_code = context_cli_output(config, query, data)
            elif args.command in {"index", "eval"}:
                if args.command == "index":
                    project = args.project
                    value = request(config, "POST", "/admin/rebuild" if args.action == "build" else "/admin/sync", {"project_id": project})
                else:
                    suite = json.loads(args.suite.read_text())
                    projects = {q["project_id"] for q in suite["questions"]}
                    if len(projects) != 1:
                        raise CmrError("SCOPE_DENIED")
                    project = projects.pop()
                    value = request(config, "POST", "/admin/eval", {"project_id": project, "suite": suite})
                operation = value["operation_id"]
                deadline = time.monotonic() + 1800
                while time.monotonic() < deadline:
                    status = request(config, "GET", "/v1/status", params={"project_id": project})
                    record = next((r for r in status["operations"] if r["operation_id"] == operation), None)
                    if record and record["state"] in {"complete", "failed"}:
                        if record["state"] == "failed":
                            raise CmrError(record["error"])
                        receipt = config.storage.runtime_dir / "data/manifests" / ("operation-" + operation + ".json")
                        data = json.loads(receipt.read_text())["result"]
                        if args.command == "eval":
                            args.output.mkdir(parents=True, exist_ok=True)
                            import shutil
                            shutil.copy2(data["report"], args.output / "evaluation.json")
                        break
                    time.sleep(1)
                else:
                    raise CmrError("INDEX_NOT_READY")
            elif args.command == "service":
                if args.action == "start":
                    try:
                        data = request(config, "GET", "/v1/status")
                    except CmrError:
                        path = config.storage.runtime_dir / "logs/daemon.log"
                        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                        with os.fdopen(fd, "ab") as log:
                            child = subprocess.Popen([sys.executable, "-m", "codex_memory_rag.cli", "--config", str(args.config), "serve"], stdout=log, stderr=log, start_new_session=True)
                        data = {"started_pid": child.pid}
                else:
                    data = request(config, "GET", "/v1/status")
                    if args.action == "stop":
                        pid = int((config.storage.runtime_dir / "run/daemon.lock").read_text())
                        if pid != data["pid"]:
                            raise CmrError("CONFIG_CONFLICT")
                        os.kill(pid, signal.SIGTERM)
                        data = {"stopped_pid": pid}
        if client_started is not None and (config.candidate_pilot.enabled or is_candidate_payload(data)):
            data = data | {"client_elapsed_ms": (time.perf_counter() - client_started) * 1000}
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if exit_code:
            sys.exit(exit_code)
    except CmrError as exc:
        error = {"error": exc.code}
        if args.command == "context":
            error["readback_status"] = "error"
        if client_started is not None and config.candidate_pilot.enabled:
            error["client_elapsed_ms"] = (time.perf_counter() - client_started) * 1000
        print(json.dumps(error))
        sys.exit(2)


if __name__ == "__main__":
    main()
