"""Private UDS service. A single priority worker owns inference and mutations."""
import asyncio
import concurrent.futures
import fcntl
import itertools
import json
import os
import queue
import stat
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .schemas import CmrError, ContextRequest, SearchRequest
from .store import atomic_json


class DaemonLease:
    def __init__(self, directory: Path):
        self.directory, self.fd, self.owned = directory, None, False

    def acquire(self):
        if not self.directory.exists():
            self.directory.mkdir(mode=0o700, parents=True)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise CmrError("SCOPE_DENIED")
        if len(str(self.directory / "daemon.sock").encode()) > 100:
            raise CmrError("CONFIG_CONFLICT")
        self.fd = os.open(self.directory / "daemon.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        info = os.fstat(self.fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            self.close()
            raise CmrError("SCOPE_DENIED")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.close()
            return False
        self.owned = True
        os.ftruncate(self.fd, 0)
        os.write(self.fd, str(os.getpid()).encode())
        return True

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.owned = False

    def bind_socket(self):
        if not self.owned:
            raise CmrError("SCOPE_DENIED")
        path = self.directory / "daemon.sock"
        if path.exists():
            info = path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise CmrError("SCOPE_DENIED")
            path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(path))
            os.chmod(path, 0o600)
            sock.listen(128)
        except BaseException:
            sock.close()
            raise
        return sock


class PriorityWorker:
    def __init__(self):
        self.tasks = queue.PriorityQueue()
        self.counter = itertools.count()
        self.thread = None
        self.closed = False

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True, name="cmr-single-writer")
            self.thread.start()

    def _run(self):
        while True:
            _, _, call, future = self.tasks.get()
            if call is None:
                return
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(call())
                except Exception as exc:
                    future.set_exception(exc)

    async def submit(self, call, priority=0):
        if self.closed:
            raise CmrError("INDEX_NOT_READY")
        self.start()
        future = concurrent.futures.Future()
        self.tasks.put((priority, next(self.counter), call, future))
        return await asyncio.wrap_future(future)

    def close(self):
        self.closed = True
        self.tasks.put((1000, next(self.counter), None, None))
        if self.thread:
            self.thread.join(timeout=1)


class ProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str


class EvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str
    suite: dict
    lock: dict | None = None
    calibration: bool = False


def create_app(config, engine=None):
    from .metadata_log import emit, metadata_logger
    logger = metadata_logger(config.storage.runtime_dir)
    worker = PriorityWorker()
    state = {"engine": engine, "loading": engine is None, "error": None, "syncing": False, "operations": {},
             "reranker": None, "reranker_error": None}

    async def synchronize(project, rebuild=False):
        if state["syncing"]:
            raise CmrError("INDEX_NOT_READY")
        current = state["engine"]
        if current is None:
            raise CmrError("MODEL_UNAVAILABLE" if state["error"] else "INDEX_NOT_READY")
        state["syncing"] = True
        old_table, old_generation = current.store.table, current.store.generation
        receipts = []
        try:
            sources = current.sources(project)
            if rebuild and current.store.active_path.exists() and not current.store.active_matches():
                sources = [s for s in config.sources if s.enabled and s.allow_cloud_context]
            # Preflight all roots before opening a staging generation.
            plans = []
            for source in sources:
                steps = current.plan_source_steps(source)

                def advance():
                    # StopIteration cannot cross an asyncio Future boundary.
                    try:
                        next(steps)
                        return False, None
                    except StopIteration as completed:
                        return True, completed.value

                try:
                    while True:
                        done, plan = await worker.submit(advance, 10)
                        if done:
                            plans.append((source, plan))
                            break
                finally:
                    # A cancelled scan may retain directory descriptors. Close
                    # on the owning worker after any running step has finished.
                    await asyncio.shield(worker.submit(steps.close, 0))
            if rebuild:
                from .sources import build_corpus_manifest
                corpus_before = await worker.submit(lambda: build_corpus_manifest(sources).corpus_manifest_hash, 10)
                await worker.submit(lambda: current.begin_rebuild(project), 10)
            for source, plan in plans:
                changed, deleted = plan
                if rebuild:
                    from .sources import scan_source
                    changed = await worker.submit(lambda s=source: scan_source(s).allowed, 10)
                    deleted = []
                for path in changed:
                    job = await worker.submit(lambda s=source, p=path: current.prepare_file(s, p, encode=False), 10)
                    for i in range(0, len(job.chunks), config.embedding.batch_size):
                        batch = job.chunks[i:i + config.embedding.batch_size]
                        await worker.submit(lambda j=job, b=batch: current.encode_job_batch(j, b), 10)
                    receipt = await worker.submit(lambda j=job: current.commit_file(j), 10)
                    receipts.append(receipt.__dict__)
                for key in deleted:
                    await worker.submit(lambda k=key: current.store.delete_source(k), 10)
                current.states[source.source_id]["state"] = "ready"
                current.states[source.source_id]["last_sync"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
            if rebuild:
                corpus_after = await worker.submit(lambda: build_corpus_manifest(sources).corpus_manifest_hash, 10)
                if corpus_after != corpus_before:
                    raise CmrError("SOURCE_CHANGED")
                await worker.submit(lambda: current.store.activate_generation(current.store.generation.generation_id), 10)
            return {"project_id": project, "commits": len(receipts), "generation": current.store.generation.generation_id}
        except BaseException:
            if rebuild:
                # Cancellation must not swap the writer table underneath an
                # already-running native transaction on the worker thread.
                await asyncio.shield(worker.submit(lambda: None, 0))
                current.store.table, current.store.generation = old_table, old_generation
            raise
        finally:
            state["syncing"] = False

    async def load_and_watch():
        try:
            def load():
                from .embedding import Encoder
                from .indexing import Indexer
                dependency_lock = Path(__file__).parents[2] / "uv.lock"
                indexer = Indexer(config, Encoder(config.storage.runtime_dir, config.runtime.backend, dependency_lock))
                if config.retrieval.reranker.enabled:
                    from .reranking import Reranker
                    try:
                        state["reranker"] = Reranker(config.retrieval.reranker, dependency_lock)
                    except CmrError as exc:
                        state["reranker_error"] = exc.code
                return indexer
            state["engine"] = await worker.submit(load)
            state["loading"] = False
        except Exception as exc:
            state["error"] = exc.code if isinstance(exc, CmrError) else "MODEL_UNAVAILABLE"
            if state["error"] == "MODEL_UNAVAILABLE":
                try:
                    from .embedding import unavailable_encoder
                    from .indexing import Indexer
                    state["engine"] = await worker.submit(lambda: Indexer(config, unavailable_encoder(config.storage.runtime_dir, config.runtime.backend)))
                except (CmrError, OSError, ValueError):
                    pass
            state["loading"] = False
            return
        while True:
            if state["engine"].store.table is not None:
                for project in sorted({s.project_id for s in config.sources if s.enabled and s.allow_cloud_context}):
                    try:
                        await synchronize(project)
                    except CmrError:
                        pass
            await asyncio.sleep(config.runtime.reconcile_seconds)

    async def watch():
        from watchfiles import DefaultFilter, awatch
        from .sources import EXCLUDED_PARTS, git_state
        roots = []
        for source in config.sources:
            if not source.enabled or not source.allow_cloud_context:
                continue
            roots.append(source.root)
            if source.kind == "code":
                try:
                    metadata = git_state(source)
                    roots.extend([Path(metadata["gitdir"]), Path(metadata["common_dir"])])
                except CmrError:
                    pass
        roots = list(dict.fromkeys(p for p in roots if p.is_dir()))
        if not roots:
            return
        ignore_dirs = tuple(set(DefaultFilter.ignore_dirs) | EXCLUDED_PARTS)
        async for _ in awatch(*roots, watch_filter=DefaultFilter(ignore_dirs=ignore_dirs), debounce=int(config.runtime.debounce_seconds * 1000), force_polling=config.runtime.force_polling, poll_delay_ms=500):
            if state["loading"] or state["engine"] is None or state["engine"].store.table is None:
                continue
            for project in sorted({s.project_id for s in config.sources if s.enabled and s.allow_cloud_context}):
                try:
                    await synchronize(project)
                except CmrError:
                    pass

    @asynccontextmanager
    async def lifespan(app):
        tasks = [asyncio.create_task(load_and_watch()), asyncio.create_task(watch())] if engine is None else []
        yield
        tasks += list(state.get("operation_tasks", set()))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await worker.submit(lambda: None, 0)
        worker.close()
        if engine is None:
            state["reranker"] = None
            state["engine"] = None

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.cmr = state
    app.state.synchronize = synchronize

    def dispatch(project, coroutine_factory):
        operation = __import__("uuid").uuid4().hex
        record = {"operation_id": operation, "project_id": project, "state": "running"}
        state["operations"][operation] = record
        async def execute():
            try:
                record["result"] = await coroutine_factory()
                record["state"] = "complete"
            except Exception as exc:
                record["error"] = exc.code if isinstance(exc, CmrError) else "INDEX_NOT_READY"
                record["state"] = "failed"
            atomic_json(config.storage.runtime_dir / "data/manifests" / ("operation-" + operation + ".json"), record)
            emit(logger, "operation", operation_id=operation, project_id=project, state=record["state"], error=record.get("error"))
        task = asyncio.create_task(execute())
        state.setdefault("operation_tasks", set()).add(task)
        task.add_done_callback(state["operation_tasks"].discard)
        return {"operation_id": operation, "state": "running"}

    @app.exception_handler(CmrError)
    async def error_handler(request: Request, exc: CmrError):
        return JSONResponse({"error": exc.code}, status_code=503 if exc.code in {"MODEL_UNAVAILABLE", "INDEX_NOT_READY"} else 409)

    @app.get("/v1/status")
    async def status(project_id: str | None = Query(default=None)):
        e = state["engine"]
        sources = [s for s in config.sources if s.project_id == project_id and s.enabled and s.allow_cloud_context]
        if project_id and not sources:
            raise CmrError("SOURCE_NOT_CONFIGURED")
        result = {"model_id": config.embedding.model_id, "dimension": 768, "pid": os.getpid(),
                "peak_rss_bytes": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * (1 if __import__("sys").platform == "darwin" else 1024),
                "loading": state["loading"], "error": state["error"], "backend": config.runtime.backend,
                "queue": worker.tasks.qsize(), "syncing": state["syncing"],
                "embedding_profile": e.encoder.profile_id if e else None,
                "reranker": {"enabled": config.retrieval.reranker.enabled, "ready": state["reranker"] is not None,
                             "profile": state["reranker"].profile_id if state["reranker"] else None,
                             "backend": config.retrieval.reranker.backend,
                             "error": state["reranker_error"]},
                "generation": e.store.generation.generation_id if e and e.store.generation else None,
                "operations": [{k: v for k, v in op.items() if k != "result"} for op in state["operations"].values() if op["project_id"] == project_id],
                "sources": {s.source_id: e.states.get(s.source_id, {"state": "not_indexed"}) if e else {"state": "not_ready"} for s in sources}}
        if config.candidate_pilot.enabled:
            from .candidate_pilot import candidate_warnings
            result["warnings"] = candidate_warnings(config)
        return result

    @app.post("/v1/search")
    async def search(request: SearchRequest):
        from .retrieval import Retriever
        e = state["engine"]
        if e is None:
            raise CmrError("MODEL_UNAVAILABLE" if state["error"] else "INDEX_NOT_READY")
        result = await worker.submit(lambda: Retriever(config, e.store, e.encoder, state["reranker"]).search(request))
        return result.model_dump()

    @app.post("/v1/context")
    async def context(request: ContextRequest):
        from .retrieval import Retriever
        e = state["engine"]
        if e is None:
            raise CmrError("INDEX_NOT_READY")
        return await worker.submit(lambda: Retriever(config, e.store, e.encoder).read_context(request))

    @app.post("/admin/sync")
    async def sync(request: ProjectRequest):
        if state["engine"] is None or state["syncing"]:
            raise CmrError("INDEX_NOT_READY")
        state["engine"].sources(request.project_id)
        return dispatch(request.project_id, lambda: synchronize(request.project_id))

    @app.post("/admin/rebuild")
    async def rebuild(request: ProjectRequest):
        if state["engine"] is None or state["syncing"]:
            raise CmrError("INDEX_NOT_READY")
        state["engine"].sources(request.project_id)
        return dispatch(request.project_id, lambda: synchronize(request.project_id, rebuild=True))

    @app.post("/admin/eval")
    async def evaluate(request: EvalRequest):
        if config.candidate_pilot.enabled:
            raise CmrError("SCOPE_DENIED")
        if state["engine"] is None or state["syncing"]:
            raise CmrError("INDEX_NOT_READY")
        state["engine"].sources(request.project_id)
        projects = {s.project_id for s in config.sources}
        if any(q.get("project_id") != request.project_id and (q.get("positive") or q.get("expected_error") != "SOURCE_NOT_CONFIGURED" or q.get("project_id") in projects) for q in request.suite.get("questions", [])):
            raise CmrError("SCOPE_DENIED")
        async def run():
            from .evaluation import Evaluator
            from .retrieval import Retriever
            from .schemas import EvaluationLock
            e = state["engine"]
            evaluator = Evaluator(config, Retriever(config, e.store, e.encoder, state["reranker"]))
            lock = EvaluationLock.model_validate(request.lock) if request.lock else await worker.submit(lambda: evaluator.freeze(request.suite), 10)
            result = await evaluator.calibration_samples_async(request.suite, lock, worker.submit) if request.calibration else await evaluator.evaluate_async(request.suite, lock, worker.submit)
            atomic_json(config.storage.runtime_dir / "reports/evaluation-latest.json", result)
            return {"report": str(config.storage.runtime_dir / "reports/evaluation-latest.json"), "lock": lock.model_dump()}
        return dispatch(request.project_id, run)

    return app


def serve(config):
    import uvicorn
    os.umask(0o077)
    directory = config.storage.runtime_dir / "run"
    lease = DaemonLease(directory)
    if not lease.acquire():
        from .client import request
        print(json.dumps(request(config, "GET", "/v1/status")))
        return
    socket_path = directory / "daemon.sock"
    bound_socket = None
    try:
        bound_socket = lease.bind_socket()
        from .metadata_log import emit, metadata_logger
        emit(metadata_logger(config.storage.runtime_dir), "startup", pid=os.getpid(), backend=config.runtime.backend)
        server = uvicorn.Server(uvicorn.Config(create_app(config), workers=1, reload=False,
                    access_log=False, ws="none", log_level="warning"))
        server.run(sockets=[bound_socket])
    finally:
        if bound_socket:
            bound_socket.close()
        if socket_path.exists() and stat.S_ISSOCK(socket_path.lstat().st_mode):
            socket_path.unlink()
        lease.close()
