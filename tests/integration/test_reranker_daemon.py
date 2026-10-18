import asyncio

import httpx
import pytest

from codex_memory_rag.daemon import create_app
from codex_memory_rag.schemas import CmrError
from test_watcher import engine, Encoder
from test_reranker_retrieval import enable, Scores


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", [False, True])
async def test_daemon_owns_one_reranker_and_reports_load_failure(engine, monkeypatch, unavailable):
    import codex_memory_rag.embedding as embedding
    import codex_memory_rag.reranking as reranking
    e, s = engine
    (s.root / "a.md").write_text("退款恢复")
    e.build("p")
    enable(e)
    calls = []
    monkeypatch.setattr(embedding, "Encoder", lambda *args: Encoder())
    def load(*args):
        calls.append(1)
        if unavailable:
            raise CmrError("RERANKER_UNAVAILABLE")
        return Scores(0.9)
    monkeypatch.setattr(reranking, "Reranker", load)
    app = create_app(e.config)
    async with app.router.lifespan_context(app):
        for _ in range(100):
            if not app.state.cmr["loading"]:
                break
            await asyncio.sleep(0.01)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            status = (await client.get("/v1/status", params={"project_id": "p"})).json()
            assert status["error"] is None
            assert status["reranker"]["ready"] is not unavailable
            assert status["reranker"]["error"] == ("RERANKER_UNAVAILABLE" if unavailable else None)
            for _ in range(2):
                out = (await client.post("/v1/search", json={"project_id": "p", "query": "退款"})).json()
                assert out["results"]
                assert ("RERANKER_UNAVAILABLE" in out["warnings"]) is unavailable
    assert calls == [1]
    assert app.state.cmr["reranker"] is None
