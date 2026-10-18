import pytest
from mcp.server.mcpserver.exceptions import ToolError

from codex_memory_rag.mcp_bridge import create_server
from codex_memory_rag.schemas import AppConfig, CmrError
from test_candidate_pilot import pilot_config


def test_candidate_configuration_cannot_create_mcp_bridge():
    with pytest.raises(CmrError, match="SCOPE_DENIED"):
        create_server(pilot_config(AppConfig()))


@pytest.mark.asyncio
@pytest.mark.parametrize("method,args,payload", [
    ("search_memory", {"query": "synthetic", "project_id": "p"},
     {"results": [{"retrieval_role": "related_candidate", "support_status": "not_checked", "summary": "SYNTHETIC_PRIVATE"}], "warnings": ["CANDIDATE_PILOT"]}),
    ("get_context", {"project_id": "p", "chunk_ids": ["a" * 64]},
     {"items": [{"retrieval_role": "related_candidate", "support_status": "not_checked", "text": "SYNTHETIC_PRIVATE"}]}),
    ("memory_status", {"project_id": "p"}, {"warnings": ["CANDIDATE_PILOT"]}),
])
async def test_legacy_bridge_cannot_forward_candidate_daemon_payload(monkeypatch, method, args, payload):
    monkeypatch.setattr("codex_memory_rag.mcp_bridge.request", lambda *a, **k: payload)
    server = create_server(AppConfig())
    with pytest.raises(ToolError, match="SCOPE_DENIED") as error:
        await server.call_tool(method, args)
    assert "SYNTHETIC_PRIVATE" not in str(error.value)
