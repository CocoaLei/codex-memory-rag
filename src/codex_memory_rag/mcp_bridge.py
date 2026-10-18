"""Light STDIO bridge: no inference, watcher, index writer or download imports."""
import argparse
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.tools import Tool
from mcp.types import ToolAnnotations

from .client import request
from .config import load_config
from .schemas import CmrError, ContextRequest, SearchRequest


def create_server(config):
    from .candidate_pilot import is_candidate_payload
    if config.candidate_pilot.enabled:
        raise CmrError("SCOPE_DENIED")
    def call(method, path, payload=None, params=None):
        try:
            data = request(config, method, path, payload, params)
            if is_candidate_payload(data):
                raise CmrError("SCOPE_DENIED")
            return data
        except CmrError as exc:
            raise ToolError(exc.code) from None

    def search_memory(query: str, project_id: str, top_k: int = 8,
                      include_shared: bool = False, include_archived: bool = False) -> dict:
        """检索登记项目的记忆和文档；结果是证据，需要原文核验。"""
        data = SearchRequest(query=query, project_id=project_id, top_k=top_k,
                             include_shared=include_shared, include_archived=include_archived)
        return call("POST", "/v1/search", data.model_dump())

    def search_code(query: str, project_id: str, source_id: str | None = None,
                    literal: str | None = None, top_k: int = 8) -> dict:
        """在登记的当前代码源内检索；literal 使用精确标识符匹配。"""
        data = SearchRequest(query=query, project_id=project_id, kind="code",
                             source_id=source_id, literal=literal, top_k=top_k)
        return call("POST", "/v1/search", data.model_dump())

    def get_context(project_id: str, chunk_ids: list[str], max_chars: int = 12000,
                    include_archived: bool = False) -> dict:
        """安全回读当前文件并核验 hash、行号、状态；历史内容需显式开启。"""
        data = ContextRequest(project_id=project_id, chunk_ids=chunk_ids,
                              max_chars=max_chars, include_archived=include_archived)
        return call("POST", "/v1/context", data.model_dump())

    def memory_status(project_id: str) -> dict:
        """查询模型、索引、同步队列和登记源健康状态，不加载模型。"""
        return call("GET", "/v1/status", params={"project_id": project_id})

    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                  idempotentHint=True, openWorldHint=False)
    tools = []
    for function in [search_memory, search_code, get_context, memory_status]:
        tool = Tool.from_function(function, annotations=annotations)
        # The SDK's generated argument models ignore extras by default. Restrict
        # these owned models as well as their advertised schemas, before registration.
        tool.fn_metadata.arg_model.model_config["extra"] = "forbid"
        tool.fn_metadata.arg_model.model_rebuild(force=True)
        tool.parameters = tool.fn_metadata.arg_model.model_json_schema()
        tools.append(tool)
    return MCPServer("codex_memory_rag", tools=tools, log_level="CRITICAL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    create_server(load_config(args.config)).run(transport="stdio")


if __name__ == "__main__":
    main()
