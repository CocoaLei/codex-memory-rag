import sys,asyncio
from pathlib import Path
import pytest
from mcp.client.stdio import stdio_client,StdioServerParameters
from mcp.client.session import ClientSession
from codex_memory_rag.mcp_bridge import create_server
from codex_memory_rag.schemas import AppConfig

@pytest.mark.asyncio
async def test_four_read_only_tools_and_strict_arguments_without_model():
    s=create_server(AppConfig());tools=await s.list_tools()
    assert {t.name for t in tools}=={'search_memory','search_code','get_context','memory_status'}
    assert all(t.annotations.read_only_hint and t.input_schema['additionalProperties'] is False for t in tools)
    assert 'torch' not in sys.modules
    for kwargs in [dict(query='x',project_id='p',path='/etc/passwd'),dict(query='x',project_id='p',top_k=99)]:
        with pytest.raises(Exception):await s.call_tool('search_memory',kwargs)

@pytest.mark.asyncio
async def test_real_stdio_initialize_list_and_call(tmp_path):
    cfg=tmp_path/'config.toml';cfg.write_text('sources=[]\n[storage]\nruntime_dir="'+str(tmp_path/'runtime')+'"\n')
    params=StdioServerParameters(command=sys.executable,args=['-m','codex_memory_rag.mcp_bridge','--config',str(cfg)])
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize();listed=await session.list_tools()
            assert len(listed.tools)==4
            result=await session.call_tool('memory_status',{'project_id':'p'})
            assert result.is_error
