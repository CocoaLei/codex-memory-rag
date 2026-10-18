import asyncio
import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_adapter import fixture,impl,search,KEY,WARN,context

ROOT=Path(__file__).parents[1]


def transport():
    path=ROOT/'candidate_transport.py'
    assert path.is_file(),'owned process and MCP transport missing'
    spec=importlib.util.spec_from_file_location('candidate_transport',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def envelope(body,code=0):
    guard={'role':'cli','input_hash':'d'*64,'metadata_verified':True,'content_verified':False,'verified_at':1.0}
    return {'exit_code':code,'body':body,'environment':{'before':guard,'after':guard},'process_wall_ms':1}


def runner(t,source,timeout=1,max_output=262144):
    return t.ProcessRunner(lambda args:[sys.executable,'-B','-c',source], 'd'*64,timeout=timeout,max_output=max_output)


def test_process_invokes_child_with_fixed_startup_flags_and_no_credentials(monkeypatch):
    t=transport();monkeypatch.setenv('OPENAI_API_KEY','SYNTHETIC_SECRET');monkeypatch.setenv('PYTHONPATH','/not-allowed')
    payload=envelope({'flags':'expected'})
    script='import os,sys,json;assert sys.flags.utf8_mode==0 and sys.flags.dont_write_bytecode;assert "OPENAI_API_KEY" not in os.environ and "PYTHONPATH" not in os.environ;print('+repr(json.dumps(payload))+')'
    assert asyncio.run(runner(t,script)(['status'],time.monotonic()+2))=={'flags':'expected'}


@pytest.mark.parametrize('bad',['exit','before','after','missing','malformed'])
def test_cli_guard_or_exit_disagreement_never_releases_body(bad):
    t=transport();v=envelope({'secret':'PRIVATE_BODY'})
    if bad=='exit':v['exit_code']=2
    elif bad in {'before','after'}:v['environment'][bad]=dict(v['environment'][bad],input_hash='e'*64)
    elif bad=='missing':v.pop('environment')
    source='print('+repr(json.dumps(v))+')' if bad!='malformed' else 'print("not json PRIVATE_BODY")'
    with pytest.raises(t.AdapterError):asyncio.run(runner(t,source)(['status'],time.monotonic()+2))


def test_child_failure_never_exposes_stderr_or_unknown_error_payload():
    t=transport();source='import sys;print("PRIVATE_STDERR",file=sys.stderr);print(\'{"error":"PRIVATE_ERROR"}\');sys.exit(2)'
    with pytest.raises(t.AdapterError) as e:asyncio.run(runner(t,source)(['status'],time.monotonic()+2))
    assert str(e.value)=='CLI_FAILED'


def test_explicit_guard_error_is_preserved_without_body():
    t=transport();source='import sys;print(\'{"error":"ENVIRONMENT_CHANGED"}\');sys.exit(2)'
    with pytest.raises(t.AdapterError,match='ENVIRONMENT_CHANGED'):asyncio.run(runner(t,source)(['status'],time.monotonic()+2))


def test_partial_context_exit_three_remains_distinguishable():
    t=transport();v=envelope(context('part',truncated=True,readback_status='partial'),3)
    source='import sys;print('+repr(json.dumps(v))+');sys.exit(3)'
    assert asyncio.run(runner(t,source)(['context'],time.monotonic()+2))['readback_status']=='partial'


def test_timeout_only_signals_unreaped_owned_child_group(monkeypatch):
    t=transport();calls=[];kill=t.os.killpg
    def own(pgid,sig):calls.append((pgid,sig));return kill(pgid,sig)
    monkeypatch.setattr(t.os,'killpg',own)
    with pytest.raises(t.AdapterError,match='CLI_TIMEOUT'):asyncio.run(runner(t,'import time;time.sleep(60)',.1)(['status'],time.monotonic()+2))
    assert len(calls)==1 and calls[0][0]>1 and calls[0][1]==signal.SIGKILL


def test_overlarge_output_is_bounded_and_rejected():
    t=transport()
    with pytest.raises(t.AdapterError,match='CLI_OUTPUT_LIMIT'):asyncio.run(runner(t,'import sys,time;sys.stdout.write("x"*100000);sys.stdout.flush();time.sleep(60)',max_output=1024)(['status'],time.monotonic()+2))


def test_cancel_waits_for_own_process_cleanup_before_releasing_admission(tmp_path,monkeypatch):
    t=transport();started=tmp_path/'pid';signals=[];kill=t.os.killpg
    def own(pgid,sig):signals.append(pgid);return kill(pgid,sig)
    monkeypatch.setattr(t.os,'killpg',own)
    cli=runner(t,'import os,time;open('+repr(str(started))+',"w").write(str(os.getpid()));time.sleep(60)',5)
    async def scenario():
        task=asyncio.create_task(cli(['status'],time.monotonic()+8))
        for _ in range(100):
            if started.exists():break
            await asyncio.sleep(.01)
        assert started.exists();pid=int(started.read_text());task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert signals==[pid]
        with pytest.raises(ProcessLookupError):os.kill(pid,0)
    asyncio.run(scenario())


def test_nondefault_sigchld_prevents_spawn(monkeypatch):
    t=transport();monkeypatch.setattr(t.signal,'getsignal',lambda s:signal.SIG_IGN)
    with pytest.raises(t.AdapterError,match='PROCESS_IDENTITY_UNVERIFIED'):asyncio.run(runner(t,'raise Exception("must not run")')(['status'],time.monotonic()+1))


def test_completed_child_never_signals_reaped_group_on_late_deadline(monkeypatch):
    t=transport()
    class Done:
        returncode=0;pid=999
    with pytest.raises(t.AdapterError,match='PROCESS_IDENTITY_UNVERIFIED'):t.cleanup_child(Done())


def test_mcp_exposes_exact_four_strict_readonly_candidate_tools(tmp_path):
    m,a,cli,f=fixture(tmp_path);t=transport();server=t.create_server(a)
    async def scenario():
        tools=await server.list_tools()
        assert {x.name for x in tools}=={'candidate_status','candidate_search_memory','candidate_search_code','candidate_get_context'}
        for x in tools:
            wire=x.model_dump(by_alias=True)
            assert wire['annotations']['readOnlyHint'] and not wire['annotations']['destructiveHint'] and not wire['annotations']['openWorldHint']
            assert wire['inputSchema'].get('additionalProperties') is False
        result=await server.call_tool('candidate_search_code',{'query':'x'})
        assert 'PRIVATE' not in str(result)
        with pytest.raises(Exception):await server.call_tool('candidate_search_code',{'query':'x','project_id':'other'})
    asyncio.run(scenario())


def test_unapproved_mcp_call_never_invokes_cli(tmp_path):
    m,a,cli,f=fixture(tmp_path,False);t=transport();server=t.create_server(a)
    async def scenario():
        with pytest.raises(Exception,match='CLOUD_SCOPE_NOT_APPROVED'):await server.call_tool('candidate_status',{})
    asyncio.run(scenario());assert not cli.calls


def test_real_local_stdio_roundtrip_uses_only_synthetic_data(tmp_path):
    # This is an SDK transport contract test, not a Codex/cloud session.
    m,a,cli,policy=fixture(tmp_path);t=transport()
    script=tmp_path/'synthetic_server.py'
    script.write_text('''import sys,time
sys.path.insert(0,PACKAGE)
sys.path.insert(0,TESTS)
from candidate_adapter import Adapter,PolicyFile
from candidate_transport import create_server
from test_adapter import search,context
class Synthetic:
 async def __call__(self,args,deadline):
  if args[0]=='search':return search()
  if args[0]=='context':return context('SYNTHETIC_CONTEXT')
  raise RuntimeError('must not reach a real CLI')
a=Adapter(PolicyFile(POLICY,'c'*64,'d'*64),Synthetic(),{'example-code':{'root':'/synthetic/code','worktree_id':'main'}})
create_server(a).run(transport='stdio')
'''.replace('PACKAGE',repr(str(ROOT))).replace('TESTS',repr(str(ROOT/'tests'))).replace('POLICY',repr(str(policy))))
    async def scenario():
        from mcp import ClientSession,StdioServerParameters
        from mcp.client.stdio import stdio_client
        params=StdioServerParameters(command=sys.executable,args=['-B',str(script)],env=dict(os.environ,PYTHONUTF8='0'))
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize();tools=await session.list_tools()
                assert len(tools.tools)==4
                s=await session.call_tool('candidate_search_code',{'query':'synthetic'})
                data=s.model_dump(by_alias=True);assert not data['isError'] and 'PRIVATE' not in json.dumps(data)
                c=await session.call_tool('candidate_get_context',{'chunk_id':KEY})
                value=c.model_dump(by_alias=True);assert not value['isError'] and 'SYNTHETIC_CONTEXT' in json.dumps(value)
                # Strict SDK argument validation is exercised over actual pipes.
                denied=await session.call_tool('candidate_search_code',{'query':'x','project_id':'other'})
                assert denied.model_dump(by_alias=True)['isError'] is True
    asyncio.run(scenario())


def test_inherited_pipe_descendant_is_killed_before_leader_is_reaped(tmp_path,monkeypatch):
    t=transport();child=tmp_path/'grandchild';kill=t.os.killpg;signalled=[]
    def own(pgid,sig):signalled.append(pgid);return kill(pgid,sig)
    monkeypatch.setattr(t.os,'killpg',own)
    # Leader exits; grandchild retains stdout. The CLI runner must keep the
    # unreaped leader reservation until it cleans the entire owned group.
    source='import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);Path('+repr(str(child))+').write_text(str(p.pid))'
    with pytest.raises(t.AdapterError,match='CLI_TIMEOUT'):asyncio.run(runner(t,source,.15)(['status'],time.monotonic()+2))
    assert child.exists() and len(signalled)==1 and signalled[0]!=int(child.read_text())
    # No pipe-holding descendant survives: its parent's group no longer exists
    # except for a possible already-dead zombie awaiting the OS parent reaper.
    pid=int(child.read_text())
    result=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
    assert result.returncode==1 or result.stdout.strip().startswith('Z')


def test_known_business_error_inside_real_cli_wrapper_is_preserved():
    t=transport();v=envelope({'error':'SOURCE_CHANGED'},2)
    source='import sys;print('+repr(json.dumps(v))+');sys.exit(2)'
    with pytest.raises(t.AdapterError,match='SOURCE_CHANGED'):asyncio.run(runner(t,source)(['context'],time.monotonic()+2))


@pytest.mark.parametrize('trigger',['cancel','timeout'])
def test_unconfirmed_cleanup_latches_runner_and_refuses_next_spawn(tmp_path,monkeypatch,trigger):
    t=transport();original=t.cleanup_child;owned=[];started=tmp_path/'child'
    def cannot_confirm(process):owned.append(process);raise t.AdapterError('PROCESS_IDENTITY_UNVERIFIED')
    monkeypatch.setattr(t,'cleanup_child',cannot_confirm)
    source='import os,time;open('+repr(str(started))+',"w").write(str(os.getpid()));time.sleep(60)'
    cli=runner(t,source,5 if trigger=='cancel' else .1)
    async def scenario():
        if trigger=='cancel':
            task=asyncio.create_task(cli(['status'],time.monotonic()+6))
            for _ in range(100):
                if started.exists():break
                await asyncio.sleep(.01)
            assert started.exists();task.cancel()
            with pytest.raises(t.AdapterError,match='PROCESS_IDENTITY_UNVERIFIED'):await task
        else:
            with pytest.raises(t.AdapterError,match='PROCESS_IDENTITY_UNVERIFIED'):await cli(['status'],time.monotonic()+1)
        assert len(owned)==1
        with pytest.raises(t.AdapterError,match='PROCESS_IDENTITY_UNVERIFIED'):await cli(['status'],time.monotonic()+1)
        assert len(owned)==1
    try:asyncio.run(scenario())
    finally:
        # Test-created child, exact unreaped Popen reservation retained here.
        for child in owned:
            if child.returncode is None:original(child)
