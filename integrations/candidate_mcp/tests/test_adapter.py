import asyncio
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[1]
KEY='a'*64
HASH='b'*64
WARN=['CANDIDATE_PILOT','SUPPORT_NOT_CHECKED','完整支持未检查']


def impl():
    path=ROOT/'candidate_adapter.py'
    assert path.is_file(), 'candidate adapter implementation missing'
    spec=importlib.util.spec_from_file_location('candidate_adapter',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def item(**kw):
    v={'chunk_id':KEY,'project_id':'example-project','source_id':'example-code','worktree_id':'main',
       'relative_path':'example.py','path':'/synthetic/code/example.py','line_start':2,'line_end':4,
       'file_sha256':HASH,'status':None,'indexed_at':'2026-10-15T00:00:00Z',
       'retrieval_state':'source_current','current_evidence':True,'evidence_basis':'verified_source_bytes',
       'retrieval_role':'related_candidate','support_status':'not_checked','rrf_score':0.1,'rerank_score':1.0,
       'summary':'PRIVATE_SUMMARY','symbol':'PRIVATE_SYMBOL','extra_secret':'PRIVATE_EXTRA'}
    return v|kw


def search(items=None,**kw):
    return {'results':[item()] if items is None else items,'generation':'g1','table_version':3,
            'mode':'hybrid_reranked','warnings':WARN,'truncated':False,'private':'PRIVATE_TOP'}|kw


def context(text='line two\nline three\nline four\n',**kw):
    return {'items':[item(text=text)],'generation':'g1','table_version':3,'warnings':WARN,
            'truncated':False,'readback_status':'complete','private':'PRIVATE_TOP'}|kw


class FakeCli:
    def __init__(self):self.calls=[];self.result=search();self.hook=None
    async def __call__(self,args,deadline):
        self.calls.append(args)
        if self.hook:await self.hook()
        return copy.deepcopy(self.result)


def fixture(tmp_path,approved=True):
    m=impl();f=tmp_path/'policy.json'
    policy={'schema':'candidate-cloud-policy-v2','approved':approved,'adapter_seal_sha256':'c'*64,
            'r4_input_hash':'d'*64,'project_id':'example-project','max_context_chars':12000,
            'files':[{'source_id':'example-code','relative_path':'example.py','file_sha256':HASH,'chunk_id':KEY,'line_start':2,'line_end':4}]}
    f.write_text(json.dumps(policy));f.chmod(0o600)
    store=m.PolicyFile(f,'c'*64,'d'*64)
    cli=FakeCli();a=m.Adapter(store,cli,{'example-code':{'root':'/synthetic/code','worktree_id':'main'},
          'example-memory':{'root':'/synthetic/memory','worktree_id':None},'example-docs':{'root':'/synthetic/code','worktree_id':None}})
    return m,a,cli,f


def run(a,name,args=None):return asyncio.run(a.dispatch(name,args or {}))


def test_unapproved_policy_never_invokes_cli(tmp_path):
    m,a,cli,f=fixture(tmp_path,False)
    with pytest.raises(m.AdapterError,match='CLOUD_SCOPE_NOT_APPROVED'):run(a,'candidate_status')
    assert cli.calls==[]


def test_search_redacts_all_body_and_unlisted_fields_and_pins_cli_arguments(tmp_path):
    m,a,cli,f=fixture(tmp_path)
    out=run(a,'candidate_search_code',{'query':'explain','literal':'normalize_size'})
    assert 'PRIVATE' not in json.dumps(out)
    assert out['results'][0]['relative_path']=='example.py'
    assert out['results'][0]['support_status']=='not_checked'
    assert out['warnings']==WARN
    assert cli.calls==[['search','--project','example-project','--kind','code','--query','explain','--source','example-code','--literal','normalize_size']]


@pytest.mark.parametrize('args',[{'query':'x','project_id':'other'},{'query':'x','include_archived':True},
 {'query':True},{'query':''},{'query':'x','mode':'vector'},{'query':'x','literal':'x'*257}])
def test_invalid_or_extra_search_parameters_reject_before_cli(tmp_path,args):
    m,a,cli,f=fixture(tmp_path)
    with pytest.raises(m.AdapterError,match='INVALID_REQUEST'):run(a,'candidate_search_code',args)
    assert cli.calls==[]


def test_scope_filtered_search_does_not_expose_unapproved_file(tmp_path):
    m,a,cli,f=fixture(tmp_path);cli.result=search([item(relative_path='other.py',path='/synthetic/code/other.py')])
    assert run(a,'candidate_search_code',{'query':'x'})['results']==[]


@pytest.mark.parametrize('bad',[{'project_id':'other'},{'relative_path':'../outside'}, {'path':'/outside'},
 {'retrieval_role':'evidence'}, {'support_status':'supported'}, {'retrieval_state':'historical'},
 {'source_id':'foreign'},{'line_start':True},{'worktree_id':'other'},{'rrf_score':float('nan')}])
def test_malformed_or_foreign_candidate_never_gets_cached(tmp_path,bad):
    m,a,cli,f=fixture(tmp_path);cli.result=search([item(**bad)])
    with pytest.raises(m.AdapterError):run(a,'candidate_search_code',{'query':'x'})
    cli.result=context()
    with pytest.raises(m.AdapterError,match='CANDIDATE_NOT_ISSUED'):run(a,'candidate_get_context',{'chunk_id':KEY})


def test_context_requires_issued_candidate_then_matches_snapshot_and_identity(tmp_path):
    m,a,cli,f=fixture(tmp_path)
    with pytest.raises(m.AdapterError,match='CANDIDATE_NOT_ISSUED'):run(a,'candidate_get_context',{'chunk_id':KEY})
    run(a,'candidate_search_code',{'query':'x'});cli.result=context()
    out=run(a,'candidate_get_context',{'chunk_id':KEY})
    assert out['items'][0]['text']=='line two\nline three\nline four\n'
    assert out['readback_status']=='complete' and 'PRIVATE' not in json.dumps(out)
    assert cli.calls[-1]==['context','--project','example-project','--chunk-id',KEY,'--max-chars','4000']


@pytest.mark.parametrize('change',[{'table_version':4},{'generation':'other'},
 {'items':[item(text='sensitive',file_sha256='e'*64)]}, {'items':[item(text='sensitive',line_end=9)]}])
def test_context_drift_never_returns_body(tmp_path,change):
    m,a,cli,f=fixture(tmp_path);run(a,'candidate_search_code',{'query':'x'});cli.result=context(**change)
    with pytest.raises(m.AdapterError,match='SNAPSHOT_OR_CANDIDATE_CHANGED'):run(a,'candidate_get_context',{'chunk_id':KEY})


def test_partial_context_preserves_partial_and_candidate_flags(tmp_path):
    m,a,cli,f=fixture(tmp_path);run(a,'candidate_search_code',{'query':'x'})
    cli.result=context('lin',truncated=True,readback_status='partial')
    out=run(a,'candidate_get_context',{'chunk_id':KEY,'max_chars':3})
    assert out['readback_status']=='partial' and out['truncated'] and out['items'][0]['text']=='lin'


def test_fresh_connection_and_expired_cache_cannot_read_previous_candidates(tmp_path):
    m,a,cli,f=fixture(tmp_path);run(a,'candidate_search_code',{'query':'x'});cli.result=context()
    a.clock=lambda:10**12
    with pytest.raises(m.AdapterError,match='CANDIDATE_NOT_ISSUED'):run(a,'candidate_get_context',{'chunk_id':KEY})
    b=m.Adapter(a.policy,cli,a.sources)
    with pytest.raises(m.AdapterError,match='CANDIDATE_NOT_ISSUED'):run(b,'candidate_get_context',{'chunk_id':KEY})


def test_policy_revoked_during_call_discards_output_and_candidates(tmp_path):
    m,a,cli,f=fixture(tmp_path)
    async def revoke():
        v=json.loads(f.read_text());v['approved']=False;f.write_text(json.dumps(v))
    cli.hook=revoke
    with pytest.raises(m.AdapterError,match='CLOUD_POLICY_CHANGED'):run(a,'candidate_search_code',{'query':'x'})
    assert not a.candidates


@pytest.mark.parametrize('bad',['mode','link','seal','input','extra'])
def test_policy_integrity_failures_reject_before_cli(tmp_path,bad):
    m,a,cli,f=fixture(tmp_path)
    if bad=='mode':f.chmod(0o644)
    elif bad=='link':
        saved=f.with_name('saved');f.rename(saved);f.symlink_to(saved)
    else:
        v=json.loads(f.read_text());v[{'seal':'adapter_seal_sha256','input':'r4_input_hash','extra':'unknown'}[bad]]='wrong';f.write_text(json.dumps(v))
    with pytest.raises(m.AdapterError):run(a,'candidate_status')
    assert cli.calls==[]


def test_status_projects_only_health_not_operations_or_source_details(tmp_path):
    m,a,cli,f=fixture(tmp_path);cli.result={'warnings':WARN,'pid':42,'dimension':768,'backend':'mps',
     'loading':False,'syncing':False,'error':None,'generation':'g1','embedding_profile':'e',
     'reranker':{'enabled':True,'ready':True,'error':None,'profile':'r','backend':'mps'},
     'sources':{'example-code':{'state':'ready','secret':'PRIVATE_SOURCE'}},'operations':[{'secret':'PRIVATE_OP'}],
     'config':'PRIVATE_CONFIG','instances':{'encoder':1,'reranker':1,'worker':1}}
    out=run(a,'candidate_status')
    assert 'PRIVATE' not in json.dumps(out) and out['sources']=={'example-code':'ready'}
    assert cli.calls==[['status','--project','example-project']]


def test_busy_call_rejects_without_queueing_or_second_cli(tmp_path):
    m,a,cli,f=fixture(tmp_path)
    async def scenario():
        entered=asyncio.Event();release=asyncio.Event()
        async def block():entered.set();await release.wait()
        cli.hook=block
        task=asyncio.create_task(a.dispatch('candidate_search_code',{'query':'x'}));await entered.wait()
        with pytest.raises(m.AdapterError,match='ADAPTER_BUSY'):await a.dispatch('candidate_search_code',{'query':'y'})
        release.set();await task
    asyncio.run(scenario());assert len(cli.calls)==1


def test_cancelled_call_discards_pending_cache_and_releases_admission(tmp_path):
    m,a,cli,f=fixture(tmp_path)
    async def scenario():
        entered=asyncio.Event()
        async def block():entered.set();await asyncio.sleep(60)
        cli.hook=block;t=asyncio.create_task(a.dispatch('candidate_search_code',{'query':'x'}));await entered.wait();t.cancel()
        with pytest.raises(asyncio.CancelledError):await t
        assert not a.candidates
        cli.hook=None;await a.dispatch('candidate_search_code',{'query':'x'})
    asyncio.run(scenario())


@pytest.mark.parametrize('source_id,root', [
    ('example-memory', '/synthetic/memory'), ('example-docs', '/synthetic/code'),
])
def test_memory_source_is_project_bound_and_summary_remains_private(tmp_path,source_id,root):
    m,a,cli,f=fixture(tmp_path);v=json.loads(f.read_text());v['files']=[{'source_id':source_id,'relative_path':'note.md','file_sha256':HASH,'chunk_id':KEY,'line_start':2,'line_end':4}];f.write_text(json.dumps(v))
    cli.result=search([item(source_id=source_id,worktree_id=None,relative_path='note.md',path=root+'/note.md',
                          retrieval_state='eligible',current_evidence=False,evidence_basis='dated_memory_assertion',status='active')])
    out=run(a,'candidate_search_memory',{'query':'what happened'})
    assert cli.calls==[['search','--project','example-project','--kind','memory','--query','what happened']]
    assert 'PRIVATE' not in json.dumps(out) and out['results'][0]['evidence_basis']=='dated_memory_assertion'


@pytest.mark.parametrize('bad',[{'max_chars':True},{'max_chars':0},{'max_chars':12001},{'max_chars':'4000'}, {'path':'/secret'}])
def test_context_budget_and_arbitrary_path_rejected_before_cli(tmp_path,bad):
    m,a,cli,f=fixture(tmp_path)
    with pytest.raises(m.AdapterError,match='INVALID_REQUEST'):run(a,'candidate_get_context',{'chunk_id':KEY}|bad)
    assert not cli.calls


def test_policy_fifo_is_rejected_without_blocking_a_connection(tmp_path):
    import subprocess
    m,a,cli,f=fixture(tmp_path);f.unlink();os.mkfifo(f,0o600)
    script='import sys;sys.path.insert(0,'+repr(str(ROOT))+');from candidate_adapter import PolicyFile,AdapterError\ntry:PolicyFile('+repr(str(f))+',"'+('c'*64)+'","'+('d'*64)+'").snapshot()\nexcept AdapterError as e:print(str(e))'
    process=subprocess.Popen([sys.executable,'-B','-c',script],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        try:out,err=process.communicate(timeout=1)
        except subprocess.TimeoutExpired:pytest.fail('nonregular policy blocks before validation')
        assert process.returncode==0 and out.strip()==b'CLOUD_POLICY_INVALID'
    finally:
        if process.poll() is None:process.kill()
        process.communicate(timeout=2)


def test_new_generation_invalidates_earlier_issued_candidate(tmp_path):
    m,a,cli,f=fixture(tmp_path);run(a,'candidate_search_code',{'query':'x'})
    cli.result=search([],generation='g2');run(a,'candidate_search_code',{'query':'y'})
    with pytest.raises(m.AdapterError,match='CANDIDATE_NOT_ISSUED'):run(a,'candidate_get_context',{'chunk_id':KEY})


def test_policy_read_access_time_change_is_not_content_or_identity_drift(tmp_path,monkeypatch):
    from types import SimpleNamespace
    m,a,cli,f=fixture(tmp_path);real=m.os.fstat;calls=0
    def accessed(fd):
        nonlocal calls
        info=real(fd);calls+=1
        fields=['st_mode','st_uid','st_size','st_dev','st_ino','st_mtime_ns','st_ctime_ns']
        return SimpleNamespace(**{k:getattr(info,k) for k in fields},st_atime_ns=1 if calls%2 else 2)
    monkeypatch.setattr(m.os,'fstat',accessed)
    out=run(a,'candidate_search_code',{'query':'x'})
    assert out['results'][0]['chunk_id']==KEY


@pytest.mark.parametrize('where',['search_top','search_item','context_top','context_item'])
def test_error_marked_payload_never_returns_body_or_keeps_candidates(tmp_path,where):
    m,a,cli,f=fixture(tmp_path)
    if where.startswith('context'):run(a,'candidate_search_code',{'query':'x'});cli.result=context()
    if where.endswith('top'):cli.result['error']='SOURCE_CHANGED'
    else:cli.result['items' if where.startswith('context') else 'results'][0]['error']='SOURCE_CHANGED'
    with pytest.raises(m.AdapterError,match='SOURCE_CHANGED'):
        run(a,'candidate_get_context' if where.startswith('context') else 'candidate_search_code',
            {'chunk_id':KEY} if where.startswith('context') else {'query':'x'})
    assert not a.candidates


@pytest.mark.parametrize('status',['archived','historical','deprecated'])
def test_noncurrent_status_cannot_be_presented_as_current_candidate(tmp_path,status):
    m,a,cli,f=fixture(tmp_path);cli.result=search([item(status=status)])
    with pytest.raises(m.AdapterError):run(a,'candidate_search_code',{'query':'x'})
    assert not a.candidates
