import asyncio
import httpx
import pytest
from codex_memory_rag.daemon import create_app,DaemonLease,PriorityWorker
from test_watcher import engine

def test_second_lease_cannot_delete_live_socket(tmp_path):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix='cmr-',dir='/private/tmp') as directory:
        root=Path(directory)/'run'
        a=DaemonLease(root);assert a.acquire()
        sock=root/'daemon.sock';sock.write_text('live')
        b=DaemonLease(root);assert not b.acquire();assert sock.read_text()=='live'
        b.close();assert sock.exists();a.close()

def test_oversize_socket_path_is_rejected(tmp_path):
    from codex_memory_rag.schemas import CmrError
    with pytest.raises(CmrError,match='CONFIG_CONFLICT'):DaemonLease(tmp_path/('x'*100)).acquire()

@pytest.mark.asyncio
async def test_status_and_schema_do_not_need_model(engine):
    e,_=engine;app=create_app(e.config,e)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://local') as c:
        r=await c.get('/v1/status',params={'project_id':'p'});assert r.status_code==200
        assert r.json()['dimension']==768
        r=await c.post('/v1/search',json={'query':'x','project_id':'p','path':'/etc/passwd'});assert r.status_code==422

@pytest.mark.asyncio
async def test_priority_queue_orders_waiting_query_before_index_batch():
    worker=PriorityWorker();worker.start()
    first=asyncio.create_task(worker.submit(lambda:__import__('time').sleep(.1),0))
    await asyncio.sleep(.01);seen=[]
    index=asyncio.create_task(worker.submit(lambda:seen.append('index'),10))
    query=asyncio.create_task(worker.submit(lambda:seen.append('query'),0))
    await asyncio.gather(first,index,query);worker.close()
    assert seen==['query','index']

def test_owned_uds_has_private_permissions():
    import tempfile,stat
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix='cmr-',dir='/private/tmp') as d:
        lease=DaemonLease(Path(d)/'run');assert lease.acquire()
        sock=lease.bind_socket()
        assert stat.S_IMODE((Path(d)/'run/daemon.sock').stat().st_mode)==0o600
        sock.close();lease.close()

@pytest.mark.asyncio
async def test_model_load_failure_preserves_scoped_lexical_fallback(engine,monkeypatch):
    import codex_memory_rag.embedding as embedding
    from codex_memory_rag.schemas import CmrError
    from test_watcher import Encoder
    e,s=engine;(s.root/'a.md').write_text('退款恢复');e.build('p')
    class Unavailable(Encoder):
        def encode_queries(self,*a):raise CmrError('MODEL_UNAVAILABLE')
        def token_count(self,*a):raise CmrError('MODEL_UNAVAILABLE')
    def fail(*a):raise CmrError('MODEL_UNAVAILABLE')
    monkeypatch.setattr(embedding,'Encoder',fail)
    monkeypatch.setattr(embedding,'unavailable_encoder',lambda *a:Unavailable(),raising=False)
    app=create_app(e.config)
    async with app.router.lifespan_context(app):
        for _ in range(100):
            if not app.state.cmr['loading']:break
            await asyncio.sleep(.01)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://local') as client:
            result=await client.post('/v1/search',json={'project_id':'p','query':'退款','kind':'memory'})
            assert result.status_code==200
            assert result.json()['mode']=='lexical_only'
            assert 'MODEL_UNAVAILABLE' in result.json()['warnings']
            assert result.json()['results'][0]['relative_path']=='a.md'
