import asyncio
import threading

import httpx
import pytest

from codex_memory_rag import indexing, sources
from codex_memory_rag.daemon import PriorityWorker, create_app
from codex_memory_rag.retrieval import Retriever
from codex_memory_rag.schemas import SearchResponse
from test_watcher import engine


@pytest.mark.asyncio
@pytest.mark.parametrize('read_stage', ['scan', 'plan'])
async def test_query_runs_between_source_file_checks(engine, monkeypatch, read_stage):
    indexer, source = engine
    for name in ('a.md', 'b.md'):
        (source.root / name).write_text(name)
    indexer.build('p')
    loop = asyncio.get_running_loop()
    first_started, query_queued = asyncio.Event(), asyncio.Event()
    release_first = threading.Event()
    seen = []
    module = sources if read_stage == 'scan' else indexing
    original_read = module.read_allowed_file
    original_submit = PriorityWorker.submit

    def read(config, path):
        seen.append(path)
        if len(seen) == 1:
            loop.call_soon_threadsafe(first_started.set)
            assert release_first.wait(3)
        return original_read(config, path)

    async def submit(self, call, priority=0):
        if priority == 0:
            query_queued.set()
        return await original_submit(self, call, priority)

    def search(self, request, snapshot=None):
        seen.append('query')
        current = self.store.get_snapshot()
        return SearchResponse(generation=current.generation_id, table_version=current.version, mode='hybrid')

    monkeypatch.setattr(module, 'read_allowed_file', read)
    monkeypatch.setattr(PriorityWorker, 'submit', submit)
    monkeypatch.setattr(Retriever, 'search', search)
    app = create_app(indexer.config, indexer)
    async with app.router.lifespan_context(app):
        sync = asyncio.create_task(app.state.synchronize('p'))
        try:
            await asyncio.wait_for(first_started.wait(), 2)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
                query = asyncio.create_task(client.post('/v1/search', json={'project_id':'p', 'query':'test'}))
                await asyncio.wait_for(query_queued.wait(), 2)
                release_first.set()
                assert (await query).status_code == 200
                await sync
        finally:
            release_first.set()
            await sync
    assert seen == ['a.md', 'query', 'b.md']


def test_incremental_plan_preserves_changed_deleted_and_force(engine):
    indexer, source = engine
    (source.root / 'a.md').write_text('old')
    (source.root / 'b.md').write_text('deleted')
    indexer.build('p')
    (source.root / 'a.md').write_text('new')
    (source.root / 'b.md').unlink()
    (source.root / 'c.md').write_text('created')
    expected = indexer.plan_source(source)
    steps = indexer.plan_source_steps(source)
    count = 0
    while True:
        try:
            next(steps)
            count += 1
        except StopIteration as completed:
            assert completed.value == expected
            break
    assert count >= 2
    assert set(expected[0]) == {'a.md', 'c.md'}
    assert len(expected[1]) == 1
    indexer.sync('p')
    assert indexer.plan_source(source) == ([], [])
    assert set(indexer.plan_source(source, force=True)[0]) == {'a.md', 'c.md'}


@pytest.mark.asyncio
async def test_cancelled_planning_closes_held_scan_on_worker(engine, monkeypatch):
    indexer, source = engine
    (source.root / 'a.md').write_text('keep')
    indexer.build('p')
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    calls, held = [], []
    original_read = sources.read_allowed_file
    original_steps = indexer.plan_source_steps

    def read(config, path):
        calls.append(('read', threading.get_ident()))
        loop.call_soon_threadsafe(started.set)
        assert release.wait(3)
        return original_read(config, path)

    def steps(config):
        def run():
            try:
                yield from original_steps(config)
            finally:
                calls.append(('close', threading.get_ident()))
        iterator = run()
        held.append(iterator)
        return iterator

    monkeypatch.setattr(sources, 'read_allowed_file', read)
    monkeypatch.setattr(indexer, 'plan_source_steps', steps)
    app = create_app(indexer.config, indexer)
    async with app.router.lifespan_context(app):
        task = asyncio.create_task(app.state.synchronize('p'))
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert [kind for kind, _ in calls] == ['read', 'close']
            assert calls[0][1] == calls[1][1]
            assert not app.state.cmr['syncing']
            assert indexer.store.get_snapshot().table.count_rows() == 1
        finally:
            release.set()
            for iterator in held:
                iterator.close()
