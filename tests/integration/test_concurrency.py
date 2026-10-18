from concurrent.futures import ThreadPoolExecutor
from codex_memory_rag.schemas import SearchRequest
from codex_memory_rag.retrieval import Retriever
from test_watcher import engine,Encoder

def test_two_readers_and_writer_keep_complete_table_versions(engine):
    e,s=engine;(s.root/'a.md').write_text('original');e.build('p')
    snapshot=e.store.get_snapshot()
    def change():
        (s.root/'a.md').write_text('new current');e.sync('p')
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(lambda:snapshot.table.search().to_list()),pool.submit(change),pool.submit(lambda:snapshot.table.search().to_list())]
        for i in [0,2]:assert jobs[i].result()[0]['text']=='original'
        jobs[1].result()
    assert e.store.get_snapshot().table.search().to_list()[0]['text']=='new current'
