from codex_memory_rag.store import Store
from codex_memory_rag.schemas import source_key
from test_store import row
from test_watcher import engine

def test_icu_long_symbol_and_unoptimized_new_row(tmp_path):
    s=Store(tmp_path,"fixture","i");g=s.open_generation();k=source_key("p","s",None,"a.md")
    symbol="VeryLongFunctionIdentifierForMemoryRetrievalTestingXYZ"
    s.replace_source(k,[row(k,"中文退款规则 RDGClientTransport "+symbol)])
    s.activate_generation(g.generation_id)
    for q in ["退款","RDGClientTransport",symbol]:assert s.get_snapshot().table.search(q,query_type="fts",fts_columns="lexical_text").to_list()
    k2=source_key("p","s",None,"b.md");s.replace_source(k2,[row(k2,"新增恢复规则")])
    assert s.get_snapshot().table.search("新增",query_type="fts",fts_columns="lexical_text").to_list()
def test_dimension_words_and_filename_metadata_are_searchable(engine):
    from codex_memory_rag.retrieval import Retriever
    from codex_memory_rag.schemas import SearchRequest
    from test_watcher import Encoder
    e,s=engine
    (s.root/'canvas.md').write_text('Finished 20cm x15cm image area.\n')
    (s.root/'2026-09-23-12-18-note.md').write_text('An independently verified change.\n')
    e.build('p');r=Retriever(e.config,e.store,Encoder())
    dimensions=r.search(SearchRequest(project_id='p',query='20 15 厘米',mode='lexical'))
    assert dimensions.results and dimensions.results[0]['relative_path']=='canvas.md'
    dated=r.search(SearchRequest(project_id='p',query='2026 12 18',mode='lexical'))
    assert dated.results and dated.results[0]['relative_path']=='2026-09-23-12-18-note.md'
