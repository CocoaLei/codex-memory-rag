import pytest
from codex_memory_rag.store import Store
from codex_memory_rag.schemas import source_key
from test_store import row

def test_transaction_failure_preserves_old_rows(tmp_path,monkeypatch):
    s=Store(tmp_path,"fixture","i");g=s.open_generation();k=source_key("p","s",None,"a.md");s.replace_source(k,[row(k)]);s.activate_generation(g.generation_id)
    def fail(*args,**kwargs):raise RuntimeError("injected pre-commit interruption")
    monkeypatch.setattr(s.table,"merge_insert",fail)
    with pytest.raises(RuntimeError):s.replace_source(k,[row(k,"new","h2")])
    assert s.get_snapshot().table.search().to_list()[0]["text"]=="old"

def test_failed_generation_is_never_activated(tmp_path):
    s=Store(tmp_path,"fixture","i");g=s.open_generation();k=source_key("p","s",None,"a.md");s.replace_source(k,[row(k)]);s.activate_generation(g.generation_id)
    before=s.get_snapshot().generation_id
    s.open_generation();assert s.get_snapshot().generation_id==before
