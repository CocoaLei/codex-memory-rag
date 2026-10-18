import json,subprocess,sys,time
from pathlib import Path
from test_store import row
from codex_memory_rag.store import Store
from codex_memory_rag.schemas import source_key

def test_actual_process_kill_cannot_expose_partial_source_replacement(tmp_path):
    store=Store(tmp_path,'fixture','i');g=store.open_generation();key=source_key('p','s',None,'a.md')
    store.replace_source(key,[row(key)]);store.activate_generation(g.generation_id)
    code=r'''
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
from test_store import row
from codex_memory_rag.store import Store
from codex_memory_rag.schemas import source_key
root=Path(sys.argv[1]);s=Store(root,'fixture','i');key=source_key('p','s',None,'a.md')
rows=[row(key, 'new-'+str(i),'h2') | {'chunk_id':'new-'+str(i)} for i in range(10000)]
original=s.table.merge_insert
class Marked:
 def __init__(self,inner):self.inner=inner
 def when_matched_update_all(self):self.inner.when_matched_update_all();return self
 def when_not_matched_insert_all(self):self.inner.when_not_matched_insert_all();return self
 def when_not_matched_by_source_delete(self,x):self.inner.when_not_matched_by_source_delete(x);return self
 def execute(self,data):
  (root/'commit-start').write_text('start');self.inner.execute(data);(root/'commit-done').write_text('done')
s.table.merge_insert=lambda *args:Marked(original(*args))
s.replace_source(key,rows)
'''
    child=subprocess.Popen([sys.executable,'-c',code,str(tmp_path),str(Path(__file__).parent)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+30
        while not (tmp_path/'commit-start').exists() and time.monotonic()<deadline and child.poll() is None:time.sleep(.005)
        assert (tmp_path/'commit-start').exists()
        assert not (tmp_path/'commit-done').exists(),'interruption occurred too late'
        child.kill();child.wait(timeout=10)
        assert child.returncode<0
        reopened=Store(tmp_path,'fixture','i');values=reopened.get_snapshot().table.search().select(['text']).to_list()
        assert (len(values)==1 and values[0]['text']=='old') or (len(values)==10000 and all(v['text'].startswith('new-') for v in values))
        hits=reopened.get_snapshot().table.search('old',query_type='fts',fts_columns='lexical_text').to_list()
        assert bool(hits)==(len(values)==1)
    finally:
        if child.poll() is None:child.kill();child.wait()
