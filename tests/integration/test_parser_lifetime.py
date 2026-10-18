import subprocess,sys

def test_repeated_native_parse_survives_gc_and_parser_lifetimes():
    code=r"""
import gc
from pathlib import Path
from codex_memory_rag.parsing import chunk_document
from codex_memory_rag.schemas import SourceConfig,SourceDocument
class Tokens:
    def token_count(self,s):return len(s)//3+1
source=SourceConfig(source_id='s',project_id='p',kind='code',root=Path('/private/tmp'),worktree_id='w')
for i in range(250):
    text='class Example:\n' + ''.join('    def method_'+str(j)+'(self):\n        \"\"\"中文说明\"\"\"\n        return '+str(j)+'\n' for j in range(20))
    doc=SourceDocument(source=source,relative_path='a.py',text=text,file_sha256='h')
    result=chunk_document(doc,Tokens())
    assert any(c.symbol.endswith('method_19') for c in result)
    gc.collect()
"""
    result=subprocess.run([sys.executable,'-X','faulthandler','-c',code],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr[-1500:]

def test_code_spans_preserve_line_numbers_above_small_integer_cache():
    from pathlib import Path
    from codex_memory_rag.parsing import _spans
    from codex_memory_rag.schemas import SourceConfig,SourceDocument
    source=SourceConfig(source_id='s',project_id='p',kind='code',root=Path('/private/tmp'),worktree_id='w')
    text='\n'*300+'def late_function():\n    return 1\n'
    doc=SourceDocument(source=source,relative_path='a.py',text=text,file_sha256='h')
    assert _spans(doc)==[(0,300,'','parsed'),(300,302,'late_function','parsed')]
