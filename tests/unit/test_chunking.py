from codex_memory_rag.schemas import SourceConfig,SourceDocument
from codex_memory_rag.parsing import chunk_document
class Tokenizer:
    def token_count(self,text):return len(text)

def test_fenced_heading_and_original_lines(tmp_path):
    text="---\ncode_project: p\nstatus: active\n---\n# Real\ntext\n```python\n# not a heading\nx=1\n```\n## Next\nend\n"
    s=SourceConfig(source_id="s",project_id="p",root=tmp_path,kind="memory")
    d=SourceDocument(source=s,relative_path="a.md",text=text,file_sha256="h",metadata={"status":"active"})
    chunks=chunk_document(d,Tokenizer())
    assert len(chunks)==1
    for c in chunks: assert c.text=="".join(text.splitlines(keepends=True)[c.line_start-1:c.line_end])
    assert '# not a heading' in chunks[0].text

def test_complete_function_and_syntax_error_fallback(tmp_path):
    s=SourceConfig(source_id="s",project_id="p",root=tmp_path,kind="code",worktree_id="w")
    for text in ["def f(x):\n    return x + 1\n", "def broken(:\n    pass\n"]:
        d=SourceDocument(source=s,relative_path="a.py",text=text,file_sha256="h")
        c=chunk_document(d,Tokenizer());assert c and c[0].line_start==1
        if 'broken' in text:assert c[0].parse_quality=="fallback"

def test_huge_block_splits_at_real_lines(tmp_path):
    s=SourceConfig(source_id="s",project_id="p",root=tmp_path,kind="docs")
    text="line "*50+"\n";text=text*30
    c=chunk_document(SourceDocument(source=s,relative_path="a.md",text=text,file_sha256="h"),Tokenizer())
    assert len(c)>1
    assert all(len(x.text)<1200 for x in c)

def test_small_methods_survive_oversized_class_split(tmp_path):
    source=SourceConfig(source_id='s',project_id='p',root=tmp_path,kind='code',worktree_id='w')
    text='class Example:\n'
    text+='    def first(self):\n'+''.join('        value = "'+('x'*100)+'"\n' for _ in range(8))+'        return value\n'
    second_line=len(text.splitlines())+1
    text+='    def second(self):\n'+''.join('        value = "'+('y'*100)+'"\n' for _ in range(8))+'        return value\n'
    document=SourceDocument(source=source,relative_path='a.py',text=text,file_sha256='h')
    chunks=chunk_document(document,Tokenizer())
    assert any(c.line_start<=second_line and c.line_end>=len(text.splitlines()) for c in chunks)

def test_short_memory_keeps_problem_and_resolution_in_one_source_span(tmp_path):
    source = SourceConfig(source_id='s', project_id='p', root=tmp_path, kind='memory')
    text = '---\nstatus: active\ncode_project: p\n---\n## Problem\nDuplicate items.\n## Resolution\nVerify the exact item set before publishing.\n## Validation\nTests passed.\n'
    document = SourceDocument(source=source, relative_path='session.md', text=text,
                              file_sha256='h', metadata={'status': 'active'})
    chunks = chunk_document(document, Tokenizer())
    assert len(chunks) == 1
    assert 'Duplicate items.' in chunks[0].text
    assert 'Verify the exact item set' in chunks[0].text
    assert chunks[0].line_start == 5 and chunks[0].line_end == len(text.splitlines())
    assert chunks[0].text == ''.join(text.splitlines(keepends=True)[4:])

def test_bilingual_terms_make_english_identifiers_searchable_without_changing_source(tmp_path):
    source=SourceConfig(source_id='s',project_id='p',root=tmp_path,kind='code',worktree_id='w')
    text='def normalizeCarrier(value):\n    """Normalize the carrier name."""\n    return value\n'
    document=SourceDocument(source=source,relative_path='shipping_record.py',text=text,file_sha256='h')
    chunk=chunk_document(document,Tokenizer())[0]
    assert '规范化' in chunk.lexical_text and '物流商' in chunk.lexical_text
    assert chunk.text==text
    assert 'normalizeCarrier' in chunk.identifiers
    assert '物流商' not in chunk.identifiers
    assert (chunk.line_start,chunk.line_end)==(1,3)

def test_bilingual_terms_match_whole_components_not_substrings(tmp_path):
    source=SourceConfig(source_id='s',project_id='p',root=tmp_path,kind='docs')
    text='outdated unnormalized\n'
    chunk=chunk_document(SourceDocument(source=source,relative_path='a.md',text=text,file_sha256='h'),Tokenizer())[0]
    assert '规范化' not in chunk.lexical_text and '日期' not in chunk.lexical_text


def test_blank_gaps_are_not_indexed_as_filename_only_evidence(tmp_path):
    source = SourceConfig(source_id='s', project_id='p', root=tmp_path,
                          kind='docs', include=['*.md'], enabled=True, allow_cloud_context=True)
    text = '\n\n# Evidence\nActual refund policy.\n'
    chunks = chunk_document(SourceDocument(source=source, relative_path='refund.md',
                                          text=text, file_sha256='h'), Tokenizer())
    assert chunks and all(c.text.strip() for c in chunks)
    assert chunks[0].line_start == 3
    assert chunks[0].text == ''.join(text.splitlines(keepends=True)[2:])
