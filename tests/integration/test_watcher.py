import numpy as np
import pytest,subprocess
from codex_memory_rag.schemas import AppConfig,SourceConfig,CmrError
from codex_memory_rag.indexing import Indexer
from codex_memory_rag.retrieval import Retriever
from codex_memory_rag.store import Store
from codex_memory_rag.schemas import SearchRequest
class Encoder:
    profile_id='fixture'
    def token_count(self,text):return len(text)//3+1
    def encode_documents(self,items):return np.ones((len(items),768),dtype=np.float32)/np.sqrt(768)
    def encode_queries(self,texts,kind):return np.ones((len(texts),768),dtype=np.float32)/np.sqrt(768)

@pytest.fixture
def engine(tmp_path):
    root=tmp_path/'docs';root.mkdir();src=SourceConfig(source_id='s',project_id='p',root=root,kind='docs',include=['*.md'],enabled=True,allow_cloud_context=True)
    cfg=AppConfig(sources=[src],storage={'runtime_dir':tmp_path/'r'})
    return Indexer(cfg,Encoder()),src

def test_reconcile_create_atomic_save_rename_delete_ignore(engine):
    e,s=engine;(s.root/'a.md').write_text('one');e.build('p')
    assert e.store.get_snapshot().table.count_rows()==1
    (s.root/'new.tmp').write_text('two');(s.root/'new.tmp').replace(s.root/'a.md');e.sync('p')
    assert e.store.get_snapshot().table.search().to_list()[0]['text']=='two'
    (s.root/'a.md').rename(s.root/'b.md');e.sync('p')
    assert e.store.get_snapshot().table.search().to_list()[0]['relative_path']=='b.md'
    (s.root/'.memoryignore').write_text('b.md\n');e.sync('p');assert e.store.get_snapshot().table.count_rows()==0
    (s.root/'.memoryignore').unlink();e.sync('p');assert e.store.get_snapshot().table.count_rows()==1
    (s.root/'b.md').unlink();e.sync('p');assert e.store.get_snapshot().table.count_rows()==0

def test_unavailable_root_does_not_clear_or_activate_incomplete_generation(engine):
    e,s=engine;(s.root/'a.md').write_text('one');e.build('p');g=e.store.get_snapshot().generation_id
    s.root.rename(s.root.with_name('away'))
    with pytest.raises(CmrError):e.sync('p')
    assert e.store.get_snapshot().table.count_rows()==1
    with pytest.raises(CmrError):e.build('p')
    assert e.store.get_snapshot().generation_id==g

def test_stale_prepared_file_cannot_commit(engine):
    e,s=engine;(s.root/'a.md').write_text('one');e.build('p')
    job=e.prepare_file(s,'a.md');(s.root/'a.md').write_text('new')
    with pytest.raises(CmrError,match='SOURCE_CHANGED'):e.commit_file(job)

def test_linked_worktree_same_content_branch_change_is_stale(tmp_path):
    main=tmp_path/'main';main.mkdir()
    def git(root,*a):return subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
    git(main,'init','-q');git(main,'config','user.email','fixture@example.invalid');git(main,'config','user.name','fixture')
    (main/'a.py').write_text('def update():\n    return 1\n');git(main,'add','a.py');git(main,'commit','-qm','fixture');git(main,'branch','other')
    linked=tmp_path/'linked';git(main,'worktree','add','-q','--detach',str(linked))
    s=SourceConfig(source_id='code',project_id='p',root=linked,kind='code',worktree_id='linked',include=['*.py'],enabled=True,allow_cloud_context=True)
    cfg=AppConfig(sources=[s],storage={'runtime_dir':tmp_path/'runtime'});e=Indexer(cfg,Encoder());e.build('p');r=Retriever(cfg,e.store,Encoder())
    assert r.search(SearchRequest(project_id='p',kind='code',query='update',literal='update')).results
    git(linked,'switch','-q','other')
    value=r.search(SearchRequest(project_id='p',kind='code',query='update',literal='update'))
    assert not value.results and 'STALE_SOURCE' in value.warnings
    e.sync('p');assert r.search(SearchRequest(project_id='p',kind='code',query='update',literal='update')).results

def test_rebuilding_second_project_preserves_first_project(engine,tmp_path):
    e,s=engine;(s.root/'a.md').write_text('project p');e.build('p')
    other=tmp_path/'other';other.mkdir();(other/'b.md').write_text('project q')
    e.config.sources.append(SourceConfig(source_id='other',project_id='q',root=other,kind='docs',include=['*.md'],enabled=True,allow_cloud_context=True))
    e.build('q')
    rows=e.store.get_snapshot().table.search().to_list()
    assert {r['project_id'] for r in rows}=={'p','q'}

def test_rebuild_cannot_activate_a_file_changed_after_its_commit(engine,monkeypatch):
    e,s=engine;(s.root/'a.md').write_text('old');e.build('p')
    before=e.store.get_snapshot().generation_id
    (s.root/'b.md').write_text('second')
    original=e.commit_file
    def mutate(job):
        result=original(job)
        if job.document.relative_path=='b.md':(s.root/'a.md').write_text('changed after commit')
        return result
    monkeypatch.setattr(e,'commit_file',mutate)
    with pytest.raises(CmrError,match='SOURCE_CHANGED'):e.build('p')
    assert e.store.get_snapshot().generation_id==before

def test_unsafe_root_ancestor_empty_directory_is_not_authoritative_deletion(tmp_path):
    registered=tmp_path/'registered';registered.mkdir();root=registered/'docs';root.mkdir();(root/'a.md').write_text('keep')
    source=SourceConfig(source_id='s',project_id='p',kind='docs',root=root,include=['*.md'],enabled=True,allow_cloud_context=True)
    cfg=AppConfig(sources=[source],storage={'runtime_dir':tmp_path/'runtime'});e=Indexer(cfg,Encoder());e.build('p')
    outside=tmp_path/'outside';outside.mkdir();(outside/'docs').mkdir()
    registered.rename(tmp_path/'saved');registered.symlink_to(outside,target_is_directory=True)
    with pytest.raises(CmrError,match='STALE_SOURCE'):e.sync('p')
    assert e.store.get_snapshot().table.count_rows()==1

def test_unreadable_ignore_policy_is_not_authoritative_empty_source(engine,tmp_path):
    e,s=engine;(s.root/'a.md').write_text('keep');e.build('p')
    target=tmp_path/'outside-policy';target.write_text('*\n')
    (s.root/'.memoryignore').symlink_to(target)
    with pytest.raises(CmrError,match='STALE_SOURCE'):e.sync('p')
    assert e.store.get_snapshot().table.count_rows()==1
