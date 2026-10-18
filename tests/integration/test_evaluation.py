import pytest
from codex_memory_rag.evaluation import Evaluator,validate_evaluation_corpus
from codex_memory_rag.retrieval import Retriever
from codex_memory_rag.schemas import CmrError
from test_watcher import engine,Encoder

def test_corpus_drift_aborts_round_and_lock_keeps_snapshot(engine):
    e,s=engine;(s.root/'a.md').write_text('退款恢复');e.build('p')
    r=Retriever(e.config,e.store,Encoder());ev=Evaluator(e.config,r)
    suite={'questions':[{'id':'q1','query':'退款','project_id':'p','kind':'memory','split':'test','positive':True,'group':'decision','evidence':[{'source_id':'s','relative_path':'a.md','line_start':1,'line_end':1}]}]}
    lock=ev.freeze(suite)
    before=lock.table_version
    (s.root/'a.md').write_text('changed');e.sync('p')
    assert e.store.checkout_snapshot(lock.generation,lock.table_version).version==before
    with pytest.raises(CmrError,match='EVAL_CORPUS_CHANGED'):validate_evaluation_corpus(e.config,lock)
    with pytest.raises(CmrError,match='EVAL_CORPUS_CHANGED'):ev.evaluate(suite,lock)

def test_each_mode_uses_same_lock_and_reports_unsupported_baseline(engine):
    e,s=engine;(s.root/'a.md').write_text('退款恢复');e.build('p')
    ev=Evaluator(e.config,Retriever(e.config,e.store,Encoder()))
    suite={'questions':[{'id':'q1','query':'退款','project_id':'p','kind':'memory','split':'test','positive':True,'group':'decision','evidence':[{'source_id':'s','relative_path':'a.md','line_start':1,'line_end':1}]}]}
    lock=ev.freeze(suite);report=ev.evaluate(suite,lock)
    assert report['basic_memory_baseline']['status']=='BASELINE_NOT_COMPARABLE'
    assert {v['table_version'] for v in report['modes'].values()}=={lock.table_version}
    assert report['modes']['lexical']['test']['recall_at_8']==1.0

def test_ignore_policy_drift_invalidates_even_when_file_set_is_unchanged(engine):
    e,s=engine;(s.root/'a.md').write_text('退款恢复');e.build('p')
    ev=Evaluator(e.config,Retriever(e.config,e.store,Encoder()))
    suite={'questions':[{'id':'q','query':'退款','project_id':'p','kind':'memory','split':'test','positive':True,'group':'decision','evidence':[{'source_id':'s','relative_path':'a.md','line_start':1,'line_end':1}]}]}
    lock=ev.freeze(suite)
    (s.root/'.memoryignore').write_text('never-present-file.md\n')
    with pytest.raises(CmrError,match='EVAL_CORPUS_CHANGED'):validate_evaluation_corpus(e.config,lock)

def test_calibration_refuses_holdout_and_returns_dev_similarity_evidence(engine):
    e,s=engine;(s.root/'a.md').write_text('退款恢复');e.build('p')
    ev=Evaluator(e.config,Retriever(e.config,e.store,Encoder()))
    item={'id':'q','query':'退款','project_id':'p','kind':'memory','split':'test','positive':True,'group':'decision','evidence':[{'source_id':'s','relative_path':'a.md','line_start':1,'line_end':1}]}
    suite={'questions':[item]};lock=ev.freeze(suite)
    with pytest.raises(CmrError,match='SCOPE_DENIED'):ev.calibration_samples(suite,lock)
    item['split']='dev';lock=ev.freeze(suite)
    result=ev.calibration_samples(suite,lock)
    assert result['samples'][0]['correct_cosine']>0.99

@pytest.mark.asyncio
async def test_long_evaluation_yields_worker_to_foreground_between_questions(engine,monkeypatch):
    import asyncio,time
    from codex_memory_rag.daemon import PriorityWorker
    from codex_memory_rag.schemas import SearchResponse
    e,s=engine;(s.root/'a.md').write_text('退款');e.build('p')
    r=Retriever(e.config,e.store,Encoder());ev=Evaluator(e.config,r)
    suite={'questions':[{'id':str(i),'query':'x','project_id':'p','kind':'memory','split':'test','positive':False,'group':'negative'} for i in range(8)]};lock=ev.freeze(suite)
    monkeypatch.setattr(r,'search',lambda *a,**kw:(time.sleep(.04) or SearchResponse(generation=lock.generation,table_version=lock.table_version,mode='lexical')))
    worker=PriorityWorker()
    try:
        background=asyncio.create_task(ev.evaluate_async(suite,lock,worker.submit));await asyncio.sleep(.02)
        start=time.monotonic();await worker.submit(lambda:'foreground',0)
        assert time.monotonic()-start<.15
        assert not background.done()
        report=await background
        assert {m['table_version'] for m in report['modes'].values()}=={lock.table_version}
    finally:worker.close()

def test_positive_labels_must_be_eligible_in_the_requested_scope(engine):
    e,s=engine;s=s.model_copy(update={'kind':'memory'});e.config.sources=[s]
    (s.root/'a.md').write_text('---\nstatus: draft\n---\n退款');e.build('p')
    ev=Evaluator(e.config,Retriever(e.config,e.store,Encoder()))
    suite={'questions':[{'id':'invalid','query':'退款','project_id':'p','kind':'memory','split':'test','positive':True,'group':'decision','evidence':[{'source_id':'s','relative_path':'a.md','line_start':4,'line_end':4}]}]}
    with pytest.raises(CmrError,match='LABEL_INELIGIBLE'):ev.freeze(suite)

@pytest.mark.asyncio
async def test_group_quality_counts_only_holdout_questions(engine):
    from codex_memory_rag.daemon import PriorityWorker
    e,s=engine
    (s.root/'a.md').write_text('退款恢复');e.build('p')
    ev=Evaluator(e.config,Retriever(e.config,e.store,Encoder()))
    base={'project_id':'p','kind':'memory','positive':True,'group':'decision',
          'evidence':[{'source_id':'s','relative_path':'a.md','line_start':1,'line_end':1}]}
    suite={'questions':[base|{'id':'dev','split':'dev','query':'退款'},
                        base|{'id':'test','split':'test','query':'不存在的词'}]}
    lock=ev.freeze(suite)
    worker=PriorityWorker()
    try:
        reports=[ev.evaluate(suite,lock),await ev.evaluate_async(suite,lock,worker.submit)]
        for report in reports:
            mode=report['modes']['lexical']
            assert mode['groups']['decision']['positive_count']==1
            assert mode['groups']['decision']['recall_at_8']==0
            assert mode['dev_groups']['decision']['recall_at_8']==1
    finally:
        worker.close()

def test_freeze_rejects_reworded_fact_across_development_and_holdout(engine):
    e,s=engine
    (s.root/'a.md').write_text('退款恢复');e.build('p')
    ev=Evaluator(e.config,Retriever(e.config,e.store,Encoder()))
    base={'project_id':'p','kind':'memory','positive':True,'group':'decision',
          'fact_id':'transactional-refund-recovery',
          'evidence':[{'source_id':'s','relative_path':'a.md','line_start':1,'line_end':1}]}
    suite={'questions':[base|{'id':'dev','split':'dev','query':'退款恢复'},
                        base|{'id':'test','split':'test','query':'如何恢复退款'}]}
    with pytest.raises(CmrError,match='EVAL_SPLIT_LEAK'):
        ev.freeze(suite)
    from codex_memory_rag.schemas import identity_hash
    import asyncio
    valid_lock=ev.freeze({'questions':suite['questions'][:1]})
    supplied_lock=valid_lock.model_copy(update={'suite_hash':identity_hash(suite)})
    with pytest.raises(CmrError,match='EVAL_SPLIT_LEAK'):
        ev.evaluate(suite,supplied_lock)
    with pytest.raises(CmrError,match='EVAL_SPLIT_LEAK'):
        asyncio.run(ev.evaluate_async(suite,supplied_lock,None))
