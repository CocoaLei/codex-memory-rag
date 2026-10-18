"""Cross-call policy changes invalidate previously issued candidate authority."""
import json

import pytest

from test_adapter import KEY, WARN, context, fixture, run, search


def health():
    return {'warnings':WARN,'pid':42,'dimension':768,'backend':'mps',
            'loading':False,'syncing':False,'error':None,'generation':'g1','embedding_profile':'e',
            'reranker':{'enabled':True,'ready':True,'error':None,'profile':'r','backend':'mps'},
            'sources':{'example-code':{'state':'ready'}}}


def reissue_and_read(adapter,cli):
    cli.result=search()
    run(adapter,'candidate_search_code',{'query':'SYNTHETIC_QUERY'})
    cli.result=context('SYNTHETIC_CONTEXT')
    assert run(adapter,'candidate_get_context',{'chunk_id':KEY})['items'][0]['text']=='SYNTHETIC_CONTEXT'


@pytest.mark.parametrize('change',['content','identity'])
def test_changed_policy_requires_search_before_old_candidate_readback(tmp_path,change):
    m,a,cli,f=fixture(tmp_path)
    run(a,'candidate_search_code',{'query':'SYNTHETIC_QUERY'})
    if change=='content':
        value=json.loads(f.read_text());value['max_context_chars']=1000
        f.write_text(json.dumps(value))
    else:
        replacement=f.with_name('replacement.json')
        replacement.write_bytes(f.read_bytes());replacement.chmod(0o600);replacement.replace(f)
    cli.result=context('SYNTHETIC_CONTEXT')
    with pytest.raises(m.AdapterError,match='CANDIDATE_NOT_ISSUED'):
        run(a,'candidate_get_context',{'chunk_id':KEY})
    assert len(cli.calls)==1
    assert not a.candidates
    reissue_and_read(a,cli)


def test_revocation_status_reapproval_never_revives_old_candidate(tmp_path):
    m,a,cli,f=fixture(tmp_path)
    run(a,'candidate_search_code',{'query':'SYNTHETIC_QUERY'});original=f.read_bytes()
    revoked=json.loads(original);revoked['files']=[];f.write_text(json.dumps(revoked))
    cli.result=health();run(a,'candidate_status')
    assert not a.candidates
    f.write_bytes(original);cli.result=context('SYNTHETIC_CONTEXT')
    with pytest.raises(m.AdapterError,match='CANDIDATE_NOT_ISSUED'):
        run(a,'candidate_get_context',{'chunk_id':KEY})
    assert len(cli.calls)==2
    reissue_and_read(a,cli)


def test_policy_changed_before_status_clears_candidate_even_when_file_still_granted(tmp_path):
    m,a,cli,f=fixture(tmp_path)
    run(a,'candidate_search_code',{'query':'SYNTHETIC_QUERY'})
    value=json.loads(f.read_text());value['max_context_chars']=1000;f.write_text(json.dumps(value))
    cli.result=health();run(a,'candidate_status')
    assert not a.candidates
    cli.result=context('SYNTHETIC_CONTEXT')
    with pytest.raises(m.AdapterError,match='CANDIDATE_NOT_ISSUED'):
        run(a,'candidate_get_context',{'chunk_id':KEY})


def test_new_policy_search_does_not_retain_old_candidates_from_same_index_snapshot(tmp_path):
    m,a,cli,f=fixture(tmp_path)
    run(a,'candidate_search_code',{'query':'SYNTHETIC_QUERY'})
    value=json.loads(f.read_text());value['max_context_chars']=1000;f.write_text(json.dumps(value))
    cli.result=search([]);run(a,'candidate_search_code',{'query':'SYNTHETIC_EMPTY'})
    cli.result=context('SYNTHETIC_CONTEXT')
    with pytest.raises(m.AdapterError,match='CANDIDATE_NOT_ISSUED'):
        run(a,'candidate_get_context',{'chunk_id':KEY})
    assert len(cli.calls)==2


def test_unchanged_policy_status_keeps_valid_candidate_readable(tmp_path):
    m,a,cli,f=fixture(tmp_path)
    run(a,'candidate_search_code',{'query':'SYNTHETIC_QUERY'})
    cli.result=health();run(a,'candidate_status')
    cli.result=context('SYNTHETIC_CONTEXT')
    assert run(a,'candidate_get_context',{'chunk_id':KEY})['readback_status']=='complete'
    assert len(cli.calls)==3
