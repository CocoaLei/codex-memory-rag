"""An approved file is not approval for another chunk or another range."""
from types import SimpleNamespace
import json

import pytest

from test_adapter import HASH, KEY, fixture, item, run, search


@pytest.mark.parametrize('change', [
    {'chunk_id': 'e' * 64}, {'line_start': 1}, {'line_end': 5},
    {'chunk_id': 'e' * 64, 'line_start': 100, 'line_end': 110},
])
def test_grant_rejects_same_file_other_chunk_or_range(tmp_path, change):
    m, adapter, cli, path = fixture(tmp_path)
    grant = SimpleNamespace(source_id='example-code', relative_path='example.py',
                            file_sha256=HASH, chunk_id=KEY, line_start=2, line_end=4)
    assert not adapter.granted(item(**change), SimpleNamespace(files=[grant]))


def test_exact_grant_allows_only_bound_identity(tmp_path):
    m, adapter, cli, path = fixture(tmp_path)
    grant = SimpleNamespace(source_id='example-code', relative_path='example.py',
                            file_sha256=HASH, chunk_id=KEY, line_start=2, line_end=4)
    assert adapter.granted(item(), SimpleNamespace(files=[grant]))


@pytest.mark.parametrize('change', [
    {'chunk_id': 'e' * 64}, {'line_start': 1}, {'line_end': 5},
])
def test_unapproved_same_file_candidate_is_not_exposed_or_issued(tmp_path, change):
    m, adapter, cli, path = fixture(tmp_path)
    cli.result = search([item(**change)])
    result = run(adapter, 'candidate_search_code', {'query': 'synthetic other region'})
    assert result['results'] == [] and result['truncated']
    assert not adapter.candidates
    with pytest.raises(m.AdapterError, match='CANDIDATE_NOT_ISSUED'):
        run(adapter, 'candidate_get_context', {'chunk_id': change.get('chunk_id', KEY)})
    assert len(cli.calls) == 1


@pytest.mark.parametrize('change', ['v1', 'missing_chunk', 'missing_range', 'reversed', 'bool', 'duplicate'])
def test_file_only_or_invalid_exact_policy_denies_before_cli(tmp_path, change):
    m, adapter, cli, path = fixture(tmp_path)
    value = json.loads(path.read_text())
    if change == 'v1':
        value['schema'] = 'candidate-cloud-policy-v1'
    elif change == 'missing_chunk':
        value['files'][0].pop('chunk_id')
    elif change == 'missing_range':
        value['files'][0].pop('line_end')
    elif change == 'reversed':
        value['files'][0]['line_start'] = 9
    elif change == 'bool':
        value['files'][0]['line_start'] = True
    else:
        value['files'].append(value['files'][0])
    path.write_text(json.dumps(value))
    with pytest.raises(m.AdapterError, match='CLOUD_POLICY_INVALID'):
        run(adapter, 'candidate_status')
    assert not cli.calls


def test_range_policy_change_invalidates_old_candidate_before_context_cli(tmp_path):
    m, adapter, cli, path = fixture(tmp_path)
    run(adapter, 'candidate_search_code', {'query': 'synthetic'})
    value = json.loads(path.read_text())
    value['files'][0]['line_end'] = 6
    path.write_text(json.dumps(value))
    with pytest.raises(m.AdapterError, match='CANDIDATE_NOT_ISSUED'):
        run(adapter, 'candidate_get_context', {'chunk_id': KEY})
    assert len(cli.calls) == 1
