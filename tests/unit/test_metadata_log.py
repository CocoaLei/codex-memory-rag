import json,pytest
from codex_memory_rag.metadata_log import metadata_logger,emit

def test_bounded_allowlisted_metadata_cannot_accept_source_or_query(tmp_path):
    logger=metadata_logger(tmp_path);handler=logger.handlers[0]
    assert handler.maxBytes==10*1024*1024 and handler.backupCount==3
    emit(logger,'startup',pid=1,state='ready');handler.flush()
    assert json.loads((tmp_path/'logs/metadata.log').read_text())['event']=='startup'
    with pytest.raises(ValueError):emit(logger,'operation',query='private note')
    with pytest.raises(ValueError):emit(logger,'private note',state='ready')
