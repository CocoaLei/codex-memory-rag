import pytest
from codex_memory_rag.schemas import identity_hash, source_key, chunk_id

def test_ambiguous_old_concatenations_are_distinct():
    assert source_key("p","a","bc","d.py") != source_key("p","ab","c","d.py")
    assert identity_hash([None]) != identity_hash(["null"])
    assert identity_hash(["é"]) != identity_hash(["e\u0301"])

def test_golden_canonical_json_hash():
    assert identity_hash(["a",1,None]) == "0ae764bffd3a17b6454c8d9ebf712311d9b92efce67a4e4ea4f095fc05ed0122"

def test_ordinal_not_bool_or_string():
    for value in [True,"1",-1]:
        with pytest.raises(ValueError):chunk_id("a","b","v1",value)
