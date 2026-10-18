from codex_memory_rag.retrieval import rrf

def test_rrf_ranks_missing_and_duplicates():
    value=rrf([["a","a","b"],["b","c"]])
    assert abs(value['a']-1/61)<1e-10
    assert abs(value['b']-(1/62+1/61))<1e-10
    assert abs(value['c']-1/62)<1e-10
