import numpy as np
import pytest
from codex_memory_rag.embedding import format_query,format_document,validate_vectors,Encoder
from codex_memory_rag.schemas import CmrError

def test_prefix_contract():
    assert format_query("查找刷新逻辑","memory")=="task: search result | query: 查找刷新逻辑"
    assert format_query("find refresh","code")=="task: code retrieval | query: find refresh"
    assert format_document("Widget.kt","fun update() {}" )=="title: Widget.kt | text: fun update() {}"
    assert format_document(None,"x")=="title: none | text: x"

@pytest.mark.parametrize("arr",[np.zeros((1,768)),np.ones((1,512)),np.ones((1,769)),np.full((1,768),np.nan),np.full((1,768),np.inf)])
def test_bad_vectors_rejected(arr):
    with pytest.raises(CmrError,match="EMBEDDING_CONTRACT_MISMATCH"):validate_vectors(arr)

def test_full_information_normalization():
    x=np.arange(1,769,dtype=np.float64)[None,:]
    y=validate_vectors(x)
    assert y.shape==(1,768) and y.dtype==np.float32
    assert abs(np.linalg.norm(y)-1)<1e-4
    assert np.all(y>0)

class Tokenizer:
    def encode(self,text,add_special_tokens=True):return list(range(len(text)))
class SpyModel:
    tokenizer=Tokenizer()
    def encode(self,texts,**kwargs):
        assert kwargs["prompt"]=="" and kwargs["truncate_dim"] is None
        assert all(t.count("task:")==1 for t in texts)
        self.inputs=texts
        return np.ones((len(texts),768))

def test_no_double_prompt_or_silent_truncation():
    model=SpyModel(); encoder=Encoder.from_verified_model(model,"test-profile",max_tokens=100)
    result=encoder.encode_queries(["查找"],"memory")
    assert result.shape==(1,768)
    assert model.inputs==["task: search result | query: 查找"]
    with pytest.raises(CmrError,match="QUERY_TOO_LONG"):encoder.encode_queries(["x"*101],"memory")

def test_profile_and_embedded_chunk_are_strong_validated_contracts():
    from codex_memory_rag.schemas import EmbeddingProfile,EmbeddedChunk
    import pytest
    with pytest.raises(ValueError):
        EmbeddingProfile(model_lock={},backend='cpu',dimension=512,profile_id='p')
    assert EmbeddingProfile(model_lock={},backend='cpu',profile_id='p').dimension==768
    assert EmbeddedChunk.model_fields['vector'].metadata
