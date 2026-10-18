import pytest
from pydantic import ValidationError
from codex_memory_rag.schemas import AppConfig, SourceConfig, SearchRequest, ContextRequest

def test_fixed_dimension_and_unknown_fields():
    for v in [{"dimension":512}, {"dimension":769}, {"allow_remote_code":True}, {"inference_dtype":"float16"}, {"surprise":1}]:
        with pytest.raises(ValidationError): AppConfig(embedding=v)

def test_source_registration_rejects_relative_duplicate_and_unbound_worktree(tmp_path):
    with pytest.raises(ValidationError): SourceConfig(source_id="s", project_id="p", kind="memory", root="relative")
    s=dict(source_id="s",project_id="p",kind="memory",root=str(tmp_path))
    with pytest.raises(ValidationError): AppConfig(sources=[s,s])
    with pytest.raises(ValidationError): SourceConfig(**(s|{"kind":"code"}))
    assert AppConfig().sources==[]

def test_request_limits_and_no_arbitrary_fields():
    for v in [dict(top_k=0),dict(top_k=21),dict(path="/etc/passwd")]:
        with pytest.raises(ValidationError): SearchRequest(project_id="p",query="x",**v)
    with pytest.raises(ValidationError): ContextRequest(project_id="p",chunk_ids=[],path="x")
