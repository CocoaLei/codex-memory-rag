import subprocess,sys,json

def test_doctor_is_installed_and_returns_selective_report(tmp_path):
    result=subprocess.run([sys.executable,'-m','codex_memory_rag.cli','doctor','--report',str(tmp_path/'r.json')],capture_output=True,text=True)
    assert result.returncode==0
    data=json.loads((tmp_path/'r.json').read_text())
    assert data['model_id']=='google/embeddinggemma-2' and data['dimension']==768
    assert 'environment' not in data
