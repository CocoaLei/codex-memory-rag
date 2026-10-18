import os
import pytest
from codex_memory_rag.sources import read_allowed_file
from codex_memory_rag.schemas import SourceConfig,CmrError

@pytest.mark.parametrize("swap",["file","parent","root"])
def test_open_time_symlink_swap_cannot_return_external_contents(tmp_path,monkeypatch,swap):
    root=tmp_path/"root";root.mkdir();(root/"sub").mkdir();(root/"sub/a.md").write_text("inside")
    outside=tmp_path/"outside";outside.mkdir();(outside/"a.md").write_text("EXTERNAL_PRIVATE_TEXT")
    s=SourceConfig(source_id="s",project_id="p",root=root,kind="docs",include=["**/*.md"],enabled=True,allow_cloud_context=True)
    original=os.open;done=False
    def race(path,*args,**kwargs):
        nonlocal done
        target= "a.md" if swap=="file" else "sub" if swap=="parent" else root.name
        if str(path)==target and not done:
            done=True
            if swap=="file":(root/"sub/a.md").unlink();(root/"sub/a.md").symlink_to(outside/"a.md")
            elif swap=="parent":(root/"sub").rename(root/"old");(root/"sub").symlink_to(outside,target_is_directory=True)
            else:root.rename(tmp_path/"oldroot");root.symlink_to(outside,target_is_directory=True)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(os,"open",race)
    with pytest.raises(CmrError):read_allowed_file(s,"sub/a.md")
    assert done

def test_root_ancestor_symlink_cannot_redirect_registered_root(tmp_path):
    parent=tmp_path/'registered';parent.mkdir();root=parent/'notes';root.mkdir();(root/'a.md').write_text('inside')
    outside=tmp_path/'outside';outside.mkdir();(outside/'notes').mkdir();(outside/'notes/a.md').write_text('OUTSIDE_FIXTURE')
    parent.rename(tmp_path/'saved');parent.symlink_to(outside,target_is_directory=True)
    s=SourceConfig(source_id='s',project_id='p',root=root,kind='docs',include=['*.md'],enabled=True,allow_cloud_context=True)
    with pytest.raises(CmrError):read_allowed_file(s,'a.md')
