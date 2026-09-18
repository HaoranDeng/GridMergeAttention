"""Keep K/V resident across multiple six-column query tiles, unchanged math."""
from pathlib import Path
import runpy
root=Path(__file__).resolve().parents[1]
old='TORCH_CHECK(tiles==1 || tiles==3 || tiles==7 || tiles==21,"Require tiles 1/3/7/21");'
runpy.run_path(str(root/'analysis/build_c10_owner_tree_native_layout.py'))
for source,target in [('fisher_c10_half_owner.cu','fisher_c10_long_tiles.cu'),('fisher_c10_owner_tree_native_layout.cu','fisher_c10_long_tiles_tree.cu')]:
    s=(root/'native_fisher'/source).read_text()
    assert old in s
    s=s.replace(old,'TORCH_CHECK(tiles==1 || tiles==3 || tiles==7 || tiles==21 || tiles==42 || tiles==84 || tiles==294,"Unsupported tile count");')
    p=root/'native_fisher'/target;p.write_text(s);print(p)
