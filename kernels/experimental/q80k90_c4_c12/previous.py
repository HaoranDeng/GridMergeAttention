"""Frozen previous implementations used only for paired engineering comparison."""
import importlib.util
from pathlib import Path

def create(capacity):
    here=Path(__file__).parent
    if capacity in (6,8):
        path=here/'previous_v3/q80k90_kernel.py'
        spec=importlib.util.spec_from_file_location('previous_v3',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        import json
        return module.Q80K90(capacity,**json.loads((here/'validated_h200_config.json').read_text()))
    if capacity in (10,12):
        path=here/f'previous_c{capacity}/q80k90c{capacity}_kernel.py'
        spec=importlib.util.spec_from_file_location(f'previous_c{capacity}',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return getattr(module,f'Q80K90C{capacity}')(merge_backend='cuda',native_tiles=105,
                    head_chunk=4,merge_groups=4,merge_loop=1,route_m=128,route_n=64)
    from q80k90_kernel import Q80K90
    return Q80K90(capacity=4,seed_tile=8,merge_backend='cuda',selector_precision='fp64',
                  owner_mode=0,fast_math=False,direct_store=False,route_m=128,route_n=64)
