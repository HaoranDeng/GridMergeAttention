"""Construct the fastest validated implementation selected for each Q80K90 capacity."""
import functools
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

@functools.lru_cache(maxsize=2)
def implementation(name):
    if name=='v4':path=HERE/'q80k90_kernel.py'
    elif name=='v3':path=HERE/'previous_v3/q80k90_kernel.py'
    else:raise ValueError(f'Unrecognized validated implementation: {name}')
    spec=importlib.util.spec_from_file_location('q80k90_selected_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Q80K90

def make_q80k90(capacity):
    """Return one stream-local, exact-capacity kernel with its measured H200 config."""
    if capacity not in (4,6,8,10,12):raise ValueError('Supported capacities: 4, 6, 8, 10, 12')
    configs=json.loads((HERE/'validated_h200_configs.json').read_text())
    selected=configs[str(capacity)]
    return implementation(selected['implementation'])(capacity=capacity,**selected['config'])
