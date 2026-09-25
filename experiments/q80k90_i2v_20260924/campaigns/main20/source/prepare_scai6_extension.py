"""Create host/shard wrappers; retain frozen generator, metrics and kernels."""
from pathlib import Path

HERE = Path(__file__).resolve().parent

def change(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

w = (HERE/'worker.py').read_text(encoding='utf-8')
w = change(w, "choices=[4,6,8,10],default=4", "choices=[6],default=6")
w = change(w, "choices=[0,7],default=0", "choices=[2,5],default=2")
w = change(w, "gpu=dict(plan['gpu']) if a.gpu==0 else dict(host='scai7',index=7,uuid='GPU-7aa670f7-95a6-1e16-c28f-80664a449ad8',model='NVIDIA H200')",
           "gpu=dict(host='scai6',index=a.gpu,uuid={2:'GPU-244e87b8-5a10-aadb-858a-f0eea028dbd1',5:'GPU-60b9a1cc-32ad-7b4d-e8eb-7fc32ed5ba3c'}[a.gpu],model='NVIDIA H200')")
w = change(w, "control=ROOT if a.gpu==0 else ROOT/'gpu7'", "control=ROOT/f'scai6_gpu{a.gpu}'")
w = change(w, "out=control/'native_gate_probe' if a.gate else ROOT/f'C{a.capacity}/video20'", "out=control/'native_gate_probe' if a.gate else ROOT/f'C6/shards/scai6_gpu{a.gpu}/video20'")
w = change(w, "        gen.main()\n        runtime=", "        sys.argv += ['--num_prompt_shards','2','--prompt_shard_id',str({2:0,5:1}[a.gpu])]\n        gen.main()\n        runtime=")
w = change(w, "assert len(runtime['records'])==len(purity)==20", "assert len(runtime['records'])==len(purity)==10\n        assert sorted(r['prompt_index'] for r in runtime['records'])==list(range({2:0,5:1}[a.gpu],20,2))")
w = change(w, "save(ROOT/f'C{a.capacity}/generation_complete.json',dict(complete=True,videos=20,time=time.time()))", "save(control/'generation_complete.json',dict(complete=True,videos=10,host='scai6',gpu=a.gpu,indices=list(range({2:0,5:1}[a.gpu],20,2)),time=time.time()))")
(HERE/'worker_scai6.py').write_text(w, encoding='utf-8')

l = (HERE/'launch.py').read_text(encoding='utf-8')
l = change(l, "{0:'GPU-4eca131f-0ba3-759d-bcb5-0418d6b7c68e',7:'GPU-7aa670f7-95a6-1e16-c28f-80664a449ad8'}", "{2:'GPU-244e87b8-5a10-aadb-858a-f0eea028dbd1',5:'GPU-60b9a1cc-32ad-7b4d-e8eb-7fc32ed5ba3c'}")
l = change(l, "choices=[0,7],default=0", "choices=[2,5],default=2")
l = change(l, "control=ROOT if a.gpu==0 else ROOT/'gpu7'", "control=ROOT/f'scai6_gpu{a.gpu}'")
l = change(l, "ROOT/'source/run_campaign.py'", "ROOT/'source/run_scai6_shard.py'")
l = change(l, "TORCH_EXTENSIONS_DIR=str(control/'torch_extensions'),", "FLASHINFER_WORKSPACE_BASE='/xuanwu-tank/east/zrl/exp-gridmergeattn/.cache/environments/monarch-rt/flashinfer_workspace',\n             TORCH_EXTENSIONS_DIR=str(control/'torch_extensions'),")
l = l.replace('borrowed=a.gpu==0', 'borrowed=False').replace('restoration_required=a.gpu==0', 'restoration_required=False')
(HERE/'launch_scai6.py').write_text(l, encoding='utf-8')
print('Created two coordination wrappers; original worker, launch and snapshot unchanged.')
