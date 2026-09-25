"""Tune and validate a bounded family inside one exclusively guarded GPU process."""
import argparse,gc,hashlib,importlib,itertools,json,os,statistics,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
DATA=Path('/local2/denghaoran/GridMergeAttention/experiments/rf0_spatial_c8_8layers_20260919_v1')
REPO=Path('/local2/denghaoran/GridMergeAttention/axis_wh_c8_source_20260918')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)

def configurations(capacity):
    base=json.loads((HERE/'validated_h200_config.json').read_text())
    base['seed_tile']=4 if capacity==4 else 8 if capacity<=8 else 16
    candidates=[base]
    for patch in [dict(native_tiles=n) for n in (21,63,84,105)]+[
        dict(selector_groups=1),dict(selector_groups=4),dict(head_chunk=8),
        dict(route_n=64,route_warps=4,route_stages=2),
        dict(route_m=64,route_n=128,route_warps=4,route_stages=2),
        dict(route_stages=2),dict(route_stages=4),
        dict(sum_groups=1),dict(sum_groups=4),dict(min_blocks=3),
        dict(min_blocks=2),dict(owner_mode=0)]:
        candidates.append(dict(base,**patch))
    if capacity==4:candidates.append(dict(base,seed_tile=8))
    unique={json.dumps(c,sort_keys=True):c for c in candidates}
    return list(unique.values())

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--capacities',type=int,nargs='+',required=True)
    a=p.parse_args();assert set(a.capacities)<={4,6,8,10,12}
    a.root.mkdir(parents=True,exist_ok=True)
    os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']
    os.environ.setdefault('TRITON_CACHE_DIR','/local2/denghaoran/GridMergeAttention/q80c4_c12_v4_triton')
    os.environ.setdefault('FLASHINFER_WORKSPACE_BASE',str(a.root/'flashinfer'))
    sys.path.insert(0,str(REPO))
    import generate
    import torch
    from q80k90_kernel import Q80K90
    from previous import create as previous
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.cu')}
    write(a.root/'manifest.json',dict(capacities=a.capacities,gpu=torch.cuda.get_device_name(),source_sha256=hashes,
          protocol='Q80K90 exact C; RF0; fixed eight layers; no video generation',started=time.time()))
    def stage(name,**extra):
        row=dict(stage=name,time=time.time(),pid=os.getpid(),**extra);write(a.root/'stage.json',row);print(json.dumps(row),flush=True)
    def clean():gc.collect();torch.cuda.empty_cache()
    def timed(fn):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();out=fn();end.record();end.synchronize();return start.elapsed_time(end),out
    for cap in a.capacities:
        root=a.root/f'C{cap}';root.mkdir(exist_ok=True)
        stage('small_shape_native_validation',capacity=cap)
        from check_selected_gpu import check
        check_result=check(cap,configurations(cap)[0]);write(root/'initial_checks.json',check_result)
        stage('shape_tuning',capacity=cap)
        payload=torch.load(DATA/'captures/p000_s0_l00.pt',map_location='cpu',weights_only=False)
        full=[payload[n].to('cuda').repeat(1,1,10,1).contiguous() for n in ('q','k','v')];del payload
        ids=torch.tensor([(t*45+h)*80+w for t,h,w in itertools.product((0,10,20),(0,22,44),(0,40,79))],device='cuda')
        old=previous(cap);old_out=old(*full);reference=old_out.index_select(1,ids).float().clone();seeds=old.seeds.clone()
        previous_times=[]
        for _ in range(5):
            ms,out=timed(lambda:old(*full));previous_times.append(ms);del out
        del old,old_out;clean()
        report=dict(complete=False,capacity=cap,input_kind='four frozen heads repeated to forty; tuning only',
                    previous_samples_ms=previous_times,previous_mean_ms=statistics.mean(previous_times),rows=[])
        def dense():return torch.nn.functional.scaled_dot_product_attention(*(x.transpose(1,2) for x in full)).transpose(1,2)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            dense();torch.cuda.synchronize()
            for index,config in enumerate(configurations(cap)):
                stage('shape_tuning',capacity=cap,index=index,count=len(configurations(cap)),config=config)
                core=None
                try:
                    core=Q80K90(cap,**config);core(*full);out=core(*full)
                    sample=out.index_select(1,ids).float();del out
                    error=float((sample-reference).norm()/reference.norm())
                    changed=float((core.seeds!=seeds).float().mean());del sample
                    assert error<.02 and changed<.005,(error,changed)
                    times=dict(method=[],dense=[])
                    for repeat in range(5):
                        funcs=[('method',lambda:core(*full)),('dense',dense)]
                        if repeat%2:funcs.reverse()
                        for name,fn in funcs:
                            ms,out=timed(fn);times[name].append(ms);del out
                    row=dict(valid=True,config=config,relative_l2_vs_previous=error,seed_changed_fraction=changed,
                             samples_ms=times,mean_attention_ms=statistics.mean(times['method']),
                             dense_mean_attention_ms=statistics.mean(times['dense']))
                except (AssertionError,torch.cuda.OutOfMemoryError) as exc:
                    row=dict(valid=False,config=config,error=repr(exc))
                report['rows'].append(row);write(root/'tune.json',report)
                print(json.dumps(dict(capacity=cap,**row)),flush=True)
                del core;clean()
        valid=[r for r in report['rows'] if r['valid']];assert valid
        selected=min(valid,key=lambda r:r['mean_attention_ms'])
        report.update(complete=True,selected=selected);write(root/'tune.json',report)
        write(root/'selected_config.json',selected['config'])
        del full,ids,reference,seeds;clean()
        stage('frozen_numerical',capacity=cap)
        numerical=importlib.import_module(f'benchmark_q80k90c{cap}')
        sys.argv=['numerical','--root',str(root/'numerical'),'--data',str(DATA),'--reference',
                  str(DATA/'compare_spatial_rf0_eight_layers_20260919.py'),'--operator',str(HERE/'compare_block_extensions_rf0.py'),
                  '--phase','validate','--config',str(root/'selected_config.json')]
        numerical.main();result=json.loads((root/'numerical/validate.json').read_text());s=result['numerical_summary']
        assert result['complete'] and s['cases']==384 and result['layers']==[0,5,10,15,20,25,30,35]
        passed=s['max_relative_l2_vs_operator']<.02 and s['mean_seed_changed_fraction']<.005 and s['mean_l2_to_dense']<=s['reference_mean_l2_to_dense']+.0005
        write(root/'numerical_gate.json',dict(approved=passed,summary=s,invariants=result['invariants'],
              thresholds_owner='engineering checks, not universal user tolerances'))
        assert passed,(cap,s)
        clean()
    stage('paired_real40',capacities=a.capacities)
    import benchmark_family_real40
    sys.argv=['real40','--repo',str(REPO),'--root',str(a.root/'real40'),'--family-root',str(a.root),
              '--capacities',*[str(c) for c in a.capacities],'--prompts','3']
    benchmark_family_real40.main()
    real=json.loads((a.root/'real40/real40.json').read_text())
    assert real['complete']
    for cap in a.capacities:
        measured=real['capacities'][str(cap)]
        f=6*40*128*cap*945*840*90+4*40*128*75600*cap*840+2*40*128*16*840*45*32
        write(a.root/f'C{cap}'/'result.json',dict(capacity=cap,complete=True,numerical_passed=True,
              gpu=torch.cuda.get_device_name(),config=json.loads((a.root/f'C{cap}'/'selected_config.json').read_text()),
              **measured,per_call_tflops=f/1e12,full_50_step_pflops=f*4000/1e15,
              protocol='3 prompts x 4 frozen timesteps x 8 fixed layers x 40 real independent heads',
              source_sha256=hashes,video_quality='pending; no videos generated'))
    stage('complete',capacities=a.capacities)

if __name__=='__main__':main()
