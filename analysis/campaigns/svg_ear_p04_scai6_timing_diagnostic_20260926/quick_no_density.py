"""Four native p=.4 denoising steps with density recording disabled."""
import importlib.util,json,os,sys,time
from pathlib import Path
import quick_diagnostic as framework

def worker():
    root=framework.CAMPAIGN;out=framework.OUT;plan=framework.read(root/'experiment_plan.json');snap=Path(plan['snapshot'])
    for folder,hs in [(snap,plan['source_sha256']),(Path(plan['source_svg_root']),plan['svg_sha256'])]:
        for name,digest in hs.items():assert framework.sha(folder/name)==digest,name
    assert framework.sha(plan['config'])==plan['config_sha256']
    os.environ.pop('SVG_DENSITY_LOG',None)
    assert os.environ['SVG_TOP_P']=='0.4'
    os.chdir(snap);sys.path.insert(0,str(snap))
    import torch
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    from attention import svg_family_wan as family
    from attention.runtime_timing import ATTENTION_RUNTIME_RECORDER as recorder
    from wan.modules.model import WanSelfAttention
    cls=family._load_svg_modules()['ear']
    cls.logging_file=None
    state=dict(calls=0)
    class Finished(Exception):pass
    original=WanSelfAttention.forward
    def forward(self,*args,**kwargs):
        n=state['calls']
        if n==320:raise Finished()
        assert self.layer_idx==n%40 and self.attention_backend=='ear_all'
        assert tuple(args[0].shape)==(1,75600,5120)
        result=original(self,*args,**kwargs);state['calls']+=1
        assert cls.logging_file is None
        if state['calls']%80==0:
            framework.save(out/'progress.json',dict(calls=state['calls'],time=time.time()))
            print(json.dumps(dict(calls=state['calls'])),flush=True)
        return result
    WanSelfAttention.forward=forward
    spec=importlib.util.spec_from_file_location('frozen_generate',snap/'generate.py');gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    sys.argv=['generate.py','--config',plan['config'],'--checkpoint','pretrained','--prompt_path',plan['prompt_path'],'--image_list',plan['image_list'],'--image_dir',plan['image_dir'],'--output_dir',str(out/'unused_video_output'),'--runtime_json',str(out/'unused_runtime.json'),'--num_videos','1','--num_samples','1','--seed','0']
    try:gen.main()
    except Finished:pass
    else:raise RuntimeError('Four-step stop was not reached')
    torch.cuda.synchronize()
    assert state['calls']==len(recorder.core_events)==len(recorder.full_events)==320
    assert not (out/'density.jsonl').exists()
    rows=[]
    for n,((a,b),(c,d)) in enumerate(zip(recorder.core_events,recorder.full_events)):
        rows.append(dict(call=n,step=n//80,branch=(n%80)//40,layer=n%40,core_ms=a.elapsed_time(b),full_ms=c.elapsed_time(d)))
    framework.save(out/'diagnostic.json',dict(valid=True,top_p=.4,steps=4,calls=320,density_logging=False,extra_iteration_audit=False,
        input=plan['input_manifest'][0],rows=rows,source_sha256=plan['source_sha256'],svg_sha256=plan['svg_sha256'],config_sha256=plan['config_sha256'],
        scope='Original native core/full CUDA-event scopes; first four steps of unchanged50-step schedule. Step0 initialization excluded from steady short-test mean; steps1-3 contain240 calls. No video output; not a full50-step/video runtime or quality estimate.',time=time.time()))

if __name__=='__main__':
    framework.OUT=framework.CAMPAIGN/'quick_no_density'
    framework.__file__=__file__
    if '--worker' in sys.argv:worker()
    else:framework.guard()
