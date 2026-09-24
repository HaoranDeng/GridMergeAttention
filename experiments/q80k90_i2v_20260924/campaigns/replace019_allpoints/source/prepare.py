"""Freeze all existing I2V plotted settings, reusing the completed new019 trials."""
import copy
import json
import time
from pathlib import Path
import yaml
from worker import ROOT, read, save, sha

OLD=ROOT.parent/'i2v_replace019_sparkler_20260924'
MAIN=ROOT.parent/'q80k90_i2v_c4_c6_c8_c10_20260924'
BASE=Path('/xuanwu-tank/east/zrl/exp-gridmergeattn')

def main():
    assert not (ROOT/'experiment_plan.json').exists(), 'Already prepared; inspect, do not duplicate'
    previous=read(OLD/'experiment_plan.json');plan=copy.deepcopy(previous)
    assert read(OLD/'restoration_verified.json')['verified']
    snap=Path(plan['snapshot'])
    for rel,h in plan['source_sha256'].items(): assert sha(snap/rel)==h,rel
    for row in plan['input_manifest']: assert sha(Path(row['image']))==row['image_sha256']
    for rel,h in plan['svg_sha256'].items(): assert sha(Path(plan['source_svg_root'])/rel)==h,rel
    for cap in [6,10]:
        rel=f'configs/q80k90_i2v_C{cap}.yaml';plan['source_sha256'][rel]=sha(snap/rel)
    ref=read(ROOT/'baseline_reference.json')
    settings={f'C{c}':{'family':'gca','capacity':c,'config':plan['configs'][str(c)]} for c in (4,6,8,10)}
    configs=ROOT/'configs';configs.mkdir(exist_ok=True)
    baseline_configs={}
    dense=yaml.safe_load(Path(plan['configs']['Dense']).read_text())
    def comparable(c):
        c=copy.deepcopy(c);c['token_merging_args'].pop('attention_backend',None);return c
    for family in ['svg','svg2','svg_ear']:
        source=BASE/f'ours/GridMergeAttention/configs/base50i_{family}_unipc.yaml'
        if family=='svg_ear':source=Path(previous['configs']['EAR_p08'])
        content=source.read_bytes();c=yaml.safe_load(content)
        assert comparable(c)==comparable(dense),family
        target=configs/source.name;target.write_bytes(content);baseline_configs[family]=str(target)
    for family,ps in [('svg_ear',[.5,.6,.7,.8]),('svg2',[.5,.6,.7,.8,.9])]:
        for value in ps:
            tag=family+'_p'+str(value).replace('.','')
            settings[tag]={'family':'ear' if family=='svg_ear' else 'svg2','top_p':value,'config':baseline_configs[family]}
    for tag,value in [('svg_s0102',.102),('svg_s0153',.153),('svg_s0205',.205),('svg_s0255',.255),('svg_s03',.3)]:
        original=next(x for x in ref['series']['SVG'] if x['tag']==tag)
        assert len(set(original['flops_per_video']))==1
        settings[tag]={'family':'svg','sparsity':value,'config':baseline_configs['svg'],
                       'static_full50_flops':original['flops_per_video'][0],
                       'static_flops_scope':original['cost_kind']}
    reuse={'Dense':'Dense','C4':'C4','C8':'C8','svg_ear_p08':'EAR_p08'}
    reuse_hashes={}
    for method,old_method in reuse.items():
        source=OLD/old_method;proof,=read(source/'video20/purity_records.json')
        assert proof['attention_calls']==4000 and proof['layer_calls']==[100]*40
        assert proof['video_sha256']==sha(source/'video20/019-00.mp4')
        assert read(source/'video20/input_manifest.json')['inputs']==plan['input_manifest']
        (ROOT/method).symlink_to(source,target_is_directory=True)
        reuse_hashes[method]=dict(source=str(source),sha256=proof['video_sha256'])
    new_methods=[k for k in settings if k not in reuse]
    assert len(settings)==18 and len(new_methods)==15
    plan.update(root=str(ROOT),created_at=time.time(),status='prepared',methods=list(settings),settings=settings,
                new_methods=new_methods,reuse=reuse_hashes,original_trial=str(OLD),original_main=str(MAIN),
                dense_reference=str(ROOT/'Dense/video20'),
                gpu={'host':'scai6','index':2,'uuid':'GPU-244e87b8-5a10-aadb-858a-f0eea028dbd1','model':'NVIDIA H200'},
                config_sha256={v['config']:sha(Path(v['config'])) for v in settings.values()},
                post_hoc=True,replacement_scope='Same new019 conditioning image/prompt and seed19 in all18 curve settings; unchanged000-018 reused.',
                timing_note='Actual new019 timings are single-video cold-start samples and remain separate by GPU. Current user plotting coordinates are preserved independently.',
                restoration_required=False)
    plan['protocol'].update(capacities=[4,6,8,10],generated_indices=[19],videos_per_capacity=1,comparison_videos_per_setting=20)
    for key in ('existing_c8','launch_capacities','ear_parameters'):plan.pop(key,None)
    save(ROOT/'experiment_plan.json',plan)
    save(ROOT/'reused_quality.json',{new:read(OLD/'quality.json')['results'][old] for new,old in reuse.items() if new!='Dense'})
    print(json.dumps({'prepared':True,'settings':18,'new_videos':15,'reused':list(reuse),'gpu':plan['gpu']}))

if __name__=='__main__':main()
