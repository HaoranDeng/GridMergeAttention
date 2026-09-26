"""Apply the user-requested short-test latency only; preserve all other points."""
import copy,hashlib,json,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PAPER=Path('C:/GMA/.codex_tmp/paper-ear-p04-20260925')
SUPPORT=ROOT.parent/'T2V_I2V_tradeoff_combined'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
def main():
    v=read(ROOT/'quick_no_density/VERIFICATION.json');assert v['valid'] and v['density_logging'] is False
    source=dict(verification=v,per_call=read(ROOT/'quick_no_density/diagnostic.json')['rows'],
        previous_full20_logged_core_ms=106.2750353262048,
        quality_and_flops_source='svg_ear_p04_i2v20.json',
        user_requested_short_test=True,full50_runtime_replacement=False)
    changes=[]
    backup=ROOT/'plot_before_no_density';backup.mkdir(exist_ok=False)
    targets=[PAPER/'figure_sources/tradeoff_combined/data.json',SUPPORT/'t2v_i2v_tradeoff_replacement019_data.json']
    for j,path in enumerate(targets):
        before=read(path);data=copy.deepcopy(before);pts=data['I2V']['series']['SVG-EAR']
        p,=[p for p in pts if p['setting']=='svg_ear_p04'];assert p['runtime']==106.2750353262048
        p.update(runtime=v['steady_core_ms'],runtime_scope='User-requested native short test without density logging: input000 seed0, steps2-4 of original50-step schedule,240 calls, exclusive scai6 GPU2. Not a full50-step/video mean.',
            runtime_source='svg_ear_p04_no_density_timing.json',runtime_samples=1,runtime_steady_calls=240,
            runtime_steady_steps_zero_based=[1,2,3],runtime_previous_logged_ms=p['runtime'])
        data['svg_ear_p04_latency_override']=dict(source='svg_ear_p04_no_density_timing.json',core_ms=v['steady_core_ms'],
            full_attention_ms=v['steady_full_ms'],scope=p['runtime_scope'],quality_flops_unchanged=True)
        restored=copy.deepcopy(data);restored.pop('svg_ear_p04_latency_override')
        oldp,=[p for p in before['I2V']['series']['SVG-EAR'] if p['setting']=='svg_ear_p04']
        for i,p in enumerate(restored['I2V']['series']['SVG-EAR']):
            if p['setting']=='svg_ear_p04':restored['I2V']['series']['SVG-EAR'][i]=copy.deepcopy(oldp)
        assert restored==before
        shutil.copy2(path,backup/f'data_{j}.json');write(path,data)
        write(path.parent/'svg_ear_p04_no_density_timing.json',source)
        changes.append(dict(file=str(path),only_changed_point='I2V/svg_ear_p04',previous=oldp['runtime'],new=v['steady_core_ms']))
    tex=PAPER/'content/experiments.tex';before=tex.read_text(encoding='utf-8')
    match=re.search(r'(\\includegraphics\[width=\\linewidth\]\{figures/gca_tradeoff_t2v_i2v.pdf\}\s*\\caption\{)(.*?)(\}\s*\\label\{fig:gca_tradeoff\})',before,re.S)
    assert match
    sentence=' For I2V SVG-EAR $p=0.4$, latency uses 240 calls from steps 2--4 of one 50-step trajectory with density logging disabled; quality and arithmetic retain the complete 20-video measurements.'
    after=before[:match.start(2)]+match.group(2)+sentence+before[match.end(2):]
    assert after.replace(sentence,'',1)==before
    shutil.copy2(tex,backup/'experiments.tex');tex.write_text(after,encoding='utf-8',newline='\n')
    readme=PAPER/'figure_sources/tradeoff_combined/README.md';text=readme.read_text(encoding='utf-8')
    text=text.replace('Latency=106.275035326ms/call is the mean','Original logged latency=106.275035326ms/call is the mean')
    text+='\n## User-requested p=.4 latency correction\n\nOnly the I2V p=.4 latency coordinate is now '+f"{v['steady_core_ms']:.9f}"+'ms/call, measured in a short native run with density recording disabled. The first four steps of the original50-step schedule were executed on exclusive scai6 GPU2; step1 initialization is excluded, leaving240 calls across steps2-4, all40 layers and both CFG branches, input000 seed0. Full attention was '+f"{v['steady_full_ms']:.9f}"+'ms/call. This is not a full50-step/video timing mean. The caption states this exception; all20-video quality and full50 arithmetic values, other points, T2V and GCA user time coordinates remain unchanged. `svg_ear_p04_no_density_timing.json` retains per-call timings, verification and the previous logged latency.\n'
    readme.write_text(text,encoding='utf-8',newline='\n')
    write(ROOT/'PLOT_LATENCY_VERIFICATION.json',dict(valid=True,changes=changes,caption_only=True,other_points_unchanged=True,T2V_unchanged=True,GCA_times_unchanged=True))
    print(json.dumps(dict(valid=True,latency_ms=v['steady_core_ms'],files=len(changes))))
if __name__=='__main__':main()
