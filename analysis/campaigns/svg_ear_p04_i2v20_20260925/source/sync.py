"""Incremental verified artifacts plus compact status; no video decoding."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/local2/denghaoran/GridMergeAttention/experiments/svg_ear_p04_i2v20_20260925'
OPTS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


known = read(ROOT/'sync_hashes.json') if (ROOT/'sync_hashes.json').exists() else {}
script = '''
import hashlib,io,json,tarfile,time
from pathlib import Path
root=Path(REMOTE)
paths=list(root.glob('*.json'))+list((root/'watchers').glob('*.json'))+list((root/'guard').glob('*.json'))
paths+=list((root/'svg_ear_p04/video20').glob('*.json'))+list((root/'svg_ear_p04/quality_cpu').glob('*.json'))
video=root/'svg_ear_p04/video20'
if (video/'runtime.json').exists() and (video/'purity_records.json').exists():
    proof={r['video']:r for r in json.loads((video/'purity_records.json').read_text())}
    for row in json.loads((video/'runtime.json').read_text())['records']:
        name=row['output']
        if name in proof:
            p=video/name
            assert hashlib.sha256(p.read_bytes()).hexdigest()==proof[name]['video_sha256']
            paths.append(p)
live=dict(time=time.time())
for name in ['generation.log','quality.log','watchers/scai6.log','watchers/scai7.log']:
    p=root/name
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-1400));live[name]=f.read().decode(errors='replace')
monitor=root/'guard/monitor.jsonl'
events=[];samples=0
if monitor.exists():
    with monitor.open() as f:
        for line in f:
            row=json.loads(line);samples+=1
            if row['foreign_pids']:events.append(row)
live.update(monitor_samples=samples,foreign_events=events)
(root/'live_status.json').write_text(json.dumps(live,indent=2)+'\\n');paths.append(root/'live_status.json')
archive=root/'sync_delta.tar.gz';manifest={};changed=0
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(set(paths)):
        if not p.is_file():continue
        name=str(p.relative_to(root));data=p.read_bytes();digest=hashlib.sha256(data).hexdigest();manifest[name]=digest
        if KNOWN.get(name)==digest:continue
        info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data));changed+=1
print(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),manifest=manifest,changed=changed)))
'''
p = subprocess.run(['ssh',*OPTS,'scai7','python -'],input='REMOTE='+repr(REMOTE)+'\nKNOWN='+repr(known)+'\n'+script,
                   text=True,encoding='utf-8',capture_output=True,timeout=180)
assert p.returncode == 0, p.stderr[-2000:]
receipt = json.loads(p.stdout.strip().splitlines()[-1])
archive = ROOT/'sync_delta.tar.gz'
subprocess.run(['scp','-q',*OPTS,'scai7:'+receipt['archive'],str(archive)],check=True,timeout=180)
assert sha(archive) == receipt['sha256']
results = ROOT/'results'
results.mkdir(exist_ok=True)
with tarfile.open(archive,'r:gz') as tar:
    for member in tar.getmembers():
        assert member.isfile() and member.name in receipt['manifest']
        dest = (results/member.name).resolve()
        assert results.resolve() in dest.parents
        data = tar.extractfile(member).read()
        assert hashlib.sha256(data).hexdigest() == receipt['manifest'][member.name]
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
(ROOT/'sync_hashes.json').write_text(json.dumps(receipt['manifest'],indent=2)+'\n',encoding='utf-8')
plan = read(results/'experiment_plan.json')
rtpath = results/'svg_ear_p04/video20/runtime.json'
qualitypath = results/'svg_ear_p04/quality_cpu/quality.json'
rt = read(rtpath) if rtpath.exists() else {}
quality = read(qualitypath) if qualitypath.exists() else {}
proofpath = results/'svg_ear_p04/video20/purity_records.json'
proof = read(proofpath) if proofpath.exists() else []
live = read(results/'live_status.json')
status = dict(generated=len(rt.get('records',[])), evaluated=len(quality.get('videos',[])), expected=20,
              input_version=plan['input_version'], watchers={}, plot_ready=False,
              foreign_events=len(live['foreign_events']), quality=quality.get('summary'), runtime=rt.get('summary'))
for host in ['scai6','scai7']:
    path=results/'watchers'/f'{host}.status.json'
    if path.exists():
        row=read(path);status['watchers'][host]={k:v for k,v in row.items() if k!='gpus'}
for name in ['active_gpu.json','launch.json','failure.json','quality_failure.json','guard/completion.json']:
    path=results/name
    if path.exists():status[name]=read(path)
if proof:
    status['actual_full50_pflops_mean_so_far']=statistics.mean(r['full_50_step_pflops'] for r in proof)
    status['scheduled_upper_full50_pflops_mean_so_far']=statistics.mean(r['scheduled_upper_full_50_step_pflops'] for r in proof)
if status['generated']==status['evaluated']==20 and status.get('guard/completion.json',{}).get('valid'):
    assert not status['foreign_events'] and len(proof)==20
    assert {r['index'] for r in proof}=={r['index'] for r in quality['videos']}==set(range(20))
    for r in proof:
        assert r['attention_calls']==4000 and r['layer_calls']==[100]*40
        assert r['dense_self_attention_calls']==0 and r['gpu']['uuid']==status['active_gpu.json']['uuid']
        trace=read(results/f'svg_ear_p04/video20/flops50_{r["index"]:03d}.json')
        assert len(trace['rows'])==trace['calls']==4000
        assert abs(sum(x['major_matmul_flops'] for x in trace['rows'])/1e15-r['full_50_step_pflops'])<1e-8
        assert abs(sum(x['scheduled_upper_flops'] for x in trace['rows'])/1e15-r['scheduled_upper_full_50_step_pflops'])<1e-8
    q={k:statistics.mean(r[k] for r in quality['videos']) for k in ['psnr','ssim','lpips']}
    latency=statistics.mean(r['mean_attention_ms'] for r in rt['records'][1:])
    assert abs(latency-rt['summary']['steady_state_mean_attention_ms'])<1e-6
    packet=dict(setting='svg_ear_p04',**q,runtime=latency,
                flops=status['scheduled_upper_full50_pflops_mean_so_far']/4,
                actual_full50_pflops=status['actual_full50_pflops_mean_so_far'],
                scheduled_upper_full50_pflops=status['scheduled_upper_full50_pflops_mean_so_far'],
                videos=20,gpu=status['active_gpu.json'],post_hoc_input_selection=True,
                source='SVG_EAR_p04_I2V20_20260925',input_version=plan['input_version'],
                quality_source_sha256=sha(qualitypath),runtime_source_sha256=sha(rtpath),
                flops_scope='Measured full50 densities plus scheduled clustering upper bound, matching existing I2V plotted points.',
                runtime_scope='Single H200 steady mean for indices001-019; no cross-GPU pooling; separate campaign from existing p=.5-.8.')
    (ROOT/'plot_update_packet.json').write_text(json.dumps(packet,indent=2)+'\n',encoding='utf-8')
    status['plot_ready']=True
(ROOT/'status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
print(json.dumps(status,ensure_ascii=False))
