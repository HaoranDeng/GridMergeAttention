"""Bounded real-key Fisher-tangent seed screen; no video or speed claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch
from contextlib import nullcontext

METHODS = ('fisher_geometric', 'fisher_mixture', 'fisher_uniform')
POPULATION_METHODS = ('fisher_mass_pivot', 'fisher_population', 'fisher_mass_population')
VALUE_METHODS = ('value_affine', 'fisher_value', 'fisher_mass_value')
SECANT_METHODS = ('fisher_anchor', 'jeffreys_secant')
TANGENT_METHODS = ('tangent_farthest', 'tangent_coverage', 'fisher_nearest')


def tangent_pairwise(logits, *, fisher=False):
    """[bank, target, seed] reverse KL, or shared-Fisher squared distance."""
    import torch
    z = logits.double()
    if z.ndim != 3 or not torch.isfinite(z).all():
        raise ValueError('finite [bank,candidate,probe] logits required')
    if fisher:
        rho = torch.softmax(z.mean(1), -1)
        x = (z - (z*rho[:, None]).sum(-1, keepdim=True))*rho[:, None].sqrt()
        norm = x.square().sum(-1)
        distance = norm[:, :, None]+norm[:, None, :]-2*torch.bmm(x,x.transpose(1,2))
    else:
        logp = torch.log_softmax(z, -1)
        prob = logp.exp()
        self_term = (prob*logp).sum(-1)
        distance = self_term[:, None, :]-torch.bmm(logp,prob.transpose(1,2))
    # Roundoff only: exact distances are nonnegative and diagonal zero.
    return distance.clamp_min(0)


def tangent_seed_indices(logits, anchor, rule, *, count=4):
    """Greedy actual-tangent coverage; no trial hard partitions or fitting."""
    import torch
    if rule not in TANGENT_METHODS:
        raise ValueError('unsupported tangent rule')
    banks, candidates, _ = logits.shape
    if not 1 <= count <= candidates:
        raise ValueError('seed count outside candidate budget')
    anchor = torch.as_tensor(anchor, device=logits.device, dtype=torch.long)
    if anchor.shape != (banks,) or ((anchor < 0) | (anchor >= candidates)).any():
        raise ValueError('one valid anchor per bank required')
    d = tangent_pairwise(logits, fisher=rule=='fisher_nearest')
    batch = torch.arange(banks,device=logits.device)
    best = d[batch, :, anchor]
    used = torch.arange(candidates,device=logits.device)[None] == anchor[:,None]
    chosen = [anchor]
    covered = (-d).exp() if rule=='tangent_coverage' else None
    for _ in range(1,count):
        if covered is not None:
            # Marginal increase in sum_i max_s exp(-KL(p_s || p_i)).
            score = (covered-(-best).exp()[:, :, None]).clamp_min(0).sum(1)
        else:
            score = best
        pick = score.masked_fill(used,-float('inf')).argmax(-1)
        chosen.append(pick)
        used[batch,pick] = True
        best = torch.minimum(best,d[batch, :, pick])
    return torch.stack(chosen,-1)


def select_tangents(q, k):
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    table, anchors = multiscale_candidate_table(45,32,q.device,geometry_count=45)
    heads=q.shape[2]; total=21*45*80*heads
    outputs={name:torch.empty((total,4),device=q.device,dtype=torch.uint8)
             for name in TANGENT_METHODS}
    for start in range(0,total,2048):
        ids=torch.arange(start,min(start+2048,total),device=q.device)
        head,col=ids%heads,(ids//heads)%80
        row,frame=(ids//(heads*80))%45,ids//(heads*80*45)
        real=table[row].long()
        qq=q[0,real*80+col[:,None],head[:,None]].float()
        kk=k[0,frame[:,None]*3600+row[:,None]*80+
             torch.arange(0,80,5,device=q.device),head[:,None]].float()
        z=torch.bmm(qq,kk.transpose(1,2)).double()*(128**-.5)
        for name in TANGENT_METHODS:
            slots=tangent_seed_indices(z,anchors[row],name)
            outputs[name][start:start+len(ids)]=real.gather(1,slots).to(torch.uint8)
    return {name:value.reshape(1,1,21,45,80,heads,4) for name,value in outputs.items()}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_population(q, k, v=None, *, secant=False, candidate_count=32, geometric_only=False, probe_count=16):
    """Greedy total quadratic-error reduction, not exact partition lookahead.

    Residual rows x_i are local Fisher tangent features. Adding normalized
    direction u_s reduces sum_i pi_i ||x_i||^2 by sum_i pi_i (x_i @ u_s)^2.
    pi is one or an inference-time G256 block-mass estimate. No fitted data.
    """
    import numpy as np
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    if probe_count not in (16,40,80) or (probe_count!=16 and not geometric_only):
        raise ValueError('extended probe counts are only for geometric Fisher')
    tables, anchors = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    if candidate_count==45:
        tables=torch.arange(45,device=q.device)[None].expand(45,-1)
        anchors=torch.arange(45,device=q.device)
    elif candidate_count!=32: raise ValueError('only C32 and full grid C45 are compared')
    heads=q.shape[2]; total=21*45*80*heads
    # Preserve original G256 locations, and compute only the RF0 query norms.
    gk=k[0,torch.arange(256,device=q.device)*(75600//256)].permute(1,0,2).float()
    gq=q[0,:3600].permute(1,0,2).float()
    norm=torch.logsumexp(torch.bmm(gq,gk.transpose(1,2)).double()*(128**-.5),-1)
    methods=(f'fisher_candidates{candidate_count}',) if geometric_only else SECANT_METHODS if secant else VALUE_METHODS if v is not None else POPULATION_METHODS
    outputs={name:torch.empty((total,4),device=q.device,dtype=torch.uint8) for name in methods}
    for start in range(0,total,2048):
        ids=torch.arange(start,min(start+2048,total),device=q.device)
        head,col=ids%heads,(ids//heads)%80
        row,frame=(ids//(heads*80))%45,ids//(heads*80*45)
        real=tables[row].long(); batch=torch.arange(len(ids),device=q.device)
        qq=q[0,real*80+col[:,None],head[:,None]].float()
        kk=k[0,frame[:,None]*3600+row[:,None]*80+torch.arange(0,80,80//probe_count,device=q.device),head[:,None]].float()
        z=torch.bmm(qq,kk.transpose(1,2)).double()*(128**-.5)
        first=anchors[row].long(); rho=torch.softmax(z.mean(1),-1)
        delta=z-z[batch,first,None,:]
        feature=(delta-(delta*rho[:,None]).sum(-1,keepdim=True))*rho[:,None].sqrt()
        if secant:
            logp=torch.log_softmax(z,-1); prob=logp.exp()
            arho=prob[batch,first]
            anchor_feature=(delta-(delta*arho[:,None]).sum(-1,keepdim=True))*arho[:,None].sqrt()
            dl=logp-logp[batch,first,None,:]; dp=prob-prob[batch,first,None,:]
            # Squared feature norm is the exact pairwise symmetrized KL.
            product=dl*dp
            if (product < -1e-12).any(): raise ValueError('nonmonotone log probability')
            secant_feature=dl.sign()*product.clamp_min(0).sqrt()
            torch.testing.assert_close(secant_feature.square().sum(-1),product.sum(-1),rtol=1e-10,atol=1e-10)
        if v is not None:
            vv=v[0,frame[:,None]*3600+row[:,None]*80+torch.arange(0,80,5,device=q.device),head[:,None]].double()
            # Center V: constant value shifts cannot change response diversity.
            vv=vv-vv.mean(1,keepdim=True)
            prob=torch.softmax(z,-1)
            response=torch.bmm(prob-prob[batch,first,None,:],vv)
            tangent=torch.bmm(feature*rho[:,None].sqrt(),vv)
        mass=torch.logsumexp(z,-1)-norm[head[:,None],real*80+col[:,None]]
        mass=torch.exp(mass-mass.max(-1,keepdim=True).values)
        for name in methods:
            residual=(anchor_feature if name=='fisher_anchor' else secant_feature if name=='jeffreys_secant' else response if name=='value_affine' else tangent if v is not None else feature).clone()
            used=torch.arange(candidate_count,device=q.device)[None]==first[:,None]
            chosen=[first]; pi=mass if 'mass' in name else torch.ones_like(mass)
            for _ in range(3):
                norms=residual.square().sum(-1)
                if name=='fisher_mass_pivot' or v is not None or secant or geometric_only: score=norms*pi
                else:
                    gram=torch.bmm(residual,residual.transpose(1,2))
                    score=(gram.square()*pi[:,:,None]).sum(1)/(norms+1e-12)
                pick=score.masked_fill(used,-float('inf')).argmax(-1)
                basis=residual[batch,pick]; basis=basis/(basis.square().sum(-1,keepdim=True)+1e-12).sqrt()
                updated=residual-(residual*basis[:,None]).sum(-1,keepdim=True)*basis[:,None]
                if start==0 and name!='fisher_mass_pivot' and v is None and not secant and not geometric_only:
                    # Independent direct recomputation of all trial reductions.
                    rr=residual[:8].cpu().numpy(); pp=pi[:8].cpu().numpy()
                    uu=rr/np.sqrt((rr*rr).sum(-1,keepdims=True)+1e-12)
                    direct=np.sum(pp[:,:,None]*np.einsum('bip,bsp->bis',rr,uu)**2,axis=1)
                    np.testing.assert_allclose(score[:8].cpu().numpy(),direct,rtol=1e-10,atol=1e-10)
                if ((updated.square().sum(-1)*pi).sum(-1) > (norms*pi).sum(-1)+1e-8).any():
                    raise ValueError('quadratic projection increased residual energy')
                used[batch,pick]=True; chosen.append(pick); residual=updated
            outputs[name][start:start+len(ids)]=real.gather(1,torch.stack(chosen,-1)).to(torch.uint8)
    return {name:value.reshape(1,1,21,45,80,heads,4) for name,value in outputs.items()}


def select_all(q, k):
    import numpy as np
    import torch
    from attention.adaptive_seed_pytorch import multiscale_candidate_table
    tables, anchors = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    heads = q.shape[2]
    total = 21 * 45 * 80 * heads
    outputs = {name: torch.empty((total, 4), device=q.device, dtype=torch.uint8) for name in METHODS}
    for start in range(0, total, 4096):
        ids = torch.arange(start, min(start+4096, total), device=q.device)
        head, col = ids % heads, (ids // heads) % 80
        row, frame = (ids // (heads*80)) % 45, ids // (heads*80*45)
        real_rows = tables[row].long()
        qq = q[0, real_rows*80 + col[:, None], head[:, None]].float()
        kk = k[0, frame[:, None]*3600 + row[:, None]*80 + torch.arange(0,80,5,device=q.device), head[:, None]].float()
        z = torch.bmm(qq, kk.transpose(1,2)) * (128 ** -.5)
        first = anchors[row].long()
        batch = torch.arange(len(ids), device=q.device)
        delta = z - z[batch, first, None, :]
        rho = {'fisher_geometric': torch.softmax(z.mean(1), -1),
               'fisher_mixture': torch.softmax(z, -1).mean(1),
               'fisher_uniform': torch.full_like(z[:, 0], 1/16)}
        for name in METHODS:
            r = rho[name]
            residual = (delta - (delta*r[:, None]).sum(-1, keepdim=True)) * r[:, None].sqrt()
            used = torch.arange(32,device=q.device)[None] == first[:, None]
            chosen = [first]
            # Independently repeat the first 16 banks in FP64 on CPU.
            if start == 0:
                zz = z[:16].double().cpu().numpy()
                rr = r[:16].double().cpu().numpy()
                dd = zz - zz[np.arange(16), first[:16].cpu().numpy(), None]
                cpu_res = (dd - (dd*rr[:, None]).sum(-1, keepdims=True))*np.sqrt(rr[:, None])
                cpu_used = used[:16].cpu().numpy().copy()
            for step in range(3):
                score = residual.square().sum(-1).masked_fill(used, -float('inf'))
                pick = score.argmax(-1)
                if start == 0:
                    cc = (cpu_res**2).sum(-1)
                    cc[cpu_used] = -np.inf
                    pp = pick[:16].cpu().numpy()
                    best = np.max(cc,axis=-1)
                    if np.any(best-cc[np.arange(16),pp] > 1e-4*(1+np.abs(best))):
                        raise ValueError('Fisher projection implementation mismatch')
                    cpu_used[np.arange(16), pp] = True
                    basis = cpu_res[np.arange(16),pp]
                    basis /= np.sqrt((basis*basis).sum(-1,keepdims=True)+1e-12)
                    cpu_res -= (cpu_res*basis[:, None]).sum(-1,keepdims=True)*basis[:, None]
                chosen.append(pick)
                used[batch, pick] = True
                basis = residual[batch, pick]
                basis = basis / (basis.square().sum(-1, keepdim=True)+1e-12).sqrt()
                residual = residual - (residual*basis[:, None]).sum(-1,keepdim=True)*basis[:, None]
            chosen = torch.stack(chosen, -1)
            outputs[name][start:start+len(ids)] = real_rows.gather(1,chosen).to(torch.uint8)
    return {name: value.reshape(1,1,21,45,80,heads,4) for name,value in outputs.items()}


def worker(args):
    import torch
    import attention.token_merging as token
    from analysis.experiment_conditional_jeffreys_native import ATTENTION_OPTIONS, metrics, pool_metrics, seed_check
    from analysis.quality_first_mode_contract import BF16_MODES
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.refined or args.bregman or args.query_medoids:
        from refined_fisher_seed import REFINED_METHODS, BREGMAN_METHODS, MEDOID_METHODS, select_refined
    if args.joint_columns:
        from joint_column_fisher_seed import select as select_joint_columns
    if args.tangent_advantage:
        from tangent_advantage_seed import select as select_advantage
    if args.anchor_full_grid:
        from anchor_fullgrid_seed import select as select_anchor_full_grid
    if args.zero_anchor:
        from zero_anchor_seed import select as select_zero_anchor, zero_queries
    if args.mean_seeds or args.zero_anchor:
        import importlib.util
        from mean_seed_pytorch import MEAN_METHODS, MeanSeedKernel, select_means
        import attention.hard_c4_wan21_14b_720p_triton as merge
        if args.merge_source is None: raise ValueError('explicit mean merge source required')
        spec=importlib.util.spec_from_file_location('attention.mean_seed_merge_screen',args.merge_source)
        mean_module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=mean_module
        spec.loader.exec_module(mean_module)
    old = json.loads((args.baseline_root/'results.json').read_text())
    report = {'complete': False, 'scope': 'L33, three prompts, four heads, all frames; not video',
              'methods': list(('fisher_geometric_control',*MEAN_METHODS) if args.mean_seeds else (args.native_key,) if args.native_cache else TANGENT_METHODS if args.tangent else ('fisher_candidates32','fisher_candidates45') if args.full_candidates else SECANT_METHODS if args.secant else VALUE_METHODS if args.value_gradient else POPULATION_METHODS if args.population else METHODS), 'records': [], 'mean_attention_ms': None, 'speedup_vs_sdpa': None}
    with torch.inference_mode():
        if args.refined: report['methods']=list(REFINED_METHODS)
        if args.bregman: report['methods']=list(BREGMAN_METHODS)
        if args.query_medoids: report['methods']=list(MEDOID_METHODS)
        if args.joint_columns: report['methods']=['fisher_joint4']
        if args.tangent_advantage: report['methods']=['tangent_advantage_gs']
        if args.anchor_full_grid: report['methods']=['fisher_anchor_candidates32','fisher_anchor_candidates45']
        if args.zero_anchor: report['methods']=['zero_fisher32']
        if args.full_probes: report['methods']=['fisher_probes40','fisher_probes80']
        for index, prior in enumerate(old['records']):
            capture = args.capture_root/f'prompt{index:03d}_call033_layer33.pt'
            if digest(capture) != prior['input']['sha256']:
                raise ValueError('capture differs from BF16 reference')
            payload = torch.load(capture, map_location='cpu', weights_only=True)
            q,k,v,reference = [payload[key].to('cuda') for key in ('q','k','v','dense_output')]
            assert q.shape == (1,75600,4,128) and q.dtype == torch.bfloat16
            if args.zero_anchor: mean_queries=zero_queries(q)
            if args.native_cache:
                import importlib.util
                if args.native_selector is None: raise ValueError('explicit native selector source required')
                spec=importlib.util.spec_from_file_location('attention.fisher_native_screen',args.native_selector)
                native_module=importlib.util.module_from_spec(spec)
                sys.modules[spec.name]=native_module
                spec.loader.exec_module(native_module)
                fisher_geometric_seed_rows_triton=getattr(native_module,args.native_function)
                seeds=fisher_geometric_seed_rows_triton(q[:,:3600].reshape(1,1,45,80,4,128),k.reshape(1,21,45,80,4,128),candidate_count=args.native_candidates)
                expected=torch.load(args.native_cache/f'prompt{index:03d}_{args.native_key}_seeds.pt',weights_only=True).to('cuda')
                agreement=float((seeds==expected).all(-1).float().mean())
                set_agreement=float((seeds.sort(-1).values==expected.sort(-1).values).all(-1).float().mean())
                comparison={'native_group_agreement':agreement,'native_set_agreement':set_agreement,
                            'equivalence_established':False}
                print(json.dumps({'prompt':index,**comparison}),flush=True)
                # Report this native implementation separately. Row identity
                # alone cannot establish or refute a tangent-score error;
                # do not relabel nonidentical outputs as prototype parity.
                rows={args.native_key:seeds}
            else:
                if args.zero_anchor:
                    rows=select_zero_anchor(q,k)
                elif args.anchor_full_grid:
                    rows=select_anchor_full_grid(q,k)
                elif args.tangent_advantage:
                    rows=select_advantage(q,k)
                elif args.joint_columns:
                    rows=select_joint_columns(q,k,joint_width=4)
                elif args.full_probes:
                    rows={f'fisher_probes{p}':select_population(q,k,geometric_only=True,probe_count=p)['fisher_candidates32'] for p in (40,80)}
                elif args.refined or args.bregman or args.query_medoids:
                    rows=select_refined(q,k,bregman=args.bregman,medoids=args.query_medoids)
                elif args.mean_seeds:
                    control=select_population(q,k,geometric_only=True)['fisher_candidates32']
                    rows,mean_queries=select_means(q,k)
                    rows={'fisher_geometric_control':control,**rows}
                elif args.tangent:
                    rows=select_tangents(q,k)
                elif args.full_candidates:
                    rows={**select_population(q,k,geometric_only=True),**select_population(q,k,candidate_count=45,geometric_only=True)}
                else:
                    rows = select_population(q,k,secant=True) if args.secant else select_population(q,k,v) if args.value_gradient else select_population(q,k) if args.population else select_all(q,k)
            record = {'prompt':index, 'metrics':{}}
            if args.native_cache: record['prototype_comparison']=comparison
            for name, seeds in rows.items():
                is_mean=(args.mean_seeds and name in MEAN_METHODS) or args.zero_anchor
                if is_mean:
                    assert seeds.shape==(1,1,21,45,80,4,4) and seeds.dtype==torch.uint8
                    assert (seeds<45).all() and (seeds[...,1:].sort(-1).values.diff(dim=-1)>0).all()
                    assert (seeds[...,0]==torch.arange(45,device=seeds.device)[None,None,None,:,None,None]).all()
                else:
                    seed_check(seeds.cpu().numpy(),4)
                torch.save(seeds.cpu(),args.output/f'prompt{index:03d}_{name}_seeds.pt')
                def selector(*a, **kw):
                    assert a[0].shape == (1,1,45,80,4,128) and kw['candidate_count'] == 32
                    return seeds
                ctx=patch.object(merge,'_hard_c4_720p_temporal_shared_merge_kernel',
                    MeanSeedKernel(mean_module._hard_c4_720p_temporal_shared_merge_kernel,mean_queries)) if is_mean else nullcontext()
                with ctx,patch.object(token,'entropy_mass_seed_rows_triton',selector):
                    out = token._hard_c4_attention(q,k,v,seed_mode=BF16_MODES['mass_rf0'],**ATTENTION_OPTIONS).contiguous()
                record['metrics'][name] = metrics(out,reference)
                print(json.dumps({'prompt':index,'method':name,'mse':record['metrics'][name]['mse']}),flush=True)
            report['records'].append(record)
            write(args.output/'results.json',report)
    report['summary'] = {name:pool_metrics([r['metrics'][name] for r in report['records']]) for name in rows}
    report['complete'] = True
    write(args.output/'results.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--population',action='store_true')
    p.add_argument('--value-gradient',action='store_true')
    p.add_argument('--secant',action='store_true')
    p.add_argument('--full-candidates',action='store_true')
    p.add_argument('--tangent',action='store_true')
    p.add_argument('--mean-seeds',action='store_true')
    p.add_argument('--refined',action='store_true')
    p.add_argument('--bregman',action='store_true')
    p.add_argument('--query-medoids',action='store_true')
    p.add_argument('--joint-columns',action='store_true')
    p.add_argument('--tangent-advantage',action='store_true')
    p.add_argument('--anchor-full-grid',action='store_true')
    p.add_argument('--zero-anchor',action='store_true')
    p.add_argument('--full-probes',action='store_true')
    p.add_argument('--merge-source',type=Path)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--native-cache',type=Path)
    p.add_argument('--native-selector',type=Path)
    p.add_argument('--native-function',default='fisher_geometric_seed_rows_triton')
    p.add_argument('--native-key',default='fisher_geometric')
    p.add_argument('--native-candidates',type=int,choices=(32,45),default=32)
    for key in ('source-root','capture-root','baseline-root','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.source_root))
    sys.path.insert(0,str(args.source_root/'analysis'))
    if args.worker:
        if os.environ.get('FISHER_SCREEN_PARENT')!=str(os.getppid()):
            raise ValueError('worker requires its monitoring parent')
        return worker(args)
    from analysis import launch_quality_first_videos as guard
    initial=guard.require_clean(args.gpu)
    if args.output.exists(): raise ValueError('fresh output required')
    args.output.mkdir(parents=True)
    source_hash=digest(Path(__file__))
    extra_sources=([Path(__file__).with_name('mean_seed_pytorch.py'),args.merge_source]
                   if args.mean_seeds or args.zero_anchor else [])
    if args.zero_anchor: extra_sources.append(Path(__file__).with_name('zero_anchor_seed.py'))
    if args.refined or args.bregman or args.query_medoids: extra_sources.append(Path(__file__).with_name('refined_fisher_seed.py'))
    if args.joint_columns: extra_sources.append(Path(__file__).with_name('joint_column_fisher_seed.py'))
    if args.tangent_advantage: extra_sources.append(Path(__file__).with_name('tangent_advantage_seed.py'))
    if args.anchor_full_grid: extra_sources.append(Path(__file__).with_name('anchor_fullgrid_seed.py'))
    if args.native_selector: extra_sources.append(args.native_selector)
    extra_hashes={str(path):digest(path) for path in extra_sources}
    write(args.output/'source.json',{'script':str(Path(__file__).resolve()),'sha256':source_hash,
          'source_root':str(args.source_root),'source_provenance_sha256':digest(args.source_root.parent/'source_provenance.json'),
          'native_selector':str(args.native_selector) if args.native_selector else None,
          'native_function':args.native_function if args.native_selector else None,
          'native_selector_sha256':digest(args.native_selector) if args.native_selector else None,
          'extra_sources':extra_hashes})
    env=guard.child_environment(args.output,initial['gpu']['uuid'])
    env['FISHER_SCREEN_PARENT']=str(os.getpid())
    def interrupted(*unused): raise KeyboardInterrupt('stopped by parent signal')
    signal.signal(signal.SIGTERM,interrupted)
    process=None; failure=None; foreign=[]
    try:
        guard.require_clean(args.gpu,expected_uuid=initial['gpu']['uuid'])
        with (args.output/'worker.log').open('x') as log,(args.output/'gpu_monitor.jsonl').open('x') as monitor:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',*sys.argv[1:]],env=env,stdout=log,stderr=subprocess.STDOUT)
            write(args.output/'process.json',{'parent_pid':os.getpid(),'worker_pid':process.pid})
            while True:
                active=guard.gpu_processes(initial['gpu']['uuid'])
                foreign=[x for x in active if x['pid']!=process.pid]
                monitor.write(json.dumps({'utc':guard.now(),'processes':active,'foreign':foreign})+'\n');monitor.flush()
                if foreign: raise RuntimeError('foreign GPU process')
                if process.poll() is not None: break
                time.sleep(5)
            if process.returncode!=0: raise RuntimeError(f'worker exited {process.returncode}')
            if digest(Path(__file__))!=source_hash: raise RuntimeError('script changed')
            if any(digest(Path(p))!=h for p,h in extra_hashes.items()): raise RuntimeError('extra source changed')
            if not json.loads((args.output/'results.json').read_text())['complete']: raise RuntimeError('incomplete')
    except BaseException as error: failure=f'{type(error).__name__}: {error}'
    finally: guard.stop_own_child(process)
    write(args.output/'completion.json',{'valid':failure is None,'failure':failure,'foreign':foreign,
          'worker_exit_code':process.returncode if process else None,'completed_utc':guard.now()})
    if failure: raise RuntimeError(failure)


if __name__=='__main__': main()
