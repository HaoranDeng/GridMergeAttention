"""Independent post-hoc sample019 sensitivity; never rewrites the main plots."""
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator, NullLocator

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parent / 'Q80K90_I2V_C4_C6_C8_C10_20260924'
COMBINED = ROOT.parent / 'T2V_I2V_tradeoff_combined'
OUT = ROOT / 'tradeoff_replacement019'
OUT.mkdir(exist_ok=True)


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    paths = {
        'replacement_summary': ROOT/'summary.json',
        'quality': ROOT/'results/quality.json',
        'old_quality': ROOT/'original_quality_reference.json',
        'plan': ROOT/'results/experiment_plan.json',
        'new_ear_trace': ROOT/'results/EAR_p08/video20/flops50_019.json',
        'baseline_reference': MAIN/'tradeoff/reference/i2v_reference.json',
        'original_plot': COMBINED/'t2v_i2v_tradeoff_full20_data.json',
    }
    hashes = {k: sha(p) for k, p in paths.items()}
    summary, quality, old, plan, trace, ref, current = [read(p) for p in paths.values()]
    assert quality['complete'] and quality['frames'] == 'all81' and quality['device'] == 'cpu'
    assert summary['restoration_verified']
    for i in range(19):
        for k in ('image_sha256', 'prompt_sha256'):
            assert old['input_manifest'][i][k] == plan['input_manifest'][i][k]
    for method in ('Dense', 'C4', 'C8', 'EAR_p08'):
        video = ROOT/f'results/{method}/video20/019-00.mp4'
        proof, = read(ROOT/f'results/{method}/video20/purity_records.json')
        assert sha(video) == proof['video_sha256']
        assert proof['attention_calls'] == 4000 and proof['layer_calls'] == [100]*40
        if method != 'Dense':
            q = quality['results'][method]
            assert q['frames'] == 81 and q['index'] == q['seed'] == 19
            assert q['method_sha256'] == proof['video_sha256']
            assert q['dense_sha256'] == sha(ROOT/'results/Dense/video20/019-00.mp4')
            assert q['source_sha256'] == '469286f936b14cce79ac142195a3c2b8296b0b1f8f8e7c2a4594b51feaf356a9'
            assert q['image_sha256'] == plan['replacement']['new']['image_sha256']
            assert all(int(r['video'][:3]) == i for i, r in enumerate(old['methods'][method]))

    ear = next(p for p in ref['series']['SVG-EAR'] if p['tag'] == 'svg_ear_p08')
    rows = trace['rows']
    assert len(rows) == trace['calls'] == 4000
    assert sum(r['initialization'] for r in rows) == 40
    dense_call = 4*75600**2*40*128
    compensation_call = 4*75600*1000*40*128
    upper_cluster = sum(2*75600*40*128*(300+1000)*(50 if r['initialization'] else 2) for r in rows)
    assert math.isclose(upper_cluster, ear['cluster_upper_bound_per_video'], rel_tol=1e-12)
    assert math.isclose(compensation_call*4000, ear['ear_compensation_per_video'], rel_tol=1e-12)
    new_ear_upper = sum(dense_call*r['density']+compensation_call for r in rows) + upper_cluster
    actual_cost = sum(r['major_matmul_flops'] for r in rows)
    assert math.isclose(actual_cost, trace['full_50_step_flops'], rel_tol=1e-12)
    assert new_ear_upper >= actual_cost - 1e6
    revised_ear_cost = (sum(ear['flops_per_video'])-ear['flops_per_video'][19]+new_ear_upper)/20

    gca = {p['setting']: p for p in current['I2V']['series']['GCA (ours)']}
    old_ear_plot = next(p for p in current['I2V']['series']['SVG-EAR'] if p['setting'] == 'svg_ear_p08')
    data = {'post_hoc': True, 'sample_count': 20, 'replaced_index': 19, 'seed': 19,
            'old_prompt': old['input_manifest'][19].get('prompt', 'a shadow of a hand reaching for a leaf'),
            'new_prompt': summary['new_sample']['prompt'], 'source_hashes': hashes,
            'series': {}, 'missing_replacement_methods': ['C6', 'C10', 'other SVG-EAR p values', 'SVG2', 'SVG', 'original Q45K80C8'],
            'latency_scope': 'Retain current figure coordinates for both versions; cold single-video replacement timings are stored separately, never pooled.',
            'flops_scope': 'TFLOPs/call averaged over actual complete50-step density traces. SVG-EAR old and replacement20 both use the same scheduled clustering upper bound.',
            'ear_cost_audit': {'old019_upper_flops': ear['flops_per_video'][19],
                               'new019_actual_flops': actual_cost, 'new019_upper_flops': new_ear_upper,
                               'old20_mean_full50_pflops': statistics.mean(ear['flops_per_video'])/1e15,
                               'replacement20_mean_full50_pflops': revised_ear_cost/1e15},
            'new_single_video_results': summary['methods']}
    for version, quality_key in [('original', 'original20'), ('replacement019', 'replacement20')]:
        data['series'][version] = {}
        for method in ('C4', 'C8', 'EAR_p08'):
            q = summary['methods'][method][quality_key]
            source = gca[method] if method != 'EAR_p08' else old_ear_plot
            cost = source['flops']
            if method == 'EAR_p08' and version == 'replacement019':
                cost = revised_ear_cost/4000/1e12
            data['series'][version][method] = {
                **q, 'runtime': source['runtime'], 'flops': cost,
                'quality_count': 20, 'runtime_scope': data['latency_scope']}

    (OUT/'data.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.1))
    fig.subplots_adjust(left=.055, right=.985, top=.78, bottom=.29, wspace=.37)
    panels = [('psnr','flops'), ('psnr','runtime'), ('ssim','flops'), ('ssim','runtime')]
    for ax, (metric, cost) in zip(axes, panels):
        if cost == 'runtime': ax.set_xscale('log')
        ax.set_axisbelow(True)
        ax.grid(color='#e5e5e5', linewidth=.65)
        for side in ('top','right'): ax.spines[side].set_visible(False)
        for side in ('left','bottom'): ax.spines[side].set_color('#c8c8c8')
        xs, ys = [], []
        for version in ('original', 'replacement019'):
            points = data['series'][version]
            alpha = .45 if version == 'original' else 1
            for methods, color, marker in [(('C4','C8'),'#d62728','*'), (('EAR_p08',),'#2a78d6','o')]:
                group = [points[m] for m in methods]
                x, y = [p[cost] for p in group], [p[metric] for p in group]
                xs.extend(x); ys.extend(y)
                ax.plot(x, y, color=color, marker=marker, alpha=alpha,
                        linestyle='--' if version=='original' else '-', linewidth=1.6,
                        markersize=9 if marker=='*' else 6,
                        markerfacecolor='white' if version=='original' else color,
                        markeredgewidth=1, zorder=5 if marker=='*' else 4)
        ratios = (6,5,4,3) if cost=='flops' else (6,5,4,3,2)
        dense = current['I2V']['dense'][cost]
        positions = [dense/r for r in ratios]
        for pos in positions: ax.axvline(pos, color='#aaa9a5', linewidth=.7, linestyle=(0,(3,3)), zorder=1)
        top = ax.secondary_xaxis('top'); top.set_xticks(positions, labels=[f'{r}×' for r in ratios])
        top.tick_params(labelsize=8, length=0, pad=3)
        top.set_xlabel('Arithmetic reduction vs Dense' if cost=='flops' else 'Speedup vs Dense', fontsize=8, labelpad=5)
        top.spines['top'].set_color('#c8c8c8'); top.xaxis.set_minor_locator(NullLocator())
        low, high = min(xs+positions), max(xs+positions)
        if cost == 'runtime':
            ax.set_xlim(low/1.07, high*1.06)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,1.5,2,3,4,6,8)))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.xaxis.set_minor_locator(NullLocator())
        else:
            pad=(high-low)*.07; ax.set_xlim(low-pad, high+pad)
            ax.xaxis.set_major_locator(MaxNLocator(4))
        low, high = min(ys), max(ys); pad=(high-low)*.17
        ax.set_ylim(low-pad, high+pad); ax.yaxis.set_major_locator(MaxNLocator(4))
        if metric=='ssim': ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(labelsize=9, length=0, pad=3, colors='#454545')
        ax.set_ylabel('PSNR (dB)' if metric=='psnr' else 'SSIM', fontsize=10)
        ax.set_xlabel('Attention arithmetic\n(TFLOPs / call)' if cost=='flops' else 'Attention latency\n(ms per attention layer)', fontsize=9)
    handles=[]
    for family, color, marker in [('GCA','#d62728','*'), ('SVG-EAR p=0.8','#2a78d6','o')]:
        for version in ('original','replacement019'):
            handles.append(Line2D([0],[0],color=color,marker=marker,markersize=8 if marker=='*' else 5,
                alpha=.45 if version=='original' else 1, linestyle='--' if version=='original' else '-',
                markerfacecolor='white' if version=='original' else color,
                label=family+(' · original' if version=='original' else ' · new 019')))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,frameon=False,fontsize=9)
    for ext in ('pdf','png','svg'): fig.savefig(OUT/f'i2v_replacement019_comparison.{ext}', dpi=220, facecolor='white')
    plt.close(fig)
    assert all(sha(paths[k]) == v for k,v in hashes.items())
    (OUT/'README.md').write_text(
        '# Independent post-hoc replacement sensitivity\n\n'
        'Only original019 is replaced;000–018 and seed19 remain unchanged. Both the conditioning image and prompt change. '
        'Original019 was selected after inspecting quality, so this is a post-hoc sensitivity result, not a newly prespecified benchmark.\n\n'
        'The four panels compare original20 and replacement20 PSNR/SSIM versus FLOPs/call and current plotting latency. '
        'GCA contains C4 and C8 only; only SVG-EAR p=.8 has been rerun. Missing settings are not joined into the replacement curve. '
        'Runtime positions retain the current plotting coordinates, including user C8=104.1; new cold-start timing is not pooled. '
        'SVG-EAR replacement cost uses its actual4000-call densities with the same scheduled clustering upper bound as the original20. '
        'The exact newly measured single-video FLOPs are separately retained in data.json.\n\n'
        'Quality is the frozen all81-frame CPU PSNR/SSIM/LPIPS evaluation against each input’s paired Dense video. '
        'The main benchmark, its plots, the paper and workbook are unchanged. Original Q45K80C8 remains pending.\n', encoding='utf-8')
    print(json.dumps({'output':str(OUT/'i2v_replacement019_comparison.pdf'), 'series':data['series'], 'ear_cost_audit':data['ear_cost_audit']}))


if __name__ == '__main__':
    main()
