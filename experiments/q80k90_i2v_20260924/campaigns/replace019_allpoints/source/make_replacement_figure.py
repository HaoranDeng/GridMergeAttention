"""Render only complete replacement20 populations, retaining the original figures."""
import copy
import hashlib
import importlib.util
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT.parent
MAIN = SCAN / 'Q80K90_I2V_C4_C6_C8_C10_20260924'
OUT = SCAN / 'T2V_I2V_tradeoff_combined'
STEM = 't2v_i2v_tradeoff_replacement019'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def missing_gca_template(setting):
    """A newly completed capacity may not yet appear in the original plot."""
    assert setting == 'C10', 'Existing capacities must retain their original timing selection.'
    qpath = MAIN / f'results/{setting}/quality_cpu/quality.json'
    rpath = MAIN / f'results/{setting}/video20/runtime.json'
    quality = {int(v['video'][:3]): v for v in read(qpath)['videos']}
    runtime = {v['prompt_index']: v for v in read(rpath)['records']}
    assert set(range(20)) <= quality.keys() and set(range(20)) <= runtime.keys()
    assert all(quality[i]['frames'] == 81 and runtime[i]['attention_calls'] == 4000 for i in range(20))
    claim = read(MAIN / f'results/claims/{setting}.json')
    assert isinstance(claim['gpu'], int)
    return dict(setting=setting, n=20, partial=False, indices=list(range(20)),
                runtime_indices=list(range(1, 20)),
                runtime=statistics.mean(runtime[i]['mean_attention_ms'] for i in range(1, 20)),
                gpu=f"{claim['host']} GPU{claim['gpu']}",
                quality_source_sha256=sha(qpath), runtime_source_sha256=sha(rpath))


def main():
    paths = {'original_combined': OUT / 't2v_i2v_tradeoff_full20_data.json',
             'replacement_summary': ROOT / 'summary.json',
             'replacement_plan': ROOT / 'results/experiment_plan.json',
             'replacement_quality': ROOT / 'results/quality.json',
             'runtime_overrides': OUT / 'runtime_overrides.json',
             'renderer': OUT / 'make_figure.py'}
    protected = [OUT / f't2v_i2v_tradeoff_{name}.{ext}'
                 for name in ('full20', 'i2v_exclude019') for ext in ('pdf', 'png', 'svg')]
    original_hashes = {str(p): sha(p) for p in protected}
    source_hashes = {k: sha(p) for k, p in paths.items()}
    base = read(paths['original_combined'])
    data = copy.deepcopy(base)
    summary = read(paths['replacement_summary'])
    plan = read(paths['replacement_plan'])
    overrides = read(paths['runtime_overrides'])
    assert summary['post_hoc'] and summary['expected_settings'] == len(plan['methods']) == 18
    assert overrides['task'] == 'I2V' and overrides['unit'] == 'ms per attention layer'
    family_names = {'gca': 'GCA (ours)', 'ear': 'SVG-EAR', 'svg2': 'SVG2', 'svg': 'SVG'}
    templates = {p['setting']: p for rows in base['I2V']['series'].values() for p in rows}
    plotted, omitted = [], {}
    data['I2V']['series'] = {family: [] for family in base['I2V']['series']}
    for setting in plan['methods']:
        result = summary['methods'][setting]
        if not result['replacement20_ready']:
            omitted[setting] = ('Original20 incomplete' if result['new019_status']['evaluated']
                                else 'New019 generation or evaluation pending')
            continue
        family = family_names[plan['settings'][setting]['family']]
        point = copy.deepcopy(templates[setting] if setting in templates else missing_gca_template(setting))
        point['original20_reference'] = copy.deepcopy(point)
        point.update(result['replacement20'])
        point.update(n=20, partial=False, full20_quality=result['replacement20'],
                     flops=result['replacement20_tflops_per_call'],
                     source_full_50_step_pflops=result['replacement20_full50_pflops'],
                     cost_kind=result['flops_scope'], post_hoc=True,
                     replacement_index=19, replacement_seed=19,
                     original_unchanged_indices=list(range(19)),
                     replacement_image_sha256=plan['input_manifest'][19]['image_sha256'],
                     replacement_summary_sha256=source_hashes['replacement_summary'])
        if 'quality_source_sha256' in point:
            point['original_quality_source_sha256'] = point.pop('quality_source_sha256')
        if setting in overrides['GCA (ours)']:
            point.setdefault('measured_runtime_ms_before_user_override', point['runtime'])
            point['runtime'] = overrides['GCA (ours)'][setting]
            point['runtime_override_source'] = overrides['source']
        assert point['runtime'] == (overrides['GCA (ours)'].get(setting, templates[setting]['runtime'])
                                    if setting in templates else overrides['GCA (ours)'][setting])
        assert math.isclose(point['flops'] * 4, point['source_full_50_step_pflops'], rel_tol=1e-12)
        assert all(math.isclose(point[k], result['replacement20'][k], abs_tol=1e-12)
                   for k in ('psnr', 'ssim', 'lpips'))
        data['I2V']['series'][family].append(point)
        plotted.append(setting)
    assert len(plotted) == summary['ready_settings'] and len(plotted) > 0
    data['I2V'].update(indices=list(range(20)), baseline_n=20, provisional=False,
        available_gca={p['setting'][1:]: 20 for p in data['I2V']['series']['GCA (ours)']},
        population_note='Every displayed point uses original000-018 plus measured replacement019. Pending settings are omitted, never filled with original-population points.',
        population='Post-hoc replacement019; same conditioning image/prompt and seed19 across settings',
        pending_settings=omitted,
        runtime_note='Original per-GPU steady timings and user plotting overrides retained. New single-video cold-start times are not pooled.',
        last_refresh={'utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Complete replacement20 points only'})
    data['I2V']['runtime_plot_overrides']['applied'] = [p['setting'] for p in data['I2V']['series']['GCA (ours)']
                                                        if p['setting'] in overrides['GCA (ours)']]
    data['I2V']['runtime_plot_overrides']['pending_settings'] = sorted(set(overrides['GCA (ours)']) - set(plotted))
    data['replacement019'] = dict(post_hoc=True, plotted=plotted, pending=omitted,
        generated=summary['generated'], evaluated=summary['evaluated'], expected=18,
        complete=len(plotted) == 18, source_hashes=source_hashes)
    # Preserve established limits for visual comparison; expand only if needed.
    for cost, key in [('runtime', 'shared_runtime_limits_ms'), ('flops', 'shared_arithmetic_limits_tflops_per_call')]:
        values = [p[cost] for rows in data['I2V']['series'].values() for p in rows if p.get(cost) is not None]
        lo, hi = data[key]
        data[key] = [min(lo, min(values) * .96), max(hi, max(values) * 1.04)]
    assert data['T2V'] == base['T2V']
    spec = importlib.util.spec_from_file_location('combined_renderer', paths['renderer'])
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                                  'ps.fonttype': 42, 'svg.fonttype': 'none'})
    (OUT / (STEM + '_data.json')).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    renderer.draw(data, STEM, False)
    assert all(sha(Path(p)) == digest for p, digest in original_hashes.items())
    assert all(sha(paths[k]) == digest for k, digest in source_hashes.items())
    verification = dict(created_at=datetime.now(timezone.utc).isoformat(),
        plotted=plotted, pending=omitted, source_hashes=source_hashes,
        all_displayed_populations_complete20=True, all_displayed_populations_replaced019=True,
        T2V_unchanged=True, original_figures_unchanged=original_hashes,
        runtime_steady_provenance_preserved=True, paper_modified=False,
        pdf_sha256=sha(OUT / (STEM + '.pdf')))
    (OUT / (STEM + '_verification.json')).write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    (OUT / 'current_replacement_figure.json').write_text(json.dumps(dict(
        path=str(OUT / (STEM + '.pdf')), data=str(OUT / (STEM + '_data.json')),
        plotted=plotted, pending=omitted, post_hoc=True), indent=2) + '\n', encoding='utf-8')
    (OUT / 'REPLACEMENT019_README.md').write_text(
        '# Current replacement019 figure\n\n'
        'The current replacement figure is t2v_i2v_tradeoff_replacement019.pdf (PNG/SVG alongside). '
        'T2V is unchanged. Every visible I2V point is a complete20 aggregate using original000-018 '
        'and the same measured new019; unfinished settings are omitted. This is a post-hoc sensitivity '
        'analysis, not a prespecified resampling protocol. Original full20 and exclude019 figures remain unchanged.\n\n'
        'Displayed settings: ' + ', '.join(plotted) + '.\n\n'
        'Pending settings: ' + ', '.join(omitted) + '.\n\n'
        'Quality and arithmetic use the revised population. Arithmetic is TFLOPs/call over the full50-step '
        '4000-call schedule, retaining matched scheduled clustering upper bounds. Runtime coordinates retain '
        'original per-GPU steady measurements and the user overrides C6=83.6, C8=104.1, C10=133.8ms. '
        'Cold-start replacement timings remain separate in the experiment summary. '
        'The figure keeps the existing log runtime axis, labels, colors and Dense guides. '
        'No LPIPS panels, global title, sample-count text or footnotes are added.\n\n'
        'Refresh after sync and summarize with ../I2V_replace019_allpoints_20260924/source/make_replacement_figure.py. '
        'Paper and workbook are unchanged. Original Q45K80C8 matched I2V data and replacement VBench remain pending.\n',
        encoding='utf-8')
    print(json.dumps({'pdf': str(OUT / (STEM + '.pdf')), 'plotted': plotted, 'pending': omitted}))


if __name__ == '__main__':
    main()
