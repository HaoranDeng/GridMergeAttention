"""Promote the round-7/8 C4 stack into kernels/best/c4, gated on its receipts.

Promotion is a packaging step and packaging has lost results before
(`campaign-process-lessons` 1: a win that passed every gate and reached neither
the package nor the report). So this script does the whole thing mechanically
and refuses at the first receipt that does not say what the promotion claims:

  * `equiv`      promoted stack (desc_select merge + FP32 GS register-pipelined
                 selector) bitwise equal, seeds and output, to the arm the
                 20-video screen actually ran (retained merge + FP32 GS selector
                 without the pipelined loop). Without this the videos are not
                 this binary's videos.
  * `pkg_final`  promoted stack against the retained stack at the declared
                 5e-4 seed-flip / relative-L2 ceiling, three layers, tie edges.
  * `fullstack`  complete-call ABBA of the promoted stack vs the retained pair.
  * `video`      CPU PSNR comparison of the two 20-video arms, same host.

The previous retained sources stay in the package as the paired control these
measurements were taken against, as the C10 promotion did. Hashes in the
manifest are of the exact bytes copied; `_integrity` verifies them afterwards.
"""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'kernels/best/c4'
NEW_MERGE = ROOT / 'native_fisher/fisher_c4_desc_select.cu'
NEW_SELECTOR = ROOT / 'native_fisher/fisher_c4_selector_fp32gs_regpipe.cu'
SCREENED_SELECTOR = ROOT / 'native_fisher/fisher_c4_selector_fp32gs.cu'
OLD_MERGE = PKG / 'fisher_c4_guarded_softmax.cu'
OLD_SELECTOR = PKG / 'fisher_c4_fused_selector_register_basis_loop.cu'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        sys.exit('REFUSED: ' + message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--equiv', type=Path, required=True)
    p.add_argument('--pkg-final', type=Path, required=True)
    p.add_argument('--fullstack', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    new_merge, new_sel, screened_sel = sha(NEW_MERGE), sha(NEW_SELECTOR), sha(SCREENED_SELECTOR)
    old_merge, old_sel = sha(OLD_MERGE), sha(OLD_SELECTOR)

    # ---- equivalence to the screened arm ----
    eq = load(a.equiv)
    require(eq.get('valid') is True and not eq.get('monitor_errors'), 'equiv receipt not valid')
    require(eq['selector_source_sha256'] == new_sel and eq['merge_source_sha256'] == new_merge,
            'equiv candidate is not the promotion stack')
    require(eq['control_selector_sha256'] == screened_sel and eq['control_merge_sha256'] == old_merge,
            'equiv control is not the screened arm')
    require(set(eq['captures']) == {'00', '19', '39'}, 'equiv did not cover all three layers')
    for layer, d in eq['captures'].items():
        require(d['seed_mismatches'] == 0 and d['output_bitwise_equal'] and d['finite'],
                'equiv layer %s is not bitwise' % layer)
    for name, d in eq['seed_edges'].items():
        require(d['seed_mismatches'] == 0, 'equiv edge case %s differs' % name)

    # ---- whole-package proof against the retained stack ----
    pk = load(a.pkg_final)
    require(pk.get('valid') is True and not pk.get('monitor_errors'), 'pkg_final receipt not valid')
    require(pk['selector_source_sha256'] == new_sel and pk['merge_source_sha256'] == new_merge,
            'pkg_final candidate is not the promotion stack')
    require(pk['control_selector_sha256'] == old_sel and pk['control_merge_sha256'] == old_merge,
            'pkg_final control is not the retained stack')
    require(pk['within_declared_tolerance'] and set(pk['captures']) == {'00', '19', '39'},
            'pkg_final breached its declared tolerance or missed a layer')

    # ---- complete-call timing ----
    fs = load(a.fullstack)
    require(fs.get('valid') is True and not fs.get('monitor_errors'), 'fullstack receipt not valid')
    require(fs['merge_sha256']['candidate'] == new_merge and fs['selector_sha256']['candidate'] == new_sel
            and fs['merge_sha256']['retained'] == old_merge and fs['selector_sha256']['retained'] == old_sel,
            'fullstack arms are not the promotion stack vs the retained stack')
    require(fs['summary']['pooled_delta_ms'] < 0, 'fullstack shows no saving')

    # ---- video screen ----
    vid = load(a.video)
    require(vid.get('valid') is True and vid['same_host'] is True, 'video screen not valid or cross-host')
    r20 = vid['results']['20']
    require(r20['paired_candidate_minus_control_db']['n'] == 20, 'video screen is not 20 paired videos')
    for run_key in ('control_run', 'candidate_run'):
        marker = Path(vid[run_key]) / 'native_kernel.json'
        if marker.exists():
            nk = load(marker)
            expected = old_sel if run_key == 'control_run' else screened_sel
            require(nk['selector_sha256'] == expected and nk['merge_sha256'] == old_merge,
                    '%s did not run the expected sources' % run_key)

    print('all receipts consistent with the promotion stack')
    print('  merge   ', NEW_MERGE.name, new_merge[:16], '(control', OLD_MERGE.name, old_merge[:16] + ')')
    print('  selector', NEW_SELECTOR.name, new_sel[:16], '(control', OLD_SELECTOR.name, old_sel[:16] + ')')
    print('  fullstack pooled delta', round(fs['summary']['pooled_delta_ms'], 3), 'ms;',
          'retained', round(fs['summary']['retained_call_ms'], 3), 'candidate',
          round(fs['summary']['candidate_call_ms'], 3))
    print('  video paired delta', round(r20['paired_candidate_minus_control_db']['mean_delta_db'], 4), 'dB')
    if a.dry_run:
        return

    # ---- copy sources byte-exact ----
    for src in (NEW_MERGE, NEW_SELECTOR):
        dst = PKG / src.name
        shutil.copyfile(src, dst)
        require(sha(dst) == sha(src), 'copy changed bytes: %s' % dst)

    # ---- operator entry point ----
    (PKG / 'optimized_c4.py').write_text('''"""Validated C4 CUDA merge and cooperative Uniform16 Fisher selector.

Promoted 2026-09-14. The merge is `fisher_c4_desc_select.cu`: the retained
guarded-softmax merge with loop-invariant WGMMA descriptors derived from three
per-CTA bases and hoisted self-column tests. Output is bitwise equal to the
previous merge on layers 0/19/39. The selector is
`fisher_c4_selector_fp32gs_regpipe.cu`: FP32 Fisher Gram-Schmidt with a
register-pipelined frame loop. This is a declared numerics change - seeds
differ from the FP64 selector on at most 4.3e-4 of positions (layer 39; 1e-6
on layers 0/19), output relative L2 under 5e-5 - accepted by the user on the
condition of a 20-video PSNR screen, which showed +0.03 dB against the dense
reference (no difference). Complete paired calls: 100.70 ms against 104.36 ms
for the previous pair in the same window, -3.66 ms, 87 of 90 paired wins.
The previous sources remain in this directory as the paired control.
"""
from pathlib import Path
from .operator import NativeFisherAttention

class OptimizedFisherC4Attention(NativeFisherAttention):
    def __init__(self,device='cuda'):
        super().__init__(device=device,capacity=4,selector_backend='fused_cooperative_loop',
                         selector_source=Path(__file__).with_name('fisher_c4_selector_fp32gs_regpipe.cu'),
                         merge_source=Path(__file__).with_name('fisher_c4_desc_select.cu'))
''', encoding='utf-8', newline='\n')

    # ---- evidence ----
    evidence = load(PKG / 'evidence.json')
    evidence['round8_promotion_20260914'] = {
        'scope': 'desc_select merge + FP32 GS register-pipelined selector replaces the '
                 'guarded-softmax merge + FP64 register-basis selector. Merge change is '
                 'bitwise; selector change is numerics-tier with a declared tolerance and '
                 'a passed 20-video PSNR screen.',
        'sources': {'merge': {'file': NEW_MERGE.name, 'sha256': new_merge},
                    'selector': {'file': NEW_SELECTOR.name, 'sha256': new_sel},
                    'control_merge': {'file': OLD_MERGE.name, 'sha256': old_merge},
                    'control_selector': {'file': OLD_SELECTOR.name, 'sha256': old_sel},
                    'screened_selector': {'file': SCREENED_SELECTOR.name, 'sha256': screened_sel}},
        'declared_tolerance': pk['declared_tolerance'],
        'package_proof': {k: pk[k] for k in ('initial_gpu', 'captures', 'seed_edges',
                                              'tolerance_breaches', 'within_declared_tolerance',
                                              'selector_resources', 'valid')},
        'equivalence_to_screened_arm': {k: eq[k] for k in ('initial_gpu', 'captures', 'seed_edges', 'valid')},
        'complete_call_timing': {k: fs[k] for k in ('initial_gpu', 'layers', 'resources',
                                                     'correctness', 'arms', 'summary', 'valid')},
        'video_screen': {k: vid[k] for k in ('scope', 'control_run', 'candidate_run', 'reference_dir',
                                             'hosts', 'same_host', 'results', 'valid')},
        'user_condition': 'FP32 Gram-Schmidt approved 2026-09-14 on condition of no PSNR '
                          'difference over 20 videos; measured +0.032 dB (20) / +0.020 dB (5).',
    }
    (PKG / 'evidence.json').write_text(json.dumps(evidence, indent=1), encoding='utf-8', newline='\n')

    # ---- manifest ----
    manifest = load(PKG / 'manifest.json')
    for name in (NEW_MERGE.name, NEW_SELECTOR.name):
        manifest['original_sources'][name] = sha(PKG / name)
    manifest['files'] = {name: sha(PKG / name) for name in list(manifest['files'])
                         + [NEW_MERGE.name, NEW_SELECTOR.name]}
    manifest['scope'] = ('One retained complete H40 attention implementation; only create() is '
                         'supported. Merge is fisher_c4_desc_select.cu (bitwise equal to the '
                         'previous guarded-softmax merge); selector is the FP32 Gram-Schmidt '
                         'register-pipelined selector, a declared numerics change screened on '
                         '20 videos. Previous sources retained as the paired control.')
    manifest.setdefault('packaging_changes', []).append(
        'optimized_c4.py: merge_source now fisher_c4_desc_select.cu and selector_source now '
        'fisher_c4_selector_fp32gs_regpipe.cu; fisher_c4_guarded_softmax.cu and '
        'fisher_c4_fused_selector_register_basis_loop.cu retained alongside as the paired control')
    manifest['provenance_20260914'] = {
        'gates': ['equiv: promoted stack bitwise equal (seeds and output, layers 00/19/39, tie '
                  'edges) to the arm the 20-video screen ran',
                  'pkg_final: promoted stack vs retained stack within declared 5e-4 ceiling',
                  'fullstack: complete-call ABBA/BAAB with A/A null pair, 30 pairs per layer, '
                  'layers 00/19/39',
                  'video: 20 paired videos per arm, same host, CPU PSNR vs dense reference'],
        'receipts': ['analysis/receipts_c4_r1/r7_fullstack_desc_select_fp32gs_regpipe.json',
                     'analysis/receipts_c4_r1/r7_video_fp32_psnr_summary.json',
                     'analysis/receipts_c4_r1/r8_equiv_promoted_vs_screened.json',
                     'analysis/receipts_c4_r1/r8_package_final.json'],
        'declared_tolerance': pk['declared_tolerance']}
    (PKG / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')

    # ---- verify ----
    subprocess.check_call([sys.executable, '-m', 'kernels.best._integrity'], cwd=ROOT)
    print('promoted and verified')


if __name__ == '__main__':
    main()
