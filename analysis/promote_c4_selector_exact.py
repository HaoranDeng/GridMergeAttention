"""Promote an exact (bitwise) C4 selector replacement into kernels/best/c4.

For a selector whose seeds and complete output are bitwise equal to the
currently packaged selector, the evidence needed is narrower than for the
round-8 numerics change: an equivalence proof at tolerance 0 on the three
captured layers plus the tie edges, a stage measurement, and a complete-call
measurement. This script refuses unless every receipt names the exact bytes
being promoted and says what the promotion claims, copies the source, points
`optimized_c4.py` at it, and re-hashes the manifest. The previous selector stays
in the directory as the paired control. `_integrity` runs at the end.
"""
import argparse, hashlib, json, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'kernels/best/c4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        sys.exit('REFUSED: ' + message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selector', type=Path, required=True, help='new selector .cu (repo-relative)')
    p.add_argument('--equiv', type=Path, required=True, help='validate_c4_selector_package_tol receipt, tolerance 0')
    p.add_argument('--stage', type=Path, required=True, help='screen_c4_selector_variants receipt')
    p.add_argument('--stage-arm', required=True, help='arm name of the new selector in the stage receipt')
    p.add_argument('--fullcall', type=Path, required=True, help='screen_c4_operator_arms receipt')
    p.add_argument('--fullcall-arm', required=True)
    p.add_argument('--label', required=True, help='evidence key, e.g. round9_selector_v8_20260915')
    p.add_argument('--describe', required=True, help='one paragraph for optimized_c4.py and the manifest')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    new_path = ROOT / a.selector
    new = sha(new_path)
    optimized = (PKG / 'optimized_c4.py').read_text(encoding='utf-8')
    current_name = re.search(r"selector_source=Path\(__file__\)\.with_name\('([^']+)'\)", optimized).group(1)
    merge_name = re.search(r"merge_source=Path\(__file__\)\.with_name\('([^']+)'\)", optimized).group(1)
    current = sha(PKG / current_name)
    merge = sha(PKG / merge_name)

    eq = load(a.equiv)
    require(eq.get('valid') is True and not eq.get('monitor_errors'), 'equiv receipt not valid')
    require(eq['selector_source_sha256'] == new, 'equiv candidate is not the selector being promoted')
    require(eq['control_selector_sha256'] == current, 'equiv control is not the packaged selector')
    require(eq['merge_source_sha256'] == merge and eq['control_merge_sha256'] == merge,
            'equiv did not hold the packaged merge fixed on both arms')
    require(eq['declared_tolerance'] == {'max_seed_mismatch_fraction': 0.0, 'max_relative_l2': 0.0},
            'equiv was not run at tolerance 0')
    require(set(eq['captures']) == {'00', '19', '39'}, 'equiv missed a layer')
    for layer, d in eq['captures'].items():
        require(d['seed_mismatches'] == 0 and d['output_bitwise_equal'] and d['finite'],
                'equiv layer %s is not bitwise' % layer)
    require(len(eq['seed_edges']) >= 2 and all(d['seed_mismatches'] == 0 for d in eq['seed_edges'].values()),
            'equiv tie edges differ or are missing')

    st = load(a.stage)
    require(st.get('valid') is True and not st.get('monitor_errors'), 'stage receipt not valid')
    require(st['selector_sha256'][a.stage_arm] == new, 'stage arm is not the selector being promoted')
    require(st['merge_sha256'] == merge, 'stage receipt used another merge')
    require(st['selector_sha256']['base'] == current,
            'stage receipt was not measured against the packaged selector')
    s = st['summary'][a.stage_arm]
    require(s['seeds_bitwise_all_layers'] and s['output_bitwise_all_layers'], 'stage arm not bitwise')
    require(s['pooled_selector_delta_ms_per_call'] < 0, 'stage shows no saving')

    fc = load(a.fullcall)
    require(fc.get('valid') is True and not fc.get('monitor_errors'), 'fullcall receipt not valid')
    arm = fc['arms'][a.fullcall_arm]
    require(arm['sha256']['selector'] == new and arm['sha256']['merge'] == merge,
            'fullcall arm is not the promotion pair')
    require(fc['base_sha256']['selector'] == current and fc['base_sha256']['merge'] == merge,
            'fullcall base is not the packaged pair')
    require(arm['summary']['bitwise_all_layers'] and arm['summary']['pooled_delta_ms'] < 0,
            'fullcall arm not bitwise or shows no saving')

    print('all receipts consistent')
    print('  packaged selector', current_name, current[:16])
    print('  new selector     ', new_path.name, new[:16])
    print('  merge held fixed ', merge_name, merge[:16])
    print('  stage delta %.3f ms (base %.3f); complete call %.3f ms (%.3f -> %.3f), wins %d/%d'
          % (s['pooled_selector_delta_ms_per_call'], st['summary']['base_selector_ms_per_call'],
             arm['summary']['pooled_delta_ms'], arm['summary']['base_call_ms'],
             arm['summary']['candidate_call_ms'], arm['summary']['wins'], arm['summary']['n']))
    if a.dry_run:
        return

    dst = PKG / new_path.name
    shutil.copyfile(new_path, dst)
    require(sha(dst) == new, 'copy changed bytes')

    optimized = optimized.replace("with_name('%s')" % current_name, "with_name('%s')" % new_path.name)
    doc_end = optimized.index('"""', 3)
    optimized = optimized[:doc_end].rstrip() + '\n\n' + a.describe.strip() + '\n' + optimized[doc_end:]
    (PKG / 'optimized_c4.py').write_text(optimized, encoding='utf-8', newline='\n')

    evidence = load(PKG / 'evidence.json')
    evidence[a.label] = {
        'scope': a.describe.strip(),
        'sources': {'selector': {'file': new_path.name, 'sha256': new},
                    'control_selector': {'file': current_name, 'sha256': current},
                    'merge': {'file': merge_name, 'sha256': merge}},
        'equivalence_proof': {k: eq[k] for k in ('initial_gpu', 'declared_tolerance', 'captures',
                                                  'seed_edges', 'selector_resources', 'valid')},
        'stage_timing': {k: st[k] for k in ('initial_gpu', 'host', 'layers', 'resources', 'correctness',
                                             'arms', 'summary', 'valid')},
        'complete_call_timing': {k: fc[k] for k in ('initial_gpu', 'host', 'layers', 'base',
                                                     'base_resources', 'valid')} | {'arm': arm},
    }
    (PKG / 'evidence.json').write_text(json.dumps(evidence, indent=1), encoding='utf-8', newline='\n')

    manifest = load(PKG / 'manifest.json')
    manifest['original_sources'][new_path.name] = new
    manifest['files'] = {name: sha(PKG / name) for name in list(manifest['files']) + [new_path.name]}
    manifest.setdefault('packaging_changes', []).append(
        'optimized_c4.py: selector_source now %s (bitwise-equal seeds and output to %s, which stays as '
        'the paired control)' % (new_path.name, current_name))
    manifest.setdefault('provenance_selector_' + a.label, {
        'gates': ['equivalence at tolerance 0: seeds and complete output bitwise on layers 00/19/39, '
                  'tie edges 0',
                  'selector stage timed in place with A/A null, interleaved',
                  'complete-call ABBA/BAAB with A/A null, 30 pairs per layer'],
        'receipts': [str(a.equiv).replace('\\', '/'), str(a.stage).replace('\\', '/'),
                     str(a.fullcall).replace('\\', '/')]})
    (PKG / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')
    subprocess.check_call([sys.executable, '-m', 'kernels.best._integrity'], cwd=ROOT)
    print('promoted and verified')


if __name__ == '__main__':
    main()
