"""Frozen 20-video, four-step validation of a screened native C4 kernel pair.

Preparation is CPU-only. Execute the frozen copy with one explicit clean GPU.
The established generation runner owns model/prompt/config checks; this module
replaces only its kernel adapter and audits all eight H5 merge/route chunks.

Both halves of the pair are arguments. The earlier revision hard-coded the
selector, which made it impossible to screen a selector change - the one thing
the FP32 Gram-Schmidt candidate needs. `--num-videos` / `--prompt-start` are
passed through so a contended cluster can be screened in shards; the dispatch
gate scales with the requested count instead of assuming twenty.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import run_quality_first_videos as base
from analysis import launch_quality_first_videos as guard

METHOD = 'fisher_geometric_rf0'
SELF = 'analysis/run_c4_native_videos.py'
RETAINED_MERGE = 'kernels/best/c4/fisher_c4_guarded_softmax.cu'
RETAINED_SELECTOR = 'kernels/best/c4/fisher_c4_fused_selector_register_basis_loop.cu'
# Python entry points plus every CUDA header either side can include. CUDA
# sources are not in the tracked-source snapshot's extension set, so each one
# has to be named here or it will not be frozen.
SOURCES = (SELF, 'analysis/launch_quality_first_videos.py',
           'kernels/best/c4/operator.py', 'kernels/best/c4/fused_selector.py',
           'kernels/best/c4/route_reference.py', 'kernels/best/c4/warp_mma.cuh',
           'native_fisher/warp_mma.cuh')
ALLOWED_PREFIXES = ('native_fisher/', 'kernels/best/c4/')
ACTIVE = {}


def relative_source(name, label):
    base.require(name.startswith(ALLOWED_PREFIXES) and Path(name).suffix == '.cu',
                 'explicit native CUDA %s source required' % label)
    base.require((ROOT / name).is_file(), 'missing %s source: %s' % (label, name))
    return name


def configure_sources(merge, selector):
    base.OWN_SOURCES = tuple(dict.fromkeys((*base.OWN_SOURCES, *SOURCES, merge, selector)))


def verify_proofs(run_dir, merge, selector):
    manifest = json.loads((run_dir / 'native_kernel.json').read_text())
    base.require(manifest['merge_source'] == merge, 'prepared merge differs')
    base.require(manifest['selector_source'] == selector, 'prepared selector differs')
    base.require(base.digest(ROOT / merge) == manifest['merge_sha256'], 'merge source changed')
    base.require(base.digest(ROOT / selector) == manifest['selector_sha256'], 'selector source changed')
    for name, record in manifest['proofs'].items():
        path = run_dir / record['file']
        base.require(base.digest(path) == record['sha256'], 'numerical proof changed')
        proof = json.loads(path.read_text())
        base.require(proof.get('valid') is True and not proof.get('monitor_errors'),
                     'invalid numerical proof')
        if name == 'package':
            # The package proof covers the pair: this merge with this selector.
            base.require(proof.get('merge_source_sha256') == manifest['merge_sha256'],
                         'package proof is for a different merge')
            base.require(proof.get('selector_source_sha256') == manifest['selector_sha256'],
                         'package proof is for a different selector')
            base.require(set(proof['captures']) == {'00', '19', '39'}, 'missing captured layer')
            # A proof signed under a looser tolerance than the manifest claims
            # would let a numerics change ride in under an exactness label.
            declared = proof.get('declared_tolerance', {'max_seed_mismatch_fraction': 0.0,
                                                        'max_relative_l2': 0.0})
            base.require(declared == manifest['declared_tolerance'],
                         'proof tolerance differs from the prepared declaration')
            base.require(proof.get('within_declared_tolerance', True), 'proof breached its tolerance')
        else:
            # The merge edge cases are a property of the merge alone.
            base.require(proof.get('candidate_sha256') == manifest['merge_sha256'],
                         'edge proof is for a different merge')
            base.require({'random', 'zero_query_ties', 'duplicated_seed_ties',
                          'large_finite_query', 'nan_query_invalid'} <= set(proof['cases']),
                         'missing merge edge case')
    return manifest


class NativeAudit(base.DispatchAudit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_route_calls = 0

    def leave_layer(self):
        count = self.calls + 1
        base.require((self.core_calls, self.selector_calls, self.merge_calls,
                      self.native_route_calls) == (count, count, count * 8, count * 8),
                     'incomplete native H40 dispatch')
        self.layers[self.active_layer] += 1
        self.calls = count
        self.active_layer = None
        self.qkv = self.seed_metadata = None

    def save(self, complete, error=None):
        # The inherited completion gate expects one merge call per attention;
        # this operator issues eight H5 chunks, so the gate is replaced with a
        # stricter one rather than relaxed. The payload is built through the
        # base method so nothing else about the record changes.
        if complete:
            expected = self.num_videos * 160
            base.require(self.videos == self.num_videos and self.calls == expected
                         and self.layers == [self.num_videos * 4] * 40, 'generation incomplete')
            base.require(self.active_layer is None
                         and self.core_calls == self.selector_calls == expected
                         and self.merge_calls == self.native_route_calls == expected * 8,
                         'incomplete native kernel coverage')
        payload = super().save(False, error)
        payload.update(complete=complete,
                       native_route_kernel_calls=self.native_route_calls,
                       native_kernel=ACTIVE['kernel'],
                       planned_videos=self.num_videos,
                       completion_gate='native eight-H5-chunk gate replaces the inherited '
                                       'one-merge-per-attention gate',
                       implementation='CUDA Fisher selector and merge; original full Triton '
                                      'route; eight H5 chunks')
        base.write_json(self.path, payload)
        return payload


def install_native_audit(stack, audit, wan, token, recorder, frozen):
    import torch
    from kernels.best.c4.operator import NativeFisherAttention
    op = NativeFisherAttention(capacity=4, selector_backend='fused_cooperative_loop',
                               selector_source=frozen / ACTIVE['selector'],
                               merge_source=frozen / ACTIVE['merge'])
    base.require(op.merge.source_sha256 == ACTIVE['kernel']['merge_sha256'], 'loaded merge differs')
    base.require(base.digest(op.native_selector.source_path) == ACTIVE['kernel']['selector_sha256'],
                 'loaded selector differs')
    originals = {'forward': wan.WanSelfAttention.forward, 'select': op.select, 'merge': op.merge.run,
                 'route': op.route, 'finish_video': recorder.finish_video}
    audit.identities = {'operator': base.callable_identity(inspect.unwrap(NativeFisherAttention.__call__), frozen),
                        'adapter': base.callable_identity(install_native_audit, frozen)}

    def select(q, k, scale):
        base.require(not ACTIVE['errors'] and audit.qkv is not None, 'selector outside clean core')
        base.require(base.tensor_metadata(q) == base.grid_metadata(audit.qkv[0]),
                     'selector query is not current Q')
        base.require(base.tensor_metadata(k) == base.grid_metadata(audit.qkv[1]) and scale == 128 ** -.5,
                     'selector K/scale differs')
        result = originals['select'](q, k, scale)
        base.require(tuple(result.shape) == base.SEED_SHAPE and result.dtype == torch.uint8,
                     'wrong native seed layout')
        audit.selector_calls += 1
        audit.seed_metadata = base.tensor_metadata(result)
        return result

    def merge(*args):
        base.require(audit.seed_metadata == base.tensor_metadata(args[3]),
                     'merge is not using current seeds')
        base.require(tuple(args[7:11]) == ((audit.merge_calls % 8) * 5, 5, 21, 128 ** -.5),
                     'wrong native merge chunk')
        result = originals['merge'](*args)
        audit.merge_calls += 1
        return result

    def route(q, k, v, out, head, scale, active_heads=5):
        base.require(head == (audit.native_route_calls % 8) * 5 and active_heads == 5
                     and scale == 128 ** -.5, 'wrong full-route chunk')
        result = originals['route'](q, k, v, out, head, scale, active_heads=active_heads)
        audit.native_route_calls += 1
        return result

    def forward(module, x, grid_size, rope_cache):
        base.require(not ACTIVE['errors'], str(ACTIVE['errors']))
        audit.enter_layer(module, grid_size)
        result = originals['forward'](module, x, grid_size, rope_cache)
        audit.leave_layer()
        return result

    def core(*args, **kwargs):
        audit.enter_core(args, kwargs)
        # Reshape is metadata-only; NativeFisherAttention rejects noncontiguous input.
        return op(*(x.view(base.GRID) for x in args[:3])).view(base.SHAPE)

    def finish_video():
        base.require(not ACTIVE['errors'], str(ACTIVE['errors']))
        timing = originals['finish_video']()
        audit.finish_video(timing)
        return timing

    for obj, name, fn in ((op, 'select', select), (op.merge, 'run', merge), (op, 'route', route),
                          (wan.WanSelfAttention, 'forward', forward), (wan, 'token_merging_attn', core),
                          (recorder, 'finish_video', finish_video)):
        stack.enter_context(patch.object(obj, name, fn))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    phase = p.add_mutually_exclusive_group(required=True)
    phase.add_argument('--prepare', action='store_true')
    phase.add_argument('--execute', action='store_true')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--merge-source', default=RETAINED_MERGE)
    p.add_argument('--selector-source', default=RETAINED_SELECTOR)
    p.add_argument('--gpu', type=int)
    p.add_argument('--package-proof', type=Path)
    p.add_argument('--edge-proof', type=Path)
    p.add_argument('--num-videos', type=int, default=20)
    p.add_argument('--prompt-start', type=int, default=0)
    p.add_argument('--label', default='',
                   help='Free text recorded in the kernel manifest, e.g. which arm of a screen.')
    a = p.parse_args()
    run = a.run_dir.resolve()
    merge = relative_source(a.merge_source, 'merge')
    selector = relative_source(a.selector_source, 'selector')
    configure_sources(merge, selector)
    if a.prepare:
        base.require(a.package_proof is not None and a.edge_proof is not None,
                     'both numerical proofs required')
        package = json.loads(Path(a.package_proof).read_text())
        base.prepare(run, METHOD, a.checkpoint, num_videos=a.num_videos, prompt_start=a.prompt_start)
        proofs = {}
        for name, path in (('package', a.package_proof), ('edges', a.edge_proof)):
            record = {'file': 'native_%s_proof.json' % name, 'sha256': base.digest(path)}
            shutil.copyfile(path, run / record['file'])
            proofs[name] = record
        kernel = {'merge_source': merge, 'merge_sha256': base.digest(ROOT / merge),
                  'selector_source': selector, 'selector_sha256': base.digest(ROOT / selector),
                  'declared_tolerance': package.get(
                      'declared_tolerance',
                      {'max_seed_mismatch_fraction': 0.0, 'max_relative_l2': 0.0}),
                  'label': a.label, 'proofs': proofs,
                  'planned': {'num_videos': a.num_videos, 'prompt_start': a.prompt_start},
                  'generation_quality_validated': False}
        base.write_json(run / 'native_kernel.json', kernel, exclusive=True)
        verify_proofs(run, merge, selector)
        print(json.dumps({'prepared': str(run), 'kernel': kernel}), flush=True)
        return
    base.require(ROOT == (run / 'source').resolve(), 'execute the frozen wrapper')
    base.require(a.gpu is not None, 'one explicit physical GPU required')
    kernel = verify_proofs(run, merge, selector)
    initial = guard.require_clean(a.gpu)
    os.environ.update(guard.child_environment(run, initial['gpu']['uuid']))
    os.environ['TORCH_EXTENSIONS_DIR'] = str(run / 'torch_extensions')
    errors = []
    samples = []
    stop = threading.Event()
    ACTIVE.update(kernel=kernel, merge=merge, selector=selector, errors=errors)

    def monitor():
        while not stop.is_set():
            try:
                processes = guard.gpu_processes(initial['gpu']['uuid'])
                samples.append({'utc': guard.now(), 'processes': processes})
                if any(row['pid'] != os.getpid() for row in processes):
                    errors.append('foreign GPU process')
            except Exception as error:
                errors.append(repr(error))
            stop.wait(3)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    complete = False
    try:
        with patch.object(base, 'DispatchAudit', NativeAudit), \
             patch.object(base, 'install_audit', install_native_audit):
            base.execute(run, METHOD, a.checkpoint, parent_purity_managed=True)
        verify_proofs(run, merge, selector)
        base.require(not errors, str(errors))
        complete = True
    finally:
        stop.set()
        thread.join(10)
        base.write_json(run / 'gpu_monitor.json',
                        {'initial_gpu': initial, 'samples': samples, 'errors': errors,
                         'valid': complete and not errors})
    print(json.dumps({'generation_complete': complete, 'quality_evaluation_pending': True}), flush=True)


if __name__ == '__main__':
    main()
