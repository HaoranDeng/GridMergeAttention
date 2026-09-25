"""Static SASS census of C4 merge/selector sources. CPU only - no GPU needed.

Compilation and disassembly need no device, so this runs while the cluster is
full. Buckets every instruction of every kernel in the module by opcode and by
functional class, and reports register/spill/occupancy from ptxas.

A static count is not a price. `never-executed-instructions` records a case where
static deltas overestimated a saving 27-fold, because most of the deleted
instructions sat in a branch that never ran. Use this to find structure - which
pipes the code is built out of, where a class concentrates - and price anything
it suggests with a measurement.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CUOBJ = '/usr/local/cuda/bin/cuobjdump'
NVDIS = '/usr/local/cuda/bin/nvdisasm'
INSTR = re.compile(r'/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\d+\s+)?([A-Z0-9._@]+)')
FUNC = re.compile(r'^\s*\.text\.(\S+):')

# Functional classes, matched against the opcode's leading mnemonic. Order
# matters: the first pattern that matches wins.
CLASSES = (
    ('wgmma', ('HGMMA', 'QGMMA', 'IGMMA', 'OGMMA', 'GMMA', 'WARPGROUP')),
    ('tma', ('UTMA', 'UBLKCP', 'BLKCP')),
    ('tensor_barrier', ('ARRIVES', 'ELECT', 'BAR', 'DEPBAR', 'MEMBAR', 'ARRIVE')),
    ('shared', ('LDS', 'STS', 'LDSM', 'STSM', 'ATOMS', 'LDGSTS')),
    ('local_spill', ('LDL', 'STL')),
    ('global', ('LDG', 'STG', 'RED', 'ATOMG', 'ATOM', 'LD', 'ST')),
    ('fp64', ('DADD', 'DMUL', 'DFMA', 'DSETP', 'DMMA', 'MUFU.RCP64', 'F2F.F64')),
    ('fp32', ('FADD', 'FMUL', 'FFMA', 'FSETP', 'FSEL', 'FMNMX', 'FCHK', 'MUFU',
              'HADD', 'HMUL', 'HFMA', 'HSETP', 'HMNMX', 'F2F', 'F2I', 'I2F')),
    ('integer', ('IADD', 'IMAD', 'IMNMX', 'ISETP', 'LOP', 'SHF', 'LEA', 'FLO',
                 'POPC', 'BREV', 'PRMT', 'SEL', 'SGXT', 'VIADD', 'VIMNMX')),
    ('warp_exchange', ('SHFL', 'VOTE', 'MATCH', 'REDUX')),
    ('uniform', ('UIADD', 'UIMAD', 'ULOP', 'USHF', 'ULEA', 'UMOV', 'UISETP',
                 'UPRMT', 'USEL', 'UFLO', 'UPOPC', 'R2UR', 'S2UR')),
    ('move', ('MOV', 'R2P', 'P2R', 'S2R', 'CS2R', 'IPA')),
    ('control', ('BRA', 'BRX', 'JMP', 'CALL', 'RET', 'EXIT', 'SSY', 'SYNC',
                 'BSSY', 'BSYNC', 'NOP', 'PMTRIG', 'YIELD', 'PREDICATE')),
)


def classify(opcode):
    head = opcode.split('.')[0]
    for name, prefixes in CLASSES:
        for prefix in prefixes:
            if opcode.startswith(prefix) or head == prefix:
                return name
    return 'other'


def resource_usage(build_dir):
    """Registers/shared/stack per kernel, read out of the cubin.

    The extension's own `resources()` runs a device query, which is exactly what
    this script exists to avoid; `cuobjdump -res-usage` reads the same numbers
    off the binary on the host.
    """
    shared = sorted(Path(build_dir).glob('*.so'))
    if not shared:
        return {}
    text = subprocess.check_output([CUOBJ, '-res-usage', str(shared[0].resolve())],
                                   text=True, stderr=subprocess.STDOUT)
    usage, current = {}, None
    for line in text.split('\n'):
        match = re.match(r'\s*Function (\S+):', line)
        if match:
            current = match.group(1)
            usage[current] = {}
            continue
        if current:
            for key, pattern in (('registers', r'REG:(\d+)'), ('shared_bytes', r'SHARED:(\d+)'),
                                 ('local_bytes', r'LOCAL:(\d+)'), ('const_bytes', r'CONSTANT\[\d+\]:(\d+)'),
                                 ('stack_bytes', r'STACK:(\d+)'), ('spill_bytes', r'SPILL:(\d+)')):
                found = re.search(pattern, line)
                if found:
                    usage[current][key] = int(found.group(1))
    return usage


def census(build_dir):
    shared = sorted(Path(build_dir).glob('*.so'))
    if not shared:
        raise FileNotFoundError('no .so in %s' % build_dir)
    tmp = tempfile.mkdtemp()
    try:
        subprocess.check_output([CUOBJ, '-xelf', 'all', str(shared[0].resolve())],
                                cwd=tmp, text=True, stderr=subprocess.STDOUT)
        text = ''.join(subprocess.check_output([NVDIS, '-g', '-c', str(entry)], text=True,
                                               stderr=subprocess.STDOUT)
                       for entry in Path(tmp).iterdir())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    function = None
    per = defaultdict(Counter)
    for line in text.split('\n'):
        match = FUNC.match(line)
        if match:
            function = match.group(1)
            continue
        match = INSTR.search(line)
        if match and function:
            per[function][match.group(1)] += 1
    return per


def summarise(counter, top):
    classes = Counter()
    for opcode, count in counter.items():
        classes[classify(opcode)] += count
    total = sum(counter.values())
    return {'total_instructions': total,
            'by_class': dict(classes.most_common()),
            'by_class_fraction': {k: round(v / total, 4) for k, v in classes.most_common()},
            'top_opcodes': dict(Counter(counter).most_common(top))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, action='append', required=True,
                   help='Repeatable. Repo-relative or absolute .cu path.')
    p.add_argument('--kind', choices=['merge', 'selector'], default='merge')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--top', type=int, default=25)
    p.add_argument('--kernel-filter', default='',
                   help='Only report kernels whose mangled name contains this.')
    a = p.parse_args()
    assert not a.output.exists()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # No device is touched: nvcc, cuobjdump and nvdisasm are host tools.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(a.output.parent / 'census_extensions'))

    import torch  # noqa: F401  (imported for its bundled build machinery)

    result = {'kind': a.kind, 'sources': {},
              'scope': 'Static SASS census. Counts what exists, not what executes; '
                       'price anything it suggests with a measurement.'}
    for index, source in enumerate(a.source):
        path = source if source.is_absolute() else ROOT / source
        if a.kind == 'merge':
            from kernels.best.c4.operator import extension
            module = extension(4, source_path=path, name_suffix='_census%d' % index)
        else:
            from kernels.best.c4.fused_selector import extension as sel_extension
            module = sel_extension(loop_frames=True, cooperative=True, source_path=path)
        sha = module.source_sha256
        # The module object does not carry its build directory, and asking the
        # extension for `resources()` would launch a device query - the one thing
        # this script exists to avoid. Take the newest directory holding a shared
        # object; sources are censused one at a time, so this is unambiguous.
        root = Path(os.environ['TORCH_EXTENSIONS_DIR'])
        candidates = [d for d in root.iterdir() if d.is_dir() and list(d.glob('*.so'))]
        candidates.sort(key=lambda d: d.stat().st_mtime)
        build = candidates[-1]
        resources = resource_usage(build)
        per = census(build)
        kernels = {k: summarise(v, a.top) for k, v in per.items()
                   if a.kernel_filter in k and sum(v.values()) > 64}
        result['sources'][str(source)] = {
            'sha256': sha, 'resources': resources, 'build_dir': str(build),
            'kernels': kernels,
            'largest_kernel': max(kernels, key=lambda k: kernels[k]['total_instructions'])
                              if kernels else None}
        entry = result['sources'][str(source)]
        print('SOURCE', source, 'sha', sha[:16], 'resources', resources, flush=True)
        if entry['largest_kernel']:
            big = kernels[entry['largest_kernel']]
            print('  largest kernel', entry['largest_kernel'][:60],
                  'instructions', big['total_instructions'], flush=True)
            print('  classes', json.dumps(big['by_class_fraction']), flush=True)
        a.output.write_text(json.dumps(result, indent=2))

    if len(a.source) == 2:
        first, second = (result['sources'][str(s)] for s in a.source)
        if first['largest_kernel'] and second['largest_kernel']:
            x = first['kernels'][first['largest_kernel']]['top_opcodes']
            y = second['kernels'][second['largest_kernel']]['top_opcodes']
            result['opcode_delta'] = {o: y.get(o, 0) - x.get(o, 0)
                                      for o in sorted(set(x) | set(y))
                                      if y.get(o, 0) != x.get(o, 0)}
            result['total_delta'] = (second['kernels'][second['largest_kernel']]['total_instructions']
                                     - first['kernels'][first['largest_kernel']]['total_instructions'])
            print('DELTA total', result['total_delta'],
                  json.dumps(result['opcode_delta']), flush=True)
    print('SUMMARY_DONE', flush=True)
    a.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
