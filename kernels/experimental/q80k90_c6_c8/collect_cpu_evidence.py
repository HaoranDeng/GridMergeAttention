"""Create a compact, portable record of completed CPU-only checks."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).parent
native=json.loads((root/'compile_checks/compile.json').read_text())
triton=json.loads((root/'triton_compile_checks/compile_triton.json').read_text())
probe=json.loads((root/'selector_cpu_probe.json').read_text())
checks=json.loads((root/'cpu_checks.json').read_text())
assert native['complete'] and native['all_compiled'] and len(native['rows'])==10
assert triton['complete'] and triton['all_compiled'] and len(triton['rows'])==21
assert probe['complete'] and not probe['gpu_used']
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in native['source_sha256'].items())
evidence=dict(gpu_execution=False,gpu_numerical_validation='pending',gpu_latency='pending',
              compiled_source_sha256=native['source_sha256'],
              source_sha256_lf={n:hashlib.sha256((root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                for n in native['source_sha256']},
              cpu_checks=checks,
              native_variants=[{k:v for k,v in row.items() if k not in ('binary','seconds')} for row in native['rows']],
              triton_variants=[{k:v for k,v in row.items() if k!='seconds'} for row in triton['rows']],
              selector_probe={k:v for k,v in probe.items() if k!='rows'})
(root/'cpu_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print('CPU evidence verified and saved')
