"""Reuse private scores with an exact eight-byte probability/owner tuple.

The earlier winner-pair trial used twelve bytes (score, probability, owner)
and reloaded scores. This version retains the current private score registers
and uses only probability plus owner in shared memory. All rounding, ties and
entropy accumulation stay unchanged. No quantization of the tuple is used.
"""
from pathlib import Path
import hashlib

root=Path(__file__).resolve().parents[1]/'native_fisher'
base=root/'fisher_c10_fp8_local_scores.cu'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='6dabac09d497f89bda35e41a43a2893fd05c99862da506f74c19478189ed1730'
s=base.read_text()
def replace(old,new):
    global s
    assert old in s,old
    s=s.replace(old,new)

replace('int* selected_owners=reinterpret_cast<int*>(normalized+64*40);',
        'uint2* selected_states=reinterpret_cast<uint2*>(normalized+64*40);')
replace('selected_owners[index]=owner;',
        'selected_states[index]=make_uint2(__float_as_uint(ex2(best)),uint32_t(owner));')
replace('''          float p0=selected_owners[index]==owner?ex2(x0):0.f;
          float p1=selected_owners[index+1]==owner?ex2(x1):0.f;''',
'''          uint2 state0=selected_states[index],state1=selected_states[index+1];
          float p0=state0.y==uint32_t(owner)?__uint_as_float(state0.x):0.f;
          float p1=state1.y==uint32_t(owner)?__uint_as_float(state1.x):0.f;''')
replace('(64*40+6*40)*4','(64*40+2*6*40)*4')
assert 'selected_owners' not in s
(root/'fisher_c10_fp8_owner_probability.cu').write_text('// Exact owner/probability tuple with private normalized scores.\n'+s)
