"""Remove redundant end-of-strip barriers in the five-strip N64 candidate.

The next strip writes only normalized scores while the preceding strip reads
only disjoint owner indices. Its score-write barrier completes all previous
owner reads before owner writes. The existing post-weight barrier protects
the final strip before FP8 output staging reuses the entire scratch region.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'native_fisher'
s=(root/'fisher_c10_fp8_n64_owner16.cu').read_text()
a=s.index('    float* normalized=reinterpret_cast<float*>(temp);')
b=s.index('    uint32_t packed_weights[20];',a)
t=s[a:b]
tail='''      __syncthreads();
    }
'''
assert t.endswith(tail)
t=t[:-len(tail)]+'''      // The next normalized-score write does not touch owner indices.
      // Its following barrier precedes any overwrite of these owners.
    }
'''
s=s[:a]+t+s[b:]
(root/'fisher_c10_fp8_n64_owner16_overlap.cu').write_text(s)
