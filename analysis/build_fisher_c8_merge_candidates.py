"""Generate bounded Fisher-C8 merge candidates from the retained r2 source.

The candidates keep the exact eight-representative contract and only rewrite
the merge arithmetic.  Generation is deterministic so every compiled source
still receives a stable content digest from the resident benchmark.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/best/c8_r2/gma_c8_d10a_fused_argmax_v0.cu"
OUT = ROOT / "native_fisher"


NORMALIZER = """    float norm_max[2],norm_sum[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float direct[20];
      #pragma unroll
      for(int i=0;i<20;++i)
        direct[i]=ex2(__fmul_rn(s[4*(i/2)+2*half+(i&1)],scale));
      float sum=reduce80(direct);
      // A full-warp vote keeps the fallback's full-mask shuffles converged.
      // If any row needs the stable path, every lane in this warp takes it.
      bool direct_ok=__all_sync(0xffffffff,sum>0x1p-64f && sum<0x1p64f);
      if(direct_ok) {
        norm_max[half]=0.f;norm_sum[half]=sum;
      } else {
      float mx=-CUDART_INF_F;
      #pragma unroll
      for(int i=0;i<10;++i) mx=fmaxf(mx,fmaxf(__fmul_rn(s[4*i+2*half],scale),__fmul_rn(s[4*i+2*half+1],scale)));
      mx=max4(mx);
      float bs[20];
      #pragma unroll
      for(int i=0;i<20;++i)
        bs[i]=ex2(__fmaf_rn(s[4*(i/2)+2*half+(i&1)],scale,-mx));
      norm_max[half]=mx;norm_sum[half]=reduce80(bs);
      }
    }
    // Each four-lane row group has the same reduced sums. Compute its two
    // independent logarithms in alternating lanes, then broadcast each.
    float norm_log=lg2((lane&1)?norm_sum[1]:norm_sum[0]);
    #pragma unroll
    for(int half=0;half<2;++half) {
      float norm=__fadd_rn(norm_max[half],__shfl_sync(0xffffffff,norm_log,(lane&~1)+half));
      // Carry the COMPLEMENT u=(norm+1)-s*scale rather than the score itself.
      // u is strictly positive and strictly decreasing in the score, so the
      // unsigned order of its fp32 bit pattern is the score order reversed --
      // which is what lets the whole winner search be one integer butterfly.
      // The multiply is written unchanged so it stays shared with the
      // direct-path ex2 argument (ptxas CSEs it; line 197 costs 40 FADD, not 80).
      float offset=__fadd_rn(norm,1.f);
      #pragma unroll
      for(int i=0;i<20;++i) {
        int ix=(i/2)*4+half*2+(i&1);
        s[ix]=__fadd_rn(offset,-__fmul_rn(s[ix],scale));
      }
    }
"""


MAX_SHIFT = """    // Cluster weights and the entropy correction are invariant to a
    // common log-probability shift.  Use the row maximum as that shift instead
    // of materializing the 80-way global softmax normalizer.  This removes an
    // extra exp2/reduction/log2 pass while keeping exponentials non-positive.
    float score_max[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
      float mx=-CUDART_INF_F;
      #pragma unroll
      for(int i=0;i<10;++i)
        mx=fmaxf(mx,fmaxf(__fmul_rn(s[4*i+2*half],scale),
                          __fmul_rn(s[4*i+2*half+1],scale)));
      score_max[half]=max4(mx);
    }
    #pragma unroll
    for(int half=0;half<2;++half) {
      float offset=score_max[half];
      #pragma unroll
      for(int i=0;i<20;++i) {
        int ix=(i/2)*4+half*2+(i&1);
        s[ix]=__fadd_rn(offset,-__fmul_rn(s[ix],scale));
      }
    }
"""


def write(
    name: str,
    *,
    max_shift: bool,
    remove_offset: bool,
    signed_complement: bool,
    approximate_reciprocal: bool,
    padded_score_n: int | None = None,
    dephase_step_ns: int | None = None,
    dephase_wave_sms: int | None = None,
    dephase_atomic_slots: bool = False,
) -> None:
    source = BASE.read_text(encoding="utf-8")
    if max_shift:
        if source.count(NORMALIZER) != 1:
            raise RuntimeError("retained normalizer block changed")
        source = source.replace(NORMALIZER, MAX_SHIFT)
        source = source.replace(
            "float best=__fadd_rn(1.f,-fminf(u0,u1));",
            "float best=-fminf(u0,u1);",
        )
        source = source.replace(
            "// u=(norm+1)-score, positive and decreasing in the score, so unsigned MIN",
            "// u=max(score)-score, nonnegative and decreasing in the score, so unsigned MIN",
        )
    elif signed_complement:
        source = source.replace(
            """      // Carry the COMPLEMENT u=(norm+1)-s*scale rather than the score itself.
      // u is strictly positive and strictly decreasing in the score, so the
      // unsigned order of its fp32 bit pattern is the score order reversed --
""",
            """      // Carry the signed normalized score w=s*scale-(norm+1).
      // w is strictly negative and increasing in the score; unsigned fp32
      // ordering reverses the negative domain, so integer MIN is score MAX --
""",
        )
        source = source.replace(
            "// u=(norm+1)-score, positive and decreasing in the score, so unsigned MIN",
            "// w=score-(norm+1), negative and increasing in the score, so unsigned MIN",
        )
        source = source.replace(
            "s[ix]=__fadd_rn(offset,-__fmul_rn(s[ix],scale));",
            "s[ix]=__fadd_rn(__fmul_rn(s[ix],scale),-offset);",
        )
        source = source.replace(
            "float u0=s[ix],u1=s[ix+2];",
            "float w0=s[ix],w1=s[ix+2];",
        )
        source = source.replace(
            "uint32_t key=min(owner_key(u0,oc0),owner_key(u1,oc1));",
            "uint32_t key=min(owner_key(w0,oc0),owner_key(w1,oc1));",
        )
        source = source.replace(
            "float best=__fadd_rn(1.f,-fminf(u0,u1));",
            "float best=fmaxf(w0,w1);",
        )
        if remove_offset:
            source = source.replace(
                "float offset=__fadd_rn(norm,1.f);",
                "float offset=norm;",
            )
            source = source.replace(
                "// Carry the signed normalized score w=s*scale-(norm+1).",
                "// Carry the signed normalized score w=s*scale-norm directly.",
            )
            source = source.replace(
                "// w=score-(norm+1), negative and increasing in the score, so unsigned MIN",
                "// w=score-norm, nonpositive and increasing in the score, so unsigned MIN",
            )
    elif remove_offset:
        source = source.replace(
            "float offset=__fadd_rn(norm,1.f);",
            "float offset=norm;",
        )
        source = source.replace(
            "float best=__fadd_rn(1.f,-fminf(u0,u1));",
            "float best=-fminf(u0,u1);",
        )
        source = source.replace(
            "// u=(norm+1)-score, positive and decreasing in the score, so unsigned MIN",
            "// u=norm-score, nonnegative and decreasing in the score, so unsigned MIN",
        )
    if approximate_reciprocal:
        source = source.replace(
            'asm("div.full.f32 %0,0f3F800000,%1;":"=f"(y):"f"(x));',
            'asm("rcp.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));',
        )
    if dephase_step_ns is not None:
        if dephase_step_ns <= 0:
            raise ValueError("dephase step must be positive")
        if dephase_wave_sms is not None and dephase_wave_sms <= 0:
            raise ValueError("dephase wave SM count must be positive")
        phase = "(blockIdx.x&3u)"
        if dephase_atomic_slots:
            source = source.replace(
                "using namespace cute::SM90::GMMA;\n",
                "using namespace cute::SM90::GMMA;\n\n"
                "__device__ unsigned fisher_c8_dephase_slots[256];\n",
            )
            phase = "phase_slot"
        elif dephase_wave_sms is not None:
            phase = (
                "(((blockIdx.x+gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z))/"
                f"{dephase_wave_sms}u)&3u)"
            )
        marker = '  int total=Q*(C/8),start=blockIdx.z*tile_count;\n'
        slot_setup = ""
        if dephase_atomic_slots:
            slot_setup = '''  // Assign the four live CTA slots independently on every physical SM;
  // this does not assume any CUDA grid-to-SM traversal order.
  if(tid==0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;":"=r"(smid));
    *reinterpret_cast<unsigned*>(temp)=
        atomicAdd(&fisher_c8_dephase_slots[smid],1u)&3u;
  }
  __syncthreads();
  unsigned phase_slot=*reinterpret_cast<unsigned*>(temp);
'''
        replacement = f'''  // Stagger the four CTAs that normally co-reside on an SM.  The D20
  // region census found their WGMMA phases nearly lock together; this one-time
  // delay changes only scheduling and is followed by a CTA barrier before any
  // score or output arithmetic.
{slot_setup}  unsigned phase_ns={phase}*{dephase_step_ns}u;
  if(phase_ns) __nanosleep(phase_ns);
  __syncthreads();
{marker}'''
        if source.count(marker) != 1:
            raise RuntimeError("retained grid prologue changed")
        source = source.replace(marker, replacement)
    if padded_score_n is not None:
        # Hopper exposes both N=80 and N=128 score WGMMA shapes.  The extra
        # accumulator columns are dead and read only the already allocated sv
        # shared-memory region.  This is a pure throughput probe: s[0:40]
        # retain the same column mapping and complete output must therefore be
        # identical to the corresponding N=80 arithmetic candidate.
        if padded_score_n == 96:
            replacements = {
                "float s[40];": "float s[48];",
                "using Score=MMA_64x80x16_F32BF16BF16_RS<Major::K,Major::K>;":
                    "using Score=MMA_64x96x16_F32BF16BF16_RS<Major::K,Major::K>;",
                "desc(sk+boff(0,d),16,1024),E40(s),":
                    "desc(sk+boff(0,d),16,1024),E40(s),E8(s,40),",
            }
        elif padded_score_n == 128:
            replacements = {
                "float s[40];": "float s[64];",
                "using Score=MMA_64x80x16_F32BF16BF16_RS<Major::K,Major::K>;":
                    "using Score=MMA_64x128x16_F32BF16BF16_RS<Major::K,Major::K>;",
                "desc(sk+boff(0,d),16,1024),E40(s),":
                    "desc(sk+boff(0,d),16,1024),E64(s),",
            }
        else:
            raise ValueError("padded score N must be 96 or 128")
        for old, new in replacements.items():
            if source.count(old) != 1:
                raise RuntimeError(f"retained score block changed: {old}")
            source = source.replace(old, new)
    (OUT / name).write_text(source, encoding="utf-8")


def main() -> None:
    write(
        "fisher_c8_merge_maxshift_v0.cu",
        max_shift=True,
        remove_offset=True,
        signed_complement=False,
        approximate_reciprocal=False,
    )
    write(
        "fisher_c8_merge_maxshift_rcp_v0.cu",
        max_shift=True,
        remove_offset=True,
        signed_complement=False,
        approximate_reciprocal=True,
    )
    write(
        "fisher_c8_merge_offset0_v0.cu",
        max_shift=False,
        remove_offset=True,
        signed_complement=False,
        approximate_reciprocal=False,
    )
    write(
        "fisher_c8_merge_offset0_rcp_v0.cu",
        max_shift=False,
        remove_offset=True,
        signed_complement=False,
        approximate_reciprocal=True,
    )
    write(
        "fisher_c8_merge_offset0_dephase1250_v0.cu",
        max_shift=False,
        remove_offset=True,
        signed_complement=False,
        approximate_reciprocal=False,
        dephase_step_ns=1250,
    )
    write(
        "fisher_c8_merge_offset0_wave132_dephase1250_v0.cu",
        max_shift=False,
        remove_offset=True,
        signed_complement=False,
        approximate_reciprocal=False,
        dephase_step_ns=1250,
        dephase_wave_sms=132,
    )
    write(
        "fisher_c8_merge_offset0_smslot_dephase1250_v0.cu",
        max_shift=False,
        remove_offset=True,
        signed_complement=False,
        approximate_reciprocal=False,
        dephase_step_ns=1250,
        dephase_atomic_slots=True,
    )
    write(
        "fisher_c8_merge_signed_complement_v0.cu",
        max_shift=False,
        remove_offset=False,
        signed_complement=True,
        approximate_reciprocal=False,
    )
    write(
        "fisher_c8_merge_signed_offset0_v0.cu",
        max_shift=False,
        remove_offset=True,
        signed_complement=True,
        approximate_reciprocal=False,
    )
    write(
        "fisher_c8_merge_signed_offset0_rcp_v0.cu",
        max_shift=False,
        remove_offset=True,
        signed_complement=True,
        approximate_reciprocal=True,
    )
    write(
        "fisher_c8_merge_offset0_n128_v0.cu",
        max_shift=False,
        remove_offset=True,
        signed_complement=False,
        approximate_reciprocal=False,
        padded_score_n=128,
    )
    write(
        "fisher_c8_merge_offset0_n96_v0.cu",
        max_shift=False,
        remove_offset=True,
        signed_complement=False,
        approximate_reciprocal=False,
        padded_score_n=96,
    )


if __name__ == "__main__":
    main()
