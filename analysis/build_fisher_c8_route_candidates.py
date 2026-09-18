"""Generate bounded Fisher-C8 route candidates from the retained r2 source."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/best/c8_r2/gma_c8_route_pipeline.cu"
OUT = ROOT / "native_fisher"


OLD = """ #pragma unroll
 for(int i=0;i<64;i+=2){
  int half=(i&3)>>1,row=warp*16+seed+8*half,d=2*(lane&3)+8*(i>>2);
  if(row<45){
   int64_t oi=(((int64_t(qo)*45+row)*80+col)*40+gh)*128+d;
   *reinterpret_cast<__nv_bfloat162*>(output+oi)=__floats2bfloat162_rn(
     __fdividef(result[i],running_sum[half]),__fdividef(result[i+1],running_sum[half]));
  }
 }
"""


SHARED_RECIPROCAL = """ // The two online-softmax denominators are shared by every output
 // dimension in their half-row.  Materialize each reciprocal once instead
 // of asking ptxas for one MUFU.RCP at every unrolled store site.
 float output_inv[2]={__fdividef(1.f,running_sum[0]),
                      __fdividef(1.f,running_sum[1])};
 #pragma unroll
 for(int i=0;i<64;i+=2){
  int half=(i&3)>>1,row=warp*16+seed+8*half,d=2*(lane&3)+8*(i>>2);
  if(row<45){
   int64_t oi=(((int64_t(qo)*45+row)*80+col)*40+gh)*128+d;
   *reinterpret_cast<__nv_bfloat162*>(output+oi)=__floats2bfloat162_rn(
     __fmul_rn(result[i],output_inv[half]),
     __fmul_rn(result[i+1],output_inv[half]));
  }
 }
"""


WARP_BIAS_REPLACEMENTS = (
    (
        """ float* route_bias=reinterpret_cast<float*>(sv+64*128);
 uint64_t* barrier=reinterpret_cast<uint64_t*>(route_bias+64);
""",
        """#if defined(FISHER_C8_ROUTE_WARP_BIAS)
 uint64_t* barrier=reinterpret_cast<uint64_t*>(sv+64*128);
#else
 float* route_bias=reinterpret_cast<float*>(sv+64*128);
 uint64_t* barrier=reinterpret_cast<uint64_t*>(route_bias+64);
#endif
""",
    ),
    (
        """  if(tid<64)route_bias[tid]=start+tid<7560?bias[int64_t(group)*7560+start+tid]:CUDART_INF_F;
  wait_tile(bar,key_group&1);
  __syncthreads();
""",
        """#if defined(FISHER_C8_ROUTE_WARP_BIAS)
  // Replicate the tiny 256-byte bias line into each warp's registers.  The
  // line is L1-hot after the first warp; this removes the CTA-wide barrier
  // that otherwise serializes every one of the 119 route iterations.
  int64_t bias_base=int64_t(group)*7560+start;
  float warp_bias_lo=start+lane<7560?bias[bias_base+lane]:CUDART_INF_F;
  float warp_bias_hi=start+lane+32<7560?bias[bias_base+lane+32]:CUDART_INF_F;
#else
  if(tid<64)route_bias[tid]=start+tid<7560?bias[int64_t(group)*7560+start+tid]:CUDART_INF_F;
#endif
  wait_tile(bar,key_group&1);
#if !defined(FISHER_C8_ROUTE_WARP_BIAS)
  __syncthreads();
#endif
""",
    ),
    (
        """   logits[i]=__fmul_rn(__fmaf_rn(logits[i],sm_scale,-route_bias[ki]),1.4426950408889634f);
""",
        """#if defined(FISHER_C8_ROUTE_WARP_BIAS)
   float warp_bias=__shfl_sync(0xffffffff,ki<32?warp_bias_lo:warp_bias_hi,ki&31);
   logits[i]=__fmul_rn(__fmaf_rn(logits[i],sm_scale,-warp_bias),1.4426950408889634f);
#else
   logits[i]=__fmul_rn(__fmaf_rn(logits[i],sm_scale,-route_bias[ki]),1.4426950408889634f);
#endif
""",
    ),
    (
        """ constexpr int bytes=3*64*128*2+64*4+16;
""",
        """#if defined(FISHER_C8_ROUTE_WARP_BIAS)
 constexpr int bytes=3*64*128*2+16;
#else
 constexpr int bytes=3*64*128*2+64*4+16;
#endif
""",
    ),
)


def main() -> None:
    source = BASE.read_text(encoding="utf-8")
    if source.count(OLD) != 1:
        raise RuntimeError("retained route output block changed")
    source = source.replace(OLD, SHARED_RECIPROCAL)
    for old, new in WARP_BIAS_REPLACEMENTS:
        if source.count(old) != 1:
            raise RuntimeError("retained route warp-bias block changed")
        source = source.replace(old, new)
    (OUT / "fisher_c8_route_shared_recip_v0.cu").write_text(
        source, encoding="utf-8"
    )
    (OUT / "fisher_c8_route_warp_bias_v0.cu").write_text(
        f'// included-source-sha256: {hashlib.sha256(source.encode("utf-8")).hexdigest()}\n'
        '#define FISHER_C8_ROUTE_WARP_BIAS 1\n'
        '#include "fisher_c8_route_shared_recip_v0.cu"\n',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
