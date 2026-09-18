"""Experimental library backend for full compressed C10 attention with bias.

This component covers compressed tokens only. A full operator must combine
its FP16 output/FP32 LSE with the unchanged BF16 SQ1 state before final
rounding. This backend does not expose FP32 output; errors must be gated.
"""
import hashlib,os
import torch
BASE='/local2/denghaoran/GridMergeAttention/experiments/gma_c10_fp8_150ms_20260910/scai7'
os.environ.setdefault('FLASHINFER_WORKSPACE_BASE',BASE)
os.environ.setdefault('FLASHINFER_CUDA_ARCH_LIST','9.0a')
os.environ.setdefault('MAX_JOBS','2')

VARIANT=r'''
struct GMACompressedBias : AttentionVariantBase {
  static constexpr bool use_softmax=true;
  float sm_scale_log2;
  uint32_t qo_len,kv_len,window_left;
  template<typename Params>
  __device__ __host__ GMACompressedBias(const Params& params,uint32_t batch_idx,uint8_t* smem_ptr) {
    sm_scale_log2=math::log2e;
    qo_len=params.get_qo_len(batch_idx);kv_len=params.get_kv_len(batch_idx);window_left=kv_len;
  }
  REGISTER_LOGITS_TRANSFORM(params,logits,batch_idx,qo_idx,kv_idx,qo_head_idx,kv_head_idx,{
    float bias=kv_idx<9450?params.gma_bias[uint64_t(batch_idx)*9450+kv_idx]:0.f;
    return logits*float(params.gma_scale)-bias;
  })
};
'''
def jit_args():
    name='gma_c10_fp8_bias_f16_i64_'+hashlib.sha256(VARIANT.encode()).hexdigest()[:12]
    return [name,torch.bfloat16,torch.float8_e4m3fn,torch.float16,torch.int64,128,128,
            ['gma_bias'],['float'],['gma_scale'],['double'],'GMACompressedBias',VARIANT]

def compile_cpu():
    from flashinfer.jit.attention import gen_customize_batch_prefill_module
    return gen_customize_batch_prefill_module('fa2',*jit_args()).build_and_load()

class CompressedRoute:
    def __init__(self,device='cuda'):
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
        self.workspace=torch.empty(128*1024*1024,dtype=torch.uint8,device=device)
        self.wrapper=BatchPrefillWithRaggedKVCacheWrapper(self.workspace,backend='fa2',jit_args=jit_args())
        # One H5 KV tensor spans 10.16 GB. The prefill kernel multiplies
        # indptr by the token stride before pointer addition; use int64.
        self.qind=torch.arange(8401,dtype=torch.int64)*45
        self.kind=torch.arange(8401,dtype=torch.int64)*9450
        self.wrapper.plan(self.qind,self.kind,1,1,128,causal=False,q_data_type=torch.bfloat16,
                          kv_data_type=torch.float8_e4m3fn,o_data_type=torch.float16,disable_split_kv=True)
        self.out=torch.empty((8400*45,1,128),dtype=torch.float16,device=device)
        self.lse=torch.empty((8400*45,1),dtype=torch.float32,device=device)

    def __call__(self,packed_q,mk,mv,bias,scale=128**-.5):
        return self.wrapper.run(packed_q.reshape(-1,1,128),mk.reshape(-1,1,128),mv.reshape(-1,1,128),
                                bias,scale,out=self.out,lse=self.lse,return_lse=True)
