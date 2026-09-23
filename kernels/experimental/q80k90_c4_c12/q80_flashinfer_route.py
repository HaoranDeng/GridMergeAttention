"""Full attention over merged banks with FP32 entropy bias in the logits hook."""
import hashlib
import torch


VARIANT = r'''
struct GMAEntropyAttention : AttentionVariantBase {
  const float* entropy;
  uint32_t kv_length;
  float scale;
  float scale_pv;
  template <typename MainloopParams, typename BlockCoord>
  __device__ GMAEntropyAttention(const MainloopParams& params,const BlockCoord& coord) {
    auto [q_tile_idx,qo_head_idx,kv_head_idx,qo_offset,kv_offset,qo_len,kv_len,batch_idx]=coord;
    entropy=params.additional_params.entropy+kv_offset;
    kv_length=kv_len;
    scale=params.additional_params.sm_scale;
    scale_pv=1.f;
  }
  template <int N>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<N, true>(math::log2e);
  }
  REGISTER_LOGITS_TRANSFORM(params,logits,batch_idx,qo_idx,kv_idx,qo_head_idx,kv_head_idx,{
    return kv_idx<kv_length ? logits*scale+entropy[kv_idx] : -__int_as_float(0x7f800000);
  })
};
'''


class EntropyRoute:
    def __init__(self,groups,length,device):
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
        name='gma_entropy_fa3_'+hashlib.sha256(VARIANT.encode()).hexdigest()[:16]
        self.workspace=torch.empty(128*1024*1024,device=device,dtype=torch.uint8)
        self.wrapper=BatchPrefillWithRaggedKVCacheWrapper(self.workspace,kv_layout='NHD',backend='fa3',
            jit_args=[name,torch.bfloat16,torch.bfloat16,torch.bfloat16,torch.int32,128,128,
                ['entropy'],['float'],['sm_scale'],['double'],'GMAEntropyAttention',VARIANT])
        ptr=torch.arange(groups+1,dtype=torch.int32)
        self.wrapper.plan(ptr*80,ptr*length,1,1,128,causal=False,
            q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,sm_scale=128**-.5)
        self.output=torch.empty((groups*80,1,128),device=device,dtype=torch.bfloat16)

    def __call__(self,q,k,v,bias):
        return self.wrapper.run(q,k,v,bias,128**-.5,out=self.output)
