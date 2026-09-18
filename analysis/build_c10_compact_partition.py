"""Native exact merge into a small key-partition workspace."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'native_fisher/fisher_c10_half_owner.cu').read_text()
s=s[:s.index('void run(torch::Tensor q,')]
s=s.replace('CUtensorMap map_v,int tile_count)', 'CUtensorMap map_v,int key_begin,int column_base)')
s=s.replace('AH=TOTAL_HEADS==40?5:1','AH=1')
s=s.replace('  int block=blockIdx.x,head=blockIdx.y,gh=HS+head;\n  int ko=block%KO,kr=block/KO;\n  int kb=ko*R+kr;',
'''  constexpr int KEY_BLOCKS=64;
  int head=0,gh=HS;
  int kb=key_begin+blockIdx.x;
  if(kb>=945)return;
  int ko=kb/45,kr=kb%45;''')
s=s.replace('  int total=Q*((C+5)/6),start=blockIdx.z*tile_count;\n  for(int tile=start;tile<min(start+tile_count,total);++tile) {\n    int qo=tile%Q,cb=tile/Q,rf=0;',
'''  for(int qo=blockIdx.y*TILES;qo<(blockIdx.y+1)*TILES;++qo) {
    int cb=column_base/6,rf=0;''')
s=s.replace('((qo*C+col)*AH+head)*(KO*R*10)+kb*10+owner',
            '(qo*6+col-column_base)*(KEY_BLOCKS*10)+(kb-key_begin)*10+owner')
s=s.replace('(((int64_t(qo)*C+col)*AH+head)*(KO*R*10)+kb*10+seed_for_row(row))',
            '((int64_t(qo)*6+col-column_base)*(KEY_BLOCKS*10)+(kb-key_begin)*10+seed_for_row(row))')
s=s.replace('"r"(kb*20),"r"(head),"r"(cb*6)', '"r"((kb-key_begin)*20),"r"(0),"r"(0)')
s+='''
void run_compact(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor seeds,
    torch::Tensor mk,torch::Tensor mv,torch::Tensor bias,int head,int column,int key_begin,int frames) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==6 && q.size(0)==1 && q.size(1)==21 && q.size(2)==45 && q.size(3)==80 && q.size(4)==40 && q.size(5)==128,"Native BF16 H40 input required");
  for(const auto& x:{k,v,seeds,mk,mv,bias})TORCH_CHECK(x.is_cuda() && x.device()==q.device() && x.is_contiguous(),"Contiguous same-device tensors required");
  TORCH_CHECK(k.sizes()==q.sizes() && v.sizes()==q.sizes() && q.is_contiguous(),"QKV shape mismatch");
  TORCH_CHECK(k.scalar_type()==at::kBFloat16 && v.scalar_type()==at::kBFloat16 && mk.scalar_type()==at::kBFloat16 && mv.scalar_type()==at::kBFloat16 && bias.scalar_type()==at::kFloat && seeds.scalar_type()==at::kByte,"Dtype mismatch");
  TORCH_CHECK(seeds.numel()==945*80*40*10 && mk.numel()==21*6*640*128 && mk.sizes()==mv.sizes() && bias.numel()==21*6*640,"Workspace shape mismatch");
  TORCH_CHECK(head>=0 && head<40 && column>=0 && column<80 && column%6==0 && key_begin>=0 && key_begin<945 && key_begin%64==0,"Slice range mismatch");
  TORCH_CHECK(frames==1 || frames==3 || frames==7 || frames==21,"Frame group must divide 21");
  constexpr int bytes=57344;
  CUtensorMap map_k{},map_v{};
  uint64_t dims[5]={64,64*20,1,6,21};
  uint64_t strides[4];uint64_t stride=128;
  for(int i=0;i<4;++i){strides[i]=stride;stride*=dims[i+1];}
  uint32_t box[5]={64,20,1,6,1},element_stride[5]={1,1,1,1,1};
  auto encode=[&](CUtensorMap& map,torch::Tensor x){
    auto status=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,5,x.data_ptr(),dims,strides,box,element_stride,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"TMA descriptor failed");
  };
  encode(map_k,mk);encode(map_v,mv);
  auto launch=[&](auto tag){
    constexpr int F=decltype(tag)::value;
    C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<F,40>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    merge_kernel<F,40><<<dim3(64,21/F),128,bytes,at::cuda::getCurrentCUDAStream()>>>(
      (const bf16*)q.data_ptr(),(const bf16*)k.data_ptr(),(const bf16*)v.data_ptr(),seeds.data_ptr<uint8_t>(),
      (bf16*)mk.data_ptr(),(bf16*)mv.data_ptr(),bias.data_ptr<float>(),21,21,45,80,40,1,head,1,float(0.08838834764831845*1.4426950408889634),nullptr,0,1,map_k,map_v,key_begin,column);
  };
  if(frames==1)launch(std::integral_constant<int,1>{});
  else if(frames==3)launch(std::integral_constant<int,3>{});
  else if(frames==7)launch(std::integral_constant<int,7>{});
  else launch(std::integral_constant<int,21>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int> resources(){
  cudaFuncAttributes a{};int active=0;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,merge_kernel<1,40>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(merge_kernel<1,40>,cudaFuncAttributeMaxDynamicSharedMemorySize,57344));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,merge_kernel<1,40>,128,57344));
  return {a.numRegs,int(a.localSizeBytes),57344,active};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run_compact",&run_compact);m.def("pack",&pack);m.def("resources",&resources);}
'''
assert 'tile_count' not in s
(root/'native_fisher/fisher_c10_compact_partition.cu').write_text(s)
