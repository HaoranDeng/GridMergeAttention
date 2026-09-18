"""Distribute duplicated transcendental/division work, preserving FP64 trees."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=(root/'native_fisher/fisher_gs_distributed.cu').read_text()
start=base.index('  double rho[PR],den_terms[PR],root[PR],anchor_z[PR];')
end=base.index('    double value=0;',start)
replacement='''  double rho[PR],root[PR],anchor_z[PR];
  // Each half-warp holds all sixteen probe coordinates. Preserve the
  // original register-first, then probe-lane balanced summation order.
  double chosen_center=0;
  #pragma unroll
  for(int p=0;p<PR;++p)if(slot%PR==p)chosen_center=center[p];
  double rho_scalar=exp(__dadd_rn(chosen_center,-mx)),den=rho_scalar;
  #pragma unroll
  for(int d=8;d;d/=2)den=__dadd_rn(den,__shfl_xor_sync(0xffffffff,den,d));
  rho_scalar=__ddiv_rn(rho_scalar,den);
  double root_scalar=sqrt(rho_scalar);
  #pragma unroll
  for(int p=0;p<PR;++p){
    rho[p]=__shfl_sync(0xffffffff,rho_scalar,p*P+probe);
    root[p]=__shfl_sync(0xffffffff,root_scalar,p*P+probe);
'''
rho=base[:start]+replacement+base[end:]
start=rho.index('      double norm=sqrt(__dadd_rn(best,1.e-12)),basis[PR];')
end=rho.index('      #pragma unroll\n      for(int c=0;c<P;++c){',start)
old=rho[start:end]
new='''      double norm=sqrt(__dadd_rn(best,1.e-12)),basis[PR],basis_scalar=0;
      #pragma unroll
      for(int p=0;p<PR;++p){
        double value=0;
        #pragma unroll
        for(int c=0;c<P;++c)if(c==picked/CS)value=res[c][p];
        value=__shfl_sync(0xffffffff,value,(picked%CS)*P+probe);
        if(slot%PR==p)basis_scalar=value;
      }
      basis_scalar=__ddiv_rn(basis_scalar,norm);
      #pragma unroll
      for(int p=0;p<PR;++p)basis[p]=__shfl_sync(0xffffffff,basis_scalar,p*P+probe);
'''
both=rho[:start]+new+rho[end:]
(root/'native_fisher/fisher_gs_distribute.cu').write_text(both)
(root/'native_fisher/fisher_gs_distribute_rho.cu').write_text(rho)
# Independent basis-only control identifies where a scheduling gain arises.
(root/'native_fisher/fisher_gs_distribute_basis.cu').write_text(base.replace(old,new))

# Keep both feature gathers and the lane-work distributor in registers.
register=(root/'native_fisher/fisher_gs_register_select.cu').read_text()
helper=register[register.index('// Opaque selects'):register.index('template<int N>')]
combined=both.replace('template<int N>',helper+'template<int N>',1)
for old,new in [
    ('if(c==anchor/CS)value=z[c][p];','value=register_select(value,z[c][p],c==anchor/CS);'),
    ('if(c==picked/CS)value=res[c][p];','value=register_select(value,res[c][p],c==picked/CS);'),
    ('if(slot%PR==p)chosen_center=center[p];','chosen_center=register_select(chosen_center,center[p],slot%PR==p);'),
    ('if(slot%PR==p)basis_scalar=value;','basis_scalar=register_select(basis_scalar,value,slot%PR==p);')]:
    assert old in combined,old;combined=combined.replace(old,new)
(root/'native_fisher/fisher_gs_register_distribute.cu').write_text(combined)
