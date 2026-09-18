"""Pack winner score, exact exponential and owner into one shared vector."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'native_fisher/fisher_c10_long_tiles_tree.cu').read_text()
old_decl='''    float* selected_winners=normalized+64*40;
    int* selected_owners=reinterpret_cast<int*>(selected_winners+6*40);'''
old_store='        selected_winners[index]=best;selected_owners[index]=owner;'
old_read='''          int index=min(col_for_row(row),5)*40+pair*8+2*(lane&3);
          float x0=selected_winners[index],x1=selected_winners[index+1];
          float p0=selected_owners[index]==owner?ex2(x0):0.f;
          float p1=selected_owners[index+1]==owner?ex2(x1):0.f;'''
assert all(base.count(x)==1 for x in (old_decl,old_store,old_read))
for stride in (40,42):
    assert 64*40*4+6*stride*16<=16384
    code=base.replace(old_decl,'    uint4* selected_vectors=reinterpret_cast<uint4*>(normalized+64*40);')
    code=code.replace(old_store,f'        selected_vectors[column*{stride}+ki]=make_uint4(__float_as_uint(best),__float_as_uint(ex2(best)),unsigned(owner),0);')
    code=code.replace(old_read,f'''          int index=min(col_for_row(row),5)*{stride}+pair*8+2*(lane&3);
          uint4 a,b;
          uint32_t address=__cvta_generic_to_shared(selected_vectors+index);
          asm volatile("ld.shared.v4.b32 {{%0,%1,%2,%3}},[%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"r"(address):"memory");
          asm volatile("ld.shared.v4.b32 {{%0,%1,%2,%3}},[%4];":"=r"(b.x),"=r"(b.y),"=r"(b.z),"=r"(b.w):"r"(address+16):"memory");
          float x0=__uint_as_float(a.x),x1=__uint_as_float(b.x);
          float p0=int(a.z)==owner?__uint_as_float(a.y):0.f;
          float p1=int(b.z)==owner?__uint_as_float(b.y):0.f;''')
    path=ROOT/f'native_fisher/fisher_c10_vector_exp{stride}.cu';path.write_text(code);print(path)
