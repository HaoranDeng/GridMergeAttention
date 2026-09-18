import unittest
import numpy as np
import ast
from pathlib import Path
from types import SimpleNamespace
try:
    import torch
except ImportError:
    torch = None
from analysis.mean_seed_pytorch import MEAN_METHODS, mean_seed_indices


def scalar_seeds(z,m,a,method):
    rho=np.exp(m-np.logaddexp.reduce(m))
    if method=='mean_hellinger32':
        r=np.sqrt(np.exp(z-np.logaddexp.reduce(z,axis=-1,keepdims=True)))-np.sqrt(rho)
    else:
        delta=z-m
        r=(delta-(delta@rho)[:,None])*np.sqrt(rho)
    chosen=[]
    for step in range(3):
        score=(r*r).sum(-1)
        score[chosen]=-np.inf
        pick=a if method=='mean_anchor_fisher32' and step==0 else int(score.argmax())
        chosen.append(pick)
        basis=r[pick]/np.sqrt(r[pick]@r[pick]+1e-12)
        r=r-(r@basis)[:,None]*basis
    return chosen


@unittest.skipIf(torch is None, 'optional PyTorch numerical tests')
class MeanSeedTests(unittest.TestCase):
    def test_scalar_greedy_and_anchor(self):
        z=np.random.default_rng(31).normal(size=(5,8,7))
        m=z.mean(1); anchors=np.arange(5)
        for method in MEAN_METHODS:
            actual=mean_seed_indices(torch.tensor(z),torch.tensor(m),anchors,method).numpy()
            expected=np.array([scalar_seeds(zz,mm,a,method) for zz,mm,a in zip(z,m,anchors)])
            np.testing.assert_array_equal(actual,expected)
            self.assertTrue(all(len(set(row))==3 for row in actual))
            if method=='mean_anchor_fisher32': np.testing.assert_array_equal(actual[:,0],anchors)

    def test_fixed_cell_mean_identity(self):
        rng=np.random.default_rng(4)
        q,k=rng.normal(size=(9,6)),rng.normal(size=(7,6))
        z=q@k.T/np.sqrt(6)
        m=q.mean(0)@k.T/np.sqrt(6)
        np.testing.assert_allclose(m,z.mean(0),atol=1e-15)
        logp=z-np.logaddexp.reduce(z,axis=-1,keepdims=True)
        logrho=m-np.logaddexp.reduce(m); rho=np.exp(logrho)
        optimum=(rho*(logrho-logp)).sum(-1).mean()
        for _ in range(20):
            w=rng.dirichlet(np.ones(7))
            objective=(w*(np.log(w)-logp)).sum(-1).mean()
            np.testing.assert_allclose(objective-optimum,w@(np.log(w)-logrho),atol=1e-14)
            self.assertGreaterEqual(objective+1e-14,optimum)

    def test_duplicate_fill_and_shift(self):
        z=torch.zeros(1,7,5,dtype=torch.float64)
        for method in MEAN_METHODS:
            out=mean_seed_indices(z,z[:,0],[3],method)
            self.assertEqual(out.tolist(),[[3,0,1]] if 'anchor' in method else [[0,1,2]])
        rng=torch.Generator().manual_seed(2)
        z=torch.randn(4,7,5,dtype=torch.float64,generator=rng);m=z.mean(1)
        for method in MEAN_METHODS:
            before=mean_seed_indices(z,m,[0,1,2,3],method)
            after=mean_seed_indices(z+torch.arange(7)[None,:,None]*5,m+13,[0,1,2,3],method)
            torch.testing.assert_close(before,after)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            mean_seed_indices(torch.zeros(1,2,5),torch.zeros(1,5),[0],MEAN_METHODS[0])


class MeanKernelContractTests(unittest.TestCase):
    def test_actual_mean_pointer_mapping_and_default_off(self):
        source=Path(__file__).resolve().parents[1]/'attention/hard_c4_wan21_14b_720p_triton.py'
        kernel=next(n for n in ast.walk(ast.parse(source.read_text()))
                    if isinstance(n,ast.FunctionDef) and n.name=='_hard_c4_720p_temporal_shared_merge_kernel')
        defaults=dict(zip([a.arg for a in kernel.args.args][-len(kernel.args.defaults):],kernel.args.defaults))
        self.assertIs(ast.literal_eval(defaults['MEAN_SEED_ZERO']),False)
        self.assertIsNone(ast.literal_eval(defaults['mean_seed_ptr']))
        block=next(n for n in ast.walk(kernel) if isinstance(n,ast.If)
                   and isinstance(n.test,ast.Name) and n.test.id=='MEAN_SEED_ZERO')
        original=np.full((64,128),-7)
        columns=np.repeat(np.arange(16)+32,4);seeds=np.arange(64)%4
        scope=dict(batch=1,Q_OUTER=21,merge_query_outer=18,COLS=80,
                   merge_query_cols=columns,TOTAL_HEADS=40,global_head=33,
                   D_HEAD=128,dims=np.arange(128),seed_ids=seeds,
                   mean_seed_ptr=0,seed_query=original.copy(),
                   tl=SimpleNamespace(load=lambda offsets:offsets+1,where=np.where))
        exec(compile(ast.Module(body=block.body,type_ignores=[]),'<actual mean kernel>','exec'),scope)
        expected=original.copy()
        for i in range(64):
            if seeds[i]==0:
                expected[i]=(((1*21+18)*80+columns[i])*40+33)*128+np.arange(128)+1
        np.testing.assert_array_equal(scope['seed_query'],expected)


if __name__=='__main__': unittest.main()
