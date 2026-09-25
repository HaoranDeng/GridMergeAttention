import numpy as np
import unittest
try:
    import torch
except ImportError:
    torch = None

from analysis.screen_fisher_seeds_bf16 import (
    TANGENT_METHODS, tangent_pairwise, tangent_seed_indices,
)


def scalar_distance(z, fisher=False):
    logp=z-np.logaddexp.reduce(z,axis=-1,keepdims=True)
    p=np.exp(logp)
    rho=np.exp(z.mean(0)-np.logaddexp.reduce(z.mean(0)))
    result=np.empty((len(z),len(z)))
    for i in range(len(z)):
        for s in range(len(z)):
            if fisher:
                delta=z[i]-z[s]
                result[i,s]=rho@(delta*delta)-(rho@delta)**2
            else:
                result[i,s]=p[s]@(logp[s]-logp[i])
    return result.clip(0)


def scalar_select(z, anchor, rule, count=4):
    d=scalar_distance(z,rule=='fisher_nearest')
    selected=[anchor]
    while len(selected)<count:
        best=d[:,selected].min(1)
        if rule=='tangent_coverage':
            scores=np.array([
                sum(max(np.exp(-d[i,s])-np.exp(-best[i]),0) for i in range(len(z)))
                for s in range(len(z))])
        else:
            scores=best.copy()
        scores[selected]=-np.inf
        selected.append(int(scores.argmax()))
    return selected


def check_independent_scalar_indices(rule):
    z=np.random.default_rng(9).normal(size=(7,9,5))*2
    anchors=np.arange(7)
    actual=tangent_seed_indices(torch.tensor(z),torch.tensor(anchors),rule).numpy()
    expected=np.array([scalar_select(zz,a,rule) for zz,a in zip(z,anchors)])
    np.testing.assert_array_equal(actual,expected)
    for row,a in zip(actual,anchors):
        assert row[0]==a and len(set(row))==4


def check_reverse_kl_orientation_and_tangent_fraction():
    z=np.array([[2.,-1.,1.],[0.,3.,-2.],[1.,2.,-1.]])
    got=tangent_pairwise(torch.tensor(z[None]))[0].numpy()
    np.testing.assert_allclose(got,scalar_distance(z),atol=1e-14)
    assert not np.allclose(got,got.T)
    p=np.exp(z-np.logaddexp.reduce(z,axis=-1,keepdims=True))
    for i in range(3):
        for s in range(3):
            tangent=p[s]@z[i]-p[s]@np.log(p[s])
            np.testing.assert_allclose(np.exp(-got[i,s]),
                np.exp(tangent-np.logaddexp.reduce(z[i])),atol=1e-14)


def check_shift_invariance(fisher):
    z=torch.randn(4,8,6,dtype=torch.float64,generator=torch.Generator().manual_seed(3))
    offset=torch.arange(8,dtype=torch.float64)[None,:,None]*20
    torch.testing.assert_close(tangent_pairwise(z,fisher=fisher),
        tangent_pairwise(z+offset,fisher=fisher),atol=1e-11,rtol=1e-11)


def check_duplicates_have_deterministic_distinct_fill(rule):
    z=torch.zeros(1,6,4,dtype=torch.float64)
    assert tangent_seed_indices(z,torch.tensor([3]),rule).tolist()==[[3,0,1,2]]


def check_coverage_monotonicity_and_diminishing_returns():
    z=torch.randn(1,8,5,dtype=torch.float64,generator=torch.Generator().manual_seed(6))
    a=(-tangent_pairwise(z)[0]).exp().numpy()
    quality=lambda s:a[:,s].max(1).sum()
    small,big,extra=[0],[0,2,4],6
    assert quality(big)>=quality(small)
    assert quality(small+[extra])-quality(small)>=quality(big+[extra])-quality(big)-1e-12


def check_invalid_count_and_anchor():
    z=torch.zeros(1,3,2)
    with unittest.TestCase().assertRaises(ValueError): tangent_seed_indices(z,[0],'tangent_coverage',count=4)
    with unittest.TestCase().assertRaises(ValueError): tangent_seed_indices(z,[3],'tangent_coverage',count=2)


@unittest.skipIf(torch is None, 'optional PyTorch numerical tests')
class TangentSeedTests(unittest.TestCase):
    test_reverse_kl_orientation_and_tangent_fraction = staticmethod(check_reverse_kl_orientation_and_tangent_fraction)
    test_coverage_monotonicity_and_diminishing_returns = staticmethod(check_coverage_monotonicity_and_diminishing_returns)
    test_invalid_count_and_anchor = staticmethod(check_invalid_count_and_anchor)

    def test_independent_scalar_indices(self):
        for rule in TANGENT_METHODS:
            with self.subTest(rule=rule): check_independent_scalar_indices(rule)

    def test_shift_invariance(self):
        for fisher in (False,True):
            with self.subTest(fisher=fisher): check_shift_invariance(fisher)

    def test_duplicates_have_deterministic_distinct_fill(self):
        for rule in TANGENT_METHODS:
            with self.subTest(rule=rule): check_duplicates_have_deterministic_distinct_fill(rule)


if __name__ == '__main__': unittest.main()
