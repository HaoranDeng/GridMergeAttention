import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch=None
from analysis.refined_fisher_seed import REFINED_METHODS,BREGMAN_METHODS,MEDOID_METHODS,bregman_features,fisher_features,query_medoid_refine,refined_indices


def scalar(z,anchor,method):
    rho=np.exp(z.mean(0)-np.logaddexp.reduce(z.mean(0)))
    delta=z-z[anchor];x=(delta-(delta@rho)[:,None])*np.sqrt(rho)
    r=x.copy();chosen=[anchor]
    for step in range(1,4):
        scores=(r*r).sum(-1);scores[chosen]=-np.inf
        pick=int(scores.argmax())
        if method=='fisher_antipodal' and step==2:
            opp=-x@x[chosen[1]];opp[chosen]=-np.inf
            if opp.max()>0:pick=int(opp.argmax())
        chosen.append(pick)
        basis=r[pick]/np.sqrt(r[pick]@r[pick]+1e-12)
        r-= (r@basis)[:,None]*basis
    initial=chosen.copy()
    volume=lambda choices:np.linalg.det(x[choices[1:]]@x[choices[1:]].T)
    if method=='fisher_maxvol1':
        for slot in range(1,4):
            scores=np.full(len(z),-np.inf)
            for candidate in range(len(z)):
                trial=chosen.copy();trial[slot]=candidate
                if len(set(trial))==4:scores[candidate]=volume(trial)
            if scores.max()>volume(chosen)+1e-10:
                chosen[slot]=int(scores.argmax())
    return chosen,volume(initial),volume(chosen)


@unittest.skipIf(torch is None,'optional PyTorch numerical tests')
class RefinedFisherTests(unittest.TestCase):
    def test_query_medoid_optimal_subproblem(self):
        z=torch.randn(5,9,7,dtype=torch.float64,generator=torch.Generator().manual_seed(51))*2
        a=torch.arange(5);x=fisher_features(z,a)
        initial=torch.tensor([[i,(i+1)%9,(i+3)%9,(i+5)%9] for i in range(5)])
        for method in MEDOID_METHODS:
            result=query_medoid_refine(z,x,initial,method).numpy()
            for bank in range(5):
                zz=z[bank].numpy();xx=x[bank].numpy();old=initial[bank].numpy()
                lp=zz-np.logaddexp.reduce(zz,axis=-1,keepdims=True);p=np.exp(lp)
                distance=np.array([[p[s]@(lp[s]-lp[i]) if method=='query_kl_medoid'
                                    else np.sum((xx[i]-xx[s])**2) for s in range(9)] for i in range(9)])
                labels=distance[:,old].argmin(1)
                expected=old.copy()
                for slot in range(1,4):
                    members=np.where(labels==slot)[0]
                    if not len(members):continue
                    candidates=set(members)|{old[slot]}
                    candidates-=set(np.delete(old,slot))|set(np.delete(expected,slot))
                    costs=distance[members].sum(0)
                    best=min(candidates,key=lambda s:(costs[s],s))
                    if costs[best]<costs[expected[slot]]-1e-10:expected[slot]=best
                np.testing.assert_array_equal(result[bank],expected)
                self.assertEqual(result[bank,0],old[0])
                self.assertEqual(len(set(result[bank])),4)
                self.assertLessEqual(sum(distance[i,result[bank,labels[i]]] for i in range(9)),
                                     sum(distance[i,old[labels[i]]] for i in range(9))+1e-10)
        for method in MEDOID_METHODS:
            self.assertEqual(refined_indices(torch.zeros(1,6,4),[3],method).tolist(),[[3,0,1,2]])

    def test_signed_reverse_kl_identity(self):
        z=torch.randn(4,8,6,dtype=torch.float64,generator=torch.Generator().manual_seed(11))*3
        anchor=torch.arange(4)
        for method in BREGMAN_METHODS:
            x,raw,lp,lr=bregman_features(z,anchor,method)
            expected=2*(lr.exp()[:,None]*(lr[:,None]-lp)).sum(-1)
            torch.testing.assert_close(raw.square().sum(-1),expected,atol=1e-13,rtol=1e-12)
            torch.testing.assert_close(x[torch.arange(4),anchor],torch.zeros(4,6,dtype=torch.float64))
            picks=refined_indices(z,anchor,method)
            self.assertTrue((picks[:,0]==anchor).all())
            self.assertTrue((picks.sort(-1).values.diff(dim=-1)>0).all())
            # Independently construct features and run scalar GS.
            for bank in range(4):
                pp=np.exp(lp[bank].numpy());rr=np.exp(lr[bank].numpy())
                delta=np.log(pp/rr)
                feat=np.sign(delta)*np.sqrt(2*np.maximum(rr*np.log(rr/pp)-rr+pp,0))
                feat-=feat[anchor[bank].item()].copy()
                selected=[anchor[bank].item()]
                for _ in range(3):
                    score=(feat*feat).sum(-1);score[selected]=-np.inf
                    pick=int(score.argmax());selected.append(pick)
                    basis=feat[pick]/np.sqrt(feat[pick]@feat[pick]+1e-12)
                    feat-=(feat@basis)[:,None]*basis
                self.assertEqual(picks[bank].tolist(),selected)

    def test_kl_local_fisher_limit(self):
        z=torch.randn(3,7,5,dtype=torch.float64,generator=torch.Generator().manual_seed(1))
        z=z[:,0,None]+1e-4*z
        a=torch.arange(3)
        _,raw,lp,lr=bregman_features(z,a,'kl_geometric_gs')
        quadratic=(lp-lr[:,None])*lr.exp()[:,None].sqrt()
        torch.testing.assert_close(raw,quadratic,atol=2e-8,rtol=1e-3)

    def test_scalar_determinant_and_signed_selection(self):
        z=np.random.default_rng(101).normal(size=(12,9,7))*2
        anchor=np.arange(12)%9
        for method in REFINED_METHODS:
            actual=refined_indices(torch.tensor(z),anchor,method).numpy()
            expected=[]
            for zz,a in zip(z,anchor):
                rows,before,after=scalar(zz,a,method)
                expected.append(rows)
                if method=='fisher_maxvol1':self.assertGreaterEqual(after+1e-8,before)
            np.testing.assert_array_equal(actual,expected)

    def test_zero_features_and_shift(self):
        for method in REFINED_METHODS:
            self.assertEqual(refined_indices(torch.zeros(1,6,4),[3],method).tolist(),[[3,0,1,2]])
        z=torch.randn(5,7,6,dtype=torch.float64,generator=torch.Generator().manual_seed(7))
        for method in REFINED_METHODS:
            torch.testing.assert_close(refined_indices(z,[0,1,2,3,4],method),
                refined_indices(z+torch.arange(7)[None,:,None]*4,[0,1,2,3,4],method))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):refined_indices(torch.zeros(1,3,4),[0],REFINED_METHODS[0])
        with self.assertRaises(ValueError):refined_indices(torch.zeros(1,5,4),[5],REFINED_METHODS[0])


if __name__=='__main__':unittest.main()
