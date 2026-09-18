"""Execute the actual native front end and GS AST against independent NumPy."""
import unittest

import numpy as np

from tests.test_affine_hellinger_seed import actual_greedy, NativeMetadataTests, MetadataTensor


def scalar_oracle(bank):
    def softmax(x):
        shifted=x-np.max(x)
        return np.exp(shifted)/np.sum(np.exp(shifted))
    # All-pair average reverse KL, independent of the implementation's
    # geometric-barycenter shortcut.
    z=np.asarray(bank,dtype=np.float64)
    logp=z-np.max(z,axis=1,keepdims=True)
    logp-=np.log(np.sum(np.exp(logp),axis=1,keepdims=True))
    p=np.exp(logp)
    costs=[np.mean([np.sum(p[s]*(logp[s]-logp[i])) for i in range(32)])
           for s in range(32)]
    chosen=[int(np.argmin(costs))]
    rho=softmax(z[chosen[0]])
    delta=z-z[chosen[0]]
    residual=np.sqrt(rho)*(delta-np.sum(delta*rho,axis=1,keepdims=True))
    trace=[]
    for step in range(3):
        scores=np.array([np.dot(x,x) for x in residual])
        trace.append(scores.copy())
        scores[chosen]=-np.inf
        pick=int(np.argmax(scores))
        chosen.append(pick)
        if step<2:
            v=residual[pick]/np.sqrt(np.dot(residual[pick],residual[pick])+1e-12)
            residual-=np.outer(residual@v,v)
    return chosen,trace


class NativeBarycentricMathTests(unittest.TestCase):
    def test_exact_first_medoid_and_three_gs_prefixes(self):
        for groups in (1,2,4,8):
            z=np.random.default_rng(35+groups).normal(size=(groups,32,16))*3
            chosen,scores=actual_greedy(z,17,feature_mode="fisher_barycentric_medoid")
            expected=[scalar_oracle(bank) for bank in z]
            np.testing.assert_array_equal(chosen,[pair[0] for pair in expected])
            np.testing.assert_allclose(scores,[pair[1] for pair in expected],atol=2e-12,rtol=2e-12)
            moved,_=actual_greedy(z,1,feature_mode="fisher_barycentric_medoid")
            np.testing.assert_array_equal(chosen,moved)

    def test_uniform_ties_use_first_slots_not_matched_row(self):
        selected,scores=actual_greedy(np.zeros((4,32,16)),19,feature_mode="fisher_barycentric_medoid")
        np.testing.assert_array_equal(selected,[[0,1,2,3]]*4)
        np.testing.assert_array_equal(scores,np.zeros((4,3,32)))

    def test_independent_row_logit_shifts_leave_choices_unchanged(self):
        z=np.random.default_rng(107).normal(size=(4,32,16))
        base,_=actual_greedy(z,3,feature_mode="fisher_barycentric_medoid")
        shifted,_=actual_greedy(z+np.arange(32)[None,:,None]*16,3,feature_mode="fisher_barycentric_medoid")
        np.testing.assert_array_equal(base,shifted)


class NativeBarycentricMetadataTests(unittest.TestCase):
    setUp=NativeMetadataTests.setUp

    def test_no_mean_or_mass_pass_and_c32_only(self):
        result=self.selector(self.q,self.k,feature_mode="fisher_barycentric_medoid")
        self.assertEqual(result.shape,(2,1,5,45,80,3,4))
        self.assertIs(self.launches[0].args[6],self.q)
        self.assertEqual(self.launches[0].kwargs["FEATURE_MODE"],"fisher_barycentric_medoid")
        with self.assertRaises(ValueError):
            self.selector(self.q,self.k,feature_mode="fisher_barycentric_medoid",candidate_count=45)
        with self.assertRaises(TypeError):
            self.selector(MetadataTensor(dtype="fp16"),MetadataTensor((2,5,45,80,3,128),dtype="fp16"),
                          feature_mode="fisher_barycentric_medoid")


if __name__=="__main__": unittest.main()
