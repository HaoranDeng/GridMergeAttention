"""Execute native kernel AST against an independent concatenated NumPy GS."""
import unittest
import numpy as np
from tests.test_affine_hellinger_seed import actual_greedy
from analysis.quality_first_mode_contract import (
    BF16_MODES, method_contract, quality_first_token_merging_args,
    validate_token_merging_args,
)


def concat_reference(z, anchor):
    center=z.mean(axis=1)
    rho=np.exp(center-center.max(axis=-1,keepdims=True))
    rho/=rho.sum(axis=-1,keepdims=True)
    delta=z-z[:,anchor:anchor+1]
    x=(delta-(delta*rho[:,None]).sum(-1,keepdims=True))*np.sqrt(rho[:,None])
    residual=x.transpose(1,0,2).reshape(32,64).copy()
    chosen=[anchor]
    for step in range(3):
        score=(residual**2).sum(-1)
        score[chosen]=-np.inf
        pick=int(score.argmax());chosen.append(pick)
        basis=residual[pick]/np.sqrt(np.sum(residual[pick]**2)+1e-12)
        residual-=np.outer(residual@basis,basis)
    return np.tile(chosen,(4,1))


class NativeJointFisherTests(unittest.TestCase):
    def test_actual_ast_equals_explicit_concatenation(self):
        rng=np.random.default_rng(906)
        for seed in range(12):
            z=rng.normal(size=(4,32,16))*(1+seed)
            actual,_=actual_greedy(z,seed,feature_mode='fisher_joint4')
            np.testing.assert_array_equal(actual,concat_reference(z,seed))

    def test_gauge_invariance_and_degenerate_distinct_rows(self):
        z=np.zeros((4,32,16))
        actual,_=actual_greedy(z,7,feature_mode='fisher_joint4')
        np.testing.assert_array_equal(actual,np.tile([7,0,1,2],(4,1)))
        rng=np.random.default_rng(98)
        z=rng.normal(size=z.shape)
        actual,_=actual_greedy(z,7,feature_mode='fisher_joint4')
        shifted,_=actual_greedy(z+rng.normal(size=(4,32,1))*100,7,
                                feature_mode='fisher_joint4')
        np.testing.assert_array_equal(actual,shifted)

    def test_video_contract_preserves_substrate(self):
        method='fisher_joint4_rf0'
        d=method_contract(method)
        self.assertEqual(d['seed_mode'],BF16_MODES['mass_rf0'])
        self.assertTrue(d['selector_override'].endswith('.fisher_joint4_seed_rows_triton'))
        validate_token_merging_args(quality_first_token_merging_args(method),method,
            dtype='bf16',selector_override=d['selector_override'])
        self.assertEqual(d['router_frame'],0)
        self.assertEqual(d['bulk_representatives'],4)
        self.assertTrue(d['all_layer_operator'])


if __name__=='__main__': unittest.main()
