"""Independent NumPy oracle versus actual native padded-45 kernel AST."""
import unittest
import numpy as np
from tests.test_affine_hellinger_seed import actual_greedy
from analysis.quality_first_mode_contract import (
    method_contract,quality_first_token_merging_args,validate_token_merging_args,
)


def oracle(z,anchor):
    rho=np.exp(z[:,anchor]-np.logaddexp.reduce(z[:,anchor],axis=-1,keepdims=True))
    delta=z-z[:,anchor:anchor+1]
    x=(delta-(delta*rho[:,None]).sum(-1,keepdims=True))*np.sqrt(rho[:,None])
    picks=np.full((len(z),4),anchor,dtype=np.int64)
    scores=[]
    for step in range(1,4):
        score=np.sum(x*x,-1);scores.append(score.copy())
        for bank in range(len(z)): score[bank,picks[bank,:step]]=-np.inf
        pick=score.argmax(-1);picks[:,step]=pick
        basis=x[np.arange(len(z)),pick]
        basis/=np.sqrt(np.sum(basis*basis,-1,keepdims=True)+1e-12)
        x-=(x*basis[:,None]).sum(-1,keepdims=True)*basis[:,None]
    return picks,np.stack(scores,axis=1)


class NativeAnchorFullGridTests(unittest.TestCase):
    def test_padded_native_ast_matches_independent_anchor_rule(self):
        rng=np.random.default_rng(506)
        for groups in (1,2,4):
            for anchor in (0,22,44):
                z=rng.normal(size=(groups,45,16))*4
                p,s=actual_greedy(z,anchor,feature_mode='fisher_anchor')
                expected,scores=oracle(z,anchor)
                np.testing.assert_array_equal(p,expected)
                np.testing.assert_allclose(s,scores,atol=1e-11,rtol=1e-11)
                self.assertTrue(np.all(p<45))

    def test_zeros_keep_anchor_and_distinct_valid_rows(self):
        p,s=actual_greedy(np.zeros((4,45,16)),44,feature_mode='fisher_anchor')
        np.testing.assert_array_equal(p,np.tile([44,0,1,2],(4,1)))
        np.testing.assert_array_equal(s,0.)

    def test_video_contract_changes_only_candidate_pool(self):
        old=quality_first_token_merging_args('fisher_anchor_rf0')
        new=quality_first_token_merging_args('fisher_anchor_full45_rf0')
        self.assertEqual(new,old|{'seed_mode':old['seed_mode'].replace('_c32_','_c45_')})
        d=method_contract('fisher_anchor_full45_rf0')
        self.assertEqual(d['selector_override'],method_contract('fisher_anchor_rf0')['selector_override'])
        validate_token_merging_args(new,'fisher_anchor_full45_rf0',dtype='bf16',
                                   selector_override=d['selector_override'])


if __name__=='__main__':unittest.main()
