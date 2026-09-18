"""Small exact entropy-compression counterexample; not a video prediction."""
import unittest
import numpy as np


def compressed_log_partition(query, seed_queries):
    keys=np.array([-1.,1.,0.])
    seed_logits=np.asarray(seed_queries)[:,None]*keys
    logp=seed_logits-np.logaddexp.reduce(seed_logits,axis=1,keepdims=True)
    owner=logp[:,:2].argmax(axis=0)
    logits=[0.]  # distinct SQ1 at key0
    for c in range(len(seed_queries)):
        members=np.flatnonzero(owner==c)
        if not len(members):continue
        z=seed_logits[c,members]
        w=np.exp(z-np.logaddexp.reduce(z))
        entropy=-np.sum(w*np.log(w))
        logits.append(query*np.dot(w,keys[members])+entropy)
    return np.logaddexp.reduce(logits)


class RF0InformationLimitTests(unittest.TestCase):
    def test_unobserved_later_row_changes_optimal_set(self):
        for special in [.01,.1,1.,5.,20.]:
            original=np.logaddexp.reduce([-special,special,0.])
            missing=compressed_log_partition(special,[0.,0.,0.,0.])
            included=compressed_log_partition(special,[0.,special,0.,0.])
            self.assertAlmostEqual(missing,np.log(3.),places=13)
            self.assertGreater(original-missing,0.)
            self.assertAlmostEqual(included,original,places=13)
            self.assertAlmostEqual(compressed_log_partition(0.,[0.,special,0.,0.]),
                                   np.log(3.),places=13)


if __name__=='__main__':unittest.main()
