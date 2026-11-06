"""P2 covariance statistics, group resampling and matching regressions."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from compare_registers import eigstats, quotas, bootstrap_weights, profiles, matched, shuffled_profiles
from analyze import overlap_matrix


class P2Tests(unittest.TestCase):
    def test_batch_pca_matches_direct_svd(self):
        x=np.random.default_rng(3).normal(size=(3,12,25))
        for standardized in [False,True]:
            result=eigstats(x,standardized)
            for j,a in enumerate(x):
                a=a-a.mean(0)
                if standardized:a=a/a.std(0)
                ev=np.linalg.svd(a,compute_uv=False)**2
                self.assertAlmostEqual(result['pc2_remaining'][j],ev[1]/ev[1:].sum())
                self.assertAlmostEqual(result['var2'][j],ev[:2].sum()/ev.sum())

    def test_raw_pca_is_rotation_invariant(self):
        rng=np.random.default_rng(9)
        x=rng.normal(size=(4,12,25))
        q,_=np.linalg.qr(rng.normal(size=(25,25)))
        a,b=eigstats(x),eigstats(x@q)
        for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-12)

    def test_quotas_and_matching_are_exact_and_unique(self):
        q=quotas([10,17,28,1,0,0,0,0,0],30)
        self.assertEqual(q.sum(),30)
        ep=pd.DataFrame(dict(stratum=np.repeat(np.arange(4),[10,17,28,1]),
                             n_chunks=6,unit_collection='x'))
        idx,actual=matched(ep,30,np.random.default_rng(0),q=q)
        self.assertEqual(len(np.unique(idx)),30)
        np.testing.assert_array_equal(np.bincount(ep.stratum.iloc[idx],minlength=9),actual)

    def test_resampling_retains_groups_and_stratum_totals(self):
        group=np.repeat(np.arange(12),4)
        strata=np.tile([0,0,1,1],12)
        w,pt,_=bootstrap_weights(group,strata,np.random.default_rng(4),n=100)
        for s in [0,1]:
            np.testing.assert_allclose(w[:,strata==s].sum(1),24)
        # Same author, same stratum must receive identical multiplicities.
        np.testing.assert_allclose(w[:,0],w[:,1])
        np.testing.assert_allclose(w[:,2],w[:,3])
        np.testing.assert_allclose(pt,np.ones(48))

    def test_author_balancing_does_not_reward_productivity(self):
        groups=np.array(['a','a','a','b'])
        _,pt,_=bootstrap_weights(groups,np.zeros(4),np.random.default_rng(1),n=20,equal_author=True)
        self.assertAlmostEqual(pt[:3].sum(),pt[3])

    def test_text_and_word_weight_estimands(self):
        a=np.array([np.full((12,3),1.),np.full((12,3),3.)])
        length=np.array([np.ones(12),np.full(12,3.)])
        np.testing.assert_allclose(profiles(a,length,np.ones((1,2)),'text'),2)
        np.testing.assert_allclose(profiles(a,length,np.ones((1,2)),'word'),2.5)

    def test_chunk_permutation_preserves_total_topic_mass(self):
        rng=np.random.default_rng(1)
        count=rng.random((12,5)).astype(np.float32)
        length=np.arange(1,13).astype(np.float32)
        w=overlap_matrix(6,12)
        sh=np.array([(w@count[i:i+6])/(w@length[i:i+6])[:,None] for i in [0,6]])
        rep=dict(starts=np.array([0,6]),n=np.array([6,6]),shares=sh,
                 raw=dict(counts=count,length=length))
        a,l=shuffled_profiles(rep,np.arange(2),rng)
        np.testing.assert_allclose((a*l[:,:,None]).sum((0,1)),count.sum(0),rtol=1e-6)
        np.testing.assert_allclose(l.sum(),length.sum())


if __name__=='__main__':unittest.main()
