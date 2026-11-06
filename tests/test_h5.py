"""Numerical regression checks for H5 clustering and grouped inference."""
from pathlib import Path
import sys
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cluster_trajectories import bh, stability
from h5_metadata import statistic


class H5Tests(unittest.TestCase):
    def test_covariance_uses_matching_one_dimensional_coordinates(self):
        rng = np.random.default_rng(2)
        a = rng.dirichlet(np.ones(3), size=80)
        b = rng.random(80)
        expected = np.sum(np.mean((a - a.mean(0)) * (b - b.mean())[:, None], axis=0) ** 2)
        np.testing.assert_allclose(statistic(a, b), expected)
        self.assertEqual(statistic(a, np.zeros(80)), 0)

    def test_permutation_batch_matches_individual_statistics(self):
        rng = np.random.default_rng(3)
        a = rng.dirichlet(np.ones(4), size=75)
        b = rng.random(75)
        perms = np.array([rng.permutation(75) for _ in range(20)])
        fast = np.sum(((b[perms] - b.mean()) @ (a - a.mean(0)) / len(a)) ** 2, axis=1)
        np.testing.assert_allclose(fast, [statistic(a, b[p]) for p in perms])

    def test_multiple_comparison_adjustment(self):
        np.testing.assert_allclose(bh([.01, .04, .03]), [.03, .04, .04])

    def test_group_resampling_recovers_separated_clusters(self):
        rng = np.random.default_rng(2)
        x = np.repeat([[-5., 0], [5., 0]], 50, axis=0) + rng.normal(scale=.1, size=(100, 2))
        with threadpool_limits(limits=1):
            avg, minimum, reference_n = stability(x, np.arange(100).astype(str), 2)
        self.assertGreater(avg, .99)
        self.assertGreater(minimum, .99)
        self.assertEqual(reference_n, 20)


if __name__ == "__main__":
    unittest.main()
