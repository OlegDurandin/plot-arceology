"""Regression checks for the author-disjoint ending experiment."""
from pathlib import Path
import sys
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from validate_endings import embedding_features, permute_internal, splits, design
from p1_report import weighted_ap_many
from sklearn.metrics import average_precision_score


class P1Tests(unittest.TestCase):
    def test_bootstrap_average_precision_matches_sklearn_with_ties(self):
        y = np.array([1, 0, 0, 1, 0, 1])
        probability = np.array([.8, .8, .5, .5, .5, .1])
        weights = np.array([[1, 1, 1, 1, 1, 1], [2, 0, 3, 1, 2, 4], [0, 1, 0, 0, 1, 0]])
        result = weighted_ap_many(y, probability, weights)
        np.testing.assert_allclose(result[:2], [average_precision_score(y, probability, sample_weight=w)
                                               for w in weights[:2]])
        self.assertTrue(np.isnan(result[2]))

    def test_shuffle_preserves_ending_composition_and_amplitude(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(30, 6, 20))
        out = permute_internal(x, 101)
        np.testing.assert_array_equal(out[:, -1], x[:, -1])
        np.testing.assert_allclose(out.mean(1), x.mean(1), atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(out - out.mean(1, keepdims=True), axis=(1, 2)),
                                   np.linalg.norm(x - x.mean(1, keepdims=True), axis=(1, 2)))
        self.assertFalse(np.array_equal(out[:, :5], x[:, :5]))

    def test_heldout_rows_do_not_change_fitted_pca_or_training_features(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(40, 6, 24))
        mean = x.mean(1); cov = rng.normal(size=(40, 4)); train = np.arange(25)
        changed = x.copy(); changed[25:] *= 100
        with threadpool_limits(limits=1):
            a, pa = embedding_features(x, mean, cov, train)
            b, pb = embedding_features(changed, mean, cov, train)
        np.testing.assert_allclose(pa.components_, pb.components_)
        for key in a:
            np.testing.assert_allclose(a[key][train], b[key][train])

    def test_outer_and_inner_splits_keep_authors_disjoint(self):
        strata = np.tile(np.arange(4), 40)
        groups = np.repeat(np.arange(80), 2)
        for train, test in splits(strata, groups, 0):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            for a, b in splits(strata[train], groups[train], 100, n=3):
                self.assertFalse(set(groups[train[a]]) & set(groups[train[b]]))

    def test_heldout_vocabulary_and_covariates_do_not_affect_training_design(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(40, 6, 24)); mean = x.mean(1)
        cov = rng.normal(size=(40, 4)); train = np.arange(25)
        text = np.array(["лес дом герой мир рассказ" for _ in range(40)], dtype=object)
        ending = np.array(["дом мир финал" for _ in range(40)], dtype=object)
        altered_text = text.copy(); altered_text[25:] = "редкиймаркер редкиймаркер"
        altered_end = ending.copy(); altered_end[25:] = "секретныймаркер"
        altered_cov = cov.copy(); altered_cov[25:] *= 100
        with threadpool_limits(limits=1):
            a, _ = design(x, mean, cov, text, ending, train)
            b, _ = design(x, mean, altered_cov, altered_text, altered_end, train)
        for key in a:
            left, right = a[key][train], b[key][train]
            if hasattr(left, "toarray"):
                left, right = left.toarray(), right.toarray()
            np.testing.assert_allclose(left, right)


if __name__ == "__main__":
    unittest.main()
