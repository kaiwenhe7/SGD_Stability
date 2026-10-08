"""Compare the iterative Gram diagnostic to explicit minibatch operators."""

import itertools
import json
import unittest

import numpy as np
from scipy.linalg import eigh

from sgd_memorization.covariance import exact_generalized_sharpness


def symmetric_basis(dimension):
    basis = []
    for i in range(dimension):
        for j in range(i, dimension):
            matrix = np.zeros((dimension, dimension))
            if i == j:
                matrix[i, i] = 1
            else:
                matrix[i, j] = matrix[j, i] = 1 / np.sqrt(2)
            basis.append(matrix)
    return np.asarray(basis)


def explicit_threshold(features, n_examples, n_outputs, batch_size):
    """Enumerate batches and construct C,D without the Gram block identity."""
    d = features.shape[0]
    h_i = []
    for i in range(n_examples):
        block = features[:, i * n_outputs : (i + 1) * n_outputs]
        h_i.append(block @ block.T)
    h = sum(h_i) / n_examples
    batches = [sum(h_i[i] for i in batch) / batch_size
               for batch in itertools.combinations(range(n_examples), batch_size)]
    basis = symmetric_basis(d)
    c = np.asarray([[np.sum(left * (h @ right + right @ h) / 2)
                     for right in basis] for left in basis])
    stochastic = np.asarray([
        [np.mean([np.sum(left * (hb @ right @ hb)) for hb in batches])
         for right in basis] for left in basis
    ])
    largest = float(eigh(stochastic, c, eigvals_only=True)[-1])
    return 2 / largest, h, batches


class CovarianceTests(unittest.TestCase):
    def test_matches_exhaustive_multiclass_symmetric_operator(self):
        rng = np.random.default_rng(72)
        features = rng.normal(size=(3, 8))
        gram = features.T @ features
        for batch in (1, 2, 3, 4):
            expected, _, _ = explicit_threshold(features, 4, 2, batch)
            result = exact_generalized_sharpness(gram, 4, 2, batch, tol=1e-11)
            self.assertAlmostEqual(result["critical_learning_rate"], expected, places=11)
            self.assertEqual(result["numerical_rank"], 3)
            self.assertEqual(result["symmetric_covariance_dimension"], 6)
            self.assertLess(result["eigenpair_relative_residual"], 1e-9)
            self.assertTrue(result["solver_converged"])
            self.assertFalse(result["numerical_rigorous_certificate"])
            json.dumps(result, allow_nan=False)

    def test_matches_independent_full_kronecker_operator(self):
        features = np.array([[2.0, 0.8, 0.1], [0.0, 1.1, 1.3]])
        expected, h, batches = explicit_threshold(features, 3, 1, 2)
        identity = np.eye(2)
        c = (np.kron(h, identity) + np.kron(identity, h)) / 2
        d = sum(np.kron(hb, hb) for hb in batches) / len(batches)
        kronecker_largest = float(eigh(d, c, eigvals_only=True)[-1])
        self.assertAlmostEqual(expected, 2 / kronecker_largest, places=12)
        result = exact_generalized_sharpness(features.T @ features, 3, 1, 2, tol=1e-11)
        self.assertAlmostEqual(result["critical_learning_rate"], expected, places=11)

    def test_rotated_scalar_example_has_point_eight_threshold(self):
        features = np.diag([2.0, 1.0]) @ (np.array([[1, -1], [1, 1]]) / np.sqrt(2))
        result = exact_generalized_sharpness(features.T @ features, 2, 1, 1, tol=1e-12)
        self.assertAlmostEqual(result["critical_learning_rate"], 0.8, places=12)
        self.assertAlmostEqual(result["generalized_sharpness"], 2.5, places=12)

    def test_orthogonal_scalar_exact_threshold(self):
        result = exact_generalized_sharpness(np.diag([4.0, 1.0]), 2, 1, 1, tol=1e-12)
        self.assertAlmostEqual(result["critical_learning_rate"], 0.5, places=12)

    def test_rank_one_with_multiclass_curvature_and_duplicate_columns(self):
        features = np.array([[1.0, 2.0, 0.3, 0.7, 1.4, 0.0]])
        expected, _, _ = explicit_threshold(features, 3, 2, 1)
        result = exact_generalized_sharpness(features.T @ features, 3, 2, 1)
        self.assertEqual(result["solver"], "analytic_rank_one")
        self.assertAlmostEqual(result["critical_learning_rate"], expected, places=12)

    def test_full_batch_shortcut_and_zero_rank(self):
        result = exact_generalized_sharpness(np.diag([4.0, 1.0]), 2, 1, 2)
        self.assertEqual(result["solver"], "analytic_full_batch")
        self.assertEqual(result["critical_learning_rate"], 1)
        empty = exact_generalized_sharpness(np.zeros((2, 2)), 2, 1, 1)
        self.assertEqual(empty["generalized_sharpness"], 0)
        self.assertIsNone(empty["critical_learning_rate"])
        self.assertTrue(empty["active_threshold_unbounded"])
        json.dumps(empty, allow_nan=False)

    def test_rank_limit_and_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "exceeds max_rank"):
            exact_generalized_sharpness(np.eye(4), 4, 1, 1, max_rank=3)
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            exact_generalized_sharpness(np.diag([1, -1]), 2, 1, 1)
        with self.assertRaises(ValueError):
            exact_generalized_sharpness(np.eye(2), 2, 1, 3)


if __name__ == "__main__":
    unittest.main()
