"""Independent feature-space checks for the Gram algebra used in the pilot."""

import itertools
import json
import unittest

import numpy as np

from sgd_memorization.geometry import analyze_gram, minibatch_moments, private_geometry


def explicit_moments(features, n_examples, n_outputs, batch_size):
    example_hessians = []
    for i in range(n_examples):
        block = features[:, i * n_outputs : (i + 1) * n_outputs]
        example_hessians.append(block @ block.T)
    full = sum(example_hessians) / n_examples
    batches = list(itertools.combinations(range(n_examples), batch_size))
    second = np.mean([
        np.sum((sum(example_hessians[i] for i in batch) / batch_size)**2)
        for batch in batches
    ])
    return float(np.trace(full)), float(second)


class GeometryTests(unittest.TestCase):
    def test_private_residuals_and_projected_moments_match_feature_space(self):
        rng = np.random.default_rng(37)
        features = rng.normal(size=(8, 6))
        k = features.T @ features
        n, c, b, eta = 3, 2, 2, 0.03
        summary, rows = analyze_gram(k, n, c, eta, b, rankone_steps=0)
        tau, second = explicit_moments(features, n, c, b)
        self.assertAlmostEqual(summary["trace_hessian"], tau, places=11)
        self.assertAlmostEqual(summary["minibatch_second_moment_trace"], second, places=10)
        for m, row in enumerate(rows):
            others = np.delete(features, m, axis=1)
            coefficients = np.linalg.lstsq(others, features[:, m], rcond=None)[0]
            residual = features[:, m] - others @ coefficients
            t = float(residual @ residual)
            self.assertAlmostEqual(row["private_sqnorm"], t, places=10)
            projector = np.eye(features.shape[0]) - np.outer(residual, residual) / t
            projected = projector @ features
            shared_tau, shared_second = explicit_moments(projected, n, c, b)
            self.assertAlmostEqual(row["shared_trace"], shared_tau, places=10)
            self.assertAlmostEqual(row["shared_second_moment_trace"], shared_second, places=10)
            self.assertAlmostEqual(row["shared_slack"], 2 * shared_tau - eta * shared_second, places=10)
            edit = residual / t
            expected = np.eye(features.shape[1])[m]
            np.testing.assert_allclose(features.T @ edit, expected, atol=1e-11)
            self.assertAlmostEqual(edit @ edit, row["unit_tangent_edit_sqnorm"], places=10)
        json.dumps({"summary": summary, "coordinates": rows}, allow_nan=False)

    def test_multiclass_minibatch_moment_includes_output_cross_terms(self):
        rng = np.random.default_rng(13)
        features = rng.normal(size=(4, 12))
        k = features.T @ features
        for b in (1, 2, 3, 4):
            moment = minibatch_moments(k, 4, 3, b)
            tau, second = explicit_moments(features, 4, 3, b)
            self.assertAlmostEqual(moment["trace_hessian"], tau, places=11)
            self.assertAlmostEqual(moment["minibatch_second_moment_trace"], second, places=10)

    def test_orthogonal_private_condition_is_tight(self):
        k = np.diag([4.0, 1.0])
        summary, rows = analyze_gram(k, 2, 1, 0.5, 1, rankone_steps=0)
        np.testing.assert_allclose([r["private_sqnorm"] for r in rows], [4, 1])
        self.assertAlmostEqual(rows[0]["universal_ratio"], 1.0)
        self.assertAlmostEqual(summary["top_hessian_rankone_ratio"], 1.0)
        self.assertAlmostEqual(summary["top_hessian_rankone_critical_learning_rate"], 0.5)
        _, beyond = analyze_gram(k, 2, 1, 0.51, 1, rankone_steps=0)
        self.assertGreater(beyond[0]["universal_ratio"], 1)

    def test_rotated_example_recovers_stronger_boundary(self):
        rotation = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2)
        features = np.diag([2.0, 1.0]) @ rotation
        k = features.T @ features
        summary, rows = analyze_gram(k, 2, 1, 0.8, 1, rankone_steps=0)
        np.testing.assert_allclose([r["private_sqnorm"] for r in rows], [1.6, 1.6], atol=1e-12)
        np.testing.assert_allclose([r["shared_sqnorm"] for r in rows], [0.9, 0.9], atol=1e-12)
        np.testing.assert_allclose([r["shared_private_ratio"] for r in rows], [1, 1], atol=1e-12)
        self.assertAlmostEqual(summary["max_universal_private_ratio"], 0.64)
        self.assertAlmostEqual(summary["top_hessian_rankone_critical_learning_rate"], 1.0)
        _, beyond = analyze_gram(k, 2, 1, 0.81, 1, rankone_steps=0)
        self.assertGreater(beyond[0]["shared_private_ratio"], 1)

    def test_duplicate_gradients_are_infeasible_without_hiding_independent_coordinate(self):
        features = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        result = private_geometry(features.T @ features, 3)
        self.assertEqual(result["rank"], 2)
        self.assertEqual(result["isolated_edit_feasible"], [False, False, True])
        np.testing.assert_allclose(result["private_sqnorm"], [0, 0, 4], atol=1e-12)
        self.assertEqual(result["unit_tangent_edit_sqnorm"][:2], [None, None])
        self.assertAlmostEqual(result["unit_tangent_edit_sqnorm"][2], 0.25)

    def test_unequal_scale_dependencies_do_not_become_feasible(self):
        # A small second column still anchors the entire direction of the
        # first: no nonzero isolated edit exists for either coordinate.
        # Testing squared null mass used to incorrectly declare column 0
        # feasible and give it unit private norm.
        for ratio in (1e-5, 1e-10):
            for scale in (1e-5, 1.0, 1e5):
                features = scale * np.array([[1.0, ratio, 0.0], [0.0, 0.0, 2.0]])
                for rank_rtol in (1e-10, 1e-3):
                    with self.subTest(ratio=ratio, scale=scale, rank_rtol=rank_rtol):
                        result = private_geometry(features.T @ features, 3, rank_rtol=rank_rtol)
                        self.assertEqual(result["rank"], 2)
                        self.assertEqual(result["isolated_edit_feasible"], [False, False, True])
                        np.testing.assert_allclose(result["private_sqnorm"], [0, 0, 4 * scale**2])
                        self.assertEqual(result["unit_tangent_edit_sqnorm"][:2], [None, None])
                        self.assertGreater(
                            result["isolated_edit_range_residual"][0],
                            result["isolated_edit_feasibility_tolerance"],
                        )

    def test_near_dependency_respects_reported_numerical_rank(self):
        for separation, expected_rank in ((1e-3, 2), (1e-6, 1)):
            features = np.array([[1.0, 1.0], [0.0, separation]])
            result = private_geometry(features.T @ features, 2)
            self.assertEqual(result["rank"], expected_rank)
            if expected_rank == 2:
                self.assertEqual(result["isolated_edit_feasible"], [True, True])
                np.testing.assert_allclose(
                    result["private_sqnorm"],
                    [separation**2 / (1 + separation**2), separation**2],
                    rtol=1e-8,
                )
            else:
                # Mathematically independent, but the default rank cutoff
                # deliberately discards the nearly singular direction.
                self.assertEqual(result["isolated_edit_feasible"], [False, False])
                self.assertEqual(result["unit_tangent_edit_sqnorm"], [None, None])
                resolved = private_geometry(features.T @ features, 2, rank_rtol=1e-14)
                self.assertEqual(resolved["rank"], 2)
                self.assertEqual(resolved["isolated_edit_feasible"], [True, True])

    def test_zero_gram_is_infeasible_and_json_safe(self):
        summary, rows = analyze_gram(np.zeros((2, 2)), 2, 1, 0.1, 1, rankone_steps=0)
        self.assertEqual(summary["numerical_rank"], 0)
        self.assertEqual(summary["isolated_edit_feasible_count"], 0)
        self.assertIsNone(summary["optimized_rankone_critical_learning_rate"])
        self.assertTrue(all(r["unit_tangent_edit_sqnorm"] is None for r in rows))
        json.dumps({"summary": summary, "coordinates": rows}, allow_nan=False)

    def test_optimized_rankone_matches_known_orthogonal_optimum(self):
        summary, _ = analyze_gram(np.diag([4.0, 1.0]), 2, 1, 0.5, 1,
                                  rankone_steps=30, rankone_restarts=3, seed=2)
        self.assertAlmostEqual(summary["optimized_rankone_ratio"], 1.0, places=10)
        self.assertGreaterEqual(summary["optimized_rankone_ratio"], summary["top_hessian_rankone_ratio"])
        self.assertFalse(summary["optimized_rankone_is_global_certificate"])

    def test_full_batch_trace_and_sharpness_scaling(self):
        summary, _ = analyze_gram(np.diag([4.0, 1.0]), 2, 1, 1.0, 2, rankone_steps=0)
        self.assertEqual(summary["p"], 0)
        self.assertAlmostEqual(summary["full_hessian_sharpness_ratio"], 1)
        self.assertAlmostEqual(summary["top_hessian_rankone_ratio"], 1)

    def test_bad_inputs_raise(self):
        for k in (np.array([[1.0, 0.3], [0.0, 1.0]]), np.diag([1.0, -1.0])):
            with self.assertRaises(ValueError):
                analyze_gram(k, 2, 1, 0.1, 1, rankone_steps=0)
        with self.assertRaises(ValueError):
            analyze_gram(np.eye(2), 2, 1, 0.1, 3, rankone_steps=0)
        with self.assertRaises(ValueError):
            analyze_gram(np.eye(2), 2, 1, 0.0, 1, rankone_steps=0)


if __name__ == "__main__":
    unittest.main()
