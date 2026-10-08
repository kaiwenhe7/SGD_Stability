"""Analytic reference cases for independent synthetic diagnostics."""

import unittest

import numpy as np

from sgd_memorization.synthetic import (
    controlled_features, direct_coordinate, evolve_exact, explicit_operator, transition_operator,
    orthogonal_concentration,
)


class SyntheticTests(unittest.TestCase):
    def test_orthogonal_threshold_and_exact_coordinate_decay(self):
        features = controlled_features(0, 4, 1)
        operator = explicit_operator(features, 1)
        self.assertAlmostEqual(operator["critical_learning_rate"], .5, places=12)
        eta = .2
        actual = evolve_exact(operator["batches"], eta, np.array([0., 1.]), 8)
        # Each step updates the private coordinate by 1-4 eta with probability 1/2.
        multiplier = (1 + (1 - 4 * eta)**2) / 2
        np.testing.assert_allclose(actual, multiplier**np.arange(9), rtol=1e-13)

    def test_direct_edit_preserves_anchor_and_cost_is_inverse_private_strength(self):
        features = controlled_features(4, .01, .25)
        operator = explicit_operator(features, 1)
        direct = direct_coordinate(features, operator["batches"], .4)
        self.assertAlmostEqual(direct["unit_tangent_edit_sqnorm"], 100., places=9)
        self.assertLess(direct["constraint_max_error"], 1e-13)
        self.assertAlmostEqual(direct["private_sqnorm"], .01, places=13)
        self.assertAlmostEqual(direct["shared_sqnorm"], 4, places=13)

    def test_covariance_transition_crosses_unit_radius_at_threshold(self):
        operator = explicit_operator(controlled_features(1, 1, 1), 1)
        eta = operator["critical_learning_rate"]
        for fraction in (.9, 1, 1.1):
            transition = transition_operator(operator["batches"], eta * fraction, operator["basis"])
            radius = max(abs(np.linalg.eigvalsh(transition)))
            if fraction < 1:
                self.assertLess(radius, 1)
            elif fraction > 1:
                self.assertGreater(radius, 1)
            else:
                self.assertAlmostEqual(radius, 1, places=12)

    def test_orthogonal_singleton_saturation_and_localization_penalty(self):
        rows = orthogonal_concentration({"n_examples": 4, "batch_sizes": [1, 4],
                                         "support_sizes": [1, 4], "learning_rate_fractions": [1., 1.1]})
        for row in rows:
            self.assertAlmostEqual(row["explicit_critical_learning_rate"], 2 * row["batch_size"], places=12)
            self.assertAlmostEqual(row["direct_edit_cost"], 1, places=12)
            if row["support_size"] == 1:
                self.assertAlmostEqual(row["lower_bound_to_cost_ratio"], row["learning_rate_fraction"], places=12)
                self.assertEqual(row["bound_violated"], row["learning_rate_fraction"] > 1)
        singleton, diffuse = rows[:2]
        self.assertGreater(singleton["directional_lower_bound"], diffuse["directional_lower_bound"])


if __name__ == "__main__":
    unittest.main()
