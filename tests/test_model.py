"""Regression checks against explicit autograd Jacobians on tiny networks."""

import unittest

import numpy as np
import torch

from sgd_memorization.model import (
    OneHiddenMLP,
    evaluate_isolated_edits,
    prediction_gram,
    tangent_parameter_edit,
)


def explicit_jacobian(model, x):
    parameters = tuple(model.parameters())
    outputs = model(x).reshape(-1)
    return torch.stack(
        [
            torch.cat([gradient.reshape(-1) for gradient in torch.autograd.grad(output, parameters, retain_graph=True)])
            for output in outputs
        ]
    )


class ModelGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = OneHiddenMLP(input_dim=3, hidden_width=7, output_dim=2).double()
        self.x = torch.randn(4, 3, dtype=torch.float64)

    def test_analytic_gram_matches_autograd_and_sample_order(self):
        jacobian = explicit_jacobian(self.model, self.x)
        kernel = prediction_gram(self.model, self.x)
        self.assertEqual(kernel.dtype, np.float64)
        self.assertEqual(kernel.shape, (8, 8))
        np.testing.assert_allclose(kernel, (jacobian @ jacobian.T).detach().numpy(), rtol=1e-12, atol=1e-12)

    def test_parameter_edit_matches_jacobian_transpose_and_tangent_change(self):
        jacobian = explicit_jacobian(self.model, self.x)
        coefficients = torch.randn(4, 2, dtype=torch.float64)
        edit = tangent_parameter_edit(self.model, self.x, coefficients)
        flat_edit = torch.cat([edit[name].reshape(-1) for name, _ in self.model.named_parameters()])
        torch.testing.assert_close(flat_edit, jacobian.T @ coefficients.reshape(-1), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            (jacobian @ flat_edit).detach().numpy(),
            prediction_gram(self.model, self.x) @ coefficients.reshape(-1).numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_isolated_edit_cost_target_leakage_and_unchanged_model(self):
        kernel = prediction_gram(self.model, self.x)
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        before_outputs = self.model(self.x).detach().clone()
        amplitudes = [1e-5, -2e-5]
        records = evaluate_isolated_edits(self.model, self.x, kernel, [1, (2, 0)], amplitudes)
        self.assertEqual(len(records), 4)
        self.assertEqual((records[2]["sample_index"], records[2]["output_index"]), (2, 0))
        for record in records:
            self.assertAlmostEqual(record["parameter_cost"], record["tangent_pred_cost"], places=14)
            self.assertLess(abs(record["achieved_target_change"] - record["amplitude"]), 2e-8)
            self.assertLess(record["leakage_max"], 2e-8)
            self.assertAlmostEqual(
                record["leakage_max_relative"], record["leakage_max"] / abs(record["amplitude"])
            )
        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(torch.equal(self.model(self.x).detach(), before_outputs))

    def test_singular_duplicate_inputs_reject_singleton_edits(self):
        x = self.x[:1].repeat(2, 1)
        kernel = prediction_gram(self.model, x)
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            evaluate_isolated_edits(self.model, x, kernel, [0], [0.1])

    def test_zero_relu_derivative_matches_autograd(self):
        with torch.no_grad():
            self.model.fc1.weight.zero_()
            self.model.fc1.bias.zero_()
        jacobian = explicit_jacobian(self.model, self.x)
        np.testing.assert_allclose(prediction_gram(self.model, self.x), (jacobian @ jacobian.T).detach().numpy())

    def test_float32_gram_preserves_native_zero_relu_gate(self):
        model = OneHiddenMLP(input_dim=2, hidden_width=1, output_dim=1)
        x = torch.ones((1, 2), dtype=torch.float32)
        with torch.no_grad():
            model.fc1.weight.copy_(torch.tensor([[0.1, 0.4]]))
            model.fc1.bias.fill_(-0.5)
            model.fc2.weight.fill_(2.0)
            model.fc2.bias.zero_()
        self.assertEqual(float(model.fc1(x).detach()), 0.0)
        # Those same stored parameters cross the boundary if promoted before
        # evaluating fc1. The Gram must describe the native float32 model.
        promoted_preactivation = x.double() @ model.fc1.weight.detach().double().T + model.fc1.bias.detach().double()
        self.assertGreater(float(promoted_preactivation), 0.0)
        jacobian = explicit_jacobian(model, x).double()
        np.testing.assert_array_equal(prediction_gram(model, x), [[1.0]])
        np.testing.assert_array_equal(prediction_gram(model, x), (jacobian @ jacobian.T).numpy())
        edit = tangent_parameter_edit(model, x, np.array([[2.0]]))
        flat_edit = torch.cat([edit[name].reshape(-1) for name, _ in model.named_parameters()]).double()
        torch.testing.assert_close(flat_edit, jacobian.T[:, 0] * 2, rtol=0, atol=0)

    def test_float32_gram_accumulates_native_jacobian_in_double(self):
        model, x = self.model.float(), self.x.float()
        jacobian = explicit_jacobian(model, x).double()
        kernel = prediction_gram(model, x)
        self.assertEqual(kernel.dtype, np.float64)
        # Autograd rounds individual float32 gradient products, whereas the
        # analytic product rounds only its native forward activations.
        np.testing.assert_allclose(kernel, (jacobian @ jacobian.T).numpy(), rtol=2e-7, atol=2e-8)

    def test_gate_radius_matches_analytic_signed_boundary(self):
        # At x=1, z=0.1 and output weight s=+/-1, J has squared norm
        # 1 + 1 + 0.1**2 + 1 = 3.01. A unit output edit changes z by
        # 2*s/3.01, so its only gate crossing is Delta=-s*0.1505.
        for output_sign in (-1.0, 1.0):
            model = OneHiddenMLP(1, 1, 1).double()
            x = torch.ones((1, 1), dtype=torch.float64)
            with torch.no_grad():
                model.fc1.weight.zero_()
                model.fc1.bias.fill_(0.1)
                model.fc2.weight.fill_(output_sign)
                model.fc2.bias.zero_()
            records = evaluate_isolated_edits(model, x, prediction_gram(model, x), [0], [-0.2, -0.1, 0.1, 0.2])
            for record in records:
                self.assertAlmostEqual(record["baseline_min_abs_preactivation"], 0.1)
                self.assertEqual(record["baseline_zero_preactivation_count"], 0)
                self.assertEqual(record["activation_gate_count"], 1)
                positive, negative = (None, -0.1505) if output_sign > 0 else (0.1505, None)
                self.assertEqual(record["first_gate_crossing_positive_amplitude"], positive)
                self.assertEqual(record["first_gate_crossing_negative_amplitude"], negative)
                crossing_direction = record["amplitude"] * output_sign < 0
                expected_radius = 0.1505 if crossing_direction else None
                self.assertEqual(record["first_gate_crossing_abs_amplitude"], expected_radius)
                self.assertEqual(record["activation_flip_count"], int(crossing_direction and abs(record["amplitude"]) > 0.1505))
                self.assertAlmostEqual(record["parameter_cost"], record["amplitude"]**2 / 3.01)
                self.assertAlmostEqual(record["target_error"], record["achieved_target_change"] - record["amplitude"])

    def test_zero_gate_has_zero_radius_only_in_activating_direction(self):
        model = OneHiddenMLP(1, 1, 1).double()
        x = torch.tensor([[2.0], [0.0]], dtype=torch.float64)
        with torch.no_grad():
            model.fc1.weight.fill_(0.1)
            model.fc1.bias.zero_()
            model.fc2.weight.fill_(1.0)
            model.fc2.bias.zero_()
        records = evaluate_isolated_edits(model, x, prediction_gram(model, x), [0], [1e-4, -1e-4])
        positive, negative = records
        self.assertEqual(positive["baseline_min_abs_preactivation"], 0)
        self.assertEqual(positive["baseline_zero_preactivation_count"], 1)
        self.assertEqual(positive["first_gate_crossing_abs_amplitude"], 0)
        self.assertIsNone(positive["amplitude_to_gate_radius"])
        self.assertEqual(positive["activation_flip_count"], 1)
        self.assertAlmostEqual(negative["first_gate_crossing_abs_amplitude"], 0.2016)
        self.assertEqual(negative["activation_flip_count"], 0)


if __name__ == "__main__":
    unittest.main()
