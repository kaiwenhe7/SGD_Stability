"""Independent autograd validation of endpoint residual curvature."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.func import functional_call

from sgd_memorization.endpoint_checks import fixed_gate_hessian_residual, inspect_endpoint
from sgd_memorization.model import OneHiddenMLP, prediction_gram


class EndpointCheckTests(unittest.TestCase):
    def test_residual_block_norm_matches_full_autograd_hessian(self):
        torch.manual_seed(313)
        model = OneHiddenMLP(2, 3, 2).double()
        with torch.no_grad():
            model.fc1.bias.copy_(torch.tensor([1.0, -0.7, 0.2], dtype=torch.float64))
        x = torch.tensor([[0.2, 0.3], [-0.4, 0.7], [0.6, -0.1]], dtype=torch.float64)
        targets = torch.tensor([[0.3, 0.1], [-0.2, 0.8], [0.4, -0.5]], dtype=torch.float64)
        shapes = [(name, value.shape, value.numel()) for name, value in model.named_parameters()]
        vector = torch.cat([value.detach().flatten() for value in model.parameters()]).requires_grad_()

        def outputs(theta, data=x):
            parameters, offset = {}, 0
            for name, shape, count in shapes:
                parameters[name] = theta[offset:offset + count].reshape(shape)
                offset += count
            return functional_call(model, parameters, (data,))

        hessian = torch.autograd.functional.hessian(lambda theta: (outputs(theta) - targets).square().sum() / (2 * len(x)), vector)
        jacobian = torch.autograd.functional.jacobian(lambda theta: outputs(theta).flatten(), vector)
        residual_hessian = hessian - jacobian.T @ jacobian / len(x)
        expected = float(torch.linalg.eigvalsh(residual_hessian).abs().max())
        measured = fixed_gate_hessian_residual(model, x, targets)
        self.assertAlmostEqual(measured["mean_residual_hessian_operator_norm"], expected, places=12)
        example_norms = []
        for index in range(len(x)):
            h_i = torch.autograd.functional.hessian(lambda theta: (outputs(theta, x[index:index+1]) - targets[index:index+1]).square().sum() / 2, vector)
            j_i = torch.autograd.functional.jacobian(lambda theta: outputs(theta, x[index:index+1]).flatten(), vector)
            example_norms.append(float(torch.linalg.eigvalsh(h_i - j_i.T @ j_i).abs().max()))
        self.assertAlmostEqual(measured["max_example_residual_hessian_operator_norm"], max(example_norms), places=12)
        self.assertFalse(measured["relu_boundary_detected"])

    def test_interpolating_residual_curvature_is_zero(self):
        torch.manual_seed(9)
        model = OneHiddenMLP(2, 2, 2).double()
        x = torch.tensor([[0.2, 0.7], [-0.2, 0.4]], dtype=torch.float64)
        targets = model(x).detach()
        measured = fixed_gate_hessian_residual(model, x, targets)
        self.assertEqual(measured["mean_residual_hessian_operator_norm"], 0)
        self.assertEqual(measured["max_example_residual_hessian_operator_norm"], 0)
        self.assertEqual(measured["train_loss_recomputed"], 0)

    def test_zero_gate_boundary_is_explicit(self):
        model = OneHiddenMLP(1, 1, 1).double()
        with torch.no_grad():
            model.fc1.weight.zero_()
            model.fc1.bias.zero_()
        measured = fixed_gate_hessian_residual(model, torch.ones((2, 1)), torch.ones((2, 1)))
        self.assertTrue(measured["relu_boundary_detected"])
        self.assertEqual(measured["baseline_zero_preactivation_count"], 2)
        self.assertEqual(measured["mean_residual_hessian_operator_norm"], 0)

    def test_saved_endpoint_retains_precision_and_writes_only_derived_checks(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            experiment = Path(temporary)
            directory = experiment / "seed0_B2_lr0.01"
            directory.mkdir()
            config = {"classes": [0, 1], "samples_per_class": 2, "dataset_seed": 0,
                      "training_dtype": "float64"}
            identity = {"seed": 0, "batch_size": 2, "learning_rate": 0.01}
            (experiment / "config.json").write_text(json.dumps(config))
            (directory / "summary.json").write_text(json.dumps({**identity, "train_loss": 0.1}))
            model = OneHiddenMLP(2, 3, 2).double()
            with torch.no_grad():
                model.fc1.weight[0, 0] = 0.123456789012345
            x = torch.tensor([[0.2, 0.3], [-0.4, 0.7], [0.6, -0.1], [0.1, 0.9]])
            labels = torch.tensor([0, 0, 1, 1])
            indices = np.arange(4)
            np.savez(experiment / "subset_indices.npz", train=indices)
            np.savez(experiment / "labels.npz", clean_train=labels.numpy(), train=labels.numpy(), corrupted=np.zeros(4, dtype=bool))
            torch.save({**identity, "config": config, "state_dict": model.state_dict()}, directory / "model.pt")
            np.save(directory / "gram.npy", prediction_gram(model, x))
            original = {path: path.read_bytes() for path in experiment.rglob("*") if path.is_file()}
            with patch("sgd_memorization.endpoint_checks._cached_training_subset", return_value=(x, labels, indices)):
                readonly = inspect_endpoint(directory, write=False)
                self.assertFalse((directory / "endpoint_checks.json").exists())
                measured = inspect_endpoint(directory)
            self.assertEqual(readonly, measured)
            self.assertEqual(measured["checkpoint_dtype"], "torch.float64")
            self.assertTrue(measured["saved_gram_agreement"]["matches_rtol_1e10_atol_1e12"])
            self.assertEqual(measured["saved_gram_agreement"]["max_absolute_difference"], 0)
            self.assertEqual(len(measured["rank_sensitivity"]), 3)
            for path, contents in original.items():
                self.assertEqual(path.read_bytes(), contents)
            new_files = {path for path in experiment.rglob("*") if path.is_file()} - original.keys()
            self.assertEqual(new_files, {directory / "endpoint_checks.json"})


if __name__ == "__main__":
    unittest.main()
