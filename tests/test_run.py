"""Protocol and artifact-integrity checks for the experiment runner."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sgd_memorization.model import OneHiddenMLP
from sgd_memorization.run import (
    corrupt_labels, learning_rate_at, run, run_directory, train_one,
    validate_config, write_json,
)


def small_config():
    return {
        "classes": [0, 1], "samples_per_class": 2,
        "test_samples_per_class": 2, "hidden_width": 3,
        "initial_weight_scale": 1.0, "dataset_seed": 7,
        "seeds": [0], "batch_sizes": [2], "learning_rates": [0.002],
        "max_epochs": 5, "warmup_epochs": 0,
        "warmup_start_fraction": 0.0001, "loss_threshold": 1e-6,
        "consecutive_epochs": 2, "checkpoint_every": 2, "log_every": 2,
        "rank_rtol": 1e-10, "rankone_steps": 2, "rankone_restarts": 1,
        "edit_targets": 2, "edit_amplitudes": [1e-6], "threads": 1,
    }


class RunnerTests(unittest.TestCase):
    def setUp(self):
        # The user requested that even test temporary files stay in this folder.
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)
        self.config = small_config()
        self.config_path = self.root / "input.json"
        self.output = self.root / "results"
        self.x = torch.zeros(4, 784)
        self.y = torch.tensor([0, 0, 1, 1])

    def tearDown(self):
        self.temporary.cleanup()

    def load_data(self, *args, **kwargs):
        return self.x.clone(), self.y.clone(), np.arange(4) + (10 if kwargs.get("test") else 0)

    def fake_train(self, config, batch_size, rate, seed, initial, x, y, xt, yt,
                   directory, device, clean_train_y=None):
        directory.mkdir(exist_ok=True)
        summary = {"seed": seed, "batch_size": batch_size,
                   "learning_rate": rate, "status": "converged", "geometry": {}}
        write_json(directory / "summary.json", summary)
        return summary

    def run_mocked(self, max_runs=None, train=None):
        write_json(self.config_path, self.config)
        with patch("sgd_memorization.run.download_mnist"), \
             patch("sgd_memorization.run.load_subset", side_effect=self.load_data), \
             patch("sgd_memorization.run.train_one", side_effect=train or self.fake_train) as trainer, \
             patch("sgd_memorization.plotting.make_plots", return_value=[]), \
             contextlib.redirect_stdout(io.StringIO()):
            run(self.config_path, self.output, self.root / "data", "cpu", max_runs)
        return trainer

    def artifact_bytes(self):
        return {str(path.relative_to(self.output)): path.read_bytes()
                for path in self.output.rglob("*") if path.is_file()}

    def test_close_learning_rates_get_distinct_runs(self):
        self.config["learning_rates"] = [0.0020000001, 0.0020000002]
        trainer = self.run_mocked()
        self.assertEqual(trainer.call_count, 2)
        names = {path.name for path in self.output.glob("seed*")}
        self.assertEqual(names, {"seed0_B2_lr0.0020000001", "seed0_B2_lr0.0020000002"})
        summaries = json.loads((self.output / "results.json").read_text())["runs"]
        self.assertEqual({row["learning_rate"] for row in summaries}, set(self.config["learning_rates"]))

    def test_completed_rerun_preserves_all_original_artifacts(self):
        self.run_mocked()
        before = self.artifact_bytes()
        with patch("sgd_memorization.run.environment_metadata", return_value={"different": True}):
            trainer = self.run_mocked()
        self.assertEqual(trainer.call_count, 0)
        self.assertEqual(before, self.artifact_bytes())

    def test_max_runs_zero_is_a_true_noop(self):
        self.run_mocked(max_runs=0)
        self.assertFalse(self.output.exists())
        self.run_mocked()
        before = self.artifact_bytes()
        self.run_mocked(max_runs=0)
        self.assertEqual(before, self.artifact_bytes())

    def test_completed_runs_rebuild_missing_aggregate_without_retraining(self):
        self.run_mocked()
        expected = (self.output / "results.json").read_bytes()
        (self.output / "results.json").unlink()
        trainer = self.run_mocked()
        self.assertEqual(trainer.call_count, 0)
        self.assertEqual((self.output / "results.json").read_bytes(), expected)

    def test_completed_runs_rebuild_stale_aggregate_without_retraining(self):
        self.run_mocked()
        expected = (self.output / "results.json").read_bytes()
        write_json(self.output / "results.json", {"runs": []})
        trainer = self.run_mocked()
        self.assertEqual(trainer.call_count, 0)
        self.assertEqual((self.output / "results.json").read_bytes(), expected)

    def test_partial_sweep_reuses_initial_and_keeps_all_summaries(self):
        self.config["learning_rates"] = [0.002, 0.01]
        self.run_mocked(max_runs=1)
        initial = (self.output / "initial_seed0.pt").read_bytes()
        environment = (self.output / "environment.json").read_bytes()
        trainer = self.run_mocked(max_runs=1)
        self.assertEqual(trainer.call_count, 1)
        self.assertEqual(initial, (self.output / "initial_seed0.pt").read_bytes())
        self.assertEqual(environment, (self.output / "environment.json").read_bytes())
        self.assertEqual(len(json.loads((self.output / "results.json").read_text())["runs"]), 2)

    def test_saved_subset_mismatch_is_rejected(self):
        self.run_mocked()
        np.savez(self.output / "subset_indices.npz", train=np.arange(4) + 99, test=np.arange(4) + 10)
        with self.assertRaisesRegex(ValueError, "Saved data selections"):
            self.run_mocked()

    def test_saved_initial_mutation_is_rejected(self):
        self.run_mocked()
        path = self.output / "initial_seed0.pt"
        initial = torch.load(path, weights_only=True)
        initial["fc1.weight"][0, 0] += 1
        torch.save(initial, path)
        with self.assertRaisesRegex(ValueError, "immutable run provenance"):
            self.run_mocked()

    def test_corruption_is_fixed_wrong_and_preserves_original(self):
        original = self.y.clone()
        corrupted, mask = corrupt_labels(self.y, 2, 0.5, 19)
        repeated, repeated_mask = corrupt_labels(self.y, 2, 0.5, 19)
        self.assertTrue(torch.equal(self.y, original))
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(torch.equal(corrupted != self.y, mask))
        self.assertTrue(torch.equal(corrupted, repeated))
        self.assertTrue(torch.equal(mask, repeated_mask))
        self.config.update(label_corruption_fraction=0.5, label_corruption_seed=19)
        trainer = self.run_mocked()
        np.testing.assert_array_equal(np.load(self.output / "labels.npz")["clean_train"], original)
        self.assertTrue(torch.equal(trainer.call_args.args[6], corrupted))
        self.assertTrue(torch.equal(trainer.call_args.kwargs["clean_train_y"], original))
        self.assertTrue(torch.equal(trainer.call_args.args[8], original))

    def test_corrupted_label_mismatch_is_rejected(self):
        self.config["label_corruption_fraction"] = 0.5
        self.run_mocked()
        path = self.output / "labels.npz"
        with np.load(path) as saved:
            values = dict(saved)
        values["train"] = self.y.numpy()
        np.savez(path, **values)
        with self.assertRaisesRegex(ValueError, "Saved data selections or labels"):
            self.run_mocked()

    def test_legacy_name_reused_only_for_matching_exact_rate(self):
        legacy = self.output / "seed0_B2_lr1"
        legacy.mkdir(parents=True)
        write_json(legacy / "summary.json", {"seed": 0, "batch_size": 2, "learning_rate": 1.0})
        self.assertEqual(run_directory(self.output, 0, 2, 1.0), legacy)
        self.assertNotEqual(run_directory(self.output, 0, 2, 1.000000001), legacy)

    def test_update_warmup_and_validation(self):
        config = {**self.config, "warmup_steps": 3}
        self.assertAlmostEqual(learning_rate_at(1, 0.1, config), 0.00001)
        self.assertAlmostEqual(learning_rate_at(2, 0.1, config), 0.001)
        self.assertEqual(learning_rate_at(3, 0.1, config), 0.1)
        self.assertEqual(learning_rate_at(8, 0.1, config), 0.1)
        for bad in ({"training_dtype": "bfloat16"}, {"hold_steps": 0},
                    {"max_steps": 2, "warmup_steps": 3}, {"label_corruption_fraction": 1.1}):
            with self.assertRaises(ValueError):
                validate_config({**self.config, **bad})

    def test_float64_training_and_exact_update_cap(self):
        config = {**self.config, "training_dtype": "float64", "max_steps": 3,
                  "warmup_steps": 2, "hold_steps": 2, "loss_threshold": 1e30}
        model = OneHiddenMLP(784, 3, 2)
        directory = self.root / "training"
        with patch("sgd_memorization.run.diagnose_endpoint", side_effect=lambda c, m, x, d, s: s), \
             contextlib.redirect_stdout(io.StringIO()):
            summary = train_one(config, 1, 1e-9, 0, model.state_dict(), self.x, self.y,
                                self.x, self.y, directory, torch.device("cpu"))
        self.assertEqual(summary["status"], "max_steps")
        self.assertEqual(summary["steps"], 3)
        checkpoint = torch.load(directory / "model.pt", weights_only=False)
        self.assertEqual(checkpoint["state_dict"]["fc1.weight"].dtype, torch.float64)
        self.assertIn("clean_train_accuracy", summary)

    def test_hold_steps_counts_intervals_between_passing_checks(self):
        config = {**self.config, "warmup_steps": 0, "hold_steps": 3, "loss_threshold": 1e30}
        model = OneHiddenMLP(784, 3, 2)
        with patch("sgd_memorization.run.diagnose_endpoint", side_effect=lambda c, m, x, d, s: s), \
             contextlib.redirect_stdout(io.StringIO()):
            summary = train_one(config, 3, 1e-9, 0, model.state_dict(), self.x, self.y,
                                self.x, self.y, self.root / "training", torch.device("cpu"))
        self.assertEqual(summary["status"], "converged")
        self.assertEqual(summary["steps"], 6)
        self.assertEqual(summary["credited_hold_steps"], 4)

    def test_interrupted_training_is_archived_before_restart(self):
        self.config["learning_rates"] = [0.002, 0.01]
        self.run_mocked(max_runs=1)
        directory = run_directory(self.output, 0, 2, 0.01)
        directory.mkdir()
        (directory / "latest.pt").write_bytes(b"interrupted checkpoint")
        self.run_mocked(max_runs=1)
        self.assertEqual((directory / "interrupted_attempt_000" / "latest.pt").read_bytes(), b"interrupted checkpoint")
        self.assertTrue((directory / "summary.json").exists())

    def test_partial_diagnostics_reuse_saved_float64_endpoint(self):
        self.config["training_dtype"] = "float64"
        self.run_mocked()
        directory = run_directory(self.output, 0, 2, 0.002)
        summary = json.loads((directory / "summary.json").read_text())
        del summary["geometry"]
        write_json(directory / "summary.json", summary)
        model = OneHiddenMLP(784, 3, 2).double()
        with torch.no_grad():
            model.fc1.weight[0, 0] = 0.123456789012345
        torch.save({"state_dict": model.state_dict(), "config": self.config,
                    "seed": 0, "batch_size": 2, "learning_rate": 0.002}, directory / "model.pt")
        with patch("sgd_memorization.run.diagnose_endpoint", return_value=summary) as diagnose:
            trainer = self.run_mocked()
        self.assertEqual(trainer.call_count, 0)
        restored = diagnose.call_args.args[1]
        self.assertEqual(restored.fc1.weight.dtype, torch.float64)
        self.assertEqual(float(restored.fc1.weight[0, 0].detach()), 0.123456789012345)


if __name__ == "__main__":
    unittest.main()
