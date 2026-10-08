"""Independent paired-seed statistics and saved-experiment integrity checks."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sgd_memorization import validation


class PairedEffectTests(unittest.TestCase):
    @staticmethod
    def record(seed, rate, value, eligible=True):
        return {"experiment": "clean", "seed": seed, "batch_size": 8,
                "learning_rate": rate, "eligible": eligible,
                **{metric: value for metric in validation.METRICS}}

    @staticmethod
    def effect(records):
        return next(row for row in validation.paired_effects(records)
                    if row["contrast"] == "learning_rate:clean:B8"
                    and row["metric"] == "median_unit_edit_cost")

    def test_pairs_by_seed_and_reports_geometric_mean_not_ratio_of_pooled_medians(self):
        records = [self.record(1, .01, 100), self.record(0, .002, 1),
                   self.record(0, .01, 4), self.record(1, .002, 100)]
        effect = self.effect(records)
        self.assertEqual(effect["paired_seeds"], [0, 1])
        np.testing.assert_allclose(effect["individual_ratios"], [4, 1])
        self.assertAlmostEqual(effect["geometric_mean_ratio"], 2.)
        # The four ordered two-seed resamples have geometric means 1,2,2,4.
        self.assertAlmostEqual(effect["bootstrap_95_low"], 1.075)
        self.assertAlmostEqual(effect["bootstrap_95_high"], 3.85)
        self.assertEqual([r["seed"] for r in effect["excluded"]], [2, 3, 4])

    def test_exclusions_retain_missing_unstable_and_undefined_seed_reasons(self):
        records = []
        for seed in range(5):
            records.append(self.record(seed, .002, 2.))
            if seed != 1:
                records.append(self.record(seed, .01, None if seed == 3 else 0. if seed == 4 else 4.,
                                           eligible=seed != 2))
        effect = self.effect(records)
        self.assertEqual(effect["paired_seeds"], [0])
        self.assertAlmostEqual(effect["geometric_mean_ratio"], 2.)
        self.assertEqual([r["seed"] for r in effect["excluded"]], [1, 2, 3, 4])
        self.assertIn("missing", effect["excluded"][0]["reason"])
        self.assertIn("undefined", effect["excluded"][2]["reason"])

    def test_no_eligible_pairs_reports_none_instead_of_zero_effect(self):
        effect = self.effect([self.record(0, .002, 1.), self.record(0, .01, 2., eligible=False)])
        self.assertEqual(effect["paired_seeds"], [])
        self.assertIsNone(effect["geometric_mean_ratio"])
        self.assertIsNone(effect["bootstrap_95_low"])
        self.assertEqual(len(effect["excluded"]), 5)


class ValidationArtifactTests(unittest.TestCase):
    def setUp(self):
        # Explicit directory keeps every temporary artifact inside the workspace.
        cache = validation.ROOT / ".cache" / "validation_tests"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="case_", dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(validation, "ROOT", self.root).start()
        self.output = self.root / "results" / "validation"

    def write_json(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))

    def write_history(self, rows):
        with (self.run_dir / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "step", "learning_rate", "train_loss"])
            writer.writeheader()
            writer.writerows(rows)

    def fixture(self):
        self.config = {"seeds": [0], "batch_sizes": [8], "learning_rates": [.002],
                       "max_steps": 20, "warmup_steps": 8, "hold_steps": 8,
                       "loss_threshold": 1e-8, "training_dtype": "float64"}
        protocol = {"experiments": [{"name": "clean", "config": "configs/validation_clean.json"}],
                    "planned_runs": 1}
        self.write_json(self.root / "configs/validation_suite.json", protocol)
        self.write_json(self.root / "configs/validation_clean.json", self.config)
        for name in ("data", "model", "geometry", "covariance", "run", "validation"):
            path = self.root / "sgd_memorization" / f"{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# immutable source fixture\n")
        (self.root / "requirements.txt").write_text("numpy\n")
        validation.prepare(self.output)
        self.write_json(self.output / "clean/config.json", self.config)
        self.run_dir = self.output / "clean/seed0_B8_lr0.002"
        self.run_dir.mkdir(parents=True)
        self.summary = {"seed": 0, "batch_size": 8, "learning_rate": .002,
                        "status": "converged", "steps": 16, "epochs": 4,
                        "train_loss": 1e-9, "train_accuracy": 1.,
                        "credited_hold_steps": 8,
                        "geometry": {"numerical_rank": 2},
                        "covariance": {"solver_converged": True, "learning_rate_ratio": .8,
                                       "eigenpair_relative_residual": 1e-12}}
        self.write_json(self.run_dir / "summary.json", self.summary)
        torch.save({"state_dict": {"weight": torch.ones(1, dtype=torch.float64)},
                    "config": self.config, "step": 16, "epoch": 4,
                    "seed": 0, "batch_size": 8, "learning_rate": .002}, self.run_dir / "model.pt")
        self.history = [
            {"epoch": 0, "step": 0, "learning_rate": 0., "train_loss": 1.},
            {"epoch": 1, "step": 4, "learning_rate": .0001, "train_loss": 1e-9},
            {"epoch": 2, "step": 8, "learning_rate": .002, "train_loss": 1e-9},
            {"epoch": 3, "step": 12, "learning_rate": .002, "train_loss": 1e-9},
            {"epoch": 4, "step": 16, "learning_rate": .002, "train_loss": 1e-9},
        ]
        self.write_history(self.history)
        np.save(self.run_dir / "gram.npy", np.eye(2))
        experiment_dir = self.run_dir.parent
        torch.save({"weight": torch.ones(1)}, experiment_dir / "initial_seed0.pt")
        np.savez(experiment_dir / "subset_indices.npz", train=np.array([0, 1]), test=np.array([2, 3]))
        np.savez(experiment_dir / "labels.npz", train=np.array([0, 1]))
        self.write_json(self.run_dir / "provenance.json", {
            key: validation.sha(experiment_dir / filename) for filename, key in [
                ("initial_seed0.pt", "initial_sha256"), ("subset_indices.npz", "subset_sha256"),
                ("labels.npz", "labels_sha256"), ("config.json", "config_sha256")]
        })

    def test_prepare_rejects_changed_training_source_before_reusing_output(self):
        self.fixture()
        (self.root / "sgd_memorization/model.py").write_text("# changed model\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            validation.prepare(self.output)

    def test_verify_accepts_boundary_checked_hold_without_counting_first_passing_interval(self):
        self.fixture()
        result = validation.verify(self.output)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["checked_runs"], 1)

    def test_verify_rejects_insufficient_successful_hold(self):
        self.fixture()
        self.write_history(self.history[:-1])
        result = validation.verify(self.output)
        self.assertFalse(result["passed"])
        self.assertTrue(any("hold" in error.lower() for error in result["errors"]))

    def test_verify_rejects_changed_frozen_source_snapshot(self):
        self.fixture()
        (self.output / "source_snapshot/sgd_memorization/model.py").write_text("# changed archived model\n")
        result = validation.verify(self.output)
        self.assertFalse(result["passed"], "A changed source snapshot cannot pass immutable-protocol verification")

    def test_verify_rejects_relaxed_output_config(self):
        self.fixture()
        self.write_json(self.output / "clean/config.json", {**self.config, "hold_steps": 1})
        result = validation.verify(self.output)
        self.assertFalse(result["passed"], "Verification must use and compare the frozen configuration")

    def test_verify_rejects_nonfinite_loss_inside_successful_hold(self):
        self.fixture()
        self.history[3]["train_loss"] = "nan"
        self.write_history(self.history)
        result = validation.verify(self.output)
        self.assertFalse(result["passed"], "NaN cannot count as a below-threshold loss")

    def test_verify_rejects_nonmonotone_steps_inside_successful_hold(self):
        self.fixture()
        self.history[3]["step"] = 17
        self.write_history(self.history)
        result = validation.verify(self.output)
        self.assertFalse(result["passed"], "An interior future step cannot substantiate an update hold")

    def test_collect_does_not_mark_invalid_covariance_ratios_eligible(self):
        self.fixture()
        for ratio in (-.1, float("-inf"), float("nan"), float("inf")):
            with self.subTest(ratio=ratio):
                self.summary["covariance"]["learning_rate_ratio"] = ratio
                self.write_json(self.run_dir / "summary.json", self.summary)
                rows, _ = validation.collect(self.output)
                self.assertFalse(rows[0]["eligible"], "Eligibility requires a finite nonnegative covariance ratio")

    def test_collect_exposes_infeasible_coordinates_without_finite_only_cost_bias(self):
        self.fixture()
        with (self.run_dir / "per_coordinate.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["isolated_edit_feasible", "unit_tangent_edit_sqnorm"])
            writer.writeheader()
            writer.writerows([
                {"isolated_edit_feasible": True, "unit_tangent_edit_sqnorm": 1.},
                {"isolated_edit_feasible": True, "unit_tangent_edit_sqnorm": 3.},
                {"isolated_edit_feasible": False, "unit_tangent_edit_sqnorm": None},
            ])
        rows, _ = validation.collect(self.output)
        self.assertEqual(rows[0]["coordinate_count"], 3)
        self.assertEqual(rows[0]["feasible_coordinate_count"], 2)
        self.assertEqual(rows[0]["infeasible_coordinate_count"], 1)
        self.assertEqual(rows[0]["finite_unit_edit_cost_count"], 2)
        self.assertEqual(rows[0]["median_feasible_unit_edit_cost"], 2.)
        self.assertIsNone(rows[0]["median_unit_edit_cost"])

    def test_verify_rejects_changed_initialization_provenance(self):
        self.fixture()
        torch.save({"weight": torch.zeros(1)}, self.run_dir.parent / "initial_seed0.pt")
        result = validation.verify(self.output)
        self.assertFalse(result["passed"])
        self.assertTrue(any("provenance" in error.lower() for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
