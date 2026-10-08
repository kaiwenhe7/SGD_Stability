"""Regression tests for outliers that pooled edit plots previously hid."""

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sgd_memorization.plotting import _Run, _edit_extrema, _edits


def edit(amplitude, relative_error, relative_leakage):
    return {"amplitude": str(amplitude),
            "achieved_target_change": str(amplitude * (1 + relative_error)),
            "leakage_max_relative": str(relative_leakage),
            "tangent_pred_cost": "1", "parameter_cost": "1"}


class EditPlotTests(unittest.TestCase):
    def test_outlier_survives_many_small_errors_and_opposite_signs(self):
        rows = [edit(1e-5, 1e-9, 1e-9) for _ in range(99)]
        rows.append(edit(-1e-5, 0.07, 0.065854))
        records = _edit_extrema(rows)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["amplitude"], 1e-5)
        self.assertAlmostEqual(records[0]["target_error_max"], 0.07)
        self.assertEqual(records[0]["anchor_leakage_max"], 0.065854)
        self.assertEqual(records[0]["anchor_observations"], 100)

    def test_zero_is_observed_and_missing_value_is_unavailable(self):
        rows = [edit(1e-6, 0, 0), {"amplitude": "1e-5"},
                {"amplitude": "0", "leakage_max_relative": "10"}]
        records = _edit_extrema(rows)
        self.assertEqual(records[0]["target_error_max"], 0)
        self.assertEqual(records[0]["anchor_leakage_max"], 0)
        self.assertEqual(records[0]["anchor_observations"], 1)
        self.assertTrue(math.isnan(records[1]["anchor_leakage_max"]))
        self.assertEqual(records[1]["anchor_observations"], 0)

    def test_legacy_absolute_leakage_uses_absolute_amplitude(self):
        records = _edit_extrema([{"amplitude": "-0.01", "achieved_target_change": "-0.01",
                                 "leakage_max": "0.002"}])
        self.assertAlmostEqual(records[0]["anchor_leakage_max"], 0.2)

    def test_learning_rates_and_seeds_remain_separate_plot_curves(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            with patch.dict(os.environ, {"MPLCONFIGDIR": temporary}):
                import matplotlib
                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt
                runs = []
                for seed, rate, leakage in [(0, 0.002, 0.06), (1, 0.002, 0.03), (0, 0.01, 1e-8)]:
                    summary = {"batch_size": 8, "learning_rate": rate, "seed": seed,
                               "train_loss": 1e-9, "threshold": 1e-8}
                    runs.append(_Run(Path(temporary) / f"{seed}_{rate}", summary, [], [],
                                     [edit(1e-5, leakage / 2, leakage)]))
                captured = []
                with patch("sgd_memorization.plotting._save", side_effect=lambda fig, *args: captured.append(fig) or []):
                    _edits(plt, runs, {8: "blue"}, Path(temporary), "All three runs included.")
                axes = captured[0].axes
                low_rate = [ax for ax in axes if "0.002" in ax.get_title(loc="left")][0]
                high_rate = [ax for ax in axes if "0.01" in ax.get_title(loc="left")][0]
                self.assertEqual(len(low_rate.lines), 4)
                self.assertEqual(len(high_rate.lines), 2)
                low_anchor = [float(line.get_ydata()[0]) for line in low_rate.lines if line.get_linestyle() == "-"]
                high_anchor = [float(line.get_ydata()[0]) for line in high_rate.lines if line.get_linestyle() == "-"]
                self.assertEqual(sorted(low_anchor), [0.03, 0.06])
                self.assertEqual(high_anchor, [1e-8])
                self.assertEqual(low_rate.get_yscale(), "symlog")


if __name__ == "__main__":
    unittest.main()
