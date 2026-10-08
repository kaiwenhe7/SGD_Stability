"""Headless figures for the MNIST local-geometry pilot.

Plots describe training endpoints and necessary tangent stability conditions.
They do not label a converged optimizer run as mean-square stable. Every run
summary contributes to the status caption, including failed and capped runs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class _Run:
    directory: Path
    summary: dict[str, Any]
    coordinates: list[dict[str, str]]
    history: list[dict[str, str]]
    edits: list[dict[str, str]]

    @property
    def batch(self) -> int:
        return int(self.summary.get("batch_size", 0))

    @property
    def rate(self) -> float:
        return _number(self.summary.get("learning_rate"))

    @property
    def geometry(self) -> dict[str, Any]:
        value = self.summary.get("geometry")
        geometry = dict(value) if isinstance(value, dict) else {}
        covariance = self.summary.get("covariance")
        if isinstance(covariance, dict) and covariance.get("solver_converged") is True:
            ratio = _number(covariance.get("learning_rate_ratio"))
            if math.isfinite(ratio):
                geometry["exact_covariance_ratio"] = ratio
        return geometry

    @property
    def near_interpolation(self) -> bool:
        loss = _number(self.summary.get("train_loss"))
        threshold = _number(self.summary.get("threshold"))
        return math.isfinite(loss) and math.isfinite(threshold) and loss <= threshold


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_csv(path: Path, notes: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        notes.append(f"Unreadable {path.parent.name}/{path.name}: {type(exc).__name__}")
        return []


def _load_runs(output_dir: Path) -> tuple[list[_Run], list[str]]:
    runs: list[_Run] = []
    notes: list[str] = []
    for path in sorted(output_dir.rglob("summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict) or "batch_size" not in summary:
                continue
            # Check identifiers here so a damaged summary cannot stop plotting
            # the remaining completed or partial runs.
            int(summary["batch_size"])
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            notes.append(f"Unreadable {path.parent.name}/summary.json: {type(exc).__name__}")
            continue
        runs.append(_Run(
            path.parent,
            summary,
            _read_csv(path.parent / "per_coordinate.csv", notes),
            _read_csv(path.parent / "history.csv", notes),
            _read_csv(path.parent / "edits.csv", notes)
            + _read_csv(path.parent / "edits_small_amplitude.csv", notes),
        ))
    return runs, notes


def _status_caption(runs: list[_Run], notes: list[str]) -> str:
    counts = Counter(str(run.summary.get("status", "unknown")) for run in runs)
    statuses = ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))
    with_geometry = sum(bool(run.geometry) for run in runs)
    near = sum(run.near_interpolation for run in runs)
    caption = (
        f"All {len(runs)} run summaries: {statuses or 'none'}. "
        f"{with_geometry} have tangent diagnostics; {near} have endpoint loss below their threshold. "
        "Crosses: above loss threshold or unavailable loss; other markers: below threshold. "
        "Curvature is Gauss-Newton; it equals the squared-loss Hessian at exact interpolation. "
        "Full covariance ratios use a numerically converged tangent-operator solve. "
        "Tangent stability diagnostics do not certify nonlinear mean-square stability."
    )
    if notes:
        caption += f" {len(notes)} unreadable result files; see plotting_notes.txt."
    return caption


def _empty(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="0.4", wrap=True)


def _finish_axis(ax: Any, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig: Any, output_dir: Path, stem: str, caption: str) -> list[Path]:
    caption_width = max(80, int((fig.get_size_inches()[0] - 1.5) * 17))
    fig.text(0.07, 0.015, "\n".join(textwrap.wrap(caption, caption_width)),
             ha="left", va="bottom", fontsize=8, color="0.3")
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    return paths


def _metric(ax: Any, runs: list[_Run], key: str, colors: dict[int, Any],
            linestyle: str = "-", positive_only: bool = False) -> int:
    """Draw endpoint points plus median curves for below-threshold endpoints."""
    count = 0
    for batch in sorted(colors):
        selected = [run for run in runs if run.batch == batch]
        medians: dict[float, list[float]] = defaultdict(list)
        for run in selected:
            value = _number(run.geometry.get(key))
            if not (math.isfinite(run.rate) and run.rate > 0 and math.isfinite(value)):
                continue
            if positive_only and value <= 0:
                continue
            ax.plot(run.rate, value, marker="o" if run.near_interpolation else "x",
                    color=colors[batch], linestyle="none", markersize=4.5,
                    alpha=0.8, markerfacecolor=(colors[batch] if linestyle == "-" else "none"))
            if run.near_interpolation:
                medians[run.rate].append(value)
            count += 1
        if medians:
            rates = sorted(medians)
            ax.plot(rates, [float(np.median(medians[rate])) for rate in rates],
                    color=colors[batch], linestyle=linestyle, linewidth=1.5, alpha=0.8)
    return count


def _overview(plt: Any, runs: list[_Run], colors: dict[int, Any],
              output_dir: Path, caption: str) -> list[Path]:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.91, hspace=0.42, wspace=0.3)
    fig.suptitle("MNIST pilot: tangent geometry at training endpoints", fontsize=15, x=0.08, ha="left")
    a, b, c, d = axes.flat
    count = _metric(a, runs, "lambda_max_hessian", colors, positive_only=True)
    valid_rates = [run.rate for run in runs if math.isfinite(run.rate) and run.rate > 0]
    if valid_rates:
        low, high = min(valid_rates), max(valid_rates)
        if low == high:
            low, high = low / 1.5, high * 1.5
        rates = np.geomspace(low, high, 100)
        a.plot(rates, 2 / rates, color="0.25", linestyle="--", label=r"$2/\eta$", linewidth=1.2)
    if not count:
        _empty(a, "No finite endpoint curvature diagnostics")
    a.set_xscale("log")
    a.set_yscale("log")
    _finish_axis(a, r"Learning rate $\eta$", r"$\lambda_{\max}(J^\top J/n)$", "a  Curvature and the full-batch limit")
    batch_handles = [Line2D([0], [0], color=colors[batch], marker="o", markersize=4,
                            linewidth=1.3, label=f"B = {batch}") for batch in sorted(colors)]
    limit_handle = Line2D([0], [0], color="0.25", linestyle="--", label=r"$2/\eta$")
    a.legend(handles=batch_handles + [limit_handle], fontsize=8, frameon=False, ncol=2)

    baselines = [
        ("full_hessian_sharpness_ratio", "-", "Gauss-Newton"),
        ("top_hessian_rankone_ratio", "--", "Top-GN rank-one test"),
        ("optimized_rankone_ratio", ":", "Best found rank-one test"),
        ("exact_covariance_ratio", "-.", "Full covariance operator"),
    ]
    count = sum(_metric(b, runs, key, colors, style) for key, style, _ in baselines)
    b.axhline(1, color="0.25", linewidth=1, linestyle="--")
    b.set_xscale("log")
    if not count:
        _empty(b, "No finite normalized curvature diagnostics")
    _finish_axis(b, r"Learning rate $\eta$", r"Necessary-condition ratio ($>1$ violates)", "b  Curvature baselines")
    b.legend(handles=[Line2D([0], [0], color="0.25", linestyle=style, label=label)
                      for _, style, label in baselines], fontsize=8, frameon=False)

    count = _metric(c, runs, "max_universal_private_ratio", colors)
    count += _metric(c, runs, "max_shared_private_ratio", colors, "--")
    c.axhline(1, color="0.25", linewidth=1, linestyle=":")
    c.set_xscale("log")
    if not count:
        _empty(c, "No applicable private-bound diagnostics")
    _finish_axis(c, r"Learning rate $\eta$", "Maximum coordinate ratio", "c  Private and shared-private tests")
    c.legend(handles=[Line2D([0], [0], color="0.25", linestyle="-", label="Universal"),
                      Line2D([0], [0], color="0.25", linestyle="--", label="Shared-private"),
                      Line2D([0], [0], color="0.25", linestyle=":", label="Necessary limit = 1")],
             fontsize=8, frameon=False)
    unavailable = sum(
        1 for run in runs if run.geometry and not math.isfinite(_number(run.geometry.get("max_shared_private_ratio")))
    )
    negative = sum(int(run.geometry.get("negative_shared_slack_count", 0)) for run in runs)
    if unavailable or negative:
        c.text(0.02, 0.03, f"Shared test unavailable: {unavailable} runs; negative slack: {negative} coordinates",
               transform=c.transAxes, fontsize=7.5, color="0.4")

    count = 0
    infeasible = 0
    for run in runs:
        if not (math.isfinite(run.rate) and run.rate > 0):
            continue
        costs = []
        for row in run.coordinates:
            value = _number(row.get("unit_tangent_edit_sqnorm"))
            if not math.isfinite(value):
                private = _number(row.get("private_sqnorm"))
                value = 1 / private if math.isfinite(private) and private > 0 else float("nan")
            if math.isfinite(value) and value > 0:
                costs.append(value)
            else:
                infeasible += 1
        if not costs:
            continue
        lo, median, hi = np.percentile(costs, [10, 50, 90])
        d.errorbar(run.rate, median, yerr=[[median - lo], [hi - median]],
                   fmt="o" if run.near_interpolation else "x", color=colors[run.batch],
                   markersize=4, linewidth=1, capsize=2, alpha=0.75)
        count += 1
    if valid_rates:
        for batch in sorted(colors):
            if batch > 0:
                d.plot(rates, rates / (2 * batch), color=colors[batch], linestyle="--",
                       linewidth=1, alpha=0.8)
    if not count:
        _empty(d, "No feasible isolated-coordinate edit costs")
    d.set_xscale("log")
    d.set_yscale("log")
    _finish_axis(d, r"Learning rate $\eta$", r"Squared cost for unit output edit", "d  Isolated tangent edit cost")
    d.legend(handles=[Line2D([0], [0], color="0.3", marker="o", label="Median; bars = 10-90%"),
                      Line2D([0], [0], color="0.3", linestyle="--", label=r"Universal lower bound $\eta/(2B)$")],
             fontsize=8, frameon=False)
    if infeasible:
        d.text(0.02, 0.03, f"{infeasible} coordinates have unavailable/infeasible costs",
               transform=d.transAxes, fontsize=7.5, color="0.4")
    paths = _save(fig, output_dir, "overview", caption)
    plt.close(fig)
    return paths


def _training(plt: Any, runs: list[_Run], colors: dict[int, Any],
              output_dir: Path, caption: str) -> list[Path]:
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.9)
    count = 0
    for run in runs:
        points = [(_number(row.get("step")), _number(row.get("train_loss"))) for row in run.history]
        points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y) and y > 0]
        if not points:
            continue
        x, y = zip(*points)
        label = f"B={run.batch}, eta={run.rate:g}, seed={run.summary.get('seed', '?')}"
        ax.plot(x, y, color=colors[run.batch], linewidth=1.05, alpha=0.7, label=label)
        count += 1
    thresholds = sorted({
        _number(run.summary.get("threshold")) for run in runs
        if math.isfinite(_number(run.summary.get("threshold"))) and _number(run.summary.get("threshold")) > 0
    })
    for threshold in thresholds:
        ax.axhline(threshold, color="0.25", linestyle="--", linewidth=1,
                   label=f"Loss threshold {threshold:g}")
    ax.set_yscale("log")
    _finish_axis(ax, "Optimizer updates", "Mean training loss", "Training trajectories and interpolation threshold")
    if not count:
        _empty(ax, "No finite positive training-loss histories")
    if count <= 12 and (count or thresholds):
        ax.legend(fontsize=7.5, frameon=False, ncol=2)
    elif count > 12:
        handles = [Line2D([0], [0], color=colors[batch], label=f"B = {batch}") for batch in sorted(colors)]
        handles += [Line2D([0], [0], color="0.25", linestyle="--", label="Loss threshold")]
        ax.legend(handles=handles, fontsize=8, frameon=False, ncol=2)
    paths = _save(fig, output_dir, "training", caption)
    plt.close(fig)
    return paths


def _edit_extrema(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    """Worst measured errors within one run, grouping only by absolute edit size.

    Each maximum covers all tested coordinates and either tested sign. Runs,
    rates, and seeds are never pooled. Nonfinite observations are unavailable,
    whereas exact zeros remain valid observations and are retained.
    """
    values: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: {"target_error": [], "anchor_leakage": []})
    for row in rows:
        signed_amplitude = _number(row.get("amplitude"))
        amplitude = abs(signed_amplitude)
        if not (math.isfinite(amplitude) and amplitude > 0):
            continue
        group = values[amplitude]
        achieved = _number(row.get("achieved_target_change"))
        target_error = abs(achieved / signed_amplitude - 1)
        if math.isfinite(target_error):
            group["target_error"].append(target_error)
        leakage = _number(row.get("leakage_max_relative"))
        if not math.isfinite(leakage):
            leakage = _number(row.get("leakage_max")) / amplitude
        if math.isfinite(leakage) and leakage >= 0:
            group["anchor_leakage"].append(leakage)
    return [
        {"amplitude": amplitude,
         "target_error_max": max(group["target_error"], default=float("nan")),
         "anchor_leakage_max": max(group["anchor_leakage"], default=float("nan")),
         "target_observations": len(group["target_error"]),
         "anchor_observations": len(group["anchor_leakage"])}
        for amplitude, group in sorted(values.items())
    ]


def _edits(plt: Any, runs: list[_Run], colors: dict[int, Any],
           output_dir: Path, caption: str) -> list[Path]:
    from matplotlib.lines import Line2D

    settings = sorted({(run.batch, run.rate) for run in runs if math.isfinite(run.rate) and run.rate > 0})
    columns = min(3, max(1, len(settings) + 1))
    rows = math.ceil((len(settings) + 1) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 4.25 * rows + 1.7), squeeze=False)
    axes = list(axes.flat)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=1.65 / fig.get_size_inches()[1],
                        top=1 - 0.7 / fig.get_size_inches()[1], wspace=0.33, hspace=0.46)
    fig.suptitle("MNIST tangent edits: worst measured error in each run", fontsize=14,
                 x=0.065, ha="left", y=1 - 0.17 / fig.get_size_inches()[1])
    cost_ax = axes[0]
    cost_values: list[float] = []
    cost_count = 0
    for run in runs:
        for row in run.edits:
            pred = _number(row.get("tangent_pred_cost"))
            actual = _number(row.get("parameter_cost"))
            if math.isfinite(pred) and pred > 0 and math.isfinite(actual) and actual > 0:
                cost_ax.scatter(pred, actual, color=colors[run.batch], s=13,
                                marker="o" if run.near_interpolation else "x", alpha=0.5)
                cost_values.extend([pred, actual])
                cost_count += 1
    if cost_values:
        low, high = min(cost_values), max(cost_values)
        if low == high:
            low, high = low / 1.5, high * 1.5
        cost_ax.plot([low, high], [low, high], color="0.25", linestyle="--", linewidth=1)
    else:
        _empty(cost_ax, "No applied tangent-edit costs")
    cost_ax.set_xscale("log")
    cost_ax.set_yscale("log")
    _finish_axis(cost_ax, r"Predicted cost $\Delta^2(K^\dagger)_{aa}$", "Applied squared parameter displacement",
                 "Tangent cost construction check")
    cost_handles = [Line2D([0], [0], color=colors[batch], marker="o", markersize=4,
                          linestyle="none", label=f"B = {batch}") for batch in sorted(colors)]
    cost_handles.append(Line2D([0], [0], color="0.25", linestyle="--", label="Identity"))
    cost_ax.legend(handles=cost_handles, fontsize=8, frameon=False)

    seeds = sorted({str(run.summary.get("seed", "?")) for run in runs})
    seed_palette = plt.get_cmap("tab10")
    seed_colors = {seed: seed_palette(index % 10) for index, seed in enumerate(seeds)}
    extrema = {run.directory: _edit_extrema(run.edits) for run in runs}
    finite_errors = [row[key] for records in extrema.values() for row in records
                     for key in ("target_error_max", "anchor_leakage_max") if math.isfinite(row[key])]
    upper = max(finite_errors, default=1e-8)
    upper = max(upper * 2, 1e-11)
    measurement_runs = 0
    for ax, (batch, rate) in zip(axes[1:], settings):
        selected = [run for run in runs if run.batch == batch and run.rate == rate]
        plotted = 0
        for run in selected:
            measurements = extrema[run.directory]
            has_measurements = False
            seed = str(run.summary.get("seed", "?"))
            for key, linestyle, marker in [("anchor_leakage_max", "-", "o"),
                                            ("target_error_max", "--", "s")]:
                points = [(row["amplitude"], row[key]) for row in measurements if math.isfinite(row[key])]
                if not points:
                    continue
                x, y = zip(*points)
                ax.plot(x, y, color=seed_colors[seed], linestyle=linestyle,
                        marker=marker if run.near_interpolation else "x", markersize=4,
                        linewidth=1.2, alpha=0.85)
                plotted += 1
                has_measurements = True
            measurement_runs += int(has_measurements)
        ax.set_xscale("log")
        # Unlike a log axis, this keeps exact zero-error observations visible.
        ax.set_yscale("symlog", linthresh=1e-12, linscale=0.5)
        ax.set_ylim(0, upper)
        _finish_axis(ax, r"Requested output change $|\Delta|$", "Worst relative error among tested edits",
                     rf"B = {batch}, $\eta$ = {rate:g}")
        if not plotted:
            _empty(ax, "No finite nonlinear edit measurements")
        ax.text(0.02, 0.97, f"{sum(bool(extrema[run.directory]) for run in selected)}/{len(selected)} runs have edit records",
                transform=ax.transAxes, va="top", fontsize=7.5, color="0.4")

    legend_handles = [Line2D([0], [0], color=seed_colors[seed], linewidth=1.5,
                             label=f"Seed {seed}") for seed in seeds]
    legend_handles += [Line2D([0], [0], color="0.25", marker="o", linestyle="-",
                              label="Maximum anchor leakage / |Δ|"),
                       Line2D([0], [0], color="0.25", marker="s", linestyle="--",
                              label="Maximum relative target error")]
    spare = axes[len(settings) + 1:]
    if spare:
        for ax in spare:
            ax.axis("off")
        spare[0].legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9)
        spare[0].text(0.02, 0.30,
                      "Each curve is one run.\nMaxima include all tested coordinates\nand both signs when available.\n"
                      "No pooling across seeds or learning rates.\nExact zeros are shown; y-axis is linear\nbelow 10⁻¹² and logarithmic above.",
                      transform=spare[0].transAxes, va="top", fontsize=9, linespacing=1.5)
    elif settings:
        axes[1].legend(handles=legend_handles, fontsize=6.8, frameon=False, loc="lower right", ncol=2)
    edit_caption = (
        f"{cost_count} applied edits with finite costs; {measurement_runs}/{len(runs)} runs have finite error measurements. "
        "The cost panel checks the tangent construction, not the minimum cost of a nonlinear constrained edit. "
        "Each error curve is one seed at the displayed batch size and learning rate. At every |Δ| it shows the actual "
        "maximum over tested target coordinates and available signs; anchor leakage also maximizes over every anchored "
        "training-output coordinate. These are observed maxima, not worst-case bounds over untested edits. "
        "No quantiles or confidence intervals are used. Original and supplemental edits are included; nonfinite "
        "measurements are unavailable. Error axes retain zero with a linear region below 1e-12. " + caption
    )
    paths = _save(fig, output_dir, "edits", edit_caption)
    plt.close(fig)
    return paths


def make_plots(output_dir: Path) -> list[Path]:
    """Create overview, training, and edit PNG/PDF pairs from runner outputs.

    Each run directory contains ``summary.json`` and optionally
    ``history.csv``, ``per_coordinate.csv``, ``edits.csv``, and supplemental
    ``edits_small_amplitude.csv``. Missing or
    malformed diagnostics are reported in captions rather than causing runs
    to disappear from the experiment count. An empty result directory still
    produces figures explaining the absent diagnostics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    runs, notes = _load_runs(output_dir)
    batches = sorted({run.batch for run in runs})
    palette = plt.get_cmap("tab10")
    colors = {batch: palette(index % 10) for index, batch in enumerate(batches)}
    caption = _status_caption(runs, notes)
    with plt.rc_context({
        "font.size": 9, "axes.labelsize": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
    }):
        paths = _overview(plt, runs, colors, output_dir, caption)
        paths += _training(plt, runs, colors, output_dir, caption)
        paths += _edits(plt, runs, colors, output_dir, caption)
    if notes:
        note_path = output_dir / "plotting_notes.txt"
        note_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
        paths.append(note_path)
    return paths
