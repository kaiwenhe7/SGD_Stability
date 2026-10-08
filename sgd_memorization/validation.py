"""Run and summarize the predeclared, workspace-contained validation suite.

Usage: python -B -m sgd_memorization.validation run|analyze|verify
All artifact/cache paths are confined to this repository. No new dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
for key, suffix in {"TMPDIR": "tmp", "MPLCONFIGDIR": "matplotlib",
                    "TORCH_HOME": "torch", "XDG_CACHE_HOME": "xdg"}.items():
    path = ROOT / ".cache" / suffix
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import torch

from .run import run, write_json, write_rows

DEFAULT_OUTPUT = ROOT / "results" / "validation_v1"
METRICS = ["median_gradient_sqnorm", "median_private_sqnorm", "median_private_fraction",
           "median_unit_edit_cost", "covariance_ratio", "max_coupling_factor",
           "max_shared_private_ratio"]


def contained(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Validation artifacts must stay inside the workspace")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path) -> dict:
    """Freeze protocol and training sources before any new experiment runs."""
    output = contained(output)
    protocol = json.loads((ROOT / "configs/validation_suite.json").read_text())
    paths = [ROOT / "sgd_memorization" / f"{name}.py"
             for name in ["data", "model", "geometry", "covariance", "run"]]
    paths += [ROOT / entry["config"] for entry in protocol["experiments"]]
    paths += [ROOT / "configs/validation_suite.json", ROOT / "requirements.txt"]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "source_hashes.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != hashes:
            raise ValueError("Training source/protocol changed: use a new output version")
    else:
        write_json(manifest, hashes)
        write_json(output / "protocol.json", protocol)
        for path in paths:
            destination = output / "source_snapshot" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        pilot = ROOT / "results/mnist_pilot"
        write_json(output / "legacy_pilot_hashes.json", {
            str(path.relative_to(ROOT)): sha(path) for path in sorted(pilot.rglob("*"))
            if path.is_file() and ".cache" not in path.parts})
    return protocol


def csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def values(rows: list[dict], key: str) -> np.ndarray:
    result = []
    for row in rows:
        try:
            number = float(row[key])
            if math.isfinite(number):
                result.append(number)
        except (KeyError, TypeError, ValueError):
            pass
    return np.asarray(result)


def median(rows: list[dict], key: str) -> float | None:
    numbers = values(rows, key)
    return float(np.median(numbers)) if len(numbers) else None


def maximum(rows: list[dict], key: str) -> float | None:
    numbers = values(rows, key)
    return float(np.max(numbers)) if len(numbers) else None


def collect(output: Path) -> tuple[list[dict], list[dict]]:
    protocol = json.loads((output / "protocol.json").read_text())
    records, edit_records = [], []
    for entry in protocol["experiments"]:
        for path in sorted((output / entry["name"]).glob("*/summary.json")):
            summary = json.loads(path.read_text())
            geometry = summary.get("geometry", {})
            covariance = summary.get("covariance", {})
            coords = csv_rows(path.parent / "per_coordinate.csv")
            edits = csv_rows(path.parent / "edits.csv")
            ratio = covariance.get("learning_rate_ratio")
            feasible_count = sum(str(c.get("isolated_edit_feasible", "")).lower() == "true" for c in coords)
            finite_cost_count = len(values(coords, "unit_tangent_edit_sqnorm"))
            row = {"experiment": entry["name"], "seed": summary["seed"],
                   "batch_size": summary["batch_size"], "learning_rate": summary["learning_rate"],
                   "status": summary["status"], "steps": summary["steps"],
                   "train_loss": summary["train_loss"], "train_accuracy": summary["train_accuracy"],
                   "clean_train_accuracy": summary.get("clean_train_accuracy"),
                   "test_accuracy": summary.get("test_accuracy"),
                   "covariance_ratio": ratio,
                   "covariance_residual": covariance.get("eigenpair_relative_residual"),
                   "covariance_converged": covariance.get("solver_converged", False),
                   "rank": geometry.get("numerical_rank"),
                   "max_universal_private_ratio": geometry.get("max_universal_private_ratio"),
                   "max_shared_private_ratio": geometry.get("max_shared_private_ratio"),
                   "median_gradient_sqnorm": median(coords, "gradient_sqnorm"),
                   "median_private_sqnorm": median(coords, "private_sqnorm"),
                   "median_private_fraction": median(coords, "private_fraction"),
                   "coordinate_count": len(coords), "feasible_coordinate_count": feasible_count,
                   "infeasible_coordinate_count": len(coords) - feasible_count,
                   "finite_unit_edit_cost_count": finite_cost_count,
                   "median_feasible_unit_edit_cost": median(coords, "unit_tangent_edit_sqnorm"),
                   "median_unit_edit_cost": median(coords, "unit_tangent_edit_sqnorm") if coords and feasible_count == finite_cost_count == len(coords) else None,
                   "max_coupling_factor": maximum(coords, "coupling_factor"),
                   "median_coupling_factor": median(coords, "coupling_factor"),
                   "tested_edit_count": len(edits),
                   "max_tested_anchor_leakage": maximum(edits, "leakage_max_relative"),
                   "max_tested_target_error": maximum(edits, "target_error_relative"),
                   "min_abs_preactivation": median(edits, "baseline_min_abs_preactivation"),
                   "path": str(path.parent.relative_to(ROOT))}
            row["eligible"] = (row["status"] == "converged" and row["covariance_converged"]
                               and ratio is not None and math.isfinite(ratio) and 0 <= ratio <= 1 + 1e-6)
            checks_path = path.parent / "endpoint_checks.json"
            if checks_path.exists():
                row_checks = json.loads(checks_path.read_text())
                for key in ["gram_condition_number", "mean_residual_hessian_operator_norm",
                            "mean_residual_hessian_to_gauss_newton_top_ratio", "baseline_zero_preactivation_count"]:
                    row[key] = row_checks.get(key)
            records.append(row)
            for amplitude in sorted(set(float(edit["amplitude"]) for edit in edits)):
                selected = [edit for edit in edits if float(edit["amplitude"]) == amplitude]
                edit_records.append({"experiment": row["experiment"], "seed": row["seed"],
                    "batch_size": row["batch_size"], "learning_rate": row["learning_rate"],
                    "amplitude": amplitude, "targets": len(selected),
                    "median_anchor_leakage": median(selected, "leakage_max_relative"),
                    "max_tested_anchor_leakage": maximum(selected, "leakage_max_relative"),
                    "max_tested_target_error": maximum(selected, "target_error_relative"),
                    "max_gate_flips": maximum(selected, "activation_gate_flip_count")})
    return records, edit_records


def paired_effects(records: list[dict]) -> list[dict]:
    """Bootstrap only independent seeds, never coordinates or amplitudes."""
    effects = []
    experiments = sorted({r["experiment"] for r in records})
    selections = []
    for experiment in experiments:
        for batch in [8, 32]:
            selections.append((f"learning_rate:{experiment}:B{batch}",
                [r for r in records if r["experiment"] == experiment and r["batch_size"] == batch and r["learning_rate"] == .002],
                [r for r in records if r["experiment"] == experiment and r["batch_size"] == batch and r["learning_rate"] == .01]))
        for rate in [.002, .01]:
            selections.append((f"batch:{experiment}:eta{rate}",
                [r for r in records if r["experiment"] == experiment and r["learning_rate"] == rate and r["batch_size"] == 8],
                [r for r in records if r["experiment"] == experiment and r["learning_rate"] == rate and r["batch_size"] == 32]))
    for experiment in experiments:
        if experiment == "clean":
            continue
        for batch, rate in itertools.product([8, 32], [.002, .01]):
            selections.append((f"{experiment}_vs_clean:B{batch}:eta{rate}",
                [r for r in records if r["experiment"] == "clean" and r["batch_size"] == batch and r["learning_rate"] == rate],
                [r for r in records if r["experiment"] == experiment and r["batch_size"] == batch and r["learning_rate"] == rate]))
    for name, before, after in selections:
        before, after = {r["seed"]: r for r in before}, {r["seed"]: r for r in after}
        for metric in METRICS:
            logs, seeds, excluded = [], [], []
            for seed in range(5):
                a, b = before.get(seed), after.get(seed)
                if a is None or b is None:
                    excluded.append({"seed": seed, "reason": "missing run"})
                elif not (a["eligible"] and b["eligible"]):
                    excluded.append({"seed": seed, "reason": "not converged and below estimated tangent threshold"})
                elif a[metric] is None or b[metric] is None or min(a[metric], b[metric]) <= 0:
                    excluded.append({"seed": seed, "reason": "undefined or nonpositive metric"})
                else:
                    logs.append(math.log(b[metric] / a[metric])); seeds.append(seed)
            if logs:
                array = np.asarray(logs)
                # Exhaust every ordered n-out-of-n bootstrap resample, n <= 5.
                indices = np.asarray(list(itertools.product(range(len(logs)), repeat=len(logs))))
                boot = np.exp(array[indices].mean(axis=1))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                mean = float(np.exp(array.mean()))
            else:
                mean = lo = hi = None
            effects.append({"contrast": name, "metric": metric, "paired_seeds": seeds,
                "individual_ratios": np.exp(logs).tolist(), "geometric_mean_ratio": mean,
                "bootstrap_95_low": None if lo is None else float(lo),
                "bootstrap_95_high": None if hi is None else float(hi), "excluded": excluded})
    return effects


def plots(output: Path, records: list[dict], edits: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {8: "#27648a", 32: "#c05c31"}
    suites = ["clean", "corrupted", "epoch_warmup", "initial_scale_1"]
    fig, axes = plt.subplots(4, 4, figsize=(14, 12), sharex=True)
    metrics = [("median_unit_edit_cost", "Squared unit edit cost", True),
               ("median_gradient_sqnorm", "Total gradient squared norm", True),
               ("median_private_fraction", "Private fraction", False),
               ("covariance_ratio", "Learning rate / tangent threshold", False)]
    for row, suite in enumerate(suites):
        for col, (metric, label, logarithmic) in enumerate(metrics):
            ax = axes[row, col]
            for batch in [8, 32]:
                for seed in range(5):
                    selected = sorted([r for r in records if r["experiment"] == suite and r["batch_size"] == batch and r["seed"] == seed and r.get(metric) is not None], key=lambda r: r["learning_rate"])
                    if not selected:
                        continue
                    ax.plot([r["learning_rate"] for r in selected], [r[metric] for r in selected], color=colors[batch], alpha=.45, lw=1)
                    for r in selected:
                        ax.scatter(r["learning_rate"], r[metric], marker="o" if r["eligible"] else "x", color=colors[batch], s=20)
                ax.plot([], [], color=colors[batch], label=f"B={batch}")
            if metric == "covariance_ratio":
                ax.axhline(1, color="black", ls="--", lw=.8)
            if logarithmic:
                ax.set_yscale("log")
            ax.set_xscale("log"); ax.set_xticks([.002, .01], [".002", ".01"])
            ax.grid(alpha=.18)
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(suite.replace("_", " "))
            if row == 3:
                ax.set_xlabel("Learning rate")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("MNIST: paired seeds, absolute sensitivity, and sensitivity fractions", fontsize=14)
    fig.text(.07, .015, "Each line is one initialization seed on the same 128 images. x = ineligible endpoint; no failed/capped run is silently excluded.\nFloat64 training; 16 fixed edit targets; five seeds are replicates, output coordinates are not.", fontsize=9)
    fig.tight_layout(rect=[0, .055, 1, .96])
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"paired_geometry.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for suite, ax in zip(suites, axes.flat):
        for batch in [8, 32]:
            for rate in [.002, .01]:
                for seed in range(5):
                    chosen = [r for r in edits if r["experiment"] == suite and r["batch_size"] == batch and r["learning_rate"] == rate and r["seed"] == seed]
                    amps = sorted({abs(r["amplitude"]) for r in chosen})
                    if amps:
                        worst = [max(r["max_tested_anchor_leakage"] for r in chosen if abs(r["amplitude"]) == amp) for amp in amps]
                        ax.plot(amps, np.maximum(worst, 1e-16), color=colors[batch], ls="-" if rate == .002 else "--", alpha=.5)
        ax.set_title(suite.replace("_", " ")); ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=.18)
        ax.axhline(.01, ls=":", color="black", lw=.8)
        ax.set_xlabel("Absolute requested edit"); ax.set_ylabel("Maximum tested anchor drift / edit")
    fig.suptitle("Nonlinear leakage: each curve retains its seed, batch, and learning rate")
    fig.text(.08, .015, "Maximum over 16 prespecified coordinates and both signs, not over all coordinates. Blue B8; orange B32.\nSolid eta=.002; dashed eta=.01. Dotted line is 1% leakage. These are actual nonlinear measurements, not nonlinear optimum costs.", fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, .95])
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"nonlinear_leakage.{ext}", dpi=160)
    plt.close(fig)


def analyze(output: Path) -> dict:
    output = contained(output)
    records, edits = collect(output)
    effects = paired_effects(records)
    write_rows(output / "run_metrics.csv", records)
    write_rows(output / "edit_metrics.csv", edits)
    write_json(output / "paired_effects.json", {"method": "paired-seed percentile bootstrap of geometric mean ratios; exploratory with five seeds", "effects": effects})
    # This secondary population avoids hiding selection caused by the conditional
    # theorem's stability premise. It is labeled separately from the frozen plan.
    all_converged = [{**row, "eligible": row["status"] == "converged"} for row in records]
    write_json(output / "paired_effects_all_converged.json", {
        "scope": "Secondary analysis added after initial endpoints included above-threshold cases; all converged runs regardless of frozen-tangent stability. Does not replace predeclared conditional contrasts.",
        "effects": paired_effects(all_converged)})
    summary = {"planned_runs": 80, "completed_runs": len(records),
               "converged_runs": sum(r["status"] == "converged" for r in records),
               "eligible_runs": sum(r["eligible"] for r in records),
               "runs": records, "ineligible": [r for r in records if not r["eligible"]]}
    write_json(output / "analysis.json", summary)
    write_json(output / "analysis_source_hashes.json", {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))})
    plots(output, records, edits)
    return summary


def verify(output: Path) -> dict:
    """Recheck saved artifacts and convergence holds, without changing inputs."""
    output = contained(output)
    protocol = json.loads((output / "protocol.json").read_text())
    errors, checked = [], 0
    hashes_path = output / "source_hashes.json"
    if not hashes_path.exists():
        errors.append("Missing frozen source hashes")
    else:
        for relative, digest in json.loads(hashes_path.read_text()).items():
            for path in [ROOT / relative, output / "source_snapshot" / relative]:
                if not path.is_file() or sha(path) != digest:
                    errors.append(f"Frozen source/config mismatch: {path}")
    frozen_protocol = output / "source_snapshot/configs/validation_suite.json"
    if not frozen_protocol.exists() or json.loads(frozen_protocol.read_text()) != protocol:
        errors.append("Protocol differs from frozen snapshot")
    for entry in protocol["experiments"]:
        cfg = json.loads((output / entry["name"] / "config.json").read_text())
        frozen_config = output / "source_snapshot" / entry["config"]
        if not frozen_config.exists() or json.loads(frozen_config.read_text()) != cfg:
            errors.append(f"Output configuration differs from frozen snapshot: {entry['name']}")
        expected = {(seed, batch, rate) for seed, batch, rate in itertools.product(cfg["seeds"], cfg["batch_sizes"], cfg["learning_rates"])}
        found = set()
        for path in sorted((output / entry["name"]).glob("*/summary.json")):
            summary = json.loads(path.read_text()); checked += 1
            identity = (summary["seed"], summary["batch_size"], summary["learning_rate"])
            if identity in found:
                errors.append(f"Duplicate identity {entry['name']} {identity}")
            found.add(identity)
            if summary["steps"] > cfg["max_steps"]:
                errors.append(f"Update cap exceeded: {path}")
            checkpoint = torch.load(path.parent / "model.pt", map_location="cpu", weights_only=False)
            if any(checkpoint.get(key) != summary.get(key) for key in ["seed", "batch_size", "learning_rate"]) or checkpoint.get("step") != summary["steps"]:
                errors.append(f"Checkpoint/summary identity mismatch: {path}")
            if checkpoint.get("config") != cfg:
                errors.append(f"Checkpoint configuration mismatch: {path}")
            if any(value.dtype != torch.float64 for value in checkpoint["state_dict"].values()):
                errors.append(f"Non-float64 checkpoint: {path}")
            provenance_path = path.parent / "provenance.json"
            if not provenance_path.exists():
                errors.append(f"Missing immutable provenance: {path}")
            else:
                provenance = json.loads(provenance_path.read_text())
                for filename, key in [(f"initial_seed{summary['seed']}.pt", "initial_sha256"),
                        ("subset_indices.npz", "subset_sha256"), ("labels.npz", "labels_sha256"),
                        ("config.json", "config_sha256")]:
                    artifact = path.parent.parent / filename
                    if not artifact.exists() or sha(artifact) != provenance.get(key):
                        errors.append(f"Immutable provenance mismatch: {artifact}")
            hist = csv_rows(path.parent / "history.csv")
            steps = [int(row["step"]) for row in hist]
            if not steps or steps[0] != 0 or steps[-1] != summary["steps"] or any(b <= a for a, b in zip(steps, steps[1:])):
                errors.append(f"Nonmonotone/incomplete history: {path}")
            finite_history = all(row.get("train_loss") not in [None, ""] and math.isfinite(float(row["train_loss"])) for row in hist)
            if summary["status"] == "converged" and not finite_history:
                errors.append(f"Nonfinite converged history: {path}")
            if summary["status"] == "converged":
                passing = []
                for row in reversed(hist):
                    loss = float(row["train_loss"]) if row.get("train_loss") not in [None, ""] else float("nan")
                    if not math.isfinite(loss) or loss >= cfg["loss_threshold"] or float(row["learning_rate"]) != summary["learning_rate"]:
                        break
                    passing.append(row)
                if len(passing) < 2 or int(passing[0]["step"]) - int(passing[-1]["step"]) < cfg["hold_steps"]:
                    errors.append(f"Insufficient successful hold: {path}")
                if passing and "warmup_steps" in cfg and int(passing[-1]["step"]) < cfg["warmup_steps"]:
                    errors.append(f"Hold started before warmup ended: {path}")
                if passing and "warmup_steps" not in cfg and int(passing[-1].get("epoch", 0)) < cfg.get("warmup_epochs", 0):
                    errors.append(f"Hold started before epoch warmup ended: {path}")
                last_loss = float(hist[-1]["train_loss"]) if hist and hist[-1].get("train_loss") not in [None, ""] else float("nan")
                if not math.isfinite(last_loss) or last_loss != summary["train_loss"]:
                    errors.append(f"Endpoint loss differs from history: {path}")
            if (path.parent / "gram.npy").exists():
                k = np.load(path.parent / "gram.npy")
                if not np.isfinite(k).all() or not np.allclose(k, k.T):
                    errors.append(f"Invalid Gram: {path}")
        if found != expected:
            errors.append(f"Run identity mismatch {entry['name']}: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    for relative, digest in json.loads((output / "legacy_pilot_hashes.json").read_text()).items():
        if not (ROOT / relative).exists() or sha(ROOT / relative) != digest:
            errors.append(f"Original pilot changed: {relative}")
    result = {"checked_runs": checked, "passed": not errors, "errors": errors}
    write_json(output / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "analyze", "diagnose", "verify"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment", choices=["clean", "corrupted", "epoch_warmup", "initial_scale_1"])
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    output = contained(args.output)
    torch.set_num_threads(1)
    if args.command == "run":
        protocol = prepare(output)
        for entry in protocol["experiments"]:
            if args.experiment and entry["name"] != args.experiment:
                continue
            run(ROOT / entry["config"], output / entry["name"], ROOT / "data", "cpu", args.max_runs)
        summary = analyze(output)
        print(json.dumps({key: summary[key] for key in ["completed_runs", "converged_runs", "eligible_runs"]}))
    elif args.command == "analyze":
        summary = analyze(output)
        print(json.dumps({key: summary[key] for key in ["completed_runs", "converged_runs", "eligible_runs"]}))
    elif args.command == "diagnose":
        from .endpoint_checks import inspect_endpoint
        for path in sorted(output.glob("*/*/model.pt")):
            inspect_endpoint(path.parent)
            print(f"Checked endpoint: {path.parent.relative_to(output)}", flush=True)
        analyze(output)
    else:
        result = verify(output)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
