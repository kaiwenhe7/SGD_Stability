"""Controlled linear features: independent covariance, edit, and coupling checks.

Run with ``python -m sgd_memorization.synthetic --config ... --output ...``.
The scalar predictor has two examples and two parameters. Feature columns
are g1=(sqrt(a),sqrt(t)), g2=(sqrt(h),0). Thus a, t, and anchor strength h
can be changed separately. There is no optimization or learned representation:
this suite checks conditional linear stability and the usefulness of its
projection compression, not causal representation selection by training.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh

from .covariance import exact_generalized_sharpness
from .geometry import analyze_gram


def controlled_features(shared: float, private: float, anchor: float) -> np.ndarray:
    if shared < 0 or private <= 0 or anchor <= 0:
        raise ValueError("Require nonnegative shared and positive private/anchor strengths")
    return np.array([[np.sqrt(shared), np.sqrt(anchor)], [np.sqrt(private), 0.0]])


def explicit_batches(features: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Enumerate actual parameter-space Hessians without Gram moment formulas."""
    if not 1 <= batch_size <= features.shape[1]:
        raise ValueError("Invalid batch size")
    individual = [np.outer(g, g) for g in features.T]
    return [sum(individual[i] for i in batch) / batch_size
            for batch in itertools.combinations(range(len(individual)), batch_size)]


def symmetric_basis(dimension: int) -> np.ndarray:
    basis = []
    for i in range(dimension):
        for j in range(i, dimension):
            matrix = np.zeros((dimension, dimension))
            matrix[i, j] = matrix[j, i] = 1.0 if i == j else 1 / np.sqrt(2)
            basis.append(matrix)
    return np.asarray(basis)


def explicit_operator(features: np.ndarray, batch_size: int) -> dict[str, Any]:
    """Build dense C,D in an orthonormal symmetric parameter-space basis."""
    batches = explicit_batches(features, batch_size)
    hessian = np.mean(batches, axis=0)
    basis = symmetric_basis(features.shape[0])
    c = np.array([[np.sum(left * (hessian @ right + right @ hessian) / 2)
                   for right in basis] for left in basis])
    d = np.array([[np.mean([np.sum(left * (hb @ right @ hb)) for hb in batches])
                   for right in basis] for left in basis])
    sharpness = float(eigh(d, c, eigvals_only=True)[-1])
    return {"batches": batches, "basis": basis, "C": c, "D": d,
            "hessian": hessian, "critical_learning_rate": 2 / sharpness}


def transition_operator(batches: list[np.ndarray], eta: float,
                        basis: np.ndarray) -> np.ndarray:
    identity = np.eye(batches[0].shape[0])
    steps = [identity - eta * hb for hb in batches]
    return np.array([[np.mean([np.sum(left * (step @ right @ step.T)) for step in steps])
                      for right in basis] for left in basis])


def direct_coordinate(features: np.ndarray, batches: list[np.ndarray],
                      eta: float, target: int = 0) -> dict[str, float]:
    """Project explicit features, enumerate projected moments, solve constraints."""
    other = np.delete(features, target, axis=1)
    u, singular, _ = np.linalg.svd(other, full_matrices=False)
    active = singular > 1e-12 * singular[0]
    projector = u[:, active] @ u[:, active].T
    residual = (np.eye(len(projector)) - projector) @ features[:, target]
    private = float(residual @ residual)
    shared = float(np.linalg.norm(projector @ features[:, target])**2)
    tau = float(np.trace(projector @ np.mean(batches, axis=0)))
    second = float(np.mean([np.sum((projector @ hb @ projector)**2) for hb in batches]))
    rhs = np.eye(features.shape[1])[:, target]
    edit = np.linalg.lstsq(features.T, rhs, rcond=None)[0]
    private_projector = np.outer(residual, residual) / private
    projectors = [projector, private_projector]
    full_hessian = np.mean(batches, axis=0)
    traces = [float(np.trace(p @ full_hessian)) for p in projectors]
    compression = np.array([[np.mean([np.sum((left @ hb @ right)**2) for hb in batches])
                             / np.sqrt(traces[i] * traces[j])
                             for j, right in enumerate(projectors)]
                            for i, left in enumerate(projectors)])
    return {"private_sqnorm": private, "shared_sqnorm": shared,
            "shared_trace": tau, "shared_second_moment_trace": second,
            "shared_slack": 2 * tau - eta * second,
            "unit_tangent_edit_sqnorm": float(edit @ edit),
            "two_projection_ratio": float(eta * np.linalg.eigvalsh(compression)[-1] / 2),
            "constraint_max_error": float(np.max(np.abs(features.T @ edit - rhs)))}


def evolve_exact(batches: list[np.ndarray], eta: float, initial: np.ndarray,
                 steps: int) -> np.ndarray:
    """Direct second-moment recursion, without generalized-eigenvalue formulas."""
    transforms = [np.eye(len(initial)) - eta * hb for hb in batches]
    covariance = np.outer(initial, initial)
    traces = [float(np.trace(covariance))]
    for _ in range(steps):
        covariance = np.mean([a @ covariance @ a.T for a in transforms], axis=0)
        traces.append(float(np.trace(covariance)))
    return np.asarray(traces)


def evolve_monte_carlo(batches: list[np.ndarray], eta: float, initial: np.ndarray,
                       steps: int, trajectories: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent minibatch draws across trajectories and steps; no clipping."""
    rng = np.random.default_rng(seed)
    transforms = np.asarray([np.eye(len(initial)) - eta * hb for hb in batches])
    errors = np.tile(initial, (trajectories, 1))
    means, standard_errors = [float(initial @ initial)], [0.0]
    for _ in range(steps):
        choices = rng.integers(len(batches), size=trajectories)
        errors = np.einsum("nij,nj->ni", transforms[choices], errors)
        norms = np.sum(errors**2, axis=1)
        means.append(float(norms.mean()))
        standard_errors.append(float(norms.std(ddof=1) / np.sqrt(trajectories)))
    return np.asarray(means), np.asarray(standard_errors)


def orthogonal_concentration(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Independent direct edit solves for predefined equal-energy k-local edits.

    G=I implies eta_crit=2B and unit-norm prediction edits cost exactly one.
    The directional bound is eta/2*((1-p)*||b||^2/n+p*||b||_4^4/||b||^2).
    The direct parameter solve supplies the observed cost, not this identity.
    """
    n = config["n_examples"]
    features = np.eye(n)
    rows = []
    for batch in config["batch_sizes"]:
        operator = explicit_operator(features, batch)
        production = exact_generalized_sharpness(features.T @ features, n, 1, batch, tol=1e-12)
        critical = operator["critical_learning_rate"]
        p = (n - batch) / (batch * (n - 1))
        for fraction, support in itertools.product(config["learning_rate_fractions"], config["support_sizes"]):
            eta = fraction * critical
            change = np.zeros(n)
            change[:support] = 1 / np.sqrt(support)
            edit = np.linalg.lstsq(features.T, change, rcond=None)[0]
            cost = float(edit @ edit)
            energy = float(change @ change)
            fourth = float(np.sum(change**4))
            lower = eta / 2 * ((1 - p) * energy / n + p * fourth / energy)
            rows.append({"n_examples": n, "batch_size": batch, "support_size": support,
                "learning_rate_fraction": fraction, "learning_rate": eta,
                "explicit_critical_learning_rate": critical,
                "analytic_critical_learning_rate": float(2 * batch),
                "production_critical_learning_rate": production["critical_learning_rate"],
                "direct_edit_cost": cost, "prediction_energy": energy,
                "directional_lower_bound": lower, "lower_bound_to_cost_ratio": lower / cost,
                "constraint_max_error": float(np.max(np.abs(features.T @ edit - change))),
                "bound_violated": bool(lower > cost + 1e-10),
                "scope": "Conditional bound applies only at or below the exact stability threshold; above-threshold failures are retained."})
    return rows


def run_suite(config: dict[str, Any]) -> dict[str, Any]:
    rows, discrepancies = [], []
    edit_errors, threshold_errors = [], []
    for shared, private, anchor, batch in itertools.product(
            config["shared_sqnorms"], config["private_sqnorms"],
            config["anchor_sqnorms"], config["batch_sizes"]):
        features = controlled_features(shared, private, anchor)
        explicit = explicit_operator(features, batch)
        critical = explicit["critical_learning_rate"]
        measured = exact_generalized_sharpness(features.T @ features, 2, 1, batch, tol=1e-12)
        relative_threshold_error = abs(measured["critical_learning_rate"] / critical - 1)
        threshold_errors.append(relative_threshold_error)
        for fraction in config["learning_rate_fractions"]:
            eta = fraction * critical
            _, coordinates = analyze_gram(features.T @ features, 2, 1, eta, batch,
                                          rankone_steps=0, rankone_restarts=1)
            row = coordinates[0]
            independent = direct_coordinate(features, explicit["batches"], eta)
            other_coordinate = direct_coordinate(features, explicit["batches"], eta, target=1)
            for field in ("private_sqnorm", "shared_sqnorm", "shared_trace",
                          "shared_second_moment_trace", "shared_slack", "unit_tangent_edit_sqnorm"):
                # Scaling by max(1,|value|) keeps near-zero slack well-defined.
                discrepancies.append(abs(row[field] - independent[field]) / max(1, abs(independent[field])))
            spectrum = np.linalg.eigvalsh(transition_operator(explicit["batches"], eta, explicit["basis"]))
            for amplitude in config["edit_amplitudes"]:
                rhs = np.array([amplitude, 0.0])
                direct_edit = np.linalg.lstsq(features.T, rhs, rcond=None)[0]
                edit_errors.append(float(np.max(np.abs(features.T @ direct_edit - rhs))))
                edit_errors.append(abs(float(direct_edit @ direct_edit) - amplitude**2 / private))
            rows.append({
                "shared_sqnorm_setting": shared, "private_sqnorm_setting": private,
                "anchor_sqnorm_setting": anchor, "batch_size": batch,
                "learning_rate_fraction": fraction, "learning_rate": eta,
                "explicit_critical_learning_rate": critical,
                "production_critical_learning_rate": measured["critical_learning_rate"],
                "threshold_relative_error": relative_threshold_error,
                "production_solver_converged": measured["solver_converged"],
                "production_eigenpair_relative_residual": measured["eigenpair_relative_residual"],
                "transition_spectral_radius": float(np.max(np.abs(spectrum))),
                "linear_stability_class": "below_threshold" if fraction < 1 else "boundary" if fraction == 1 else "above_threshold",
                "shared_slack_fraction": independent["shared_slack"] / (2 * independent["shared_trace"]),
                "direct_unit_edit_cost": independent["unit_tangent_edit_sqnorm"],
                "direct_constraint_max_error": independent["constraint_max_error"],
                "target0_two_projection_ratio": independent["two_projection_ratio"],
                "max_all_coordinates_two_projection_ratio": max(independent["two_projection_ratio"], other_coordinate["two_projection_ratio"]),
                **{key: row[key] for key in ("private_sqnorm", "shared_sqnorm", "shared_slack",
                    "universal_capacity", "universal_ratio", "coupling_factor", "shared_private_capacity",
                    "shared_private_ratio", "shared_private_status")},
            })
    candidates = [r for r in rows if r["learning_rate_fraction"] <= .99 and r["shared_slack"] > 0]
    strongest = max(candidates, key=lambda r: r["coupling_factor"] or 0)
    dynamics_config = config["dynamics"]
    features = controlled_features(dynamics_config["shared_sqnorm"], dynamics_config["private_sqnorm"],
                                   dynamics_config["anchor_sqnorm"])
    explicit = explicit_operator(features, dynamics_config["batch_size"])
    initial = np.asarray(dynamics_config["initial_error"])
    dynamics = []
    for index, fraction in enumerate(dynamics_config["learning_rate_fractions"]):
        eta = fraction * explicit["critical_learning_rate"]
        expected = evolve_exact(explicit["batches"], eta, initial, dynamics_config["steps"])
        empirical, standard_error = evolve_monte_carlo(explicit["batches"], eta, initial,
            dynamics_config["steps"], dynamics_config["trajectories"], dynamics_config["seed"] + index)
        dynamics.append({"learning_rate_fraction": fraction, "learning_rate": eta,
            "critical_learning_rate": explicit["critical_learning_rate"],
            "steps": list(range(len(expected))), "exact_mean_square": expected.tolist(),
            "mc_mean_square": empirical.tolist(), "mc_standard_error": standard_error.tolist(),
            "final_mc_to_exact_ratio": float(empirical[-1] / expected[-1]),
            "final_mc_relative_standard_error": float(standard_error[-1] / empirical[-1]),
            "seed": dynamics_config["seed"] + index})
    below = [r for r in rows if r["learning_rate_fraction"] <= 1]
    above = [r for r in rows if r["learning_rate_fraction"] > 1]
    orthogonal = orthogonal_concentration(config["orthogonal"])
    # Fixed counterexample requested before execution: this distinguishes the
    # eta/2 consequence proved by stability from the manuscript's 2/eta typo.
    identity_features = np.eye(8)
    singleton = np.eye(8)[:, 0]
    direct_counterexample_edit = np.linalg.lstsq(identity_features.T, singleton, rcond=None)[0]
    counterexample = {
        "features": "G=I_8", "batch_size": 1, "learning_rate": .5,
        "critical_learning_rate": explicit_operator(identity_features, 1)["critical_learning_rate"],
        "direct_unit_edit_cost": float(direct_counterexample_edit @ direct_counterexample_edit),
        "correct_eta_over_two_lower_bound": .25,
        "incorrect_two_over_eta_lower_bound": 4.,
        "finding": "The printed 2/eta directional-edit prefactor fails at a stable point; the proved eta/2 prefactor holds. The suite tests eta/2; no manuscript source was edited.",
    }
    return {
        "config": config,
        "scope": "Fixed scalar linear features, exact interpolation. Tests conditional tangent statements, not SGD representation learning, nonlinear edits, or generalization.",
        "independence": "Parameter-space Hessians and all batches explicitly enumerated; dense generalized eigenvalues and direct covariance recursion independently check production Gram formulas. Direct parameter-space least-squares independently checks edit constraints and costs.",
        "monte_carlo_caveat": "Monte Carlo is a finite-sample descriptive check; multiplicative noise can have large variance. Standard errors are sample estimates, not rigorous guarantees. All outcomes retained.",
        "summary": {
            "geometry_count": len(threshold_errors), "grid_row_count": len(rows),
            "max_threshold_relative_error": max(threshold_errors),
            "max_scaled_coordinate_discrepancy": max(discrepancies),
            "max_direct_edit_absolute_error": max(edit_errors),
            "zero_coupling_rows": sum(r["coupling_factor"] == 0 for r in rows),
            "negative_or_zero_shared_slack_rows": sum(r["shared_slack"] <= 0 for r in rows),
            "stable_rows_violating_universal": sum(r["universal_ratio"] > 1 + 1e-9 for r in below),
            "stable_rows_violating_shared_private": sum(r["shared_private_ratio"] is not None and r["shared_private_ratio"] > 1 + 1e-9 for r in below),
            "unstable_rows_passing_target0_shared_private_ratio": sum(r["shared_private_ratio"] is not None and r["shared_private_ratio"] <= 1 + 1e-9 for r in above),
            "unstable_rows_passing_both_coordinates_two_projection_tests": sum(r["max_all_coordinates_two_projection_ratio"] <= 1 + 1e-9 for r in above),
            "strongest_stable_by_predeclared_rule": strongest,
            "orthogonal_directional_case_count": len(orthogonal),
            "orthogonal_stable_directional_violations": sum(r["bound_violated"] for r in orthogonal if r["learning_rate_fraction"] <= 1),
            "orthogonal_unstable_directional_violations": sum(r["bound_violated"] for r in orthogonal if r["learning_rate_fraction"] > 1),
        },
        "grid": rows, "dynamics": dynamics, "orthogonal_concentration": orthogonal,
        "manuscript_prefactor_counterexample": counterexample,
    }


def make_plots(result: dict[str, Any], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".cache" / "matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    fig.subplots_adjust(bottom=.23, wspace=.34, top=.84)
    fig.suptitle("Controlled linear features: covariance threshold, coupling, and actual SGD", fontsize=13)
    all_rows = result["grid"]
    unique = all_rows[::len(result["config"]["learning_rate_fractions"]) ]
    x = [r["explicit_critical_learning_rate"] for r in unique]
    y = [r["production_critical_learning_rate"] for r in unique]
    axes[0].scatter(x, y, s=18, alpha=.6)
    axes[0].plot([min(x), max(x)], [min(x), max(x)], "k--", linewidth=1)
    axes[0].set(xscale="log", yscale="log", xlabel="Explicit parameter-space threshold",
                ylabel="Production Gram threshold", title="Independent operators agree")
    # These three slices were fixed in advance; strongest-row summary is separately labelled.
    for a, t, h, label in [(0, 1, 1, "No shared component"), (1, .1, 1, "Moderate coupling"),
                           (4, .01, .25, "Small private component")]:
        selected = [r for r in all_rows if r["shared_sqnorm_setting"] == a
                    and r["private_sqnorm_setting"] == t and r["anchor_sqnorm_setting"] == h
                    and r["batch_size"] == 1 and r["learning_rate_fraction"] <= .99]
        axes[1].plot([r["learning_rate_fraction"] for r in selected],
                     [1 + (r["coupling_factor"] or 0) for r in selected], "o-", label=label)
    axes[1].set(yscale="log", xlabel=r"Learning rate / exact $\eta_{crit}$",
                ylabel="Stronger / universal cost lower bound", title="Shared slack can matter substantially")
    axes[1].legend(fontsize=7)
    for entry in result["dynamics"]:
        line, = axes[2].plot(entry["steps"], entry["exact_mean_square"],
                             label=f"Exact, fraction {entry['learning_rate_fraction']:g}")
        axes[2].errorbar(entry["steps"][::4], entry["mc_mean_square"][::4],
                        yerr=entry["mc_standard_error"][::4], fmt="o", markersize=3,
                        linewidth=.8, color=line.get_color(),
                        label=f"Monte Carlo ±1 SE, fraction {entry['learning_rate_fraction']:g}")
    axes[2].set(yscale="log", xlabel="SGD updates", ylabel=r"$E\|e_t\|^2$", title="Below / above the exact threshold")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.05, .04, "Two fixed scalar examples; no representation learning. Every predefined grid row is saved, including weak/zero coupling and unstable cases.\n"
             "Monte Carlo uses independent minibatches across 100,000 paths; sample means and estimated SEs can miss rare multiplicative-noise events. Exact propagation is the reference.", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"synthetic_validation.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    fig.subplots_adjust(bottom=.22, wspace=.32)
    for batch in result["config"]["orthogonal"]["batch_sizes"]:
        for axis, fraction in zip(axes, (1., 1.1)):
            selected = [r for r in result["orthogonal_concentration"]
                        if r["batch_size"] == batch and r["learning_rate_fraction"] == fraction]
            axis.plot([r["support_size"] for r in selected],
                      [r["lower_bound_to_cost_ratio"] for r in selected], "o-", label=f"B={batch}")
    for axis, title in zip(axes, ("At exact stability threshold", "Above threshold (bound not assumed)")):
        axis.axhline(1, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="Support size k, fixed unit prediction energy", ylabel="Directional lower bound / direct edit cost",
                 title=title, xticks=[1, 2, 4, 8])
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.06, .05, "Eight orthogonal scalar features. Direct constrained parameter solves give cost 1. Singleton edits saturate at eta=2B.\n"
             "Localized edits incur the strongest stability-implied penalty; above-threshold failures show why the stability assumption cannot be dropped.", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"orthogonal_concentration.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic_validation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_validation"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    result = run_suite(config)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.output / "grid.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["grid"][0]))
        writer.writeheader()
        writer.writerows(result["grid"])
    with (args.output / "orthogonal_concentration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["orthogonal_concentration"][0]))
        writer.writeheader()
        writer.writerows(result["orthogonal_concentration"])
    make_plots(result, args.output)
    print(json.dumps(result["summary"], indent=2))
    for row in result["dynamics"]:
        print(json.dumps({key: row[key] for key in ("learning_rate_fraction", "final_mc_to_exact_ratio", "final_mc_relative_standard_error")}))


if __name__ == "__main__":
    main()
