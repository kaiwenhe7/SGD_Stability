"""Gram diagnostics for local SGD stability and isolated prediction edits.

Columns of the prediction Jacobian are ordered sample first, output second.
The per-example loss is ``0.5 * sum_c (f_c - y_c)**2``; consequently the
interpolating Hessian is ``H = sum_i sum_c g_ic g_ic.T / n``.  A minibatch is
uniformly sampled without replacement, independently at every iteration.

The diagnostics below are necessary stability conditions. Passing them does
not establish mean-square stability. The optimized rank-one diagnostic is a
numerical lower bound on the maximum rank-one test, not a global certificate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _validate_gram(
    gram: np.ndarray, n_examples: int, n_outputs: int, rank_rtol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if n_examples < 1 or n_outputs < 1:
        raise ValueError("n_examples and n_outputs must be positive")
    if not 0 < rank_rtol < 1:
        raise ValueError("rank_rtol must be between zero and one")
    k = np.asarray(gram, dtype=np.float64)
    size = n_examples * n_outputs
    if k.shape != (size, size):
        raise ValueError(f"expected a ({size}, {size}) Gram matrix, got {k.shape}")
    if not np.isfinite(k).all():
        raise ValueError("Gram matrix contains nonfinite entries")
    scale = max(float(np.max(np.abs(k))), np.finfo(float).tiny)
    if float(np.max(np.abs(k - k.T))) > 1e-8 * scale:
        raise ValueError("Gram matrix is not symmetric")
    k = (k + k.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(k)
    spectral_scale = max(float(eigenvalues[-1]), scale)
    if float(eigenvalues[0]) < -1e-8 * spectral_scale:
        raise ValueError("Gram matrix is not positive semidefinite")
    cutoff = rank_rtol * spectral_scale
    active = eigenvalues > cutoff
    # Roundoff below the cutoff determines the reported numerical rank. No
    # ridge is introduced, so unavailable isolated edits stay unavailable.
    return k, eigenvalues[active], eigenvectors[:, active], cutoff


def minibatch_moments(
    gram: np.ndarray, n_examples: int, n_outputs: int, batch_size: int
) -> dict[str, float]:
    """Return tr(H) and E[tr(H_batch**2)] from the prediction Gram.

    This helper assumes the Gram was validated by the caller. Its block term
    includes all output cross terms within each example's Hessian.
    """
    if not 1 <= batch_size <= n_examples:
        raise ValueError("batch_size must lie between 1 and n_examples")
    k = np.asarray(gram, dtype=np.float64)
    if k.shape != (n_examples * n_outputs,) * 2:
        raise ValueError("Gram shape does not match n_examples * n_outputs")
    p = (
        (n_examples - batch_size) / (batch_size * (n_examples - 1))
        if n_examples > 1
        else 0.0
    )
    full_norm_sq = float(np.sum(k * k))
    diagonal_block_norm_sq = 0.0
    for i in range(n_examples):
        block = k[
            i * n_outputs : (i + 1) * n_outputs,
            i * n_outputs : (i + 1) * n_outputs,
        ]
        diagonal_block_norm_sq += float(np.sum(block * block))
    return {
        "p": float(p),
        "trace_hessian": float(np.trace(k) / n_examples),
        "hessian_frobenius_sq": full_norm_sq / n_examples**2,
        "sum_example_hessian_frobenius_sq": diagonal_block_norm_sq,
        "minibatch_second_moment_trace": (
            (1.0 - p) * full_norm_sq / n_examples**2
            + p * diagonal_block_norm_sq / n_examples
        ),
    }


def _private_components(
    k: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    size = len(k)
    q = np.maximum(np.diag(k).copy(), 0.0)
    rank = len(eigenvalues)
    # Range membership is an amplitude test, independent of the eigenvalue
    # cutoff used to define numerical rank. Squared null mass, especially
    # 1 - sum(V**2), both loses tiny dependencies to cancellation and applies
    # an inadvertently square-root-sized feasibility tolerance.
    feasibility_tol = 10 * size * np.finfo(float).eps
    if rank == size:
        inverse_diagonal = np.sum(eigenvectors**2 / eigenvalues[None, :], axis=1)
        feasible = np.ones(size, dtype=bool)
        range_residual = np.zeros(size)
    elif rank:
        # e_m must belong to range(K) for an edit that preserves every other
        # output coordinate. Only then is 1/(K^dagger)_mm a private norm.
        null_projector = np.eye(size) - eigenvectors @ eigenvectors.T
        range_residual = np.linalg.norm(null_projector, axis=0)
        feasible = range_residual <= feasibility_tol
        inverse_diagonal = np.sum(eigenvectors**2 / eigenvalues[None, :], axis=1)
    else:
        feasible = np.zeros(size, dtype=bool)
        inverse_diagonal = np.zeros(size)
        range_residual = np.ones(size)
    private = np.zeros(size)
    positive = feasible & (inverse_diagonal > 0)
    private[positive] = 1.0 / inverse_diagonal[positive]
    private = np.minimum(private, q)
    shared = np.maximum(q - private, 0.0)
    return private, shared, feasible, range_residual, feasibility_tol


def private_geometry(
    gram: np.ndarray, n_examples: int, n_outputs: int = 1,
    rank_rtol: float = 1e-10,
) -> dict[str, Any]:
    """Compute isolated-coordinate geometry at the reported numerical rank.

    ``private_sqnorm[m]`` is the squared residual after projecting gradient m
    onto all other prediction gradients. A rank-deficient matrix can have
    both feasible and infeasible coordinates; the latter receive zero private
    norm and ``None`` edit cost. No regularization is used. Eigenvalues below
    ``rank_rtol`` times the spectral scale are discarded; this can classify
    mathematically feasible but ill-conditioned edits as unavailable. Within
    that retained subspace, range membership uses the reported float64
    roundoff tolerance on the *norm* of the coordinate's null projection,
    rather than the eigenvalue cutoff. Values below this resolution cannot
    distinguish exact from approximate dependencies.
    """
    k, eig, vec, cutoff = _validate_gram(gram, n_examples, n_outputs, rank_rtol)
    private, shared, feasible, range_residual, feasibility_tol = _private_components(k, eig, vec)
    return {
        "rank": int(len(eig)),
        "rank_cutoff": float(cutoff),
        "private_sqnorm": private.tolist(),
        "shared_sqnorm": shared.tolist(),
        "isolated_edit_feasible": feasible.tolist(),
        "isolated_edit_range_residual": range_residual.tolist(),
        "isolated_edit_feasibility_tolerance": float(feasibility_tol),
        "unit_tangent_edit_sqnorm": [
            float(1.0 / t) if ok and t > 0 else None
            for t, ok in zip(private, feasible)
        ],
    }


def _rankone_values(
    vectors: np.ndarray, features: np.ndarray, n_examples: int,
    n_outputs: int, p: float,
) -> np.ndarray:
    projections = vectors @ features
    h_i = np.sum(projections.reshape(len(vectors), n_examples, n_outputs)**2, axis=2)
    mean_h = h_i.mean(axis=1)
    moment = (1 - p) * mean_h**2 + p * np.mean(h_i**2, axis=1)
    return np.divide(moment, 2 * mean_h, out=np.zeros_like(moment), where=mean_h > 0)


def _optimize_rankone(
    features: np.ndarray, n_examples: int, n_outputs: int, p: float,
    steps: int, restarts: int, seed: int,
) -> tuple[float, float]:
    """Search sphere directions; return top-H and best found critical ratios.

    Returned values omit eta: eta * value > 1 violates the corresponding
    necessary condition. Features use the active Hessian eigenbasis, ordered
    by ascending eigenvalue, so the last basis direction is top-H.
    """
    rank = features.shape[0]
    if rank == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    starts = rng.standard_normal((restarts, rank))
    starts[0] = 0.0
    starts[0, -1] = 1.0
    starts /= np.linalg.norm(starts, axis=1, keepdims=True)
    initial = _rankone_values(starts, features, n_examples, n_outputs, p)
    top_value = float(initial[0])
    best = float(np.max(initial))
    if steps == 0 or rank == 1:
        return top_value, best
    z = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float64)
    directions = torch.nn.Parameter(torch.as_tensor(starts, dtype=torch.float64))
    optimizer = torch.optim.Adam([directions], lr=0.05)
    scale = max(float(np.sum(features[-1]**2) / n_examples), np.finfo(float).tiny)
    for _ in range(steps):
        optimizer.zero_grad()
        v = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
        prediction_changes = v @ z
        h_i = prediction_changes.reshape(restarts, n_examples, n_outputs).square().sum(2)
        mean_h = h_i.mean(1)
        moment = (1 - p) * mean_h.square() + p * h_i.square().mean(1)
        values = moment / (2 * mean_h)
        best = max(best, float(values.detach().max()))
        loss = -values.mean() / scale
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            directions.div_(torch.linalg.vector_norm(directions, dim=1, keepdim=True))
    final = _rankone_values(directions.detach().numpy(), features, n_examples, n_outputs, p)
    best = max(best, float(np.max(final)))
    return top_value, best


def analyze_gram(
    K: np.ndarray, n_examples: int, n_outputs: int,
    learning_rate: float, batch_size: int, rank_rtol: float = 1e-10,
    rankone_steps: int = 300, rankone_restarts: int = 6, seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return JSON-safe summary and sample/output-coordinate diagnostics.

    Shared-private quantities refer to preserving every *other output
    coordinate*, including the other outputs of the same training example.
    A ``ratio`` above one violates a necessary condition for local mean-square
    stability at interpolation. All metrics use the supplied learning rate
    and minibatch size; this function does not assess residual interpolation
    error or simulate perturbations.
    """
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if rankone_steps < 0 or rankone_restarts < 1:
        raise ValueError("rankone_steps >= 0 and rankone_restarts >= 1 are required")
    k, eig, vec, cutoff = _validate_gram(K, n_examples, n_outputs, rank_rtol)
    moments = minibatch_moments(k, n_examples, n_outputs, batch_size)
    private, shared, feasible, range_residual, feasibility_tol = _private_components(k, eig, vec)
    q = np.maximum(np.diag(k), 0.0)
    p = moments["p"]
    tau = moments["trace_hessian"]
    second_moment = moments["minibatch_second_moment_trace"]
    full_frob = moments["hessian_frobenius_sq"] * n_examples**2
    block_frob = moments["sum_example_hessian_frobenius_sq"]
    features = np.sqrt(eig)[:, None] * vec.T
    top_rankone, optimized_rankone = _optimize_rankone(
        features, n_examples, n_outputs, p, rankone_steps, rankone_restarts, seed
    )
    lambda_max = float(eig[-1] / n_examples) if len(eig) else 0.0
    universal_capacity = float(2 * batch_size / learning_rate)
    per_coordinate: list[dict[str, Any]] = []
    for m in range(len(k)):
        t, a = float(private[m]), float(shared[m])
        # Projecting out r_m alters only the m,m Gram entry: L = K - t e_m e_m.T.
        norm_reduction = 2 * t * float(q[m]) - t * t
        tau_p = max(float(tau - t / n_examples), 0.0)
        d_p = max(float(
            (1 - p) * (full_frob - norm_reduction) / n_examples**2
            + p * (block_frob - norm_reduction) / n_examples
        ), 0.0)
        slack = 2 * tau_p - learning_rate * d_p
        if slack > 0:
            coupling = learning_rate * a * a / (batch_size * n_examples * slack)
            stronger_capacity = universal_capacity / (1 + coupling)
            stronger_ratio = t / stronger_capacity
            stronger_status = "applicable"
        elif tau_p == 0 and d_p == 0 and a == 0:
            # Empty shared space gives the universal private condition exactly.
            coupling = 0.0
            stronger_capacity = universal_capacity
            stronger_ratio = t / stronger_capacity
            stronger_status = "empty_shared_space"
        else:
            coupling = None
            stronger_capacity = None
            stronger_ratio = None
            stronger_status = "negative_shared_slack" if slack < 0 else "zero_shared_slack"
        per_coordinate.append({
            "coordinate_index": int(m),
            "example_index": int(m // n_outputs),
            "output_index": int(m % n_outputs),
            "gradient_sqnorm": float(q[m]),
            "private_sqnorm": t,
            "shared_sqnorm": a,
            "private_fraction": float(t / q[m]) if q[m] > 0 else 0.0,
            "isolated_edit_feasible": bool(feasible[m]),
            "isolated_edit_range_residual": float(range_residual[m]),
            "unit_tangent_edit_sqnorm": float(1 / t) if feasible[m] and t > 0 else None,
            "shared_trace": tau_p,
            "shared_second_moment_trace": d_p,
            "shared_slack": float(slack),
            "universal_capacity": universal_capacity,
            "universal_ratio": float(learning_rate * t / (2 * batch_size)),
            "coupling_factor": None if coupling is None else float(coupling),
            "shared_private_capacity": None if stronger_capacity is None else float(stronger_capacity),
            "shared_private_ratio": None if stronger_ratio is None else float(stronger_ratio),
            "shared_private_status": stronger_status,
        })
    shared_ratios = [r["shared_private_ratio"] for r in per_coordinate if r["shared_private_ratio"] is not None]
    trace_ratio = learning_rate * second_moment / (2 * tau) if tau > 0 else 0.0
    summary: dict[str, Any] = {
        "n_examples": int(n_examples),
        "n_outputs": int(n_outputs),
        "n_coordinates": int(len(k)),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "numerical_rank": int(len(eig)),
        "rank_rtol": float(rank_rtol),
        "rank_cutoff": float(cutoff),
        "isolated_edit_feasible_count": int(np.sum(feasible)),
        "isolated_edit_feasibility_tolerance": float(feasibility_tol),
        "lambda_max_hessian": lambda_max,
        **moments,
        "full_hessian_sharpness_ratio": float(learning_rate * lambda_max / 2),
        "trace_stochastic_ratio": float(trace_ratio),
        "trace_stochastic_critical_learning_rate": float(2 * tau / second_moment) if second_moment > 0 else None,
        "top_hessian_rankone_ratio": float(learning_rate * top_rankone),
        "top_hessian_rankone_critical_learning_rate": float(1 / top_rankone) if top_rankone > 0 else None,
        "optimized_rankone_ratio": float(learning_rate * optimized_rankone),
        "optimized_rankone_critical_learning_rate": float(1 / optimized_rankone) if optimized_rankone > 0 else None,
        "optimized_rankone_is_global_certificate": False,
        "rankone_steps": int(rankone_steps),
        "rankone_restarts": int(rankone_restarts),
        "rankone_seed": int(seed),
        "max_universal_private_ratio": float(max(r["universal_ratio"] for r in per_coordinate)),
        "max_shared_private_ratio": float(max(shared_ratios)) if shared_ratios else None,
        "negative_shared_slack_count": int(sum(r["shared_slack"] < 0 for r in per_coordinate)),
        "zero_shared_slack_count": int(sum(r["shared_slack"] == 0 for r in per_coordinate)),
        "interpretation": "Ratios above one violate necessary local mean-square stability conditions; passing is not a stability certificate.",
    }
    return summary, per_coordinate
