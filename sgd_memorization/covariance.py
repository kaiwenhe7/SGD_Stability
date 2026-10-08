"""Optional numerical mean-square threshold for an interpolating tangent model.

This computes the full generalized covariance eigenvalue, rather than a
rank-one necessary-condition search. The stochastic Hessians are those of
the prediction-gradient Gauss-Newton model with independently sampled
uniform minibatches of distinct examples. They equal the loss Hessians at
exact squared-loss interpolation. Numerical eigenvalue estimates are not
rigorous stability certificates for the tangent model or actual network.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh


def exact_generalized_sharpness(
    K: np.ndarray, n_examples: int, n_outputs: int, batch_size: int,
    rank_rtol: float = 1e-10, tol: float = 1e-7, maxiter: int = 300,
    max_rank: int = 512,
) -> dict[str, Any]:
    """Estimate the exact tangent covariance threshold from its full operator.

    For H = E[H_batch], define C(X) = (H X + X H)/2 and
    D(X) = E[H_batch X H_batch]. The threshold is
    ``eta_star = 2 / lambda_max(C**(-1/2) D C**(-1/2))`` on the active
    symmetric feature space. Null feature directions remain neutral and are
    excluded; no ridge regularization is added.

    The mathematical operator characterizes the tangent model exactly, but
    its largest eigenvalue is found by an iterative numerical solver. The
    returned residual and convergence flag must accompany the estimate.
    Flattened antisymmetric matrices are annihilated to avoid inadvertently
    testing covariance directions outside the symmetric space.
    """
    started = time.perf_counter()
    if n_examples < 1 or n_outputs < 1 or not 1 <= batch_size <= n_examples:
        raise ValueError("Positive dimensions and 1 <= batch_size <= n_examples are required")
    if not 0 < rank_rtol < 1 or not 0 < tol < 1 or maxiter < 1 or max_rank < 1:
        raise ValueError("Require rank_rtol and tol in (0, 1), positive maxiter and max_rank")
    k = np.asarray(K, dtype=np.float64)
    m = n_examples * n_outputs
    if k.shape != (m, m) or not np.isfinite(k).all():
        raise ValueError(f"K must be a finite matrix of shape {(m, m)}")
    scale = max(float(np.max(np.abs(k))), np.finfo(float).tiny)
    if np.max(np.abs(k - k.T)) > 1e-8 * scale:
        raise ValueError("K must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh((k + k.T) / 2)
    spectral_scale = max(float(eigenvalues[-1]), scale)
    if eigenvalues[0] < -1e-8 * spectral_scale:
        raise ValueError("K must be positive semidefinite")
    cutoff = rank_rtol * spectral_scale
    active = eigenvalues > cutoff
    eig = eigenvalues[active]
    vectors = eigenvectors[:, active]
    rank = len(eig)
    if rank > max_rank:
        raise ValueError(
            f"Active Gram rank {rank} exceeds max_rank={max_rank}; "
            "the covariance diagnostic is intended for small systems"
        )
    p = (n_examples - batch_size) / (batch_size * (n_examples - 1)) if n_examples > 1 else 0.0
    common: dict[str, Any] = {
        "numerical_rank": int(rank),
        "rank_rtol": float(rank_rtol),
        "rank_cutoff": float(cutoff),
        "n_examples": int(n_examples),
        "n_outputs": int(n_outputs),
        "batch_size": int(batch_size),
        "sampling_p": float(p),
        "symmetric_covariance_dimension": int(rank * (rank + 1) // 2),
        "solver_tolerance": float(tol),
        "solver_maxiter": int(maxiter),
        "numerical_rigorous_certificate": False,
        "curvature": "prediction-gradient Gauss-Newton; equals loss Hessian at exact squared-loss interpolation",
        "interpretation": "Numerical full covariance threshold for the frozen interpolating tangent model; not a rigorous certificate or a stability test of nonlinear network dynamics.",
    }
    if rank == 0:
        return {
            **common, "generalized_sharpness": 0.0,
            "critical_learning_rate": None,
            "active_threshold_unbounded": True,
            "solver": "zero_active_rank", "solver_converged": True,
            "eigenpair_absolute_residual": 0.0,
            "eigenpair_relative_residual": 0.0,
            "operator_matvec_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
    lam = eig / n_examples
    if batch_size == n_examples:
        # With one deterministic Hessian, the covariance condition is precisely
        # eta * lambda_max(H) <= 2, including the marginal equality case.
        largest = float(lam[-1])
        return {
            **common, "generalized_sharpness": largest,
            "critical_learning_rate": float(2 / largest),
            "active_threshold_unbounded": False,
            "solver": "analytic_full_batch", "solver_converged": True,
            "eigenpair_absolute_residual": 0.0,
            "eigenpair_relative_residual": 0.0,
            "operator_matvec_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }

    features = np.ascontiguousarray(np.sqrt(eig)[:, None] * vectors.T)
    grouped = features.reshape(rank, n_examples, n_outputs)
    inverse_sqrt_c = np.sqrt(2 / (lam[:, None] + lam[None, :]))
    matvec_count = 0

    def matvec(value: np.ndarray) -> np.ndarray:
        nonlocal matvec_count
        matvec_count += 1
        x = np.asarray(value, dtype=np.float64).reshape(rank, rank)
        x = (x + x.T) * 0.5
        x = x * inverse_sqrt_c
        transformed = (x @ features).reshape(rank, n_examples, n_outputs)
        # For each i, construct Z_i.T X Z_i in output coordinates. Multiplying
        # the assembled blocks back by Z_i avoids storing n dense F_i matrices.
        blocks = np.einsum("rnc,rnd->ncd", grouped, transformed, optimize=True)
        weighted_features = np.einsum("rnc,ncd->rnd", grouped, blocks, optimize=True)
        block_term = weighted_features.reshape(rank, m) @ features.T
        d_x = (1 - p) * lam[:, None] * x * lam[None, :] + (p / n_examples) * block_term
        result = d_x * inverse_sqrt_c
        return ((result + result.T) * 0.5).reshape(-1)

    dimension = rank * rank
    operator = LinearOperator((dimension, dimension), matvec=matvec, rmatvec=matvec, dtype=np.float64)
    initial = np.eye(rank).reshape(-1) / np.sqrt(rank)
    converged = True
    if rank == 1:
        vector = initial
        largest = float(matvec(vector)[0])
        solver = "analytic_rank_one"
    else:
        solver = "scipy_eigsh_full_symmetric_covariance"
        try:
            values, directions = eigsh(operator, k=1, which="LA", v0=initial,
                                       tol=tol, maxiter=maxiter)
        except ArpackNoConvergence as exc:
            converged = False
            if exc.eigenvalues is None or len(exc.eigenvalues) == 0:
                raise RuntimeError(
                    "Covariance eigensolver did not converge and returned no usable eigenpair"
                ) from exc
            values, directions = exc.eigenvalues, exc.eigenvectors
        index = int(np.argmax(values))
        largest = float(values[index])
        vector = directions[:, index]
    residual = float(np.linalg.norm(matvec(vector) - largest * vector))
    if not np.isfinite(largest) or largest <= 0:
        raise RuntimeError("Covariance eigensolver returned a nonpositive or nonfinite active eigenvalue")
    return {
        **common, "generalized_sharpness": largest,
        "critical_learning_rate": float(2 / largest),
        "active_threshold_unbounded": False,
        "solver": solver, "solver_converged": bool(converged),
        "eigenpair_absolute_residual": residual,
        "eigenpair_relative_residual": float(residual / largest),
        "operator_matvec_count": int(matvec_count),
        "elapsed_seconds": time.perf_counter() - started,
    }
