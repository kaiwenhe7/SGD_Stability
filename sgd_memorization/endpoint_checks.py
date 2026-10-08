"""Independent numerical checks on saved MNIST endpoints.

This module never trains or changes a checkpoint. Its only optional output is
``endpoint_checks.json`` beside the saved checkpoint. All Gram products and
residual-Hessian block norms accumulate in float64; activation gates and
predictions are evaluated in the checkpoint's original precision.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .data import load_subset
from .model import OneHiddenMLP, prediction_gram


@lru_cache(maxsize=8)
def _cached_training_subset(
    data_root: str, classes: tuple[int, ...], per_class: int, dataset_seed: int,
) -> tuple[Tensor, Tensor, np.ndarray]:
    # Consumers only read these tensors. This avoids repeatedly decompressing
    # MNIST across the paired endpoints while preserving saved sample order.
    return load_subset(data_root, list(classes), per_class, dataset_seed)


@torch.no_grad()
def fixed_gate_hessian_residual(
    model: OneHiddenMLP, x: Tensor, targets: Tensor,
) -> dict[str, Any]:
    """Measure ``||H_loss - H_GN||_2`` for half-sum MSE, averaged over n.

    With all ReLU gates fixed, residual curvature couples each hidden unit's
    first-layer parameters only to that unit's output weights. Its block is
    ``[[0, M_h], [M_h.T, 0]]``, where
    ``M_h = X_aug.T @ (gate_h * (prediction - target)) / n``.
    Thus the exact operator norm is ``max_h sigma_max(M_h)``. At a zero
    preactivation this is a fixed-gate derivative convention, not a claim
    that a classical Hessian exists.
    """
    parameter = model.fc1.weight
    x = x.detach().to(device=parameter.device, dtype=parameter.dtype)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != model.fc1.in_features:
        raise ValueError("x must contain at least one correctly shaped example")
    prediction = model(x)
    targets = targets.detach().to(device=prediction.device, dtype=prediction.dtype)
    if targets.shape != prediction.shape:
        raise ValueError("targets must have the same shape as model predictions")
    preactivation = model.fc1(x)
    residual = (prediction - targets).to(device="cpu", dtype=torch.float64)
    z = preactivation.to(device="cpu", dtype=torch.float64)
    gate = (z > 0).to(torch.float64)
    x64 = x.to(device="cpu", dtype=torch.float64)
    if not (torch.isfinite(residual).all() and torch.isfinite(z).all()):
        raise ValueError("Nonfinite predictions or preactivations cannot be diagnosed")
    x_aug = torch.cat((x64, torch.ones((len(x64), 1), dtype=torch.float64)), dim=1)
    weighted_residual = gate.T[:, :, None] * residual[None, :, :]
    blocks = torch.einsum("ni,hnc->hic", x_aug, weighted_residual) / len(x64)
    block_norms = torch.linalg.svdvals(blocks)[:, 0]
    # Individual-example loss Hessians have the same hidden-unit block
    # structure, without averaging by n. Every active block has this norm.
    example_norms = torch.linalg.vector_norm(x_aug, dim=1) * torch.linalg.vector_norm(residual, dim=1)
    example_norms *= gate.any(dim=1)
    zero_count = int((z == 0).sum())
    return {
        "mean_residual_hessian_operator_norm": float(block_norms.max()),
        "max_example_residual_hessian_operator_norm": float(example_norms.max()),
        "train_loss_recomputed": float(residual.square().sum() / (2 * len(x64))),
        "max_abs_training_residual": float(residual.abs().max()),
        "baseline_min_abs_preactivation": float(z.abs().min()),
        "baseline_zero_preactivation_count": zero_count,
        "activation_gate_count": int(z.numel()),
        "baseline_near_zero_preactivation_counts": {
            f"{threshold:g}": int((z.abs() <= threshold).sum())
            for threshold in (1e-12, 1e-10, 1e-8)
        },
        "relu_boundary_detected": zero_count > 0,
        "hessian_interpretation": (
            "Exact residual-Hessian block norm within the current ReLU region; "
            "Gauss-Newton equals the loss Hessian only when residual curvature vanishes. "
            "At zero preactivations this is a fixed-gate convention. A small mean "
            "residual norm alone does not certify stochastic nonlinear stability."
        ),
    }


def inspect_endpoint(
    run_directory: str | Path, *, data_root: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Inspect one saved run, optionally writing its derived checks JSON.

    The saved subset indices and training labels are validated before use.
    Corrupted-label runs require their original ``labels.npz``. Writes are
    restricted to this project's folder; ``write=False`` is entirely read-only.
    """
    directory = Path(run_directory).resolve()
    project = Path(__file__).resolve().parents[1]
    if write and not directory.is_relative_to(project):
        raise ValueError("Endpoint checks may only be written inside the project folder")
    experiment = directory.parent
    config = json.loads((experiment / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    checkpoint_path = directory / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["config"] != config:
        raise ValueError("Checkpoint configuration differs from its saved experiment")
    for key in ("seed", "batch_size", "learning_rate"):
        if checkpoint[key] != summary[key]:
            raise ValueError(f"Checkpoint and summary disagree on {key}")
    state = checkpoint["state_dict"]
    dtype = state["fc1.weight"].dtype
    if dtype not in (torch.float32, torch.float64) or any(value.dtype != dtype for value in state.values()):
        raise ValueError("Endpoint parameters must consistently use float32 or float64")
    model = OneHiddenMLP(state["fc1.weight"].shape[1], state["fc1.weight"].shape[0], state["fc2.weight"].shape[0]).to(dtype=dtype)
    model.load_state_dict(state)
    model.eval()
    root = Path(data_root).resolve() if data_root is not None else project / "data"
    x, clean_labels, indices = _cached_training_subset(
        str(root), tuple(config["classes"]), config["samples_per_class"], config["dataset_seed"]
    )
    subset_path = experiment / "subset_indices.npz"
    with np.load(subset_path) as saved:
        if not np.array_equal(saved["train"], indices):
            raise ValueError("Loaded training subset differs from saved indices")
    labels_path = experiment / "labels.npz"
    if labels_path.exists():
        with np.load(labels_path) as saved:
            if not np.array_equal(saved["clean_train"], clean_labels.numpy()):
                raise ValueError("Loaded clean labels differ from saved labels")
            labels = torch.from_numpy(saved["train"].copy()).long()
            if labels.shape != clean_labels.shape or not np.array_equal(saved["corrupted"], labels.numpy() != clean_labels.numpy()):
                raise ValueError("Saved label corruption mask is inconsistent")
    elif config.get("label_corruption_fraction", 0.0):
        raise ValueError("Corrupted-label endpoint is missing its saved labels")
    else:
        labels = clean_labels
    targets = torch.nn.functional.one_hot(labels, model.fc2.out_features).to(dtype=dtype)
    kernel = prediction_gram(model, x)
    eigenvalues = np.linalg.eigvalsh((kernel + kernel.T) / 2)
    spectral_scale = max(float(eigenvalues[-1]), float(np.abs(kernel).max()), np.finfo(float).tiny)
    rank_checks = []
    for rtol in (1e-8, 1e-10, 1e-12):
        active = eigenvalues[eigenvalues > rtol * spectral_scale]
        rank_checks.append({
            "rank_rtol": rtol, "rank_cutoff": float(rtol * spectral_scale),
            "numerical_rank": int(len(active)),
            "min_retained_eigenvalue": float(active[0]) if len(active) else None,
            "retained_condition_number": float(active[-1] / active[0]) if len(active) else None,
        })
    hessian = fixed_gate_hessian_residual(model, x, targets)
    gn_top = max(float(eigenvalues[-1] / len(x)), 0.0)
    gram_path = directory / "gram.npy"
    if gram_path.exists():
        saved_gram = np.load(gram_path)
        if saved_gram.shape != kernel.shape:
            raise ValueError("Saved Gram shape differs from the endpoint prediction Gram")
        delta = saved_gram - kernel
        saved_agreement = {
            "max_absolute_difference": float(np.abs(delta).max()),
            "relative_frobenius_difference": float(np.linalg.norm(delta) / max(np.linalg.norm(kernel), np.finfo(float).tiny)),
            "matches_rtol_1e10_atol_1e12": bool(np.allclose(saved_gram, kernel, rtol=1e-10, atol=1e-12)),
            "saved_gram_sha256": hashlib.sha256(gram_path.read_bytes()).hexdigest(),
        }
    else:
        saved_agreement = {"status": "missing_saved_gram"}
    result = {
        "schema_version": 1,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_dtype": str(dtype),
        "precision": "Native checkpoint predictions and gates; float64 Gram and residual-block accumulation",
        "n_examples": len(x), "n_outputs": model.fc2.out_features,
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "rank_sensitivity": rank_checks,
        "gram_min_eigenvalue": float(eigenvalues[0]),
        "gram_max_eigenvalue": float(eigenvalues[-1]),
        "gram_condition_number": float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else None,
        "saved_gram_agreement": saved_agreement,
        "lambda_max_gauss_newton": gn_top,
        "mean_residual_hessian_to_gauss_newton_top_ratio": hessian["mean_residual_hessian_operator_norm"] / gn_top if gn_top > 0 else None,
        "saved_train_loss": summary.get("train_loss"),
        **hessian,
    }
    if write:
        destination = directory / "endpoint_checks.json"
        # Atomic replacement uses a temporary file beside the sole output.
        temporary = directory / ".endpoint_checks.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)
    return result
