"""Small ReLU network and exact, sample-major prediction tangent geometry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import functional_call


class OneHiddenMLP(nn.Module):
    """One-hidden-layer network with ordinary PyTorch linear initialization."""

    def __init__(
        self, input_dim: int = 784, hidden_width: int = 1024, output_dim: int = 4
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_width, output_dim) < 1:
            raise ValueError("All network dimensions must be positive.")
        self.fc1 = nn.Linear(input_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _model_input(model: OneHiddenMLP, x: Tensor) -> Tensor:
    if x.ndim != 2 or x.shape[1] != model.fc1.in_features:
        raise ValueError("x must have shape (number_of_samples, input_dim).")
    return x.detach().to(device=model.fc1.weight.device, dtype=model.fc1.weight.dtype)


@torch.no_grad()
def prediction_gram(model: OneHiddenMLP, x: Tensor) -> np.ndarray:
    """Return ``K = J J.T`` without constructing the parameter Jacobian.

    Rows of J enumerate (sample, output) in sample-major order. K includes
    derivatives with respect to both weight matrices and both bias vectors.
    The forward pass uses the model's dtype and device, preserving its ReLU
    gates and hidden activation values. Gram products accumulate in float64
    on CPU. Promoting the model before this call instead measures the
    promoted model, whose gates may differ near a ReLU boundary.
    The ReLU derivative at zero is zero, matching PyTorch autograd.
    """
    x = _model_input(model, x)
    preactivation = model.fc1(x)
    hidden = preactivation.clamp_min(0).to(device="cpu", dtype=torch.float64)
    gate = (preactivation > 0).to(device="cpu", dtype=torch.float64)
    x = x.to(device="cpu", dtype=torch.float64)
    weight2 = model.fc2.weight.detach().to(device="cpu", dtype=torch.float64)

    # The first-layer contribution includes shared output-weight factors;
    # the second-layer contribution has disjoint parameters across outputs.
    gated_weights = gate[:, None, :] * weight2[None, :, :]
    first_layer = torch.einsum("ich,jdh->icjd", gated_weights, gated_weights)
    input_inner = x @ x.T + 1.0
    first_layer *= input_inner[:, None, :, None]
    hidden_inner = hidden @ hidden.T + 1.0
    output_identity = torch.eye(model.fc2.out_features, dtype=x.dtype, device=x.device)
    kernel = first_layer + hidden_inner[:, None, :, None] * output_identity[None, :, None, :]
    n_rows = x.shape[0] * model.fc2.out_features
    return kernel.reshape(n_rows, n_rows).cpu().numpy()


@torch.no_grad()
def tangent_parameter_edit(
    model: OneHiddenMLP, x: Tensor, coefficients: Tensor | np.ndarray
) -> dict[str, Tensor]:
    """Construct ``J.T @ coefficients`` in the model's parameter layout.

    Coefficients have shape (number_of_samples, number_of_outputs). Neither
    model parameters nor existing parameter gradients are modified.
    """
    x = _model_input(model, x)
    coefficients = torch.as_tensor(coefficients, dtype=x.dtype, device=x.device)
    expected_shape = (x.shape[0], model.fc2.out_features)
    if tuple(coefficients.shape) != expected_shape:
        raise ValueError(f"coefficients must have shape {expected_shape}.")
    preactivation = model.fc1(x)
    hidden = preactivation.clamp_min(0)
    gate = (preactivation > 0).to(x.dtype)
    upstream = (coefficients @ model.fc2.weight) * gate
    return {
        "fc1.weight": upstream.T @ x,
        "fc1.bias": upstream.sum(dim=0),
        "fc2.weight": coefficients.T @ hidden,
        "fc2.bias": coefficients.sum(dim=0),
    }


def _target_rows(
    targets: Iterable[int | tuple[int, int]], n_samples: int, n_outputs: int
) -> list[int]:
    result = []
    for target in targets:
        if isinstance(target, (tuple, list)):
            if len(target) != 2:
                raise ValueError("Each coordinate target must be a (sample, output) pair.")
            sample, output = int(target[0]), int(target[1])
            if not (0 <= sample < n_samples and 0 <= output < n_outputs):
                raise ValueError(f"Target coordinate {(sample, output)} is out of range.")
            row = sample * n_outputs + output
        else:
            row = int(target)
            if not 0 <= row < n_samples * n_outputs:
                raise ValueError(f"Target row {row} is out of range.")
        result.append(row)
    return result


def _first_gate_crossing_radius(preactivation: Tensor, direction: Tensor) -> float | None:
    """First gate change along ``z(t) = preactivation + t * direction``, t>=0.

    ``None`` means no finite crossing. An inactive gate starting exactly at
    zero has radius zero only if it becomes active in this direction. This
    is an infimum: a negative gate activates strictly after its zero crossing.
    """
    z, dz = preactivation.double(), direction.double()
    toward_boundary = ((z > 0) & (dz < 0)) | ((z < 0) & (dz > 0))
    if bool(((z == 0) & (dz > 0)).any()):
        return 0.0
    if not bool(toward_boundary.any()):
        return None
    return float((-z[toward_boundary] / dz[toward_boundary]).min())


@torch.no_grad()
def evaluate_isolated_edits(
    model: OneHiddenMLP,
    x: Tensor,
    K: np.ndarray,
    targets: Iterable[int | tuple[int, int]],
    amplitudes: Iterable[float],
    rank_rtol: float = 1e-10,
) -> list[dict[str, float | int | None]]:
    """Apply minimum-norm tangent edits and measure actual network changes.

    Each edit changes one output coordinate in the tangent model and anchors
    every other training output coordinate, including outputs of the target
    sample. The model itself is never mutated: edited predictions use
    ``torch.func.functional_call``. The supplied K must correspond to the
    entire x batch, include every parameter, and be numerically full rank.
    Parameter costs are squared Euclidean norms. Relative leakage divides
    output drift by the absolute requested edit amplitude. Gate radii are
    measured in requested *output-amplitude* units along this fixed tangent
    edit direction, over every hidden activation in the supplied x batch.
    Positive/negative radii carry the sign of that direction; the per-row
    absolute radius selects the sign of the requested amplitude. ``None``
    denotes no finite crossing. Network evaluation stays in model precision.
    """
    x = _model_input(model, x)
    n_samples, n_outputs = x.shape[0], model.fc2.out_features
    n_rows = n_samples * n_outputs
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (n_rows, n_rows) or not np.isfinite(K).all():
        raise ValueError(f"K must be a finite matrix of shape {(n_rows, n_rows)}.")
    if rank_rtol < 0:
        raise ValueError("rank_rtol must be nonnegative.")
    if not np.allclose(K, K.T, rtol=1e-10, atol=1e-12):
        raise ValueError("K must be symmetric.")
    rows = _target_rows(targets, n_samples, n_outputs)
    amplitudes = [float(value) for value in amplitudes]
    if any(not np.isfinite(value) or value == 0 for value in amplitudes):
        raise ValueError("Edit amplitudes must be finite and nonzero.")
    if not rows or not amplitudes:
        return []
    kernel = (K + K.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(kernel)
    if eigenvalues[-1] <= 0 or eigenvalues[0] <= rank_rtol * eigenvalues[-1]:
        raise ValueError(
            "K is numerically rank deficient; arbitrary singleton edits are not guaranteed feasible."
        )
    basis = np.zeros((n_rows, len(rows)), dtype=np.float64)
    basis[rows, np.arange(len(rows))] = 1.0
    inverse_columns = np.linalg.solve(kernel, basis)
    baseline = model(x).reshape(-1)
    baseline_preactivation = model.fc1(x)
    baseline_gates = baseline_preactivation > 0
    minimum_preactivation = float(baseline_preactivation.abs().min())
    original_parameters: Mapping[str, Tensor] = dict(model.named_parameters())
    result = []
    for column, row in enumerate(rows):
        unit_edit = tangent_parameter_edit(
            model, x, inverse_columns[:, column].reshape(n_samples, n_outputs)
        )
        unit_preactivation_change = torch.nn.functional.linear(
            x, unit_edit["fc1.weight"], unit_edit["fc1.bias"]
        )
        positive_radius = _first_gate_crossing_radius(baseline_preactivation, unit_preactivation_change)
        negative_radius = _first_gate_crossing_radius(baseline_preactivation, -unit_preactivation_change)
        for amplitude in amplitudes:
            edit = {name: amplitude * value for name, value in unit_edit.items()}
            edited_parameters = {
                name: parameter + edit[name] for name, parameter in original_parameters.items()
            }
            changed = functional_call(model, edited_parameters, (x,)).reshape(-1) - baseline
            edited_preactivation = torch.nn.functional.linear(
                x, edited_parameters["fc1.weight"], edited_parameters["fc1.bias"]
            )
            flip_count = int(((edited_preactivation > 0) != baseline_gates).sum())
            direction_radius = positive_radius if amplitude > 0 else negative_radius
            other = torch.cat((changed[:row], changed[row + 1 :]))
            rms = float(other.square().mean().sqrt()) if other.numel() else 0.0
            maximum = float(other.abs().max()) if other.numel() else 0.0
            result.append(
                {
                    "sample_index": row // n_outputs,
                    "output_index": row % n_outputs,
                    "amplitude": amplitude,
                    "parameter_cost": sum(float(value.double().square().sum()) for value in edit.values()),
                    "tangent_pred_cost": float(amplitude**2 * inverse_columns[row, column]),
                    "achieved_target_change": float(changed[row]),
                    "target_error": float(changed[row]) - amplitude,
                    "target_error_relative": abs(float(changed[row]) - amplitude) / abs(amplitude),
                    "leakage_rms": rms,
                    "leakage_max": maximum,
                    "leakage_rms_relative": rms / abs(amplitude),
                    "leakage_max_relative": maximum / abs(amplitude),
                    "baseline_min_abs_preactivation": minimum_preactivation,
                    "baseline_zero_preactivation_count": int((baseline_preactivation == 0).sum()),
                    "activation_gate_count": int(baseline_preactivation.numel()),
                    "activation_flip_count": flip_count,
                    "first_gate_crossing_positive_amplitude": positive_radius,
                    "first_gate_crossing_negative_amplitude": None if negative_radius is None else -negative_radius,
                    "first_gate_crossing_abs_amplitude": direction_radius,
                    "amplitude_to_gate_radius": (
                        abs(amplitude) / direction_radius
                        if direction_radius is not None and direction_radius > 0 else None
                    ),
                }
            )
    return result
