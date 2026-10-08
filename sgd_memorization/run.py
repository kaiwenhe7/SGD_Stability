"""Run a paired MNIST learning-rate/batch-size experiment.

Training uses fresh independent batches of distinct examples. Diagnostics are
computed from prediction gradients and retain all outputs of each example.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import download_mnist, load_subset
from .geometry import analyze_gram
from .model import OneHiddenMLP, evaluate_isolated_edits, prediction_gram


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def covariance_diagnostic(kernel: np.ndarray, summary: dict, max_rank: int = 512) -> dict:
    from .covariance import exact_generalized_sharpness
    rank = summary["geometry"]["numerical_rank"]
    if rank > max_rank:
        return {"status": "skipped", "reason": f"Numerical rank {rank} exceeds covariance limit {max_rank}"}
    measured = exact_generalized_sharpness(
        kernel, summary["n_examples"], summary["n_outputs"], summary["batch_size"],
        rank_rtol=summary["geometry"]["rank_rtol"], max_rank=max_rank)
    critical = measured["critical_learning_rate"]
    measured["learning_rate_ratio"] = summary["learning_rate"] / critical if critical is not None else 0.0
    measured["status"] = "measured"
    return measured


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def learning_rate_at(epoch: int, target: float, config: dict) -> float:
    """Geometric schedule; the index counts updates when warmup_steps is set."""
    warmup = config.get("warmup_steps", config.get("warmup_epochs", 0))
    if warmup <= 1 or epoch >= warmup:
        return target
    # Geometric warmup gives several orders of magnitude of small initial steps.
    fraction = config["warmup_start_fraction"]
    return target * fraction ** (1.0 - (epoch - 1) / (warmup - 1))


def corrupt_labels(labels: torch.Tensor, outputs: int, fraction: float,
                   seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace exactly floor(fraction*n) labels with seeded incorrect classes."""
    corrupted = labels.clone()
    mask = torch.zeros(len(labels), dtype=torch.bool)
    count = math.floor(fraction * len(labels))
    if count:
        if outputs < 2:
            raise ValueError("Label corruption requires at least two classes")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = torch.randperm(len(labels), generator=generator)[:count]
        offsets = torch.randint(1, outputs, (count,), generator=generator)
        corrupted[selected] = (labels[selected] + offsets) % outputs
        mask[selected] = True
    return corrupted, mask


def training_dtype(config: dict) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[config.get("training_dtype", "float32")]


def run_identity(seed: int, batch_size: int, rate: float) -> dict:
    return {"seed": seed, "batch_size": batch_size, "learning_rate": float(rate)}


def validate_identity(saved: dict, expected: dict) -> None:
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Saved run identity does not match requested parameters: {expected}")


def run_directory(output: Path, seed: int, batch_size: int, rate: float) -> Path:
    """Round-trip float names avoid collisions; safely recognize legacy names."""
    exact = output / f"seed{seed}_B{batch_size}_lr{float(rate)!r}"
    legacy = output / f"seed{seed}_B{batch_size}_lr{rate:g}"
    if not exact.exists() and legacy != exact and (legacy / "summary.json").exists():
        saved = json.loads((legacy / "summary.json").read_text())
        if all(saved.get(key) == value for key, value in run_identity(seed, batch_size, rate).items()):
            return legacy
    return exact


def environment_metadata(device: torch.device, config: dict) -> dict:
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(Path(__file__).parent.glob("*.py"))}
    return {"torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": str(device),
            "training_dtype": config.get("training_dtype", "float32"),
            "initialization": "PyTorch float32 initialization, reused exactly and cast to training dtype",
            "loss": "0.5 * squared output residuals summed over outputs, averaged over examples",
            "sampling": "fresh independent uniform subset of B distinct examples each step",
            "pixels": "uint8 / 255; no centering or augmentation",
            "warmup": "geometric; updates" if "warmup_steps" in config else "geometric; epochs",
            "convergence_hold": "epoch-boundary full-data checks; credited target-rate updates between successful boundaries" if "hold_steps" in config else "consecutive epoch-boundary full-data checks",
            "source_sha256": sources}


def validate_saved_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with np.load(path) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
            raise ValueError(f"Saved data selections or labels do not match this experiment: {path}")


def load_or_create_initial(output: Path, config: dict, seed: int) -> dict:
    path = output / f"initial_seed{seed}.pt"
    # Instantiate in float32 regardless of requested training precision, making
    # float32/64 experiments share the same numerical starting parameters.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = OneHiddenMLP(784, config["hidden_width"], len(config["classes"])).float()
    if path.exists():
        initial = torch.load(path, map_location="cpu", weights_only=True)
        expected = model.state_dict()
        if set(initial) != set(expected) or any(initial[key].shape != value.shape or initial[key].dtype != value.dtype or not torch.isfinite(initial[key]).all() for key, value in expected.items()):
            raise ValueError(f"Invalid saved initial checkpoint: {path}")
        return initial
    if any(output.glob(f"seed{seed}_B*")):
        raise ValueError(f"Cannot recreate missing initialization for existing seed {seed}")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("weight"):
                parameter.mul_(config["initial_weight_scale"])
    initial = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
    torch.save(initial, path)
    return initial


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, labels: torch.Tensor,
             outputs: int) -> tuple[float, float]:
    total_loss, total_correct = 0.0, 0
    for start in range(0, len(x), 1024):
        pred = model(x[start:start + 1024])
        target = torch.nn.functional.one_hot(labels[start:start + 1024], outputs).to(pred)
        total_loss += float(0.5 * (pred - target).square().sum())
        total_correct += int((pred.argmax(dim=1) == labels[start:start + 1024]).sum())
    return total_loss / len(x), total_correct / len(x)


def train_one(config: dict, batch_size: int, rate: float, seed: int,
              initial_state: dict, train_x: torch.Tensor, train_y: torch.Tensor,
              test_x: torch.Tensor, test_y: torch.Tensor, directory: Path,
              device: torch.device, clean_train_y: torch.Tensor | None = None) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = len(config["classes"])
    dtype = training_dtype(config)
    model = OneHiddenMLP(784, config["hidden_width"], outputs).to(device=device, dtype=dtype)
    model.load_state_dict(initial_state)
    x, labels = train_x.to(device=device, dtype=dtype), train_y.to(device)
    xt, yt = test_x.to(device=device, dtype=dtype), test_y.to(device)
    clean_labels = (clean_train_y if clean_train_y is not None else train_y).to(device)
    targets = torch.nn.functional.one_hot(labels, outputs).to(x)
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    # Identical streams for learning rates with the same seed and batch size.
    generator = torch.Generator(device="cpu").manual_seed(seed + 100000)
    n = len(x)
    steps_per_epoch = math.ceil(n / batch_size)
    history = []
    below_threshold = 0
    held_steps = 0
    previous_below = False
    hold_steps = config.get("hold_steps")
    hold_epochs = config.get("consecutive_epochs", 1)
    status = "max_epochs"
    started = time.perf_counter()
    step = 0
    print(f"START {directory.name}: n={n}, width={config['hidden_width']}, device={device}", flush=True)
    train_loss, accuracy = evaluate(model, x, labels, outputs)
    _, clean_accuracy = evaluate(model, x, clean_labels, outputs)
    history.append({"epoch": 0, "step": 0, "learning_rate": 0.0,
                    "train_loss": train_loss, "train_accuracy": accuracy,
                    "clean_train_accuracy": clean_accuracy,
                    "test_loss": None, "test_accuracy": None, "elapsed_seconds": 0.0})
    epoch = 0
    for epoch in range(1, config["max_epochs"] + 1):
        lr = learning_rate_at(epoch, rate, config) if "warmup_steps" not in config else learning_rate_at(step + 1, rate, config)
        target_steps = 0
        for _ in range(steps_per_epoch):
            if config.get("max_steps") is not None and step >= config["max_steps"]:
                break
            if "warmup_steps" in config:
                lr = learning_rate_at(step + 1, rate, config)
            optimizer.param_groups[0]["lr"] = lr
            indices = torch.randperm(n, generator=generator)[:batch_size].to(device)
            pred = model(x[indices])
            loss = 0.5 * (pred - targets[indices]).square().sum(dim=1).mean()
            if not torch.isfinite(loss) or float(loss.detach()) > 1e12:
                status = "diverged"
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            target_steps += int(lr == rate)
        train_loss, accuracy = evaluate(model, x, labels, outputs)
        _, clean_accuracy = evaluate(model, x, clean_labels, outputs)
        if not math.isfinite(train_loss) or train_loss > 1e12:
            status = "diverged"
        warmup_done = step >= config["warmup_steps"] if "warmup_steps" in config else epoch >= config.get("warmup_epochs", 0)
        below_now = warmup_done and math.isfinite(train_loss) and train_loss < config["loss_threshold"]
        if below_now:
            below_threshold += 1
            # Loss is checked at epoch boundaries. The first passing check
            # starts the clock; subsequent passing checks credit intervening
            # target-rate updates, without claiming every update was checked.
            held_steps = held_steps + target_steps if previous_below else 0
        else:
            below_threshold = 0
            held_steps = 0
        previous_below = below_now
        held = held_steps >= hold_steps if hold_steps is not None else below_threshold >= hold_epochs
        log_now = epoch == 1 or epoch % config["log_every"] == 0 or held
        test_loss, test_accuracy = evaluate(model, xt, yt, outputs) if log_now else (None, None)
        history.append({"epoch": epoch, "step": step, "learning_rate": lr,
                        "train_loss": train_loss if math.isfinite(train_loss) else None,
                        "train_accuracy": accuracy, "test_loss": test_loss,
                        "clean_train_accuracy": clean_accuracy,
                        "test_accuracy": test_accuracy,
                        "elapsed_seconds": time.perf_counter() - started})
        if log_now:
            print(f"  epoch={epoch:5d} lr={lr:.4g} loss={train_loss:.4g} acc={accuracy:.3f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if epoch % config["checkpoint_every"] == 0:
            write_rows(directory / "history.csv", history)
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "step": step, "batch_rng": generator.get_state(),
                        "config": config}, directory / "latest.pt")
        if status == "diverged":
            break
        if held:
            status = "converged"
            break
        if config.get("max_steps") is not None and step >= config["max_steps"]:
            status = "max_steps"
            break
    training_seconds = time.perf_counter() - started
    test_loss, test_accuracy = evaluate(model, xt, yt, outputs)
    write_rows(directory / "history.csv", history)
    torch.save({"state_dict": model.cpu().state_dict(), "epoch": epoch,
                "step": step, "config": config, "batch_size": batch_size,
                "learning_rate": rate, "seed": seed}, directory / "model.pt")
    summary = {"batch_size": batch_size, "learning_rate": rate, "seed": seed,
               "status": status, "train_loss": train_loss if math.isfinite(train_loss) else None,
               "train_accuracy": accuracy,
               "clean_train_accuracy": clean_accuracy,
               "label_corruption_fraction": config.get("label_corruption_fraction", 0.0),
               "corrupted_training_examples": int((labels != clean_labels).sum()),
               "training_dtype": config.get("training_dtype", "float32"),
               "convergence_hold_steps": hold_steps,
               "credited_hold_steps": held_steps if hold_steps is not None else None,
               "test_loss": test_loss if math.isfinite(test_loss) else None,
               "test_accuracy": test_accuracy, "epochs": epoch, "steps": step,
               "training_seconds": training_seconds, "n_examples": n,
               "n_outputs": outputs, "threshold": config["loss_threshold"],
               "diagnostics_applicable": status == "converged",
               "curvature": "prediction-gradient Gauss-Newton; equals Hessian at exact interpolation",
               "private_edit_definition": "one output; preserve all other outputs of all training examples",
               "mean_square_stability_certified": False}
    write_json(directory / "summary.json", summary)
    if status == "diverged":
        print(f"END {directory.name}: diverged; retained failure", flush=True)
        return summary
    return diagnose_endpoint(config, model, train_x, directory, summary)


def diagnose_endpoint(config: dict, model: OneHiddenMLP, train_x: torch.Tensor,
                      directory: Path, summary: dict) -> dict:
    """Measure a saved endpoint; safe to repeat after interrupted diagnostics."""
    n, outputs = summary["n_examples"], summary["n_outputs"]
    seed, batch_size, rate = summary["seed"], summary["batch_size"], summary["learning_rate"]
    diagnostic_index = len(list(directory.glob("diagnostic_provenance_*.json")))
    write_json(directory / f"diagnostic_provenance_{diagnostic_index:03d}.json", {
        "environment": environment_metadata(torch.device("cpu"), config),
        "model_sha256": hashlib.sha256((directory / "model.pt").read_bytes()).hexdigest(),
        "purpose": "endpoint diagnostics; training provenance is stored separately"})
    print(f"  Computing full prediction Gram and geometry for {directory.name}", flush=True)
    measured_at = time.perf_counter()
    # Edit evaluation in double precision avoids float32 cancellation dominating
    # small requested prediction changes. Float64 training avoids a dtype
    # transition and any associated changes to near-zero ReLU gates.
    model.cpu().double()
    summary["diagnostic_precision"] = "float64; trained parameters cast without reoptimization" if config.get("training_dtype", "float32") == "float32" else "float64; same precision as training"
    kernel = prediction_gram(model, train_x)
    np.save(directory / "gram.npy", kernel)
    geometry, coordinates = analyze_gram(
        kernel, n, outputs, rate, batch_size, rank_rtol=config["rank_rtol"],
        rankone_steps=config["rankone_steps"], rankone_restarts=config["rankone_restarts"], seed=seed)
    summary["geometry"] = geometry
    summary["covariance"] = covariance_diagnostic(kernel, summary, config.get("covariance_max_rank", 512))
    write_rows(directory / "per_coordinate.csv", coordinates)
    feasible = [i for i, row in enumerate(coordinates) if row["isolated_edit_feasible"]]
    if feasible and geometry["numerical_rank"] == n * outputs:
        if config.get("edit_target_policy", "private_quantiles") == "fixed_random":
            generator = np.random.default_rng(config.get("edit_target_seed", config["dataset_seed"]))
            selected = sorted(generator.choice(n * outputs, size=min(config["edit_targets"], n * outputs), replace=False).tolist())
        else:
            sorted_indices = sorted(feasible, key=lambda index: coordinates[index]["private_sqnorm"])
            selected = np.unique(np.asarray(sorted_indices)[
                np.linspace(0, len(sorted_indices) - 1, min(config["edit_targets"], len(feasible)), dtype=int)]).tolist()
        edits = evaluate_isolated_edits(model, train_x, kernel, selected,
                                       config["edit_amplitudes"], rank_rtol=config["rank_rtol"])
        write_rows(directory / "edits.csv", edits)
        summary["nonlinear_edit_measurements"] = len(edits)
        summary["edit_target_policy"] = config.get("edit_target_policy", "private_quantiles")
        summary["edit_target_coordinates"] = selected
    else:
        summary["nonlinear_edit_measurements"] = 0
        summary["edit_measurement_note"] = "Full Gram rank required by the pilot edit evaluator. See per-coordinate feasibility."
    summary["diagnostic_seconds"] = time.perf_counter() - measured_at
    write_json(directory / "summary.json", summary)
    print(f"END {directory.name}: {summary['status']}, loss={summary['train_loss']:.4g}, private ratio max={geometry['max_universal_private_ratio']}", flush=True)
    return summary


def validate_config(config: dict) -> None:
    n = len(config["classes"]) * config["samples_per_class"]
    if len(set(config["classes"])) != len(config["classes"]) or not all(0 <= c <= 9 for c in config["classes"]):
        raise ValueError("Classes must be distinct MNIST digits")
    if min(config["batch_sizes"]) < 1 or max(config["batch_sizes"]) > n:
        raise ValueError("Every batch size must be between one and n")
    if not all(math.isfinite(rate) and rate > 0 for rate in config["learning_rates"]) or not 0 < config["warmup_start_fraction"] <= 1:
        raise ValueError("Learning rates must be positive and warmup fraction in (0,1]")
    if config["max_epochs"] < 1 or config.get("consecutive_epochs", 1) < 1:
        raise ValueError("Epoch cap must cover warmup; convergence hold must be positive")
    if "warmup_steps" in config:
        if not isinstance(config["warmup_steps"], int) or config["warmup_steps"] < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if any(config["max_epochs"] * math.ceil(n / batch) < config["warmup_steps"] for batch in config["batch_sizes"]):
            raise ValueError("Epoch cap must cover update-counted warmup for every batch size")
    elif config["max_epochs"] < config.get("warmup_epochs", 0):
        raise ValueError("Epoch cap must cover warmup")
    if "hold_steps" in config and (not isinstance(config["hold_steps"], int) or config["hold_steps"] < 1):
        raise ValueError("hold_steps must be a positive integer")
    if "max_steps" in config and (not isinstance(config["max_steps"], int) or config["max_steps"] < 1 or config["max_steps"] < config.get("warmup_steps", 0)):
        raise ValueError("max_steps must be positive and cover warmup_steps")
    if config.get("training_dtype", "float32") not in {"float32", "float64"}:
        raise ValueError("training_dtype must be float32 or float64")
    fraction = config.get("label_corruption_fraction", 0.0)
    if not 0 <= fraction <= 1 or (fraction > 0 and len(config["classes"]) < 2):
        raise ValueError("Label corruption fraction must be in [0,1] with at least two classes")
    if config.get("edit_target_policy", "private_quantiles") not in {"private_quantiles", "fixed_random"}:
        raise ValueError("edit_target_policy must be private_quantiles or fixed_random")
    for name in ("samples_per_class", "test_samples_per_class", "hidden_width", "threads", "log_every", "checkpoint_every"):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if len(set(config["seeds"])) != len(config["seeds"]) or len(set(config["batch_sizes"])) != len(config["batch_sizes"]) or len(set(config["learning_rates"])) != len(config["learning_rates"]):
        raise ValueError("Seeds, batch sizes, and learning rates must be unique")


def measure_small_edits(output: Path, data_root: Path, amplitudes: list[float]) -> None:
    """Supplement a saved experiment with smaller-amplitude edit measurements."""
    config = json.loads((output / "config.json").read_text())
    torch.set_num_threads(config["threads"])
    x, _, indices = load_subset(data_root, config["classes"], config["samples_per_class"], config["dataset_seed"])
    saved_indices = np.load(output / "subset_indices.npz")["train"]
    if not np.array_equal(indices, saved_indices):
        raise ValueError("Selected MNIST examples do not match the saved experiment")
    for directory in sorted(output.glob("seed*_*")):
        if not (directory / "edits.csv").exists():
            continue
        checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        dtype = next(iter(checkpoint["state_dict"].values())).dtype
        model = OneHiddenMLP(784, config["hidden_width"], len(config["classes"])).to(dtype=dtype)
        model.load_state_dict(checkpoint["state_dict"])
        model.double()
        with (directory / "edits.csv").open() as handle:
            original_edits = list(csv.DictReader(handle))
        targets = sorted({int(row["sample_index"]) * len(config["classes"]) + int(row["output_index"])
                          for row in original_edits})
        kernel = np.load(directory / "gram.npy")
        edits = evaluate_isolated_edits(model, x, kernel, targets, amplitudes, config["rank_rtol"])
        write_rows(directory / "edits_small_amplitude.csv", edits)
        summary = json.loads((directory / "summary.json").read_text())
        summary["supplemental_edit_amplitudes"] = amplitudes
        summary["supplemental_edit_measurements"] = len(edits)
        write_json(directory / "summary.json", summary)
        print(f"Small edits: {directory.name}, {len(edits)} measurements", flush=True)
    summaries = [json.loads(path.read_text()) for path in sorted(output.glob("*/summary.json"))]
    write_json(output / "results.json", {"runs": summaries})
    from .plotting import make_plots
    make_plots(output)


def measure_stability(output: Path, max_rank: int) -> None:
    config = json.loads((output / "config.json").read_text())
    torch.set_num_threads(config["threads"])
    for directory in sorted(output.glob("seed*_*")):
        if not (directory / "gram.npy").exists():
            continue
        summary = json.loads((directory / "summary.json").read_text())
        kernel = np.load(directory / "gram.npy")
        summary["covariance"] = covariance_diagnostic(kernel, summary, max_rank)
        write_json(directory / "summary.json", summary)
        print(f"Covariance: {directory.name}, {summary['covariance']}", flush=True)
    summaries = [json.loads(path.read_text()) for path in sorted(output.glob("*/summary.json"))]
    write_json(output / "results.json", {"runs": summaries})
    from .plotting import make_plots
    make_plots(output)


def refresh_aggregate(output: Path) -> None:
    """Repair derived outputs without touching current experiment artifacts."""
    payload = {"runs": [json.loads(path.read_text()) for path in sorted(output.glob("*/summary.json"))]}
    aggregate = output / "results.json"
    try:
        current = json.loads(aggregate.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        current = None
    changed = current != payload
    if changed:
        write_json(aggregate, payload)
    plots_missing = any(not (output / f"{stem}.{suffix}").exists()
                        for stem in ("overview", "training", "edits") for suffix in ("png", "pdf"))
    if changed or plots_missing:
        from .plotting import make_plots
        for path in make_plots(output):
            print(f"Plot: {path}", flush=True)


def run(config_path: Path, output: Path, data_root: Path, device_name: str,
        max_runs: int | None = None) -> None:
    config = json.loads(config_path.read_text())
    validate_config(config)
    if max_runs is not None and max_runs < 0:
        raise ValueError("max_runs must be nonnegative")
    saved_config = output / "config.json"
    if saved_config.exists() and json.loads(saved_config.read_text()) != config:
        raise ValueError("Output directory contains a different configuration; choose a new output directory")
    # A zero-work request is a true no-op, including metadata and plots.
    if max_runs == 0:
        return
    records = []
    for seed in config["seeds"]:
        for batch_size in config["batch_sizes"]:
            for rate in config["learning_rates"]:
                directory = run_directory(output, seed, batch_size, rate)
                existing = json.loads((directory / "summary.json").read_text()) if (directory / "summary.json").exists() else None
                if existing is not None:
                    validate_identity(existing, run_identity(seed, batch_size, rate))
                complete = existing is not None and ("geometry" in existing or existing["status"] == "diverged")
                records.append((seed, batch_size, rate, directory, existing, complete))
    has_runs = any(output.glob("seed*_B*"))
    if has_runs and not saved_config.exists():
        raise ValueError("Existing runs have no saved configuration; refusing to reconstruct provenance")
    pending = [record for record in records if not record[-1]]
    torch.set_num_threads(config["threads"])
    device = choose_device(device_name)
    if device.type == "mps" and training_dtype(config) == torch.float64:
        raise ValueError("MPS does not support float64 training; choose CPU or CUDA")
    download_mnist(data_root)
    x, clean_y, indices = load_subset(data_root, config["classes"], config["samples_per_class"], config["dataset_seed"])
    xt, yt, test_indices = load_subset(data_root, config["classes"], config["test_samples_per_class"], config["dataset_seed"], test=True)
    y, corrupted = corrupt_labels(clean_y, len(config["classes"]), config.get("label_corruption_fraction", 0.0), config.get("label_corruption_seed", config["dataset_seed"]))
    subsets = {"train": indices, "test": test_indices}
    labels = {"clean_train": clean_y.numpy(), "train": y.numpy(), "test": yt.numpy(), "corrupted": corrupted.numpy()}
    subset_path, labels_path = output / "subset_indices.npz", output / "labels.npz"
    if subset_path.exists():
        validate_saved_arrays(subset_path, subsets)
    elif has_runs:
        raise ValueError("Existing runs have no saved subset indices; refusing to reconstruct provenance")
    if labels_path.exists():
        validate_saved_arrays(labels_path, labels)
    elif has_runs and config.get("label_corruption_fraction", 0.0) > 0:
        raise ValueError("Existing corrupted-label runs have no saved labels")
    for seed in config["seeds"]:
        if any(record[0] == seed and record[5] for record in records):
            initial = load_or_create_initial(output, config, seed)
            initial_digest = hashlib.sha256((output / f"initial_seed{seed}.pt").read_bytes()).hexdigest()
            for record in records:
                if record[0] == seed and (record[3] / "provenance.json").exists():
                    provenance = json.loads((record[3] / "provenance.json").read_text())
                    if provenance["initial_sha256"] != initial_digest:
                        raise ValueError("Saved initialization differs from immutable run provenance")
    if not pending:
        for record in records:
            print(f"SKIP completed {record[3].name}", flush=True)
        refresh_aggregate(output)
        return
    output.mkdir(parents=True, exist_ok=True)
    if not saved_config.exists():
        write_json(saved_config, config)
    if not subset_path.exists():
        np.savez(subset_path, **subsets)
    if not labels_path.exists():
        np.savez(labels_path, **labels)
    environment = environment_metadata(device, config)
    if not (output / "environment.json").exists():
        write_json(output / "environment.json", environment)
    initial_states = {}
    for seed, batch_size, rate, directory, existing, complete in pending[:max_runs]:
        if seed not in initial_states:
            initial_states[seed] = load_or_create_initial(output, config, seed)
        initial = initial_states[seed]
        initial_path = output / f"initial_seed{seed}.pt"
        initial_digest = hashlib.sha256(initial_path.read_bytes()).hexdigest()
        if (directory / "provenance.json").exists():
            provenance = json.loads((directory / "provenance.json").read_text())
            if provenance["initial_sha256"] != initial_digest:
                raise ValueError("Saved initialization differs from immutable run provenance")
        if existing is not None and (directory / "model.pt").exists():
            checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
            validate_identity(checkpoint, run_identity(seed, batch_size, rate))
            if checkpoint["config"] != config:
                raise ValueError("Saved endpoint configuration does not match requested configuration")
            dtype = next(iter(checkpoint["state_dict"].values())).dtype
            model = OneHiddenMLP(784, config["hidden_width"], len(config["classes"])).to(dtype=dtype)
            model.load_state_dict(checkpoint["state_dict"])
            print(f"RESUME diagnostics {directory.name}", flush=True)
            diagnose_endpoint(config, model, x, directory, existing)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        old_files = [path for path in directory.iterdir() if path.is_file()]
        if old_files:
            # Training restarts from its preserved initial state. Keep the
            # interrupted attempt and its provenance without mixing histories.
            archive = directory / f"interrupted_attempt_{len(list(directory.glob('interrupted_attempt_*'))):03d}"
            archive.mkdir()
            for path in old_files:
                destination = "summary.partial.json" if path.name == "summary.json" else path.name
                path.rename(archive / destination)
        write_json(directory / "provenance.json", {
            **run_identity(seed, batch_size, rate), "environment": environment,
            "initial_sha256": initial_digest,
            "subset_sha256": hashlib.sha256(subset_path.read_bytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(saved_config.read_bytes()).hexdigest()})
        train_one(config, batch_size, rate, seed, initial, x, y, xt, yt, directory, device, clean_train_y=clean_y)
    # Keep the aggregate complete when a bounded call advances only part of a
    # sweep that already contains completed runs.
    refresh_aggregate(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("run")
    training.add_argument("--config", type=Path, default=Path("configs/mnist_pilot.json"))
    training.add_argument("--output", type=Path, default=Path("results/mnist_pilot"))
    training.add_argument("--data", type=Path, default=Path("data"))
    training.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    training.add_argument("--max-runs", type=int, help="Maximum new runs in this call; completed runs are reused")
    plotting = subparsers.add_parser("plot")
    plotting.add_argument("output", type=Path)
    editing = subparsers.add_parser("edit-check")
    editing.add_argument("output", type=Path)
    editing.add_argument("--data", type=Path, default=Path("data"))
    editing.add_argument("--amplitudes", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4])
    stability = subparsers.add_parser("stability-check")
    stability.add_argument("output", type=Path)
    stability.add_argument("--max-rank", type=int, default=512)
    args = parser.parse_args()
    if args.command == "run":
        run(args.config, args.output, args.data, args.device, args.max_runs)
    elif args.command == "edit-check":
        measure_small_edits(args.output, args.data, args.amplitudes)
    elif args.command == "stability-check":
        measure_stability(args.output, args.max_rank)
    else:
        from .plotting import make_plots
        for path in make_plots(args.output):
            print(path)


if __name__ == "__main__":
    main()
