# MNIST: SGD stability and private prediction geometry

This experiment follows the MNIST approach in [Mulayoff and Michaeli, COLT 2024, Section 4 and Appendix K](https://proceedings.mlr.press/v247/mulayoff24a/mulayoff24a.pdf) and adds the private-gradient and shared-private diagnostics from the SGD memorization draft.

## Start here

Use Python 3.10 or newer and install `requirements.txt` in your preferred environment. The existing `/usr/local/bin/python3` environment on this machine has the required packages. Torchvision is unnecessary: the loader reads the original MNIST IDX files, downloads over verified HTTPS, and checks their published MD5 values.

```bash
python3 -m sgd_memorization.data --root data
python3 -m unittest discover -s tests -v
python3 -m sgd_memorization.run run --config configs/mnist_pilot.json --output results/mnist_pilot
```

The default device selection prefers CUDA, then MPS, then CPU. Pass `--device cpu` or `--device cuda` explicitly when needed. Training is float32; endpoint Gram and edit measurements use float64 on CPU. A completed run is reused when rerunning the same configuration. Interrupted training restarts that run from its shared initialization; `latest.pt` preserves intermediate state for inspection. A changed configuration requires a new output directory.

For the larger experiment on a GPU:

```bash
python3 -m sgd_memorization.run run --config configs/mnist_reference.json --output results/mnist_reference --device cuda
```

Use `--max-runs 1` to check the first configuration before the full sweep. Regenerate plots with `python3 -m sgd_memorization.run plot results/mnist_pilot`.

Supplement saved runs with smaller edit sizes using `python3 -m sgd_memorization.run edit-check results/mnist_pilot`. This records amplitudes 1e-6, 1e-5, and 1e-4 separately, using the same target coordinates as the original edit measurements.

Small runs also compute the complete Gauss-Newton covariance stability operator numerically, with an active-rank limit of 512. Add this measurement to already saved runs using `python3 -m sgd_memorization.run stability-check results/mnist_pilot`. The reference configuration exceeds that limit and retains the cheaper necessary conditions.

## Protocol and deviations from the source

| Setting | Reference configuration | CPU pilot |
|---|---|---|
| Digits | 0, 1, 2, 3 | 0, 1, 2, 3 |
| Training examples per digit | 256 | 16 |
| Hidden ReLU units | 1,024 | 128 |
| Output | Four raw outputs; one-hot targets | Same |
| Initialization | Default PyTorch initialization; weight matrices multiplied by 15 | Same |
| Batch sizes | 1, 2, 4, 8, 16, 32, 64, 128 | 8, 32 |
| Final learning rates | 0.002, 0.012, 0.043 | 0.002, 0.01 |
| Epoch cap | 40,000 | 4,000 |
| Loss threshold | 1e-6 | 1e-6 |
| Consecutive epochs below threshold | 200 | 25 |
| Warmup | 1,000 epochs | 300 epochs |

The paper specifies four classes, model width, enlarged initialization, warmup, loss threshold, hold duration, and epoch cap. It does not specify which classes, pixel preprocessing, the exact loss scaling, warmup schedule, or the complete learning-rate grid. Our explicit choices are digits 0-3, pixels divided by 255 with no centering/augmentation, and geometric warmup from 1e-4 times the final learning rate. The reference learning rates are the three values identified in Figure 1(b), rather than a claim to reproduce the complete original sweep. The reference architecture has exactly 807,940 parameters.

Every SGD step draws a fresh uniform subset of B distinct examples, independently of previous steps. An epoch is `ceil(n/B)` such updates; no shuffled-epoch sampler is used. There is no momentum, weight decay, clipping, dropout, softmax, or learning-rate decay. All runs with a given seed start from identical parameters, and equal-batch runs share the same minibatch random stream. Convergence is checked only after warmup is complete.

The objective is explicitly

`0.5 * (prediction - one_hot_target).square().sum(dim=1).mean()`.

This sums over output coordinates and averages over examples. PyTorch's default `MSELoss` uses a different normalization. The normalization is essential when interpreting learning-rate bounds.

## What the measurements mean

For four outputs, an isolated coordinate edit changes one output of one example while preserving **every other output of every training example**, including the other outputs of that same example. This extends the scalar proof because the per-example interpolating curvature is `H_i = sum_c g_ic g_ic.T`; batch size and n still count examples, not output coordinates.

The prediction Gram includes derivatives with respect to every weight and bias. A closed-form kernel for the one-hidden-layer ReLU model avoids materializing an enormous parameter Jacobian. Independent tests compare it to explicit autograd Jacobians and verify grouped minibatch moments against enumeration.

Endpoint measurements include:

- Full-data Gauss-Newton sharpness, stochastic trace ratio, top-Hessian-direction test, and the best rank-one test found by a multiple-start numerical search.
- For active rank at most 512, the full generalized covariance operator's estimated critical learning rate, eigenpair residual, and solver status. This solves the exact tangent operator numerically, rather than relying only on necessary compressed tests.
- Per-coordinate gradient norm, private residual norm, private fraction, isolated unit edit cost, shared stability slack, and universal/shared-private bound ratios.
- Actual nonlinear target changes and drift on all anchored training outputs for three edit amplitudes.

Ratios exceeding one violate a necessary tangent stability condition at interpolation. Passing these tests does **not** certify mean-square stability. Ordinary training convergence is also not a stability certificate. The rank-one search is a numerical lower bound on the optimum, with optimization settings recorded.

At small residual loss, the prediction-gradient matrix is Gauss-Newton curvature; it equals the full Hessian at exact interpolation. Runs above the loss threshold still have endpoint geometry measured for diagnosis, but those measurements are marked as outside the interpolating protocol. All capped and divergent runs remain in the summaries and figures.

Gram rank is evaluated using a stated relative eigenvalue threshold. No ridge is added. Infeasible singleton edits are reported as such. The pilot nonlinear evaluator requires a numerically full-rank Gram; rank-deficient runs retain coordinate feasibility diagnostics and a reason for unavailable edit measurements. Tests of a few rank tolerances are advisable before drawing conclusions from a nearly singular endpoint.

The nonlinear editing measurement applies the minimum-norm tangent solution. It measures its actual behavior; it does not claim to solve the global nonlinear minimum-edit problem. Parameter costs are squared Euclidean norms in the fixed network parameterization.

## Outputs

Each experiment directory contains the resolved configuration, environment metadata, selected original MNIST indices, shared initial checkpoints, and a folder for each seed/batch/learning-rate setting. Run folders contain `model.pt`, `history.csv`, `summary.json`, `gram.npy`, `per_coordinate.csv`, and `edits.csv` when feasible.

- `overview.png` / `.pdf`: sharpness, curvature tests, private bounds, and isolated edit-cost distributions.
- `training.png` / `.pdf`: full-training-set loss versus SGD updates.
- `edits.png` / `.pdf`: tangent cost consistency and nonlinear anchor leakage versus edit size.

These are a first pilot, not a paper-ready statistical comparison. Add independent initialization seeds and enlarge the subset before assessing optimizer-dependent representation selection.

## First pilot observations

All four runs converged below 1e-6 for the configured 25-epoch hold. The original pilot uses 64 examples, width 128, and one initialization seed. Numerical covariance measurements put the endpoints below the estimated tangent threshold:

| B | Learning rate | Final loss | Learning rate / estimated tangent threshold | Median squared unit edit cost |
|---|---|---|---|---|
| 8 | 0.002 | 1.31e-7 | 0.9425 | 9.62e-4 |
| 8 | 0.01 | 4.81e-7 | 0.9737 | 2.73e-3 |
| 32 | 0.002 | 5.73e-7 | 0.9888 | 8.73e-4 |
| 32 | 0.01 | 4.78e-7 | 0.9915 | 1.76e-3 |

Larger learning rates produced larger isolated tangent edit costs in these paired runs. The maximum coupling factor was below 0.008, so the aggregate shared-private bound tightened the universal cap by less than 1% here. This pilot provides a baseline; a construction with less shared stability slack is needed to demonstrate a substantial coupling effect.

At edit amplitude 1e-6, the tested target changes and anchor drift agree with the tangent prediction to relative errors below 6e-8. At larger amplitudes some edits cross nearby ReLU activation boundaries and anchor leakage grows, reaching approximately 8.6% of the requested change among the original edits. These observations describe local validity, rather than a global nonlinear edit-cost guarantee.
