# Normalising Flow Model

A normalising flow built from scratch in PyTorch — affine coupling layers, an analytic
log-determinant, machine-precision invertibility, and a tuned training pipeline on the
two-moons density.

**Deep Learning · MPhil in Data Intensive Science, University of Cambridge (2025/26)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-ee4c2c)
![Optuna](https://img.shields.io/badge/Optuna-TPE-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

A normalising flow learns a density by composing invertible maps from a simple base
distribution. The model is only as trustworthy as the exactness of those inverses and the
Jacobian bookkeeping behind them, so this project treats correctness as a result to be
demonstrated rather than assumed.

Four pieces of work: building the flow and proving it genuinely inverts; training it with
regularisation choices justified by ablation rather than habit; performing *surgery* on the
trained flow to produce a one-parameter family of densities with no retraining; and deriving
an analytic FLOP count for the inverse pass.

## Results

### Correctness

Both checks run in float64 on an *untrained* flow, isolating implementation correctness from
anything the model has learned.

| Check | Error | Interpretation |
| --- | --- | --- |
| Invertibility, `x → f⁻¹(x) → f(·)` across all 800 points | **4.44 × 10⁻¹⁶** | float64 machine epsilon |
| Analytic log\|det J\| vs central differences (ε = 1e-6) | **1.66 × 10⁻¹¹** | finite-difference truncation |

The forward and inverse maps are exact inverses to machine precision, and the analytic
Jacobian is correct.

### Training

| Stage | Train NLL | Validation NLL | Test NLL |
| --- | --- | --- | --- |
| Tiny-subset sanity run (128 points) | — | 0.449 | — |
| Unregularised capacity baseline | 0.366 | 0.612 | — |
| **Tuned final model** | **0.261** | **0.320** | **0.590** |

Training was staged deliberately. A 128-point overfitting run came first, purely to confirm
the pipeline could memorise a small sample and beat a fitted Gaussian baseline — if it cannot,
a bug is hiding somewhere. Only then was the full dataset used.

Regularisation was introduced one change at a time. **Early stopping helped most; weight decay
and dropout both made validation worse** and were dropped. An Optuna sweep (30 trials, TPE
sampler, median pruner) then searched learning rate, scheduler, depth, width and batch size,
and a post-hoc ablation attributed the gain to its source: raising the learning rate from
3×10⁻⁴ to ≈4.1×10⁻³ was the single largest improvement, not the architecture changes.

Final configuration: 8 coupling layers, hidden width 96, OneCycleLR, no weight decay, no
dropout — retrained from scratch and selected on validation NLL alone.

### Flow surgery

The shear `y₁ = x₁ + αx₂`, `y₂ = x₂` applied after the trained sampling map has an
upper-triangular Jacobian with determinant 1, so `log|det| = 0` for every α. The family
`p_α(x) = p₀(g_α⁻¹(x))` is therefore an exactly volume-preserving deformation of the learned
density — a continuum of valid densities obtained with **zero retraining**.

### FLOP count

An analytic cost model for the inverse pass used by `log_prob`. Each coupling layer's MLP
contributes `2HD + H + 4DH`, plus `10D` for the tanh on the scale output; the inverse
transform adds `D` subtractions, `10D` exponentiations and `D` divisions; log-determinant
accumulation costs `2D − 1` per layer. Permutations, slicing and Python overhead are treated
as free.

## Repository structure

```
.
├── coursework.ipynb          # Full analysis — 73 cells, code and written discussion
├── results.json              # Correctness metrics, training metrics, written write-up
├── report.pdf                # Assessed report
├── checkpoints/flow_full.pt  # Trained flow weights
├── logs/training_curves.json # Loss curves
├── figs/                     # Generated figures
├── data/                     # two-moons train / validation / test splits
└── pyproject.toml
```

## Notebook contents

| Section | Content |
| --- | --- |
| 0 | Preprocessing and data exploration |
| 1 | Affine coupling flow, analytic log-determinant, correctness checks |
| 2 | Tiny-subset sanity stage, full-dataset optimisation, model saving |
| 3 | Flow surgery — a one-parameter family of densities |
| 4 | FLOP counting for the inverse pass |

## Setup

```bash
python3 -m venv m2_env
source m2_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m ipykernel install --user --name normalising-flow --display-name "Python (normalising-flow)"
```

Open `coursework.ipynb`, select the `Python (normalising-flow)` kernel and run top to bottom.
Outputs regenerate into `figs/`, `checkpoints/`, `logs/` and `results.json`.

## Reproducibility

The notebook runs CPU-only by design. GPU execution is disabled via `CUDA_VISIBLE_DEVICES=-1`,
because non-deterministic kernel scheduling would undermine the bit-level reproducibility the
correctness checks depend on. A global seed is set before any stochastic operation.

## Use of Generative Tools

This project has utilised auto-generative tools in the development of the solutions notebook.

## Author

Harvey Bermingham — MPhil in Data Intensive Science, University of Cambridge

## License

Released under the MIT License. See [LICENSE](LICENSE).
