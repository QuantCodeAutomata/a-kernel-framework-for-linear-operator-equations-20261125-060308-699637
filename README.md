# A Kernel Framework for Linear Operator Equations

Replication of selected numerical experiments from Filipovic, D. and
Tissot-Daguette, L., "A Kernel Framework for Linear Operator Equations"
(arXiv:2611.31626).

The paper develops a reproducing-kernel-Hilbert-space method for solving
systems of linear operator equations (ODEs, PDEs, PIDEs) by collocation: a
linear operator equation `A h = f` is turned into a representer-based
least-squares problem, solved in closed form by a regularized normal
equation, with a Nystrom reduction for large collocation sets. This
repository reproduces the paper's worked examples, Section 4.

## Experiments

1. `exp1` — First-order ODE with a Gaussian kernel, overdetermined boundary
   conditions (Sec. 4.1, p.16).
2. `exp2` — Second-order ODE with a truncated Fourier kernel adapted to the
   operator (Sec. 4.1, p.17).
3. `exp3` — One-dimensional heat equation with a parabolic kernel (Sec.
   4.2.1, p.18).
4. `exp4` — Bachelier option-pricing PDE with Delta and Gamma sensitivities
   (Sec. 4.2.2, p.18-19).
5. `exp5` — Multi-asset Black-Scholes max-call option with Nystrom
   approximation, d = 2, 3, 5 (Sec. 4.2.3, p.19-20).
6. `exp6` — Bates stochastic-volatility-with-jumps pricing PIDE, closed-form
   jump integral on Gaussian kernels (Sec. 4.2.4, p.21-23).
7. `exp7` — Variance Gamma pricing PIDE with Gauss-Laguerre quadrature of
   the jump integral (Sec. 4.2.5, p.23-25).

No market data is used: every equation is solved on synthetic collocation
points drawn from the probability measures the paper specifies.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
python -m exp.exp6
python -m exp.exp7
```

Each command writes aggregate metrics (`results/expN_metrics.json`) and
static plots (`results/expN_*.png`) to `results/`. Experiments 5-7 accept a
`--full` flag to run the paper's full configuration (see
`results/RESULTS.md` for the executed vs. paper configuration and runtime
estimates).

See `results/RESULTS.md` for the comparison against the paper's reported
values, deviations, and items not reproduced. See `AGENTS.md` for repository
conventions.
