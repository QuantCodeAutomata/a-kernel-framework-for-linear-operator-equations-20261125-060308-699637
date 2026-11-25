# Repository guide

This repository reproduces selected numerical experiments from Filipovic, D.
and Tissot-Daguette, L., "A Kernel Framework for Linear Operator Equations"
(arXiv:2611.31626): a reproducing-kernel-Hilbert-space collocation method for
systems of linear operator equations (ODEs, PDEs, PIDEs), with a Nystrom
reduction for large collocation sets.

## Layout

- `src/` — flat library modules: `kernel_solver.py` (generic block
  least-squares solver and Nystrom reduction, Sec. 2-3), `jax_utils.py`
  (JAX-based derivative kernels), `fourier_kernel.py` (truncated Fourier
  kernel), `parabolic_kernel.py` (parabolic/backward-warp kernel),
  `bachelier.py`, `black_scholes.py`, `bates.py`, `variance_gamma.py`
  (pricing-model operators and closed-form benchmarks), `fft_pricing.py`
  (Carr-Madan FFT benchmark), `config.py` (all experiment parameters as
  module-level `UPPER_CASE` variables).
- `exp/` — one script per experiment (`exp1.py`-`exp7.py`), each calling
  `src/` functions and writing `results/expN_metrics.json` and
  `results/expN_*.png`.
- `data/` — `generate_synthetic.py` generates the collocation points
  (uniform draws, equally spaced grids, scrambled Sobol sequences) each
  experiment uses; no market data is used anywhere in this repository.
- `tests/` — unit tests of `src/` on synthetic fixtures.
- `results/` — `RESULTS.md` plus the aggregate metrics (`.json`) and static
  plots (`.png`) written by the experiment scripts.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -m "not market_data"
python -m exp.exp1   # ... through exp7
python -m exp.exp5 --full   # paper's full Nystrom configuration
python -m exp.exp7 --full   # paper's full convergence/CI grid
```

No API keys are used by this repository; it does not read `DATALOADER_API_KEY`
or `MASSIVE_API_KEY` and there is no `.env.example`, since no experiment loads
market data.

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` variables with a comment giving their meaning, unit, and the
  paper section (or "not specified in the paper"). Do not hardcode
  parameters elsewhere.
- Each experiment script calls the shared `src/kernel_solver.py` block
  least-squares solve (`solve_block_system`) and Nystrom reduction
  (`solve_nystrom_system`); it does not reimplement the linear-algebra core.
- Running an experiment script overwrites its `results/expN_*` files.
  `results/RESULTS.md` is not regenerated automatically: after a rerun,
  check its numbers and verdicts against the new files and update it by
  hand if they changed.
- Tests run on synthetic fixtures only; there is no `market_data` marker in
  this repository because no experiment depends on live market data, but
  `python -m pytest -m "not market_data"` still works (the marker is simply
  unused).
