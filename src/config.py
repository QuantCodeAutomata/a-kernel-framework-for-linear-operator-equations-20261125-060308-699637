"""Experiment parameters for the kernel framework replication.

All parameters are module-level constants grouped by experiment. Each
comment states the paper section/equation that specifies the value, or
"not specified in the paper" when the paper leaves it open (in which case a
reasonable value is used and the choice is flagged as an assumption in
results/RESULTS.md).

Reference: Filipovic & Tissot-Daguette, "A Kernel Framework for Linear
Operator Equations" (arXiv:2611.31626).
"""

import numpy as np

# Use float64 throughout: the linear systems are mildly ill-conditioned
# (small lambda) and JAX defaults to float32 unless enabled otherwise.
JAX_ENABLE_X64 = True

RANDOM_SEED = 2026  # not specified in the paper for any experiment; fixed for reproducibility

# ---------------------------------------------------------------------------
# Experiment 1: First-order ODE with Gaussian kernel (paper Sec. 4.1, p.16)
# ---------------------------------------------------------------------------
EXP1_GAMMA = 100.0          # Gaussian kernel inverse squared length scale, paper Sec. 4.1 (gamma^{-1/2}=0.1)
EXP1_LAMBDA = 1e-8          # regularization parameter lambda, paper Sec. 4.1
EXP1_W1 = 1.0               # objective weight on interior ODE equation, paper Sec. 4.1
EXP1_W2 = 1.0               # objective weight on boundary condition equation, paper Sec. 4.1
EXP1_N1 = 100               # number of interior collocation points in (-1,1), paper Sec. 4.1
EXP1_N2 = 2                 # number of boundary collocation points {-1,1}, paper Sec. 4.1
EXP1_N_QUERY = 400          # query grid size for plotting/error evaluation, not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 2: Second-order ODE with truncated Fourier kernel (paper Sec. 4.1, p.17)
# ---------------------------------------------------------------------------
EXP2_L = 20                 # Fourier truncation order L, paper Sec. 4.1
EXP2_LAMBDA = 1e-10         # regularization parameter lambda, paper Sec. 4.1
EXP2_W1 = 1.0               # objective weight on interior ODE equation, paper Sec. 4.1
EXP2_W2 = 1.0               # objective weight on boundary condition equation, paper Sec. 4.1
EXP2_N1 = 150               # number of interior collocation points in [0.01,0.99], paper Sec. 4.1
EXP2_N2 = 2                 # number of boundary collocation points {0,1}, paper Sec. 4.1
EXP2_N_VALUES = (1, 2, 5)   # frequency indices n used in zeta_n = (2n+1/2)^2 pi^2, paper Sec. 4.1
EXP2_N_QUERY = 400          # query grid size for plotting/error evaluation, not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 3: 1D heat equation with parabolic kernel (paper Sec. 4.2.1, p.18)
# ---------------------------------------------------------------------------
EXP3_T = 0.5                # terminal time T, paper Sec. 4.2.1
EXP3_X_MIN = -3.0            # left edge of spatial box, paper Sec. 4.2.1
EXP3_X_MAX = 3.0             # right edge of spatial box, paper Sec. 4.2.1
EXP3_SIGMA0 = np.sqrt(0.5)   # initial condition phi_{1/2}: standard deviation at t=0, paper Sec. 4.2.1
EXP3_GAMMA0 = 4.0            # parabolic kernel time bandwidth, paper Sec. 4.2.1 (Eq. 53)
EXP3_GAMMA1 = 1.0            # parabolic kernel spatial bandwidth (in warped coordinates), paper Sec. 4.2.1
EXP3_EPS_WARP = 0.25         # warp parameter epsilon in psi(t,x)=x/sqrt(t+eps), paper Sec. 4.2.1
EXP3_LAMBDA = 1e-7           # regularization parameter lambda, paper Sec. 4.2.1
EXP3_W1 = 1.0                # objective weight on interior PDE equation, paper Sec. 4.2.1
EXP3_W2 = 1.0                # objective weight on initial condition equation, paper Sec. 4.2.1
EXP3_N1 = 256                # number of interior collocation points in (0,T)x[-3,3], paper Sec. 4.2.1
EXP3_N2 = 32                 # number of initial-condition collocation points in {0}x[-3,3], paper Sec. 4.2.1
EXP3_GRID_NT = 51            # time grid resolution for the reported relative L2 error, not specified in the paper
EXP3_GRID_NX = 121           # space grid resolution for the reported relative L2 error, not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 4: Bachelier option pricing PDE with Greeks (paper Sec. 4.2.2, p.18-19)
# ---------------------------------------------------------------------------
EXP4_T = 1.0                 # maturity T (years), paper Sec. 4.2.2
EXP4_SIGMA = 20.0             # Bachelier normal volatility sigma, paper Sec. 4.2.2
EXP4_X0 = 100.0               # at-the-money strike/spot x0, paper Sec. 4.2.2
EXP4_X_MIN = -20.0             # left edge of spatial box, paper Sec. 4.2.2
EXP4_X_MAX = 220.0             # right edge of spatial box, paper Sec. 4.2.2
EXP4_GAMMA0 = 8.0             # parabolic kernel time bandwidth, paper Sec. 4.2.2
EXP4_GAMMA1 = 1.0 / EXP4_SIGMA ** 2  # spatial bandwidth 1/sigma^2 matching diffusion scale, paper Sec. 4.2.2
EXP4_EPS_WARP = 0.5           # warp epsilon in psi(t,x)=x/sqrt(T-t+eps); not specified in the paper. A small value gives
                              # a near-singular (AA*+lambda*G) system close to maturity (condition number grows by
                              # several orders of magnitude as eps_warp shrinks); 0.5 keeps the system well conditioned
                              # while still localizing the kernel bandwidth near the terminal slice, paper Sec. 4.2.2
EXP4_LAMBDA = 1e-7            # regularization parameter lambda, paper Sec. 4.2.2
EXP4_W1 = 1.0                 # objective weight on interior PDE equation, paper Sec. 4.2.2
EXP4_W2 = 10.0                # objective weight on terminal condition equation, paper Sec. 4.2.2
EXP4_N1 = 1024                # number of interior collocation points, paper Sec. 4.2.2 (2^10)
EXP4_N2 = 1024                # number of terminal collocation points, paper Sec. 4.2.2 (2^10)
EXP4_GRID_NT = 9              # number of time slices reported for Delta/Gamma comparison, not specified in the paper
EXP4_GRID_NX = 161            # spatial grid resolution for Delta/Gamma comparison, not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 5: Multi-asset Black-Scholes max-call with Nystrom approximation
# (paper Sec. 4.2.3, p.19-20)
# ---------------------------------------------------------------------------
EXP5_S0 = 100.0                # initial price per asset, paper Sec. 4.2.3
EXP5_K = 100.0                 # strike, paper Sec. 4.2.3
EXP5_T = 1.0                   # maturity, paper Sec. 4.2.3
EXP5_SIGMA = 0.20              # volatility per asset, paper Sec. 4.2.3
EXP5_RHO = 0.50                # pairwise correlation, paper Sec. 4.2.3
EXP5_EPS_PAYOFF = 2.0          # log-sum-exp payoff smoothing scale, paper Sec. 4.2.3
EXP5_EPS_WARP = 0.10           # warp parameter epsilon, paper Sec. 4.2.3
EXP5_GAMMA0 = 4.0              # parabolic kernel time bandwidth, paper Sec. 4.2.3
EXP5_GAMMA1 = 1.0 / 16.0       # parabolic kernel spatial bandwidth, paper Sec. 4.2.3
EXP5_LAMBDA = 1e-7             # regularization parameter lambda, paper Sec. 4.2.3
EXP5_W1 = 1.0                  # objective weight on interior PDE equation, paper Sec. 4.2.3
EXP5_W2 = 10.0                 # objective weight on terminal condition equation, paper Sec. 4.2.3
EXP5_DIMS = (2, 3, 5)          # number of underlying assets d, paper Sec. 4.2.3

# Paper-scale collocation/Nystrom/Monte Carlo configuration (Sec. 4.2.3)
EXP5_N_PAPER = 2 ** 13         # n1 = n2 Sobol collocation points, paper Sec. 4.2.3
EXP5_R_GRID_PAPER = (2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11)  # Nystrom sizes r, paper Sec. 4.2.3
EXP5_MC_PATHS_PAPER = 2 ** 20  # Monte Carlo benchmark paths, paper Sec. 4.2.3

# Executed configuration: reduced to fit the available compute budget while
# preserving the method (Nystrom with nested index sets, same kernel and
# operator). See results/RESULTS.md, "Deviations from the paper".
EXP5_N_RUN = 2 ** 11           # executed n1 = n2 Sobol collocation points
EXP5_R_GRID_RUN = (2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9)  # executed Nystrom sizes r
EXP5_MC_PATHS_RUN = 2 ** 16    # executed Monte Carlo benchmark paths
EXP5_R_SELECTED_FRACTION_TERMINAL = 0.4  # fraction r2/r taken from terminal condition, paper Sec. 4.2.3 (r2=floor(0.4r))

# ---------------------------------------------------------------------------
# Experiment 6: Bates model pricing PIDE with closed-form jump integral
# (paper Sec. 4.2.4, p.21-23)
# ---------------------------------------------------------------------------
EXP6_ZETA = 0.3                # jump intensity, paper Sec. 4.2.4
EXP6_ALPHA = -0.1              # mean log-jump size, paper Sec. 4.2.4
EXP6_BETA2 = 0.15 ** 2         # log-jump variance beta, paper Sec. 4.2.4
EXP6_KAPPA = 1.5               # variance mean-reversion speed, paper Sec. 4.2.4
EXP6_VBAR = 0.04               # long-run variance, paper Sec. 4.2.4
EXP6_ETA = 0.3                 # vol-of-vol, paper Sec. 4.2.4
EXP6_RHO = -0.75                # correlation between price and variance Brownian motions, paper Sec. 4.2.4
EXP6_R = 0.0                    # risk-free rate; not specified in the paper for this experiment, set to 0
EXP6_S0 = 100.0                 # spot price, paper Sec. 4.2.4
EXP6_K = 100.0                  # strike, paper Sec. 4.2.4
EXP6_T = 1.0                    # maturity (one-year option), paper Sec. 4.2.4
EXP6_V0 = 0.04                  # initial variance used for the price comparison (t=0, v=0.04), paper Sec. 4.2.4
EXP6_X_BOX = (np.log(100.0) - 0.6, np.log(100.0) + 0.6)  # log-price box, paper Sec. 4.2.4
EXP6_V_BOX = (0.01, 0.16)       # variance box, paper Sec. 4.2.4
EXP6_GAMMA0 = 1.0               # kernel time bandwidth, paper Sec. 4.2.4
EXP6_GAMMA1 = 25.0              # kernel log-price bandwidth, paper Sec. 4.2.4
EXP6_GAMMA2 = 14.0              # kernel variance bandwidth (paper: "approximately 14", selected by cross-validation)
EXP6_LAMBDA = 1e-6              # regularization parameter lambda, paper Sec. 4.2.4
EXP6_N_PAPER = 2 ** 11          # n1 = n2 Sobol collocation points, paper Sec. 4.2.4
EXP6_N_RUN = 2 ** 10            # executed n1 = n2 Sobol collocation points (reduced to fit the compute budget)
EXP6_N_PRICE_POINTS = 41        # number of spot prices for the FFT comparison curve, not specified in the paper

# ---------------------------------------------------------------------------
# Experiment 7: Variance Gamma model pricing PIDE with Gauss-Laguerre quadrature
# (paper Sec. 4.2.5, p.23-25)
# ---------------------------------------------------------------------------
EXP7_THETA = -0.14              # VG drift parameter theta, paper Sec. 4.2.5
EXP7_SIGMA = 0.12                # VG volatility parameter sigma, paper Sec. 4.2.5
EXP7_NU = 0.2                    # VG variance-rate parameter nu, paper Sec. 4.2.5
EXP7_R = 0.03                    # risk-free rate, paper Sec. 4.2.5
EXP7_S0 = 100.0                  # spot price (at-the-money), paper Sec. 4.2.5
EXP7_K = 100.0                   # strike (at-the-money), paper Sec. 4.2.5
EXP7_T = 1.0                     # maturity (one-year option), paper Sec. 4.2.5
EXP7_GAMMA0 = 4.0                # kernel time bandwidth, paper Sec. 4.2.5
EXP7_GAMMA1 = 1.0 / (2 * EXP7_SIGMA * np.sqrt(EXP7_T)) ** 2  # kernel log-price bandwidth, paper Sec. 4.2.5
EXP7_LAMBDA = 1e-7               # regularization parameter lambda, paper Sec. 4.2.5
EXP7_N_GAUSS_LAGUERRE = 12       # Gauss-Laguerre quadrature nodes per half-line, paper Sec. 4.2.5
EXP7_N_CONVERGENCE_PAPER = (6, 20, 50, 100, 200, 400, 800, 1600)  # collocation sizes for the convergence study, paper Sec. 4.2.5
EXP7_N_CONVERGENCE_RUN = (6, 20, 50, 100, 200, 400, 800)  # executed sizes (reduced to fit the compute budget; see RESULTS.md)
EXP7_N_CI_PAPER = 2 ** 10        # collocation size for the confidence-interval study, paper Sec. 4.2.5
EXP7_N_CI_RUNS_PAPER = 16        # number of randomized Sobol runs for the CI, paper Sec. 4.2.5
EXP7_N_CI_RUNS_RUN = 16          # executed number of randomized runs (kept at the paper value)
EXP7_FFT_BENCHMARK_PRICE = 6.8398  # FFT benchmark reported by the paper, paper Sec. 4.2.5, Table 1

# FFT (Carr-Madan) benchmark settings shared by experiments 6 and 7; not
# specified in the paper beyond "FFT benchmark of Carr and Madan".
FFT_N = 2 ** 16                 # number of FFT grid points
FFT_ALPHA = 1.5                 # Carr-Madan damping factor
FFT_ETA = 0.05                  # frequency grid spacing
