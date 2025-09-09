# configure.py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

# ── Problem setup ─────────────────────────────────────────────────────────────
# Choices: 'OU' | 'double_well' | 'CIR'
problem   = 'double_well'
seed      = 17
device    = torch.device('cuda:0')
aic_ratio = 1

# Optional knobs used elsewhere in your code (safe defaults)
error_metric = 'mse'    # 'mse' | 'kl' | 'wasserstein'


# ── SDE library (drift a(x), diffusion b(x)) ──────────────────────────────────
def get_problem_spec(name):
    """
    Returns (a_fn, b_fn, x_domain, default_params) for the chosen problem.
    b(x) is the *diffusion amplitude* (so b(x)^2 is the KM diffusion coefficient).
    """
    name = name.lower()
    if name == 'ou':
        # dX = -kappa*(X - mu) dt + sigma dW
        params = dict(kappa=1.0, mu=1.0, sigma=1.0)
        def a(x, p=params): return -p['kappa'] * (x - p['mu'])
        def b(x, p=params): return p['sigma'] * np.ones_like(x)
        x_domain = (-3.0, 5.0)  # wide enough to capture mean=1 fluctuations
        return a, b, x_domain, params

    if name == 'double_well':
        # dX = -(dV/dx) dt + sigma dW,  V(x) = (x^2 - 1)^2 / 4  ⇒ dV/dx = x^3 - x
        # so a(x) = -(x^3 - x) = x - x^3
        params = dict(sigma=3)
        def a(x, p=params): return x - x**3
        def b(x, p=params): return p['sigma'] * np.ones_like(x)
        x_domain = (-2.5, 2.5)
        return a, b, x_domain, params

    if name == 'cir':
        # Cox–Ingersoll–Ross: dX = kappa*(theta - X) dt + sigma*sqrt(X) dW,  X>=0
        # We use "full truncation" to keep sqrt argument nonnegative.
        params = dict(kappa=1.0, theta=1.0, sigma=3)
        def a(x, p=params): return p['kappa'] * (p['theta'] - x)
        def b(x, p=params): return p['sigma'] * np.sqrt(np.clip(x, 0.0, None))
        x_domain = (0.0, 4.0)
        return a, b, x_domain, params

    raise ValueError(f"Unknown problem: {name}")

# ── Simulation (Euler–Maruyama) ───────────────────────────────────────────────
def simulate_sde(a_fn, b_fn, x0, dt, steps):
    """
    Euler–Maruyama with 'full truncation' for CIR-like diffusions (b uses clip inside).
    x0: array of initial states, shape (n_traj,)
    returns: all_states with shape (steps+1, n_traj)
    """
    x = np.asarray(x0, float).copy()
    n_traj = x.shape[0]
    out = np.empty((steps + 1, n_traj), dtype=float)
    out[0] = x
    sqrt_dt = np.sqrt(dt)
    for t in range(steps):
        xi = np.random.randn(n_traj)
        drift = a_fn(x)
        diff  = b_fn(x)
        x = x + drift * dt + diff * sqrt_dt * xi
        # positivity guard is embedded in b_fn for CIR via np.clip
        out[t + 1] = x
    return out

# ── KM Estimator (1D, unconditional bins on X_t) ──────────────────────────────
def km_estimation(all_states, dt, bins, x_min, x_max):
    """
    Returns: (bin_edges, bin_centers, drift_est, diffusion_est, counts)
    diffusion_est approximates b(x)^2 via E[(ΔX)^2 | X]/Δt  (Ito KM-2 coefficient).
    """
    x_t  = all_states[:-1, :].ravel()
    dxt  = (all_states[1:, :] - all_states[:-1, :]).ravel()
    edges = np.linspace(x_min, x_max, bins + 1)
    idx   = np.digitize(x_t, edges) - 1
    idx   = np.clip(idx, 0, bins - 1)

    counts   = np.bincount(idx, minlength=bins).astype(float)
    sum_dx   = np.bincount(idx, weights=dxt, minlength=bins)
    sum_dx2  = np.bincount(idx, weights=dxt * dxt, minlength=bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        drift_est     = np.where(counts > 0, sum_dx  / (counts * dt), 0.0)
        diffusion_est = np.where(counts > 0, sum_dx2 / (counts * dt), 0.0)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers, drift_est, diffusion_est, counts

# ── Global data generation & KM preprocessing ─────────────────────────────────
np.random.seed(seed)

# Problem spec
a_fn, b_fn, (x_lo, x_hi), params = get_problem_spec(problem)

# Simulation settings (tune as needed)
dt         = 0.01
T          = 5.0
steps      = int(T / dt)
n_traj     = 10000

# Initial states
if problem.lower() == 'cir':
    x0 = np.full(n_traj, params['theta'], dtype=float)  # start at mean
    x0 += 0.1 * np.random.randn(n_traj)                 # small jitter
    x0 = np.clip(x0, 0.0, None)
else:
    # N(center, spread^2)
    x0 = 0.5 * (x_lo + x_hi) + 0.25 * (x_hi - x_lo) * np.random.randn(n_traj)

# Simulate
all_states = simulate_sde(a_fn, b_fn, x0, dt, steps)

# KM estimation on X_t with uniform bins over domain
bins = 30
bin_edges, bin_centers, drift_est, diffusion_est, counts = km_estimation(
    all_states, dt, bins, x_lo, x_hi
)

# Trim a few edge bins where counts are small
n_trim = max(1, int(0.05 * bins))  # 5% on each side
mask   = np.ones(bins, dtype=bool)
mask[:n_trim]  = False
mask[-n_trim:] = False

bin_centers_trimmed = bin_centers[mask]
drift_trimmed       = drift_est[mask]
diffusion_trimmed   = diffusion_est[mask]

# Indices of the “central” bins (in the full, untrimmed indexing)
KM_central_valid_indices = np.where(mask)[0]

# ── Export to the rest of the pipeline ────────────────────────────────────────
KM_bin_edges             = bin_edges               # full, untrimmed edges
drift_target             = drift_trimmed.reshape(-1, 1)
diffusion_target         = diffusion_trimmed.reshape(-1, 1)
all_states               = all_states             # raw SDE samples
x                        = bin_centers_trimmed     # for visualization / setup
dx                       = x[1] - x[0]             # uniform spacing
dt                       = dt
