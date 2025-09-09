import numpy as np
from scipy.linalg import lstsq
import configure as config
from inspect import isfunction
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Random seed (for reproducibility)
rand = config.seed
print(f'random seed: {rand}')
np.random.seed(rand)
random.seed(rand)

# Spatial grid: use ALL state points so our term evaluations match data length
x_bins = config.x.copy()         # the 28 bin-centers (sorted)
dx = x_bins[1] - x_bins[0]
dt = 0.01
config.dx = dx
config.dt = dt
x = config.x
t = np.ones_like(x)  # Dummy time array

zeros = np.zeros_like(x)
ones  = np.ones_like(x)

# ------------------------------------------------------------------------------
# Finite‐difference kernels
def FiniteDiff(u, dx):
    """First derivative using centered differences."""
    n = len(u)
    ux = np.zeros(n)
    ux[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    ux[0]    = (-3*u[0] + 4*u[1] - u[2]) / (2 * dx)
    ux[-1]   = (3*u[-1] - 4*u[-2] + u[-3]) / (2 * dx)
    return ux

def FiniteDiff2(u, dx):
    """Second derivative using centered differences."""
    n = len(u)
    uxx = np.zeros(n)
    uxx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    uxx[0]    = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
    uxx[-1]   = (2*u[-1] - 5*u[-2] + 4*u[-3] - u[-4]) / dx**2
    return uxx

# ------------------------------------------------------------------------------
# Wrapper to handle 1D & 2D arrays and allow 'x'/'t' axis names
def Diff(u, dx, axis):
    # 1D: only ∂/∂x makes sense
    if u.ndim == 1:
        if axis in (0, 'x'):
            return FiniteDiff(u, dx)
        else:
            raise ValueError("Cannot take time‐derivative of 1D array")

    # allow axis as int or string
    if isinstance(axis, str):
        a = axis.lower()
        if   a == 'x': axis = 0
        elif a == 't': axis = 1
        else:          raise ValueError("Axis must be 0 (x) or 1 (t)")

    if axis == 0:
        # differentiate along x‐dimension
        return np.array([FiniteDiff(u[:, i], dx) for i in range(u.shape[1])]).T
    elif axis == 1:
        # differentiate along t‐dimension
        return np.array([FiniteDiff(u[i, :], dx) for i in range(u.shape[0])])
    else:
        raise ValueError("Axis must be 0 (x) or 1 (t)")

def Diff2(u, dx, axis):
    # 1D: only ∂²/∂x² makes sense
    if u.ndim == 1:
        if axis in (0, 'x'):
            return FiniteDiff2(u, dx)
        else:
            raise ValueError("Cannot take time‐derivative of 1D array")

    # allow axis as int or string
    if isinstance(axis, str):
        a = axis.lower()
        if   a == 'x': axis = 0
        elif a == 't': axis = 1
        else:          raise ValueError("Axis must be 0 (x) or 1 (t)")

    if axis == 0:
        return np.array([FiniteDiff2(u[:, i], dx) for i in range(u.shape[1])]).T
    elif axis == 1:
        return np.array([FiniteDiff2(u[i, :], dx) for i in range(u.shape[0])])
    else:
        raise ValueError("Axis must be 0 (x) or 1 (t)")


# def Evaluate2(drift_val, diffusion_val, Ut_drift, Ut_diffusion,
#               lam, d_tol, AIC_ratio=1, maxit=10, STR_iters=10, normalize=0, sparse='STR'):
#     """
#     Fixed sparse regression with correct AIC: only non-zero coeffs > d_tol are counted,
#     and MSE is recomputed after STRidge.
#     """
#     # 1) Flatten target vectors
#     Ut_drift     = Ut_drift.reshape(-1)
#     Ut_diffusion = Ut_diffusion.reshape(-1)

#     # 2) AIC calculator uses thresholded nonzero count
#     def AIC(w, err):
#         # count coefficients whose magnitude exceeds sparsity tolerance
#         k = np.sum(np.abs(w) > d_tol)
#         return AIC_ratio * 2 * k + 2 * np.log(err)

#     # 3) Initial least-squares fit
#     w_drift,     _, _, _ = lstsq(drift_val,     Ut_drift)
#     mse_drift   = np.mean((Ut_drift     - drift_val.dot(w_drift))**2)
#     w_diffusion, _, _, _ = lstsq(diffusion_val, Ut_diffusion)
#     mse_diffusion = np.mean((Ut_diffusion - diffusion_val.dot(w_diffusion))**2)

#     # 4) Sparse regression (STRidge)
#     if sparse == 'STR':
#         w_drift     = STRidge(drift_val,     Ut_drift,     lam, STR_iters, d_tol, normalize)
#         w_diffusion = STRidge(diffusion_val, Ut_diffusion, lam, STR_iters, d_tol, normalize)
#         # Recompute MSE after sparsification
#         mse_drift     = np.mean((Ut_drift     - drift_val.dot(w_drift))**2)
#         mse_diffusion = np.mean((Ut_diffusion - diffusion_val.dot(w_diffusion))**2)

#     # 5) Compute AICs
#     aic_drift     = AIC(w_drift,     mse_drift)
#     aic_diffusion = AIC(w_diffusion, mse_diffusion)
#     total_mse     = mse_drift + mse_diffusion
#     total_aic     = aic_drift + aic_diffusion

#     # 6) Return: weights, MSEs, AICs
#     return (
#         np.concatenate([w_drift, w_diffusion]),
#         total_mse,
#         total_mse,
#         total_aic,
#         aic_drift,
#         aic_diffusion
#     )

def _kl_divergence(p, q):
    # Empty or mismatched → invalid; return huge penalty instead of crashing
    if p.size == 0 or q.size == 0 or p.size != q.size:
        return float('inf')
    return float(np.sum(p * np.log(p / q)))

def _wasserstein1(p, q, dx):
    if p.size == 0 or q.size == 0 or p.size != q.size:
        return float('inf')
    P = np.cumsum(p)  # CDFs
    Q = np.cumsum(q)
    return float(np.sum(np.abs(P - Q)) * dx)


def Evaluate2(drift_val, diffusion_val, Ut_drift, Ut_diffusion,
              lam, d_tol, AIC_ratio=1, maxit=10, STR_iters=10, normalize=0,
              sparse='STR', metric='mse'):
    """
    Sparse regression as before; only the scalar 'error' used for AIC switches
    between MSE, KL, or Wasserstein W1.
    """
    # 1) Targets
    Ut_drift     = Ut_drift.reshape(-1)
    Ut_diffusion = Ut_diffusion.reshape(-1)

    # 2) AIC calculator
    def AIC(w, err):
        k = np.sum(np.abs(w) > d_tol)   # count nonzero coeffs
        #err = max(float(err), 1e-12)    # guard -inf
        return AIC_ratio * 2 * k + 2 * np.log(err)

    # 3) Initial least-squares
    w_drift,     _, _, _ = lstsq(drift_val,     Ut_drift)
    w_diffusion, _, _, _ = lstsq(diffusion_val, Ut_diffusion)

    # Count active (non-tiny) coefficients using the same sparsity threshold
    k_d = int(np.sum(np.abs(w_drift)     > d_tol))
    k_f = int(np.sum(np.abs(w_diffusion) > d_tol))

    # Hard reject empty expression: both parts have zero active terms
    if getattr(config, 'reject_empty_model', True) and (k_d + k_f == 0):
        n_cols = (getattr(drift_val, 'shape', (0,0))[1] +
                getattr(diffusion_val, 'shape', (0,0))[1])
        huge = float(getattr(config, 'empty_model_penalty', 1e12))
        # Return zero weights and a huge score so this individual is discarded
        return (np.zeros(n_cols), huge, huge, huge, huge/2.0, huge/2.0)


    # 4) STRidge sparsification
    if sparse == 'STR':
        w_drift     = STRidge(drift_val,     Ut_drift,     lam, STR_iters, d_tol, normalize)
        w_diffusion = STRidge(diffusion_val, Ut_diffusion, lam, STR_iters, d_tol, normalize)

    # 5) Predictions (used for errors below)
    yhat_d = drift_val.dot(w_drift)
    yhat_f = diffusion_val.dot(w_diffusion)

    # 6) Compute error by selected metric
    if metric == 'mse':
        err_d = np.mean((Ut_drift     - yhat_d)**2)
        err_f = np.mean((Ut_diffusion - yhat_f)**2)
    elif metric == 'kl':
        err_d = _kl_divergence(Ut_drift,     yhat_d)
        err_f = _kl_divergence(Ut_diffusion, yhat_f)
    elif metric == 'wasserstein':
        dx = getattr(config, 'dx', 1.0)
        err_d = _wasserstein1(Ut_drift,     yhat_d, dx)
        err_f = _wasserstein1(Ut_diffusion, yhat_f, dx)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    total_err = err_d + err_f

    # 7) AICs from the chosen error
    aic_drift     = AIC(w_drift,     err_d)
    aic_diffusion = AIC(w_diffusion, err_f)
    total_aic     = aic_drift + aic_diffusion

    # 8) Return (keep shapes/signature the same)
    return (
        np.concatenate([w_drift, w_diffusion]),
        total_err,          # placeholder for "mse_d" slot
        total_err,          # placeholder for "mse_f" slot
        total_aic,
        aic_drift,
        aic_diffusion
    )


def STRidge(X0, y, lam, maxit, tol, normalize=0, print_results=False):
    """
    Sequential Threshold Ridge Regression (STRidge) with guarded normalization.

    Parameters
    ----------
    X0 : ndarray, shape (n_samples, n_features)
        Original design matrix.
    y : ndarray, shape (n_samples,)
        Target vector.
    lam : float
        Ridge regularization parameter.
    maxit : int
        Maximum number of STRidge iterations.
    tol : float
        Threshold below which coefficients are set to zero.
    normalize : int or float
        If nonzero, uses this norm to normalize columns of X0.
    print_results : bool
        If True, prints intermediate results (unused here).

    Returns
    -------
    w : ndarray, shape (n_features,)
        Learned coefficients (un-scaled if normalize=0, scaled back if normalize!=0).
    """
    n, d = X0.shape

    # Step 1: Normalize columns if requested, guarding against zero-norm columns
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        X = np.zeros((n, d))
        for i in range(d):
            norm_i = np.linalg.norm(X0[:, i], normalize)
            if norm_i > 1e-12:
                Mreg[i] = 1.0 / norm_i
                X[:, i] = X0[:, i] * Mreg[i]
            else:
                # Zero-variance feature: leave unscaled
                Mreg[i] = 1.0
                X[:, i] = X0[:, i]
    else:
        X = X0.copy()

    # Step 2: Initial ridge or least-squares fit
    if lam != 0:
        w = lstsq(X.T @ X + lam * np.eye(d), X.T @ y)[0]
    else:
        w = lstsq(X, y)[0]

    # Step 3: STRidge iterations
    biginds = np.where(np.abs(w) > tol)[0]
    for _ in range(maxit):
        smallinds = np.setdiff1d(np.arange(d), biginds)
        # Zero out small coefficients
        w[smallinds] = 0
        if biginds.size > 0:
            if lam != 0:
                w[biginds] = lstsq(
                    X[:, biginds].T @ X[:, biginds] + lam * np.eye(biginds.size),
                    X[:, biginds].T @ y
                )[0]
            else:
                w[biginds] = lstsq(X[:, biginds], y)[0]
        new_biginds = np.where(np.abs(w) > tol)[0]
        if np.array_equal(new_biginds, biginds):
            break
        biginds = new_biginds

    # Step 4: Final refit on nonzero coefficients
    if biginds.size > 0:
        w[biginds] = lstsq(X[:, biginds], y)[0]

    # Step 5: Un-normalize if necessary
    if normalize != 0:
        w = Mreg.ravel() * w

    return w


def divide(up, down, eta=1e-12):
    down[down == 0] = eta
    return up / down

def cubic(inputs):
    return np.power(inputs, 3)


# ------------------------------------------------------------------------------
# Library of candidate operations
ALL = [
    ['+',    2, np.add],
    ['-',    2, np.subtract],
    ['*',    2, np.multiply],
    ['/',    2, divide],
    ['d',    2, Diff],
    ['d^2',  2, Diff2],
    ['x',    0, x],
    ['^2',   1, np.square],
    ['^3',   1, cubic],
    ['1',    0, ones]
]

OPS = [
    ['+',   2, np.add],
    ['-',   2, np.subtract],
    ['*',   2, np.multiply],
    ['/',   2, divide],
    ['d',   2, Diff],
    ['d^2', 2, Diff2],
    ['^2',  1, np.square],
    ['^3',  1, cubic]
]

ROOT = [
    ['*',    2, np.multiply],
    ['d',    2, Diff],
    ['d^2',  2, Diff2],
    ['/',    2, divide],
    ['^2',   1, np.square],
    ['^3',   1, cubic]
]

OP1 = [
    ['^2',  1, np.square],
    ['^3',  1, cubic],
    ['^0.5',1, np.sqrt]
]

OP2 = [
    ['+',   2, np.add],
    ['-',   2, np.subtract],
    ['*',   2, np.multiply],
    ['/',   2, divide],
    ['d',   2, Diff],
    ['d^2', 2, Diff2]
]

VARS = [
    ['x', 0, x],
    ['1', 0, ones]
]

den = [
    ['x', 0, x]
]

# Containers for storing discovered PDEs and errors
pde_lib = []
err_lib = []
