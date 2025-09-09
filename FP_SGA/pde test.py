import numpy as np
import configure as config
from requests import get
from inspect import isfunction
import math
import pdb
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def FiniteDiff(u, dx):
    n = u.size
    ux = np.zeros(n)
    for i in range(1, n - 1):
        ux[i] = (u[i + 1] - u[i - 1]) / (2 * dx)
    ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
    ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
    return ux

def FiniteDiff2(u, dx):
    n = u.size
    ux = np.zeros(n)
    for i in range(1, n - 1):
        ux[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2
    ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
    ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
    return ux

def Diff(u, dxt, name):
    """
    Here dx is a scalar, name is a str indicating what it is
    """
    n, m = u.shape
    uxt = np.zeros((n, m))
    if name == 'x':
        for i in range(m):
            uxt[:, i] = FiniteDiff(u[:, i], dxt)

    elif name == 't':
        for i in range(n):
            uxt[i, :] = FiniteDiff(u[i, :], dxt)

    else:
        NotImplementedError()

    return uxt

def Diff2(u, dxt, name):
    """
    Here dx is a scalar, name is a str indicating what it is
    """
    n, m = u.shape
    uxt = np.zeros((n, m))
    if name == 'x':
        for i in range(m):
            uxt[:, i] = FiniteDiff2(u[:, i], dxt)

    elif name == 't':
        for i in range(n):
            uxt[i, :] = FiniteDiff2(u[i, :], dxt)

    else:
        NotImplementedError()

    return uxt

def smooth_pdf(u_orig, spatial_sigma=1.0, temp_sigma=1.0):
    """2D Gaussian smoothing in both spatial and temporal directions"""
    # Apply 2D Gaussian smoothing using separable 1D filters
    u_smooth = gaussian_filter(u_orig, sigma=(spatial_sigma, temp_sigma), mode='mirror', axes=(0, 1))
    
    return u_smooth

def truncate_edges(u, x, t, percent=0.05):
    """
    Truncate `percent` from each edge in BOTH spatial (axis 0) and temporal (axis 1).
    Returns:
      - u_trunc:  truncated 2D array
      - x_trunc:  truncated spatial vector
      - t_trunc:  truncated temporal vector
    """
    n_x, n_t = u.shape
    cut_x = int(n_x * percent)
    cut_t = int(n_t * percent)
    
    u_trunc = u[cut_x:-cut_x, cut_t:-cut_t]
    x_trunc = x[cut_x:-cut_x]
    t_trunc = t[cut_t:-cut_t]
    return u_trunc, x_trunc, t_trunc

# Load data and compute derivatives (using your existing code)
u = config.u
x = config.x
t = config.t

dx = x[1] - x[0]
dt = t[1] - t[0]

u  = smooth_pdf(u, 1, 5)
u,x,t = truncate_edges(u, x, t, 0.05)
n, m = u.shape

# Compute derivatives
ut = Diff(u, dt, 't')
ux = Diff(u, dx, 'x')
uxx = Diff2(u, dx, 'x')


# # Compute Fokker-Planck RHS for OU process: pt = 2*d²p/dx² + 3*(d/dx(x*p))
# pt = np.zeros((n, m))
# for idx in range(m):
#     # First term: 2*d²p/dx² (pure diffusion)
#     diffusion_term = 2.11 * FiniteDiff2(u[:, idx], dx)
    
#     # Second term: 3*d/dx(x*p) (drift term)
#     xp = x * u[:, idx]
#     drift_derivative = 2.63 * FiniteDiff(xp, dx)

#     pt[:, idx] = diffusion_term + drift_derivative

# Compute Fokker-Planck RHS (Double well)
pt = np.zeros((n, m))
for idx in range(m):
    flux = (1*x - x**3) * u[:, idx]
    flux_derivative = FiniteDiff(flux, dx)
    diffusion_term = 0.5 * uxx[:, idx]
    pt[:, idx] = -flux_derivative + diffusion_term
    
pt2 = np.zeros((n, m))
for idx in range(m):
    flux = (0.63*x - 0.7*x**3) * u[:, idx]
    flux_derivative = FiniteDiff(flux, dx)
    diffusion_term = 0.823/2 * uxx[:, idx]
    pt2[:, idx] = -flux_derivative + diffusion_term   

# Compute Fokker-Planck RHS with new form: pt = -d/dx[(1-x)p] + 0.045 d²/dx²(x*p) (CIR)
# pt = np.zeros((n, m))
# for idx in range(m):
#     # First term: -d/dx[(1-x)*p]
#     flux = (1 - x) * u[:, idx]
#     flux_derivative = FiniteDiff(flux, dx)
    
#     # Second term: 0.045 d²/dx²(x*p)
#     xp = x * u[:, idx]
#     xp_second_deriv = FiniteDiff2(xp, dx)
#     diffusion_term = 0.045 * xp_second_deriv

#     pt[:, idx] = -flux_derivative + diffusion_term

# Find global min/max for consistent color scaling
vmin = min(np.min(ut), np.min(pt))
vmax = max(np.max(ut), np.max(pt))

# Create figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

# Plot ut
contour1 = ax1.contourf(t, x, ut, levels=50, vmin=vmin, vmax=vmax)
ax1.set_title('Time Derivative (ut)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Space')
fig.colorbar(contour1, ax=ax1)

# Plot pt
contour2 = ax2.contourf(t, x, pt, levels=50, vmin=vmin, vmax=vmax)
ax2.set_title('Theoretical Fokker-Planck RHS (pt)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Space')
fig.colorbar(contour2, ax=ax2)

# Plot pt2
contour3 = ax3.contourf(t, x, pt2, levels=50, vmin=vmin, vmax=vmax)
ax3.set_title('Learned Fokker-Planck RHS (pt)')
ax3.set_xlabel('Time')
ax3.set_ylabel('Space')
fig.colorbar(contour3, ax=ax3)

# Adjust layout and show
plt.tight_layout()
plt.show()