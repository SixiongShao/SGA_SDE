import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

# =============================================
# 1. Parameters for Ornstein-Uhlenbeck (OU) Process
# =============================================
theta = 1.0      # Mean reversion strength
mu = 0.5         # Long-term mean
sigma = 0.2      # Volatility
X0 = 0.0         # Initial state

# Simulation parameters
n_trajectories = 2000  # Number of realizations
T = 5.0                # Total time
dt = 0.01              # Time step
n_steps = int(T/dt)    # Number of steps
t = np.linspace(0, T, n_steps)

# =============================================
# 2. Simulate OU Process Trajectories
# =============================================
trajectories = np.zeros((n_trajectories, n_steps))
for i in range(n_trajectories):
    x = np.zeros(n_steps)
    # Small random initial condition variation
    x[0] = X0 + np.random.normal(0, 0.05)
    
    # Generate Wiener increments
    dW = np.random.normal(0, np.sqrt(dt), n_steps-1)
    
    # Euler-Maruyama integration
    for j in range(1, n_steps):
        x[j] = x[j-1] + theta*(mu - x[j-1])*dt + sigma*dW[j-1]
    
    trajectories[i] = x

# Plot sample trajectories
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.plot(t, trajectories[i], alpha=0.7)
plt.xlabel('Time')
plt.ylabel('State')
plt.title('OU Process Sample Trajectories')
plt.grid(True)
plt.show()

# =============================================
# 3. MANTRA's KM Coefficient Estimation
# =============================================
def mantra_km_estimation(trajectories, dt, bins=30, min_points=20):
    """Estimate drift and diffusion coefficients using MANTRA's binning approach"""
    # Flatten all trajectory points (excluding last point)
    all_states = trajectories[:, :-1].flatten()
    all_deltas = (trajectories[:, 1:] - trajectories[:, :-1]).flatten()
    
    # Create state bins
    state_min, state_max = np.min(all_states), np.max(all_states)
    bin_edges = np.linspace(state_min, state_max, bins+1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    # Initialize arrays
    drift = np.zeros(bins)
    diffusion = np.zeros(bins)
    counts = np.zeros(bins)
    
    # Bin assignment and moment calculation
    for state, delta in zip(all_states, all_deltas):
        bin_idx = np.digitize(state, bin_edges) - 1
        if 0 <= bin_idx < bins:
            drift[bin_idx] += delta
            diffusion[bin_idx] += delta**2
            counts[bin_idx] += 1
    
    # Apply minimum points threshold
    valid = counts > min_points
    drift[valid] = drift[valid] / (counts[valid] * dt)
    diffusion[valid] = diffusion[valid] / (counts[valid] * dt) - drift[valid]**2 * dt
    
    # Set invalid bins to NaN
    drift[~valid] = np.nan
    diffusion[~valid] = np.nan
    
    return bin_centers, drift, diffusion, counts

# Estimate KM coefficients
bin_centers, drift_est, diffusion_est, counts = mantra_km_estimation(
    trajectories, dt, bins=30, min_points=20
)

# Filter out bins with insufficient points
valid_mask = ~np.isnan(drift_est)
bin_centers = bin_centers[valid_mask]
drift_est = drift_est[valid_mask]
diffusion_est = diffusion_est[valid_mask]

# =============================================
# 4. Analytical Solutions
# =============================================
drift_true = theta * (mu - bin_centers)
diffusion_true = sigma**2 * np.ones_like(bin_centers)

# =============================================
# 5. Plot Results
# =============================================
plt.figure(figsize=(12, 10))

# Drift coefficient comparison
plt.subplot(221)
plt.plot(bin_centers, drift_est, 'bo', markersize=6, label='Estimated')
plt.plot(bin_centers, drift_true, 'r-', linewidth=2, label='True')
plt.xlabel('State')
plt.ylabel('Drift Coefficient $\mu(x)$')
plt.legend()
plt.grid(True)
plt.title('Drift Coefficient Comparison')

# Diffusion coefficient comparison
plt.subplot(222)
plt.plot(bin_centers, diffusion_est, 'bo', markersize=6, label='Estimated')
plt.plot(bin_centers, diffusion_true, 'r-', linewidth=2, label='True')
plt.xlabel('State')
plt.ylabel('Diffusion Coefficient $\sigma^2(x)$')
plt.legend()
plt.grid(True)
plt.title('Diffusion Coefficient Comparison')

# Data density visualization
plt.subplot(223)
plt.bar(bin_centers, counts[valid_mask], width=(bin_centers[1]-bin_centers[0])*0.8)
plt.xlabel('State')
plt.ylabel('Point Count per Bin')
plt.title('Data Density in State Bins')
plt.grid(True)

# Sample trajectory density
plt.subplot(224)
plt.hist(trajectories[:, :-1].flatten(), bins=50, density=True, alpha=0.7)
plt.xlabel('State')
plt.ylabel('Density')
plt.title('State Distribution (All Trajectories)')
plt.grid(True)

plt.tight_layout()
plt.show()

# =============================================
# 6. Error Calculation and Output
# =============================================
# Calculate errors
drift_error = np.mean(np.abs(drift_est - drift_true))
diffusion_error = np.mean(np.abs(diffusion_est - diffusion_true))

print("="*50)
print("Ornstein-Uhlenbeck Process Analysis")
print("="*50)
print(f"Parameters: theta={theta}, mu={mu}, sigma={sigma}")
print(f"Simulation: {n_trajectories} trajectories, dt={dt}, T={T}")
print(f"Drift MAE: {drift_error:.6f}")
print(f"Diffusion MAE: {diffusion_error:.6f}")
print(f"Average points per bin: {np.mean(counts):.1f}")
print(f"Minimum points per bin: {np.min(counts)}")

# =============================================
# 7. Uncertainty Quantification (Bayesian)
# =============================================
def bayesian_uncertainty(trajectories, dt, bin_edges):
    """Compute Bayesian uncertainty for KM coefficients"""
    bins = len(bin_edges) - 1
    drift_mean = np.zeros(bins)
    drift_std = np.zeros(bins)
    diffusion_mean = np.zeros(bins)
    diffusion_std = np.zeros(bins)
    counts = np.zeros(bins)
    
    # Collect deltas per bin
    bin_deltas = [[] for _ in range(bins)]
    
    for traj in trajectories:
        for j in range(len(traj)-1):
            x = traj[j]
            delta = traj[j+1] - x
            bin_idx = np.digitize(x, bin_edges) - 1
            if 0 <= bin_idx < bins:
                bin_deltas[bin_idx].append(delta)
                counts[bin_idx] += 1
    
    # Compute moments and uncertainties
    for i in range(bins):
        if counts[i] > 10:  # Minimum for uncertainty estimation
            deltas = np.array(bin_deltas[i])
            n = len(deltas)
            
            # Drift uncertainty
            drift_mean[i] = np.mean(deltas) / dt
            drift_std[i] = np.std(deltas) / (np.sqrt(n) * dt)
            
            # Diffusion uncertainty
            diffusion_mean[i] = np.mean(deltas**2) / dt
            diffusion_std[i] = np.std(deltas**2) / (np.sqrt(n) * dt)
    
    return bin_edges, drift_mean, drift_std, diffusion_mean, diffusion_std, counts

# Compute Bayesian uncertainties
bin_edges = np.linspace(np.min(trajectories), np.max(trajectories), 31)
_, drift_bayes, drift_std, diffusion_bayes, diffusion_std, counts_bayes = bayesian_uncertainty(
    trajectories, dt, bin_edges
)

# Filter valid bins
valid_bayes = counts_bayes > 10
bin_centers_bayes = (bin_edges[1:] + bin_edges[:-1]) / 2
bin_centers_bayes = bin_centers_bayes[valid_bayes]
drift_bayes = drift_bayes[valid_bayes]
drift_std = drift_std[valid_bayes]
diffusion_bayes = diffusion_bayes[valid_bayes]
diffusion_std = diffusion_std[valid_bayes]

# Plot with uncertainty
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.errorbar(bin_centers_bayes, drift_bayes, yerr=2*drift_std, fmt='bo', 
             capsize=5, label='Estimated ±2σ')
plt.plot(bin_centers, drift_true, 'r-', linewidth=2, label='True')
plt.xlabel('State')
plt.ylabel('Drift Coefficient $\mu(x)$')
plt.legend()
plt.grid(True)
plt.title('Drift with Bayesian Uncertainty')

plt.subplot(122)
plt.errorbar(bin_centers_bayes, diffusion_bayes, yerr=2*diffusion_std, fmt='bo', 
             capsize=5, label='Estimated ±2σ')
plt.plot(bin_centers, diffusion_true, 'r-', linewidth=2, label='True')
plt.xlabel('State')
plt.ylabel('Diffusion Coefficient $\sigma^2(x)$')
plt.legend()
plt.grid(True)
plt.title('Diffusion with Bayesian Uncertainty')

plt.tight_layout()
plt.show()