# SGA for diffusion and jump-diffusion inverse problems

This repository is a **drop-in extended version** of the original 4-file symbolic genetic algorithm (SGA) project. It keeps the same overall structure—simulation, conditional-moment estimation, symbolic feature generation, sparse regression, and three-pool evolutionary search—but adds three important capabilities:

1. **More Brownian-driven benchmark SDEs**
2. **Optional finite-activity compound-Poisson jumps**
3. **Thresholded conditional-moment estimation** for separating diffusion-scale increments from jump-scale increments

The code is organized as:

- `configure.py` — benchmark definitions, jump models, simulation, conditional-moment estimation, and runtime globals
- `setup.py` — safe algebra, finite-difference operators, sparse regression, symbolic library
- `pde.py` — symbolic trees / symbolic models and scoring against the estimated targets
- `sga.py` — the full three-pool SGA driver with command-line switches

---

## 1. Mathematical problem

The base diffusion model is

\[
dX_t = a(X_t)\,dt + b(X_t)\,dW_t,
\]

where:

- \(a(x)\) is the **drift**
- \(b(x)\) is the **diffusion amplitude**
- \(b(x)^2\) is the quantity estimated by the second Kramers–Moyal conditional moment in this code
- \(W_t\) is standard Brownian motion

The jump-extended model implemented here is a **finite-activity jump-diffusion**:

\[
dX_t = a(X_t)\,dt + b(X_t)\,dW_t + dJ_t,
\]

with compound-Poisson jump process

\[
J_t = \sum_{k=1}^{N_t} Y_k,
\qquad N_t \sim \mathrm{Poisson}(\lambda t),
\]

where:

- \(\lambda\) is the **jump intensity**
- \(Y_k\) are i.i.d. jump marks sampled from one of the implemented jump laws
- the code simulates the diffusion part by **Euler–Maruyama** and the jump part by **exact per-step Poisson counts plus mark summation**

For a time step \(\Delta t\), the simulator uses

\[
X_{n+1} = X_n + a(X_n)\,\Delta t + b(X_n)\,\Delta W_n + \sum_{j=1}^{N_n} Y_{n,j},
\qquad N_n \sim \mathrm{Poisson}(\lambda \Delta t),
\]

with \(\Delta W_n \sim \mathcal N(0, \Delta t)\).

---

## 2. What the SGA is trying to identify

The symbolic search is not done directly on trajectories. Instead, the code first estimates state-dependent targets on a 1D grid of bins and then searches for symbolic expressions that fit those targets.

### Diffusion-only mode
The target channels are:

1. **Drift target** \(a(x)\)
2. **Diffusion target** \(b(x)^2\)

### Jump-enabled mode
When `--enable-jumps --fit-jump-intensity` is active, the target channels are:

1. **Drift target** \(a(x)\)
2. **Diffusion target** \(b(x)^2\)
3. **Jump-intensity target** \(\lambda(x)\)

In the current implementation, jumps are **state-independent compound-Poisson jumps**, so the true jump intensity target is a constant:

\[
\lambda(x) \equiv \lambda.
\]

The code also computes diagnostic jump statistics:

- empirical conditional jump mean
n\[
\mathbb E[\Delta X \mid |\Delta X| > u_\Delta, X_t \approx x],
\]

- empirical conditional jump second moment

\[
\mathbb E[(\Delta X)^2 \mid |\Delta X| > u_\Delta, X_t \approx x],
\]

but those are **not yet** fitted as symbolic target channels by the SGA.

---

## 3. Conditional-moment estimators used in the code

Let

\[
\Delta X_t = X_{t+\Delta t} - X_t.
\]

The code bins observations by the current state \(X_t\approx x\).

### 3.1 Raw conditional-moment estimator
If `--use-raw-km` is selected, the code estimates

\[
\hat a(x)
\approx
\frac{\mathbb E[\Delta X_t \mid X_t \approx x]}{\Delta t},
\]

and

\[
\widehat{b^2}(x)
\approx
\frac{\mathbb E[(\Delta X_t)^2 \mid X_t \approx x]}{\Delta t}.
\]

This is the same diffusion-style Kramers–Moyal idea as in the original project.

### 3.2 Thresholded conditional-moment estimator
If `--use-threshold-km` is selected, the code treats **small increments** as diffusion-dominated and **large increments** as jump candidates.

For each bin, it builds a threshold of the form

\[
u_\Delta(x) = c\,\hat\sigma(x)\,\Delta t^{\beta},
\]

where the defaults are:

- \(c = 4.0\) (`threshold_scale`)
- \(\beta = 0.49\) (`threshold_beta`)

and \(\hat\sigma(x)\) is a robust local scale estimate obtained from the median absolute increment in that bin.

Then the code estimates

\[
\hat a(x)
\approx
\frac{\mathbb E\left[\Delta X_t\,\mathbf 1\{ |\Delta X_t| \le u_\Delta(x) \} \mid X_t \approx x\right]}{\Delta t},
\]

\[
\widehat{b^2}(x)
\approx
\frac{\mathbb E\left[(\Delta X_t)^2\,\mathbf 1\{ |\Delta X_t| \le u_\Delta(x) \} \mid X_t \approx x\right]}{\Delta t},
\]

and jump intensity by the exceedance frequency

\[
\hat\lambda(x)
\approx
\frac{\mathbb P\left(|\Delta X_t| > u_\Delta(x) \mid X_t \approx x\right)}{\Delta t}.
\]

The large increments also produce empirical jump-mark summaries:

\[
\widehat{m_1}(x)
=
\mathbb E\left[\Delta X_t \mid |\Delta X_t| > u_\Delta(x), X_t \approx x\right],
\]

\[
\widehat{m_2}(x)
=
\mathbb E\left[(\Delta X_t)^2 \mid |\Delta X_t| > u_\Delta(x), X_t \approx x\right].
\]

---

## 4. Benchmark SDEs implemented in `configure.py`

Below, each benchmark is written in **two forms**:

- **without jumps**: the diffusion-only model
- **with jumps**: the same drift/diffusion structure plus the optional compound-Poisson jump term

For all jump-enabled cases, the code uses the same generic extension

\[
dX_t = a(X_t)\,dt + b(X_t)\,dW_t + dJ_t.
\]

### 4.1 Brownian motion
Implemented parameters:

\[
\mu = 0.5,
\qquad \sigma = 1.0.
\]

Without jumps:

\[
dX_t = \mu\,dt + \sigma\,dW_t.
\]

With jumps:

\[
dX_t = \mu\,dt + \sigma\,dW_t + dJ_t.
\]

So the symbolic targets are

\[
a(x) = \mu,
\qquad b(x) = \sigma,
\qquad b(x)^2 = \sigma^2.
\]

### 4.2 Ornstein–Uhlenbeck (OU)
Implemented parameters:

\[
\kappa = 1.0,
\qquad \mu = 1.0,
\qquad \sigma = 1.0.
\]

Without jumps:

\[
dX_t = -\kappa(X_t - \mu)\,dt + \sigma\,dW_t.
\]

With jumps:

\[
dX_t = -\kappa(X_t - \mu)\,dt + \sigma\,dW_t + dJ_t.
\]

Targets:

\[
a(x) = -\kappa(x-\mu),
\qquad b(x) = \sigma,
\qquad b(x)^2 = \sigma^2.
\]

### 4.3 Double-well diffusion
The implemented drift is the negative gradient of the quartic potential

\[
V(x) = \frac{(x^2-1)^2}{4},
\qquad -V'(x) = x - x^3.
\]

Implemented parameter:

\[
\sigma = 3.0.
\]

Without jumps:

\[
dX_t = (X_t - X_t^3)\,dt + \sigma\,dW_t.
\]

With jumps:

\[
dX_t = (X_t - X_t^3)\,dt + \sigma\,dW_t + dJ_t.
\]

Targets:

\[
a(x) = x - x^3,
\qquad b(x) = \sigma,
\qquad b(x)^2 = \sigma^2.
\]

### 4.4 Geometric Brownian motion (GBM)
Implemented parameters:

\[
\mu = 0.6,
\qquad \sigma = 0.35.
\]

Without jumps:

\[
dX_t = \mu X_t\,dt + \sigma X_t\,dW_t.
\]

With jumps:

\[
dX_t = \mu X_t\,dt + \sigma X_t\,dW_t + dJ_t.
\]

Targets:

\[
a(x) = \mu x,
\qquad b(x) = \sigma x,
\qquad b(x)^2 = \sigma^2 x^2.
\]

The code clips states to remain positive after each step.

### 4.5 CIR (Cox–Ingersoll–Ross)
Implemented parameters:

\[
\kappa = 1.5,
\qquad \theta = 1.0,
\qquad \sigma = 1.25.
\]

Without jumps:

\[
dX_t = \kappa(\theta - X_t)\,dt + \sigma\sqrt{X_t}\,dW_t.
\]

With jumps:

\[
dX_t = \kappa(\theta - X_t)\,dt + \sigma\sqrt{X_t}\,dW_t + dJ_t.
\]

Targets:

\[
a(x) = \kappa(\theta - x),
\qquad b(x) = \sigma\sqrt{x},
\qquad b(x)^2 = \sigma^2 x.
\]

The simulator clips the state to stay positive after each update.

### 4.6 CEV (constant elasticity of variance)
Implemented parameters:

\[
\mu = 0.4,
\qquad \sigma = 0.45,
\qquad \gamma = 1.5.
\]

Without jumps:

\[
dX_t = \mu X_t\,dt + \sigma X_t^{\gamma}\,dW_t.
\]

With jumps:

\[
dX_t = \mu X_t\,dt + \sigma X_t^{\gamma}\,dW_t + dJ_t.
\]

Targets:

\[
a(x) = \mu x,
\qquad b(x) = \sigma x^{\gamma},
\qquad b(x)^2 = \sigma^2 x^{2\gamma}.
\]

The simulator clips the state to stay positive after each update.

### 4.7 Stochastic logistic diffusion
Implemented parameters:

\[
r = 2.0,
\qquad K = 1.0,
\qquad \sigma = 0.35.
\]

Without jumps:

\[
dX_t = rX_t\left(1 - \frac{X_t}{K}\right)dt + \sigma X_t\,dW_t.
\]

With jumps:

\[
dX_t = rX_t\left(1 - \frac{X_t}{K}\right)dt + \sigma X_t\,dW_t + dJ_t.
\]

Targets:

\[
a(x) = rx\left(1 - \frac{x}{K}\right),
\qquad b(x) = \sigma x,
\qquad b(x)^2 = \sigma^2 x^2.
\]

The simulator clips the state to stay positive after each update.

### 4.8 Jacobi / Wright–Fisher diffusion
Implemented parameters:

\[
\kappa = 3.0,
\qquad \theta = 0.55,
\qquad \sigma = 0.55.
\]

Without jumps:

\[
dX_t = \kappa(\theta - X_t)\,dt + \sigma\sqrt{X_t(1-X_t)}\,dW_t,
\qquad X_t \in [0,1].
\]

With jumps:

\[
dX_t = \kappa(\theta - X_t)\,dt + \sigma\sqrt{X_t(1-X_t)}\,dW_t + dJ_t.
\]

Targets:

\[
a(x) = \kappa(\theta - x),
\qquad b(x) = \sigma\sqrt{x(1-x)},
\qquad b(x)^2 = \sigma^2 x(1-x).
\]

The simulator clips the state into \([0,1]\) after each update.

---

## 5. Jump models implemented in `configure.py`

All jump-enabled benchmarks use the same compound-Poisson clock but with different mark laws.

### 5.1 Gaussian jumps
Implemented parameters:

\[
\lambda = 0.75,
\qquad Y \sim \mathcal N(0, 0.8^2).
\]

So

\[
J_t = \sum_{k=1}^{N_t} Y_k,
\qquad N_t \sim \mathrm{Poisson}(0.75\,t).
\]

### 5.2 Double-exponential jumps
Implemented parameters:

\[
\lambda = 0.5,
\qquad \mathbb P(Y>0)=0.35,
\qquad Y\mid Y>0 \sim \mathrm{Exp}(\eta_+) \text{ with } \eta_+=5,
\]

\[
Y\mid Y<0 = -Z,
\qquad Z \sim \mathrm{Exp}(\eta_-) \text{ with } \eta_-=7.
\]

Equivalently,

\[
Y =
\begin{cases}
E_+, & \text{with probability } 0.35, \\
-E_-, & \text{with probability } 0.65,
\end{cases}
\]

with

\[
E_+ \sim \mathrm{Exp}(5),
\qquad E_- \sim \mathrm{Exp}(7).
\]

### 5.3 Fixed signed jumps
Implemented parameters:

\[
\lambda = 0.65,
\qquad Y = \pm 0.9,
\qquad \mathbb P(Y=+0.9)=0.5,
\qquad \mathbb P(Y=-0.9)=0.5.
\]

This is the simplest finite-activity non-Gaussian jump benchmark in the code.

---

## 6. What is actually optimized by the SGA

The symbolic genetic algorithm builds trees from the operator library in `setup.py`, evaluates those trees on the trimmed KM bin centers, and then fits sparse linear combinations of the resulting symbolic features.

### 6.1 Symbolic operator library
The current search library includes:

- binary operators: `+`, `-`, `*`, `/`
- derivative operators: `d`, `d^2`
- unary powers: `^2`, `^3`, `^0.5`
- variables/constants: `x`, `1`

So the SGA is learning expressions of the form

\[
\hat a(x) \approx \sum_j c_j \phi_j(x),
\qquad
\widehat{b^2}(x) \approx \sum_j d_j \psi_j(x),
\]

and, if jump fitting is enabled,

\[
\hat\lambda(x) \approx \sum_j e_j \chi_j(x).
\]

### 6.2 Multi-channel fitting
The total symbolic model contains an equal number of terms per target channel.

- diffusion-only mode: total terms are split into **drift** and **diffusion** blocks
- jump-enabled mode with jump fitting: total terms are split into **drift**, **diffusion**, and **jump-intensity** blocks

Each channel is fitted by sparse linear regression, and the resulting AIC-like scores are summed.

---

## 7. Three-pool search strategy

The code keeps the original **three-pool SGA structure**.

- **Pool 1**: exploitation-oriented, with crossover + mutation + optional replace
- **Pool 2**: stronger randomization, full mutate + replace on copies
- **Pool 3**: aggressive random exploration, mutate + replace on in-pool individuals

At each generation:

1. Pool 1 performs crossover and mutation
2. Pool 2 performs aggressive copy-based randomization
3. Pool 3 performs aggressive in-place exploration
4. The best models from Pools 2 and 3 are injected back into Pool 1
5. The globally best model is reported

This preserves the spirit of the original implementation while supporting either 2-target or 3-target discovery.

---

## 8. Runtime switches and usage

### 8.1 Main CLI flags
Use `sga.py` as the main entry point.

```bash
python sga.py --benchmark ou --disable-jumps --use-threshold-km
```

Important flags:

- `--benchmark {brownian_motion,ou,double_well,gbm,cir,cev,logistic,jacobi}`
- `--enable-jumps` / `--disable-jumps`
- `--jump-model {gaussian,double_exponential,fixed}`
- `--fit-jump-intensity` / `--no-fit-jump-intensity`
- `--use-threshold-km` / `--use-raw-km`
- `--dt`, `--T`, `--n-traj`, `--bins`
- `--threshold-scale`, `--threshold-beta`, `--min-bin-count`
- `--num`, `--depth`, `--width`, `--p-var`, `--p-mute`, `--p-cro`, `--p-rep`, `--gen`
- `--error-metric {mse,kl,wasserstein}`

### 8.2 Example runs

Pure diffusion OU:

```bash
python sga.py --benchmark ou --disable-jumps --use-threshold-km
```

GBM without jumps:

```bash
python sga.py --benchmark gbm --disable-jumps --use-threshold-km
```

OU with Gaussian compound-Poisson jumps and symbolic jump-intensity fitting:

```bash
python sga.py --benchmark ou --enable-jumps --jump-model gaussian --fit-jump-intensity
```

Double-well with double-exponential jumps:

```bash
python sga.py --benchmark double_well --enable-jumps --jump-model double_exponential --fit-jump-intensity
```

Jacobi diffusion without jumps:

```bash
python sga.py --benchmark jacobi --disable-jumps
```

Small quick smoke test:

```bash
python sga.py --benchmark ou --n-traj 1000 --num 5 --gen 2 --depth 3 --width 4
```

---

## 9. Important implementation details

### 9.1 What “diffusion target” means here
The second target is **not** the diffusion amplitude \(b(x)\) itself. It is the conditional second-moment coefficient used by the code:

\[
\widehat{b^2}(x) \approx \frac{\mathbb E[(\Delta X)^2 \mid X_t \approx x]}{\Delta t}.
\]

So if a benchmark has

\[
b(x) = \sigma \sqrt{x},
\]

then the true target is

\[
b(x)^2 = \sigma^2 x.
\]

### 9.2 Positivity and bounded-state handling
Some benchmarks require state constraints, so the code applies a `post_step` map after every step:

- GBM: clip to positive values
- CIR: clip to positive values
- CEV: clip to positive values
- logistic: clip to positive values
- Jacobi: clip to \([0,1]\)

### 9.3 Bin trimming and validity masking
The code:

1. estimates conditional moments on `bins` uniform state bins
2. trims a fraction `trim_ratio` from each edge
3. removes bins with fewer than `min_bin_count` samples
4. falls back to the trimmed core if count-pruning becomes too aggressive

This keeps the symbolic regression focused on the part of the state domain with enough data support.

---

## 10. Suggested reading of outputs

At runtime, `sga.py` prints:

- a runtime summary from `configure.py`
- the active target channels
- the best AIC in each generation
- the best raw symbolic model
- a concise symbolic model with fitted coefficients

In diffusion-only mode, the best concise model reports something like:

- `Drift: ...`
- `Diffusion: ...`

In jump-enabled mode with jump fitting, it reports:

- `Drift: ...`
- `Diffusion: ...`
- `Jump intensity: ...`

---

## 11. Current limitations

This code is already a substantial extension, but a few things are still intentionally simplified:

1. **Jump intensity is state-independent in the simulator**
   - the target can still be fitted as a function of \(x\), but the ground truth is currently constant

2. **Jump mark law is not yet symbolically identified**
   - only jump intensity is fitted by the SGA
   - `jump_mean_target` and `jump_second_target` are exported for future work

3. **The simulator uses Euler–Maruyama for the diffusion part**
   - this is fine for the current benchmark set, but can be replaced later by positivity-preserving or higher-order schemes for selected models

4. **The code is still 1D**
   - all benchmark cases and symbolic operators currently assume a scalar state variable

---

## 12. Minimal workflow summary

If you want the shortest conceptual summary of the pipeline, it is:

1. Choose a benchmark SDE or jump-diffusion
2. Simulate many trajectories
3. Estimate conditional moment targets on state bins
4. Build symbolic candidate features on the bin centers
5. Fit sparse linear models to drift / diffusion / optional jump intensity
6. Evolve the symbolic model with the three-pool SGA

That is exactly what this extended codebase now supports.
