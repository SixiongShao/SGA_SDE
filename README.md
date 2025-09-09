# SGA\_SDE

Symbolic Genetic Algorithms for learning **open-form stochastic differential equations (SDEs)** from trajectory data. This repo focuses on two identification routes:

* **Fokker–Planck (FPE) transformation** — learn the density-evolution PDE for $p(x,t)$ and back out drift $a(x)$ and diffusion $\beta(x)=b^2(x)$.
* **Kramers–Moyal (KM) expansion** — estimate drift and diffusion directly from conditional moments of trajectory increments, then learn symbolic forms for each.



## Method 1 — Fokker–Planck transformation

### Derivation (one dimension, brief)

Starting from Itô’s lemma and integrating by parts with a test function, the **forward Kolmogorov / Fokker–Planck equation** for the density $p(x,t)$ is

$$
\partial_t p = -\partial_x\big(a(x)p\big) + \tfrac{1}{2}\partial_x^2\big(\beta(x)p\big).
$$


### Pipeline

1. **Trajectories → density.** Simulate or collect many paths $X^{(i)}_{t_n}$. On a grid $(x_m,t_n)$, estimate $p(x,t)$ (e.g., KDE).
2. **2D smoothing.** Denoise $p$ with a separable Gaussian in $x$ and $t$ to stabilize derivatives; trim margins to avoid convolution edge effects.
3. **Differentiate.** Compute $\partial_t p$, $\partial_x p$, $\partial_x^2 p$ on the trimmed grid.
4. **Symbolic fit (STRidge + AIC).** Build features from a symbolic library for $a(x)$ and $\beta(x)$; fit the discretized FPE in a single linear system and sparsify with STRidge; score with MSE/KL/Wasserstein + AIC for parsimony.&#x20;

### Practical tips

* **Bandwidths matter.** The smoothing scales $(h_x,h_t)$ control a bias–variance trade-off for density derivatives; expect system-specific optima.
* **Conditioning.** On many time slices $p, p', p''$ are highly collinear (near-Gaussian shapes), making the regression ill-conditioned—especially when $\beta' \neq 0$.


---

## Method 2 — Kramers–Moyal (KM) expansion

### Derivation (first two coefficients)

For small $\Delta t$, the increment $\Delta X = X_{t+\Delta t}-X_t = a(X_t)\Delta t + b(X_t)\sqrt{\Delta t}\,\xi$ with $\xi\sim\mathcal N(0,1)$ implies

$$
D^{(1)}(x)=a(x), \qquad D^{(2)}(x)=\tfrac{1}{2}\beta(x).
$$

Hence the **conditional moments** give pointwise estimators:

$$
\hat a(x) \approx \tfrac{1}{\Delta t}\,\mathbb E[\Delta X\mid X_t\approx x],\qquad
\hat \beta(x) \approx \tfrac{1}{\Delta t}\,\mathbb E[(\Delta X)^2\mid X_t\approx x].
$$

These become unbiased as $\Delta t\to 0$ and concentrate with more samples near $x$.&#x20;

### Pipeline

1. **Trajectories.** Collect $N$ paths on a grid with step $\Delta t$.
2. **State binning.** Partition the state space; for each bin $B_j$ (center $x_j$), gather increments with $X_t\in B_j$; trim under-populated edge bins.
3. **KM targets.** Compute

   $$
   \hat a(x_j)=\frac{1}{\Delta t}\frac{1}{N_x(j)}\sum_{i:\,X_{t_i}\in B_j}\Delta X_i,\quad
   \hat \beta(x_j)=\frac{1}{\Delta t}\frac{1}{N_x(j)}\sum_{i:\,X_{t_i}\in B_j}(\Delta X_i)^2.
   $$
4. **Decoupled symbolic fits.** Fit $x\mapsto a(x)$ and $x\mapsto \beta(x)$ **separately** via STRidge + AIC using symbolic bases; apply weights and pruning.&#x20;


### Improvements

* **Decoupling:** Mean and variance are learned independently, removing drift–diffusion confounding.
* **Better conditioning:** No density derivatives; regressions use simple univariate bases $\Phi_k(x_j)$, avoiding collinearity of $p,p',p''$.
* **Sample efficiency:** Works with **thousands** (not $10^5$–$10^6$) of trajectories; no smoothing hyperparameters.
* **Statistics:** For bin count $N_x$,

  $$
  \mathrm{Var}[\hat a(x)] \approx \frac{\beta(x)}{\Delta t\,N_x},\qquad
  \mathrm{Var}[\hat \beta(x)] \approx \frac{2\beta^2(x)}{N_x},
  $$

  so diffusion precision scales with $1/N_x$ and is leading-order independent of $\Delta t$.&#x20;


---

## When to choose which

* Choose **KM** for most problems (robust, data-efficient, minimal tuning).
* Consider **FPE** when diffusion is **constant** and you have very large ensembles; tune $(h_x,h_t)$ carefully and monitor conditioning.



## Limitations

* **KM:** Needs enough visits per bin; drift estimation variance grows as $\Delta t\to 0$ unless $N_x$ scales; extremely low noise may obscure diffusion.
* **FPE:** Sensitive to smoothing; can be ill-conditioned for state-dependent diffusion because $p,p',p''$ are nearly collinear and couple with $\beta',\beta''$; boundary layers (e.g., CIR near $x=0$) are problematic.&#x20;

---

## References

Please cite the SGA\_SDE paper if you use this work. Full derivations, figures, and tables (including the OU/double-well/CIR studies and the statistical analysis of KM estimators) are in the PDF.&#x20;

---

*Maintainers:* E S (@shaosixiong) and collaborators.
