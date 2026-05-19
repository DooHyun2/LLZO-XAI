import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


# 0. Load data & prepare oracle
CSV = "synthetic_LLZO.csv"
if not os.path.exists(CSV):
    raise FileNotFoundError(f"{CSV} not found. Run data_synth_LLZO.py first.")

df = pd.read_csv(CSV)

X_all = df.drop(columns=["sigma_ion", "dopant_code"])
y_all = df["sigma_ion"].values

feat_names = list(X_all.columns)
dim = X_all.shape[1]

# RF as oracle (surrogate for real experiments)
rf_oracle = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf_oracle.fit(X_all, y_all)

def f_oracle(x: np.ndarray) -> float:
    x = np.asarray(x).reshape(1, -1)
    return float(rf_oracle.predict(x)[0])



# 1. Search space bounds
bounds = []
for col in feat_names:
    lo = X_all[col].min()
    hi = X_all[col].max()
    bounds.append((float(lo), float(hi)))
bounds = np.array(bounds)



# 2. Initial random sampling
rng = np.random.default_rng(42)

def sample_random(n: int) -> np.ndarray:
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    u = rng.random((n, dim))
    return lows + u * (highs - lows)

n_init = 10
X_sample = sample_random(n_init)
y_sample = np.array([f_oracle(x) for x in X_sample])



# 3. GPR + Expected Improvement (EI)
kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(
    length_scale=np.ones(dim),
    length_scale_bounds=(0.1, 10.0)
) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))

def fit_gpr(X, y):
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=42,
        n_restarts_optimizer=3,
    )
    gpr.fit(X, y)
    return gpr

def expected_improvement(X_cand, gpr, y_best, xi=0.01):
    from scipy.stats import norm
    mu, sigma = gpr.predict(X_cand, return_std=True)
    sigma = sigma.reshape(-1)
    sigma_safe = np.where(sigma < 1e-12, 1e-12, sigma)
    imp = mu - y_best - xi
    Z = imp / sigma_safe
    ei = imp * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    ei[sigma < 1e-12] = 0.0
    return ei

def propose_next(gpr, n_candidates=2000):
    X_cand = sample_random(n_candidates)
    y_best = float(np.max(y_sample))
    ei = expected_improvement(X_cand, gpr, y_best)
    return X_cand[int(np.argmax(ei))]



# 4. BO loop
n_iter = 30

history = []

for t in range(n_iter):
    gpr = fit_gpr(X_sample, y_sample)

    best_idx = int(np.argmax(y_sample))
    best_x = X_sample[best_idx]
    best_y = y_sample[best_idx]
    print(f"[iter {t:02d}] best sigma_ion = {best_y:.4f}")

    history.append({
        "iter": t,
        "best_sigma_ion": best_y,
        **{f"best_{name}": float(val) for name, val in zip(feat_names, best_x)}
    })

    x_next = propose_next(gpr)
    y_next = f_oracle(x_next)

    X_sample = np.vstack([X_sample, x_next])
    y_sample = np.append(y_sample, y_next)



# 5. Final result
best_idx = int(np.argmax(y_sample))
best_x = X_sample[best_idx]
best_y = y_sample[best_idx]

print("\n=== GPR-BO Result ===")
print(f"best sigma_ion: {best_y:.4f}")
for name, val in zip(feat_names, best_x):
    print(f"  {name}: {val:.4f}")

os.makedirs("results", exist_ok=True)
hist_df = pd.DataFrame(history)
hist_df.to_csv("results/bo_gpr_history.csv", index=False)


# 6. Convergence curve
os.makedirs("figures", exist_ok=True)

iters = [h["iter"] for h in history]
bests = [h["best_sigma_ion"] for h in history]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(iters, bests, marker='o', color='steelblue', linewidth=2, markersize=5)
ax.set_xlabel('BO Iteration')
ax.set_ylabel('Best σ_ion (normalized score) found so far')
ax.set_title('BO Convergence: GPR + Expected Improvement on Synthetic LLZO')
plt.tight_layout()
plt.savefig("results/bo_convergence.png", dpi=150, bbox_inches='tight')
plt.close()

print("Saved → results/bo_gpr_history.csv, results/bo_convergence.png")
