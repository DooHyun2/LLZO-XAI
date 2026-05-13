"""
LLZO-Synthetic: Full analysis pipeline.

Loads synthetic_LLZO.csv, trains a Random Forest, evaluates with 5-fold CV,
computes feature importances and SHAP values, and saves all artifacts.

Run after data_synth_LLZO.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

os.makedirs("results", exist_ok=True)

# 1. Load data 
CSV = "synthetic_LLZO.csv"
if not os.path.exists(CSV):
    raise FileNotFoundError(f"{CSV} not found. Run data_synth_LLZO.py first.")

df = pd.read_csv(CSV)
X = df.drop(columns=["sigma_ion"])
y = df["sigma_ion"]

# 2. Train Random Forest 
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
).fit(X_tr, y_tr)

# 3. Evaluate 
pred = rf.predict(X_te)
test_r2 = r2_score(y_te, pred)
test_mae = mean_absolute_error(y_te, pred)

cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2", n_jobs=-1)
cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

metrics = {
    "Test R2":  round(test_r2, 4),
    "Test MAE": round(test_mae, 4),
    "5-fold CV R2 mean": round(cv_mean, 4),
    "5-fold CV R2 std":  round(cv_std, 4),
}
print(metrics)

with open("results/metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

# 4. Feature importance
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("feature importance:\n", imp)

imp.plot(kind="bar")
plt.title("Feature Importance (RF impurity-based)")
plt.tight_layout()
plt.savefig("results/feat_importance.png", dpi=160)
plt.close()

# 5. SHAP analysis
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)   # shape = [n_samples, n_features]

# Beeswarm summary
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("results/LLZO_shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

# Dependence plots for top-3 features
top_feats = ["sinter_temp", "dopant_frac", "Li_excess"]
for feat in top_feats:
    shap.dependence_plot(feat, shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f"results/shap_depend_{feat}.png", dpi=200)
    plt.close()

print("saved → results/feat_importance.png, LLZO_shap_beeswarm.png, shap_depend_*.png, metrics.txt")