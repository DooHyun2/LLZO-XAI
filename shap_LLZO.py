import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# directory 
os.makedirs("results", exist_ok=True)

# Load data
CSV = "synthetic_LLZO.csv"
if not os.path.exists(CSV):
    raise FileNotFoundError(f"{CSV} not found. Run data_synth.py first.")

df = pd.read_csv("synthetic_LLZO.csv")
X = df.drop(columns=["sigma_ion"])
y = df["sigma_ion"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

# Train RF model
rf = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
rf.fit(Xtr, ytr)

# Compute SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(Xte)   # shape = [n_samples, n_features]

# SHAP summary plot
shap.summary_plot(shap_values, Xte, show=False)
plt.tight_layout()
plt.savefig("results/LLZO_shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

# SHAP dependence plots (top features)
top_feats = ["sinter_temp", "dopant_frac", "Li_excess"]
for feat in top_feats:
    shap.dependence_plot(feat, shap_values, Xte, show=False)
    plt.tight_layout()
    plt.savefig(f"results/shap_depend_{feat}.png", dpi=200)
    plt.close()

print("saved SHAP plots -> results/shap_summary.png, shap_depend_*.png")
