# SHAP beeswarm plot for LLZO ionic conductivity (synthetic)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("synthetic_LLZO.csv")
X = df.drop(columns=["sigma_ion"])
y = df["sigma_ion"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestRegressor(
    n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
).fit(X_tr, y_tr)

pred = model.predict(X_te)
r2 = r2_score(y_te, pred)
mae = mean_absolute_error(y_te, pred)

print({"R2": round(r2,4), "MAE": round(mae,4)})

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("feature importance:\n", imp)

imp.plot(kind="bar")
plt.title("Feature Importance (RF)")
plt.tight_layout()
plt.savefig("results/feat_importance.png", dpi=160)
plt.close()
