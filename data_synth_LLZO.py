# Synthetic LLZO dataset generator
import numpy as np
import pandas as pd

rng = np.random.default_rng(43)
n = 400

# Input features
dopant_code = rng.integers(0, 3, n)

dopant_frac = rng.uniform(0.0, 0.25, n)
Li_excess   = rng.uniform(0.0, 0.20, n)
sinter_temp = rng.uniform(900, 1250, n)
grain_size  = rng.uniform(1.0, 20.0, n)

# Dopant-dependent offset
dopant_shift = np.select(
    [dopant_code == 0, dopant_code == 1, dopant_code == 2],
    [0.00, 0.10, 0.08],
    default=0.0
)

noise = rng.normal(0, 0.05, n)

# Target property (trend-based synthetic model)
sigma_ion = (
    0.5
    + dopant_shift
    + 1.6 * dopant_frac - 3.0 * (dopant_frac ** 2)
    + 0.0012 * (sinter_temp - 900) - 0.0000015 * (sinter_temp - 1100) ** 2
    + 1.2 * Li_excess - 2.5 * (Li_excess ** 2)
    + 0.03 * np.log1p(grain_size)
    + noise)

df = pd.DataFrame({
    "dopant_code": dopant_code,
    "dopant_frac": dopant_frac,
    "Li_excess": Li_excess,
    "sinter_temp": sinter_temp,
    "grain_size": grain_size,
    "sigma_ion": sigma_ion
})

df.to_csv("synthetic_LLZO.csv", index=False)
print("saved: synthetic_LLZO.csv", df.shape)

