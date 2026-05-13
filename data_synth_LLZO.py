"""
Synthetic LLZO dataset generator.

Generates N=400 samples with 5 input features (dopant type, dopant fraction,
Li excess, sintering temperature, grain size) and a target ionic conductivity
score for SHAP pipeline validation.

Coefficient magnitudes are physically motivated but arbitrary:
- dopant_frac quadratic peaks near optimal substitution (~0.27)
- sintering quadratic peaks near densification optimum (~1160°C)
- Li_excess quadratic captures volatilization compensation saturation (~0.24)
- grain_size log term reflects weak monotonic boundary contribution
- dopant_code adds categorical type-specific offset

The target 'sigma_ion' is a normalized score, not actual S/cm.
Linear (not log) target is used due to the bounded positive range;
LiSSE-SHAP applies log10 to real experimental conductivity data.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(43)
n = 400

# Input features
dopant_code = rng.integers(0, 5, n)     # {0,1,2,3,4} → {Al, Ga, Nb, Ta, Y}
dopant_frac = rng.uniform(0.0, 0.25, n) # B-site dopant fraction
Li_excess   = rng.uniform(0.0, 0.20, n) # nominal Li excess (mol)
sinter_temp = rng.uniform(900, 1250, n) # °C
grain_size  = rng.uniform(1.0, 20.0, n) # μm

# Dopant-dependent offset: {Al, Ga, Nb, Ta, Y}
# Ta highest (most effective LLZO dopant in literature), Al as baseline
dopant_offsets = np.array([0.00, 0.05, 0.10, 0.12, 0.07])
dopant_shift = dopant_offsets[dopant_code]

noise = rng.normal(0, 0.05, n)

# Target property (trend-based synthetic model)
sigma_ion = (
    0.5
    + dopant_shift
    # dopant_frac quadratic peaks near optimal substitution (~0.25)
    + 1.6 * dopant_frac - 3.0 * (dopant_frac ** 2)
    # sintering quadratic peaks near densification optimum (~1100-1160°C)
    + 0.0012 * (sinter_temp - 900) - 0.00001 * (sinter_temp - 1100) ** 2
    # Li_excess quadratic captures volatilization compensation saturation
    + 1.2 * Li_excess - 2.5 * (Li_excess ** 2)
    # grain_size log term reflects weak monotonic boundary contribution
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

