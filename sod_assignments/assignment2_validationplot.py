###############################################################
# VALIDATION ANALYSIS SCRIPT
# Reads estimation_summary_results.csv
# Computes correlation and generates validation plots
###############################################################

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ------------------------------------------------------------
# LOAD RESULTS
# ------------------------------------------------------------

results_file = "estimation_summary_results.csv"
df = pd.read_csv(results_file)

# Remove failed runs (NaN values)
df = df.dropna(subset=[
    "rms_residual_mps",
    "rms_3d_km",
    "rms_S_km",
    "rms_R_km",
    "rms_W_km"
])

print("\nLoaded runs:")
print(df["run_id"].values)

# ------------------------------------------------------------
# CORRELATION ANALYSIS
# ------------------------------------------------------------

metrics = ["rms_3d_km", "rms_S_km", "rms_R_km", "rms_W_km"]

print("\nValidation Correlation Results (Pearson r):")

for m in metrics:
    r, p = pearsonr(df["rms_residual_mps"], df[m])
    print(f"{m} vs residual: r = {r:.4f}, p = {p:.4e}")

# Full correlation matrix
corr_matrix = df[[
    "rms_residual_mps",
    "rms_3d_km",
    "rms_S_km",
    "rms_R_km",
    "rms_W_km"
]].corr()

print("\nFull Correlation Matrix:")
print(corr_matrix)

## ------------------------------------------------------------
# SAVE INDIVIDUAL VALIDATION PLOTS
# ------------------------------------------------------------

os.makedirs("figures", exist_ok=True)

metrics = {
    "rms_3d_km": "RMS 3D [km]",
    "rms_S_km": "RMS S (Along-track) [km]",
    "rms_R_km": "RMS R (Radial) [km]",
    "rms_W_km": "RMS W (Cross-track) [km]"
}

for key, label in metrics.items():

    plt.figure(figsize=(6,5))
    plt.scatter(df["rms_residual_mps"], df[key])

    # regression line
    z = np.polyfit(df["rms_residual_mps"], df[key], 1)
    p = np.poly1d(z)
    plt.plot(df["rms_residual_mps"], p(df["rms_residual_mps"]))

    plt.xlabel("Residual RMS [m/s]")
    plt.ylabel(label)
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"figures/validation_{key}.png", dpi=300)
    plt.close()

print("Saved 4 validation plots in figures/ folder.")