import numpy as np
import pandas as pd

true_errors = [ 5.45284095e+02,  3.77773706e+02,  1.02867874e+02, -3.34991256e-01,
 -6.81234842e-01,  2.00998538e-01,  6.06166273e+03,  1.06762108e+04,
 -4.12470241e+03, -1.35138992e+01, -9.79488796e+00,  2.97763491e+00,
 -1.34097442e+04, -1.43047033e+04,  2.38339092e+03,  4.16730205e+00,
  5.56240418e+00, -4.96884610e-01,  6.40159241e+02,  1.81482751e+02,
 -6.33412892e+01, -1.07016608e-01, -2.05445399e-01, -2.75170862e-01,
  7.03392688e+02, -4.13517133e+01, -2.84340203e+02, -7.64987561e-01,
 -8.14262121e-01, -3.39122757e-01, -3.03439016e+02,  1.54785489e+02,
  1.83274207e+02,  9.29711308e-02, -2.39877063e-01,  2.84207787e-01,
 -4.44614599e+00, -3.75606162e+02, -3.93056724e+02, -4.73709712e-01,
 -1.07250702e-01, -1.46252238e-01,  4.55071379e-06,  3.79008320e-07]

formal_errors = [7.45952332e+02, 7.59945246e+02, 2.21542067e+02, 4.06853215e-01,
 5.31122334e-01, 1.57859802e-01, 2.54257021e+03, 4.56301672e+03,
 1.82823862e+03, 5.98720167e+00, 4.60134720e+00, 1.37569101e+00,
 1.24321799e+04, 1.56507762e+04, 3.35310419e+03, 6.86239526e+00,
 6.83820825e+00, 2.10921343e+00, 6.68943390e+02, 6.61690411e+02,
 2.72600812e+02, 7.39554062e-01, 7.18939165e-01, 3.05873151e-01,
 4.62700447e+02, 4.05083969e+02, 2.54423368e+02, 8.50547707e-01,
 9.20210750e-01, 4.14727612e-01, 2.27497297e+02, 1.81271373e+02,
 2.62507760e+02, 8.52815839e-01, 1.03869081e+00, 2.70655306e-01,
 2.91551784e+02, 4.83749665e+02, 2.71601741e+02, 7.85237991e-01,
 9.16710349e-01, 1.00980791e-01, 1.74336967e-05, 1.12399623e-06]


# --- Compute true-to-formal ratios ---
ratios = np.abs(true_errors) / formal_errors

nb_arcs = 7

# Number of arcs
nb_state_params = nb_arcs * 6

# Reshape state part: (nb_arcs, 6)
state_ratios = ratios[:nb_state_params].reshape(nb_arcs, 6)

# Component names
components = ["x0", "y0", "z0", "vx0", "vy0", "vz0"]

# Compute statistics
means = np.mean(state_ratios, axis=0)
stds = np.std(state_ratios, axis=0)

# Create dataframe
df_stats = pd.DataFrame({
    "Component": components,
    "Mean Ratio": means,
    "Std Dev": stds
})

print("\n===== TRUE-TO-FORMAL RATIO STATISTICS =====")
print(df_stats)

# Gravity parameters (last entries)
gravity_ratios = ratios[nb_state_params:]

print("\n===== GRAVITY RATIOS =====")
for i, val in enumerate(gravity_ratios):
    print(f"Gravity parameter {i}: {val}")

# Save everything
df_stats.to_csv("state_ratio_statistics.csv", index=False)

np.savetxt("all_true_to_formal_ratios.txt", ratios)

print("\nSaved:")
print(" - state_ratio_statistics.csv")
print(" - all_true_to_formal_ratios.txt")
