###############################################################
### ASSIGNMENT 2 : STATE ESTIMATION (CLEAN VERSION)
###############################################################

import sys
sys.path.append("../")

import statistics
import numpy as np
import math
import os
import pandas as pd
from matplotlib import pyplot as plt

from propagation_functions.environment import *
from propagation_functions.propagation import *
from estimation_functions.estimation import *
from estimation_functions.observations_data import *

from utility_functions.time import *
from utility_functions.tle import *
from utility_functions.data import extract_tar

from tudatpy import constants
from tudatpy.astro import element_conversion, frame_conversion
from tudatpy.estimation import estimation_analysis

# Extract data
extract_tar("./metadata.tar.xz")
extract_tar("./data.tar.xz")

metadata_folder = 'metadata/'
data_folder = 'data/'

# Files
metadata = [
'Delfi-C3_32789_202004011044.yml','Delfi-C3_32789_202004011219.yml',
'Delfi-C3_32789_202004021953.yml','Delfi-C3_32789_202004022126.yml',
'Delfi-C3_32789_202004031031.yml','Delfi-C3_32789_202004031947.yml',
'Delfi-C3_32789_202004041200.yml',
'Delfi-C3_32789_202004061012.yml','Delfi-C3_32789_202004062101.yml',
'Delfi-C3_32789_202004072055.yml','Delfi-C3_32789_202004072230.yml',
'Delfi-C3_32789_202004081135.yml'
]

data = [
'Delfi-C3_32789_202004011044.csv','Delfi-C3_32789_202004011219.csv',
'Delfi-C3_32789_202004021953.csv','Delfi-C3_32789_202004022126.csv',
'Delfi-C3_32789_202004031031.csv','Delfi-C3_32789_202004031947.csv',
'Delfi-C3_32789_202004041200.csv',
'Delfi-C3_32789_202004061012.csv','Delfi-C3_32789_202004062101.csv',
'Delfi-C3_32789_202004072055.csv','Delfi-C3_32789_202004072230.csv',
'Delfi-C3_32789_202004081135.csv'
]

# ---- YOU CHANGE THIS MANUALLY EACH RUN ----
indices_files_to_load = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

run_id = "baseline_per_pass_full_model"
arc_type_used = "per_pass"
drag_on = True
srp_on = True
earth_SH = True
bias_estimated = True

passes_used = str(indices_files_to_load)

# ------------------------------------------------------------
# INITIAL ORBIT
# ------------------------------------------------------------

initial_epoch, initial_state_teme, b_star = \
    get_tle_initial_conditions(metadata_folder + metadata[0], old_yml=False)

propagation_time = 10.0 * constants.JULIAN_DAY
final_epoch = get_start_next_day(initial_epoch) + propagation_time
mid_epoch = (initial_epoch + final_epoch) / 2.0

initial_state = propagate_sgp4(
    metadata_folder + metadata[0],
    initial_epoch,
    [mid_epoch],
    old_yml=False
)[0,1:]

recording_start_times = extract_recording_start_times_yml(
    metadata_folder,
    [metadata[i] for i in indices_files_to_load],
    old_yml=False
)

passes_start_times, passes_end_times, observation_times, observations_set = \
    load_and_format_observations(
        "Delfi", data_folder,
        [data[i] for i in indices_files_to_load],
        recording_start_times,
        old_obs_format=False
    )

# ------------------------------------------------------------
# ARC DEFINITION
# ------------------------------------------------------------

arc_start_times, arc_mid_times, arc_end_times = \
    define_arcs(arc_type_used, passes_start_times, passes_end_times)

# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------

mass = 2.2
ref_area = (4 * 0.3 * 0.1 + 2 * 0.1 * 0.1) / 4
srp_coef = 1.2
drag_coef = 1.2

bodies = define_environment(
    mass, ref_area, drag_coef, srp_coef,
    "Delfi", multi_arc_ephemeris=False
)

accelerations = dict(
    Sun={
        'point_mass_gravity': True,
        'solar_radiation_pressure': srp_on
    },
    Moon={'point_mass_gravity': True},
    Earth={
        'point_mass_gravity': not earth_SH,
        'spherical_harmonic_gravity': earth_SH,
        'drag': drag_on
    }
)


orbit = propagate_initial_state(
    initial_state, initial_epoch, final_epoch,
    bodies, accelerations, "Delfi"
)

arc_wise_initial_states = get_initial_states(
    bodies, arc_mid_times, "Delfi"
)

bodies = define_environment(
    mass, ref_area, drag_coef, srp_coef,
    "Delfi", multi_arc_ephemeris=True
)

multi_arc_propagator_settings = \
    define_multi_arc_propagation_settings(
        arc_wise_initial_states,
        arc_start_times, arc_end_times,
        bodies, accelerations, "Delfi"
    )

define_doptrack_station(bodies)

bias_definition = 'per_pass'

Doppler_models = dict(
    constant_absolute_bias={'activated':bias_estimated,'time_interval':bias_definition},
    linear_absolute_bias={'activated':bias_estimated,'time_interval':bias_definition}
)

observation_settings = define_observation_settings(
    "Delfi", Doppler_models,
    passes_start_times, arc_start_times
)

parameters_list = dict(
    initial_state={'estimate':True},
    constant_absolute_bias={'estimate':bias_estimated},
    linear_absolute_bias={'estimate':bias_estimated}
)

parameters_to_estimate = define_parameters(
    parameters_list, bodies,
    multi_arc_propagator_settings,
    "Delfi", arc_start_times, arc_mid_times,
    [(get_link_ends_id("DopTrackStation","Delfi"),
      passes_start_times)],
    Doppler_models
)

estimator = estimation_analysis.Estimator(
    bodies, parameters_to_estimate,
    observation_settings,
    multi_arc_propagator_settings
)

# ------------------------------------------------------------
# RUN ESTIMATION
# ------------------------------------------------------------

nb_iterations = 10
nb_arcs = len(arc_start_times)

pod_output = run_estimation(
    estimator,
    parameters_to_estimate,
    observations_set,
    nb_arcs,
    nb_iterations
)

residuals = pod_output.residual_history
final_residuals = residuals[:,nb_iterations-1]

# ------------------------------------------------------------
# PLOT 1 — RESIDUALS PER PASS
# ------------------------------------------------------------

residuals_per_pass = get_residuals_per_pass(
    observation_times, residuals, passes_start_times
)

n_passes = len(residuals_per_pass)
n_cols = 3
n_rows = math.ceil(n_passes / n_cols)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))

# Force axs to always be 2D
if n_rows == 1:
    axs = np.atleast_2d(axs)

for i in range(n_passes):
    row = i // n_cols
    col = i % n_cols
    axs[row, col].plot(residuals_per_pass[i])
    axs[row, col].set_title(f'Pass {i+1}')
    axs[row, col].grid()

# Remove unused subplots
for j in range(n_passes, n_rows*n_cols):
    fig.delaxes(axs.flatten()[j])

plt.tight_layout()
plt.savefig(f"plot1_residuals_per_pass_{run_id}.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# PLOT 2 — HISTOGRAM
# ------------------------------------------------------------

plt.figure()
plt.hist(final_residuals,100)
plt.xlabel("Doppler residual [m/s]")
plt.ylabel("Occurrences")
plt.grid()
plt.tight_layout()
plt.savefig(f"plot2_residual_histogram_{run_id}.png", dpi=300)
plt.close()


# ------------------------------------------------------------
# VALIDATION (ARC 0 ONLY)
# ------------------------------------------------------------

updated_parameters = parameters_to_estimate.parameter_vector
arc_index = 0

estimated_state = updated_parameters[0:6]
TLE_state = arc_wise_initial_states[0]

bodies_validation = define_environment(
    mass, ref_area, drag_coef, srp_coef,
    "Delfi", multi_arc_ephemeris=False
)

estimated_orbit = propagate_initial_state(
    estimated_state,
    arc_start_times[0],
    arc_end_times[0],
    bodies_validation,
    accelerations,
    "Delfi"
)[0]

TLE_orbit = propagate_initial_state(
    TLE_state,
    arc_start_times[0],
    arc_end_times[0],
    bodies_validation,
    accelerations,
    "Delfi"
)[0]

# RMS 3D
diff_xyz = estimated_orbit[:,1:4] - TLE_orbit[:,1:4]
rms_3d_km = np.sqrt(np.mean(np.sum(diff_xyz**2,axis=1))) / 1000

# RSW components
rsw_R, rsw_S, rsw_W = [], [], []

for i in range(len(TLE_orbit)):
    state_diff = estimated_orbit[i,1:] - TLE_orbit[i,1:]
    rot = frame_conversion.inertial_to_rsw_rotation_matrix(
        TLE_orbit[i,1:]
    )
    rsw = rot @ state_diff[0:3]
    rsw_R.append(rsw[0])
    rsw_S.append(rsw[1])
    rsw_W.append(rsw[2])

rms_R_km = np.sqrt(np.mean(np.array(rsw_R)**2))/1000
rms_S_km = np.sqrt(np.mean(np.array(rsw_S)**2))/1000
rms_W_km = np.sqrt(np.mean(np.array(rsw_W)**2))/1000

print("------------------------------------------------")
print("RMS Residual [m/s]:", np.sqrt(np.mean(final_residuals**2)))
print("RMS 3D Position [km]:", rms_3d_km)
print("RMS Radial [km]:", rms_R_km)
print("RMS Along-track [km]:", rms_S_km)
print("RMS Cross-track [km]:", rms_W_km)
print("------------------------------------------------")

# ------------------------------------------------------------
# PLOT 3 — RSW POSITION DIFFERENCE
# ------------------------------------------------------------

time_axis = TLE_orbit[:,0] - TLE_orbit[0,0]

plt.figure(figsize=(10,6))
plt.plot(time_axis,np.array(rsw_R)/1000,label='Radial')
plt.plot(time_axis,np.array(rsw_S)/1000,label='Along-track')
plt.plot(time_axis,np.array(rsw_W)/1000,label='Cross-track')
plt.xlabel("Time [s]")
plt.ylabel("Position difference [km]")
plt.title("RSW Position Difference (Estimated - TLE)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"plot3_rsw_difference_{run_id}.png", dpi=300)
plt.close()

# ============================================================
# SAVE RESULTS TO CSV (APPEND MODE)
# ============================================================

# Final residual RMS
rms_residual = np.sqrt(np.mean(final_residuals**2))

# Create dictionary
result_row = {
    "run_id": run_id,
    "arc_type": arc_type_used,
    "bias_estimated": bias_estimated,
    "drag": drag_on,
    "srp": srp_on,
    "earth_spherical_harmonics": earth_SH,
    "passes_used": passes_used,
    "rms_residual_mps": rms_residual,
    "rms_3d_km": rms_3d_km,
    "rms_R_km": rms_R_km,
    "rms_S_km": rms_S_km,
    "rms_W_km": rms_W_km
}

results_file = "estimation_summary_results.csv"

# Append or create
if os.path.exists(results_file):
    df_existing = pd.read_csv(results_file)
    df_new = pd.concat([df_existing, pd.DataFrame([result_row])],
                       ignore_index=True)
else:
    df_new = pd.DataFrame([result_row])

df_new.to_csv(results_file, index=False)

print("Results saved to:", results_file)

# ============================================================
# CORRELATION ANALYSIS + SIMPLE PLOT
# ============================================================

# Reload full results file
results_file = "estimation_summary_results.csv"
df_all = pd.read_csv(results_file)

# Only proceed if we have at least 2 runs
if len(df_all) >= 2:

    from scipy.stats import pearsonr

    metrics = ["rms_3d_km", "rms_S_km", "rms_R_km", "rms_W_km"]

    print("\nValidation Correlation Results (Pearson r):")

    for m in metrics:
        r, p = pearsonr(df_all["rms_residual_mps"], df_all[m])
        print(f"{m} vs residual: r = {r:.4f}, p = {p:.4e}")

    # Simple scatter plots
    plt.figure(figsize=(15,4))

    # 3D RMS
    plt.subplot(1,3,1)
    plt.scatter(df_all["rms_residual_mps"], df_all["rms_3d_km"])
    plt.xlabel("Residual RMS [m/s]")
    plt.ylabel("RMS 3D [km]")
    plt.grid()

    # Along-track RMS
    plt.subplot(1,3,2)
    plt.scatter(df_all["rms_residual_mps"], df_all["rms_S_km"])
    plt.xlabel("Residual RMS [m/s]")
    plt.ylabel("RMS S (Along-track) [km]")
    plt.grid()

    # Radial RMS
    plt.subplot(1,3,3)
    plt.scatter(df_all["rms_residual_mps"], df_all["rms_R_km"])
    plt.xlabel("Residual RMS [m/s]")
    plt.ylabel("RMS R (Radial) [km]")
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"plot4_validation_correlation_{run_id}.png", dpi=300)
    plt.close()

else:
    print("Not enough runs yet for correlation analysis.")