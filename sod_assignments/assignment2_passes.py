###############################################################
### PASS QUALITY INSPECTION TOOL
###############################################################

import sys
sys.path.append("../")

import numpy as np
import math
import statistics
import pandas as pd
import matplotlib.pyplot as plt

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


###############################################################
# DATA LOADING
###############################################################

extract_tar("./metadata.tar.xz")
extract_tar("./data.tar.xz")

metadata_folder = 'metadata/'
data_folder = 'data/'

metadata = [
'Delfi-C3_32789_202004011044.yml',
'Delfi-C3_32789_202004011219.yml',
'Delfi-C3_32789_202004021953.yml',
'Delfi-C3_32789_202004022126.yml',
'Delfi-C3_32789_202004031031.yml',
'Delfi-C3_32789_202004031947.yml',
'Delfi-C3_32789_202004041200.yml',
'Delfi-C3_32789_202004061012.yml',
'Delfi-C3_32789_202004062101.yml',
'Delfi-C3_32789_202004072055.yml',
'Delfi-C3_32789_202004072230.yml',
'Delfi-C3_32789_202004081135.yml'
]

data = [f.replace(".yml", ".csv") for f in metadata]

indices_files_to_load = list(range(len(metadata)))  # use ALL passes


###############################################################
# INITIAL ORBIT FROM TLE
###############################################################

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
)[0, 1:]


###############################################################
# LOAD OBSERVATIONS
###############################################################

recording_start_times = extract_recording_start_times_yml(
    metadata_folder,
    [metadata[i] for i in indices_files_to_load],
    old_yml=False
)

passes_start_times, passes_end_times, observation_times, observations_set = \
    load_and_format_observations(
        "Delfi",
        data_folder,
        [data[i] for i in indices_files_to_load],
        recording_start_times,
        old_obs_format=False
    )


###############################################################
# ARC DEFINITION
###############################################################

arc_start_times, arc_mid_times, arc_end_times = define_arcs(
    'per_pass',
    passes_start_times,
    passes_end_times
)


###############################################################
# ENVIRONMENT
###############################################################

mass = 2.2
ref_area = (4 * 0.3 * 0.1 + 2 * 0.1 * 0.1) / 4
srp_coef = 1.2
drag_coef = 1.2

bodies = define_environment(
    mass, ref_area, drag_coef, srp_coef,
    "Delfi", multi_arc_ephemeris=False
)

accelerations = dict(
    Sun={'point_mass_gravity': True, 'solar_radiation_pressure': True},
    Moon={'point_mass_gravity': True},
    Earth={'point_mass_gravity': False,
           'spherical_harmonic_gravity': True,
           'drag': True}
)

propagate_initial_state(
    initial_state,
    initial_epoch,
    final_epoch,
    bodies,
    accelerations,
    "Delfi"
)

arc_wise_initial_states = get_initial_states(
    bodies,
    arc_mid_times,
    "Delfi"
)

bodies = define_environment(
    mass, ref_area, drag_coef, srp_coef,
    "Delfi", multi_arc_ephemeris=True
)

multi_arc_propagator_settings = define_multi_arc_propagation_settings(
    arc_wise_initial_states,
    arc_start_times,
    arc_end_times,
    bodies,
    accelerations,
    "Delfi"
)

define_doptrack_station(bodies)


###############################################################
# ESTIMATION SETUP
###############################################################

Doppler_models = dict(
    constant_absolute_bias={'activated': True, 'time_interval': 'per_pass'},
    linear_absolute_bias={'activated': True, 'time_interval': 'per_pass'}
)

observation_settings = define_observation_settings(
    "Delfi",
    Doppler_models,
    passes_start_times,
    arc_start_times
)

parameters_list = dict(
    initial_state={'estimate': True},
    constant_absolute_bias={'estimate': True},
    linear_absolute_bias={'estimate': True}
)

parameters_to_estimate = define_parameters(
    parameters_list,
    bodies,
    multi_arc_propagator_settings,
    "Delfi",
    arc_start_times,
    arc_mid_times,
    [(get_link_ends_id("DopTrackStation", "Delfi"), passes_start_times)],
    Doppler_models
)

estimator = estimation_analysis.Estimator(
    bodies,
    parameters_to_estimate,
    observation_settings,
    multi_arc_propagator_settings
)


###############################################################
# RUN ESTIMATION
###############################################################

pod_output = run_estimation(
    estimator,
    parameters_to_estimate,
    observations_set,
    len(arc_start_times),
    10
)

residuals = pod_output.residual_history
final_residuals = residuals[:, -1]

residuals_per_pass = get_residuals_per_pass(
    observation_times,
    residuals,
    passes_start_times
)


###############################################################
# PASS STATISTICS
###############################################################

pass_stats = []

for i, pass_res in enumerate(residuals_per_pass):

    pass_res = np.array(pass_res)  # ensure numpy array

    rms = np.sqrt(np.mean(pass_res**2))
    mean = np.mean(pass_res)
    std = np.std(pass_res)
    max_abs = np.max(np.abs(pass_res))

    pass_stats.append({
        "pass_index": i,
        "RMS": rms,
        "Mean": mean,
        "Std": std,
        "MaxAbs": max_abs
    })

df_pass_stats = pd.DataFrame(pass_stats)

print("\nPASS STATISTICS:\n")
print(df_pass_stats.sort_values("RMS", ascending=False))


###############################################################
# SUGGESTIONS
###############################################################

worst_passes = df_pass_stats.sort_values("RMS", ascending=False)["pass_index"].tolist()
best_pass = df_pass_stats.sort_values("RMS", ascending=True)["pass_index"].iloc[0]

print("\nWorst passes (high RMS first):")
print(worst_passes)

print("\nBest pass (lowest RMS):")
print(best_pass)


###############################################################
# RESIDUAL PLOTS
###############################################################

number_of_passes = len(residuals_per_pass)

rows = int(np.ceil(number_of_passes / 3))

fig, axs = plt.subplots(rows, 3, figsize=(12, 8))

for i in range(number_of_passes):

    pass_res = np.array(residuals_per_pass[i])

    ax = axs[i//3, i%3] if rows > 1 else axs[i%3]

    ax.plot(pass_res)
    ax.set_title(f'Pass {i}')
    ax.set_xlabel('Observation index')
    ax.set_ylabel('Residual [m/s]')
    ax.grid()

# Hide empty subplots
for j in range(number_of_passes, rows*3):
    axs[j//3, j%3].axis('off')

plt.tight_layout()
plt.savefig('fig2.png', dpi=300)