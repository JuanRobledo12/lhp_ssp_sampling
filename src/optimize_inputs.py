import copy
import datetime as dt
import importlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pathlib
import sys
import time
import pickle
from typing import Union
import warnings
from datetime import datetime
from pyswarm import pso  # Install with: pip install pyswarm
warnings.filterwarnings("ignore")
from info_grupos import empirical_vars_to_avoid, frac_vars_special_cases_list
from utils import HelperFunctions, SSPModelForCalibartion
from sisepuede.manager.sisepuede_examples import SISEPUEDEExamples



# Record the start time
start_time = time.time()

# Initialize helper functions
helper_functions = HelperFunctions()

# # Load parameters from YAML
# experiment_id = int(sys.argv[1])
# param_file_name = sys.argv[2]

target_country = 'croatia'

print(f"Optimizing scaling vector")

# Paths
FILE_PATH = os.getcwd()
build_path = lambda PATH: os.path.abspath(os.path.join(*PATH))
DATA_PATH = build_path([FILE_PATH, "..", "data"])
OUTPUT_PATH = build_path([FILE_PATH, "..", "output"])
REAL_DATA_FILE_PATH = build_path([DATA_PATH, "real_data.csv"])
SSP_OUTPUT_PATH = build_path([OUTPUT_PATH, "ssp"])

# Load input dataset
examples = SISEPUEDEExamples()
cr = examples("input_data_frame")

df_input = pd.read_csv(REAL_DATA_FILE_PATH)
df_input = df_input.rename(columns={'period': 'time_period'})
df_input = helper_functions.add_missing_cols(cr, df_input.copy())
df_input = df_input.drop(columns='iso_code3')

# Columns to scale
columns_all_999 = df_input.columns[(df_input == -999).any()].tolist()
pij_cols = [col for col in df_input.columns if col.startswith('pij')]
cols_to_avoid = pij_cols + frac_vars_special_cases_list + columns_all_999 + empirical_vars_to_avoid
cols_to_stress = helper_functions.get_indicators_col_names(df_input, cols_with_issue=cols_to_avoid)


# Define bounds for scaling
n_vars = len(cols_to_stress)
lb = np.zeros(n_vars)  # Lower bound (0)
ub = np.ones(n_vars) * 2  # Upper bound (2)

# Simulation model
def simulation_model(df_scaled: pd.DataFrame) -> np.ndarray:
    """
    Function that simulates outputs based on the scaled inputs.
    Replace this with the actual simulation function.
    """
    ssp_model = SSPModelForCalibartion(SSP_OUTPUT_PATH=SSP_OUTPUT_PATH , target_country=target_country)
    emissions_df = ssp_model.run_ssp_simulation(df_scaled)
    emissions = emissions_df.iloc[0]

    # Convert the series into a DataFrame and extract the subsector suffix
    emissions = emissions.rename_axis('index').reset_index()
    emissions['Subsector'] = emissions['index'].str.replace('emission_co2e_subsector_total_', '', regex=False)
    emissions.rename(columns={0: 'sim_value'}, inplace=True)

    return emissions

# Objective function
def objective_function(scaling_vector: np.ndarray) -> float:
    """
    Evaluates the error between simulated and ground truth outputs.
    """
    print("Executing objective function...")

    stressed_df = df_input.copy()
    stressed_df[cols_to_stress] = df_input[cols_to_stress] * scaling_vector

    # TODO: Abstract this part in a method

    # Normalizing frac_ var groups using softmax
    df_frac_vars = pd.read_excel('frac_vars.xlsx', sheet_name='frac_vars_no_special_cases')
    need_norm_prefix = df_frac_vars.frac_var_name_prefix.unique()

    random_scale = 1e-2  # Scale for random noise
    epsilon = 1e-6

    for subgroup in need_norm_prefix:
        subgroup_cols = [i for i in stressed_df.columns if subgroup in i]
        
        # Skip normalization for columns in cols_to_avoid
        if any(col in cols_to_avoid for col in subgroup_cols):
            continue

        # Check if the sum of the group is zero or too small
        group_sum = stressed_df[subgroup_cols].sum(axis=1)
        is_zero_sum = group_sum < epsilon

        # Add random variability for zero-sum groups
        if is_zero_sum.any():
            noise = np.random.uniform(0, random_scale, size=(is_zero_sum.sum(), len(subgroup_cols)))
            stressed_df.loc[is_zero_sum, subgroup_cols] = noise

        # Apply softmax normalization
        stressed_df[subgroup_cols] = stressed_df[subgroup_cols].apply(
            lambda row: np.exp(row) / np.exp(row).sum(), axis=1
        )

    # Special case for ce_problematic
    ce_problematic = [
        'frac_waso_biogas_food',
        'frac_waso_biogas_sludge',
        'frac_waso_biogas_yard',
        'frac_waso_compost_food',
        'frac_waso_compost_methane_flared',
        'frac_waso_compost_sludge',
        'frac_waso_compost_yard'
    ]

    # Apply softmax normalization for ce_problematic
    stressed_df[ce_problematic] = stressed_df[ce_problematic].apply(
        lambda row: np.exp(row) / np.exp(row).sum(), axis=1
    )

    
    # Simulate outputs
    simulated_outputs = simulation_model(stressed_df)
    sectoral_diff_report_df = pd.read_csv('sectoral_diff_report.csv')

    # Merge the series DataFrame with the original DataFrame
    merged_df = pd.merge(sectoral_diff_report_df, simulated_outputs[['Subsector', 'sim_value']], on='Subsector', how='left')
    
    # Calculate error (e.g., Mean Squared Error)
    mse = np.mean((merged_df['sim_value'] - merged_df['Edgar_value']) ** 2)
    print(30*'*')
    print(f"===================  Current MSE: {mse} ==================")
    print(30*'*')
    return mse


# Run PSO to find optimal scaling vector
best_scaling_vector, best_error = pso(
    objective_function,  # Objective function
    lb,  # Lower bounds
    ub,  # Upper bounds
    swarmsize = 50,  # Number of particles in the swarm
    maxiter = 2,  # Maximum iterations
    debug=True  # Display progress
)

helper_functions.print_elapsed_time(start_time)

# Save the best scaling vector and its MSE
results = np.append(best_scaling_vector, best_error)  # Append the MSE to the scaling vector
header = ','.join([f"scale_{i}" for i in range(len(best_scaling_vector))]) + ',mse'  # Create a header
np.savetxt("best_scaling_vector.csv", [results], delimiter=",", header=header, comments="")  # Save with header
print(f"Best scaling vector: {best_scaling_vector}")
print(f"Best error (MSE): {best_error}")
