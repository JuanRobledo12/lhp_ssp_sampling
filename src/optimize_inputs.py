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
def simulation_model(df_scaled: pd.DataFrame) -> pd.DataFrame:
    """
    Function that simulates outputs based on the scaled inputs.
    """
    emissions_df = ssp_model.run_ssp_simulation(df_scaled)
    
    # Handle empty DataFrame
    if emissions_df is None or emissions_df.empty:
        print("Warning: Emissions DataFrame is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

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

    # Normalize fractional columns
    stressed_df_with_norm = helper_functions.normalize_frac_vars(stressed_df, cols_to_avoid)

    # Simulate outputs
    simulated_outputs = simulation_model(stressed_df_with_norm)
    
    # Handle empty simulation outputs
    if simulated_outputs.empty:
        high_mse = 1e6  # Assign a high MSE for garbage outputs
        print("Simulation returned an empty DataFrame. Setting MSE to a high value.")
        log_to_csv(scaling_vector, high_mse)
        return high_mse

    # Load ground truth data
    sectoral_diff_report_df = pd.read_csv('sectoral_diff_report.csv')

    # Merge the series DataFrame with the original DataFrame
    merged_df = pd.merge(sectoral_diff_report_df, simulated_outputs[['Subsector', 'sim_value']], on='Subsector', how='left')

    # Calculate error (e.g., Mean Squared Error)
    mse = np.mean((merged_df['sim_value'] - merged_df['Edgar_value']) ** 2)
    print(30 * '*')
    print(f"===================  Current MSE: {mse} ==================")
    print(30 * '*')

    # Log the results
    log_to_csv(scaling_vector, mse)
    return mse

# Function to log MSE and scaling vector to a CSV file
def log_to_csv(scaling_vector: np.ndarray, mse: float):
    """
    Logs the MSE and scaling vector to a CSV file.
    """
    log_data = {'MSE': [mse], **{f'scale_{i}': [val] for i, val in enumerate(scaling_vector)}}
    log_df = pd.DataFrame(log_data)
    
    # Append to the CSV file or create it if it doesn't exist
    log_file = 'optimization_log.csv'
    try:
        log_df.to_csv(log_file, mode='a', header=not pd.io.common.file_exists(log_file), index=False)
        print(f"Logged current MSE and scaling vector to {log_file}")
    except Exception as e:
        print(f"Error logging data to CSV: {e}")

ssp_model = SSPModelForCalibartion(SSP_OUTPUT_PATH=SSP_OUTPUT_PATH , target_country=target_country)

# Run PSO to find optimal scaling vector
best_scaling_vector, best_error = pso(
    objective_function,  # Objective function
    lb,  # Lower bounds
    ub,  # Upper bounds
    swarmsize = 1,  # Number of particles in the swarm
    maxiter = 1,  # Maximum iterations
    debug=True  # Display progress
)

helper_functions.print_elapsed_time(start_time)

# Save the best scaling vector and its MSE
results = np.append(best_scaling_vector, best_error)  # Append the MSE to the scaling vector
header = ','.join([f"scale_{i}" for i in range(len(best_scaling_vector))]) + ',mse'  # Create a header
np.savetxt("best_scaling_vector.csv", [results], delimiter=",", header=header, comments="")  # Save with header
print(f"Best scaling vector: {best_scaling_vector}")
print(f"Best error (MSE): {best_error}")
