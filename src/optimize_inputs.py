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
from utils import HelperFunctions, SSPModelForCalibartion, SectoralDiffReport
from sisepuede.manager.sisepuede_examples import SISEPUEDEExamples
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Record the start time
start_time = time.time()

# Initialize helper functions
helper_functions = HelperFunctions()

# Paths
SRC_FILE_PATH = os.getcwd()
build_path = lambda PATH: os.path.abspath(os.path.join(*PATH))
DATA_PATH = build_path([SRC_FILE_PATH, "..", "data"])
OUTPUT_PATH = build_path([SRC_FILE_PATH, "..", "output"])
REAL_DATA_FILE_PATH = build_path([DATA_PATH, "real_data.csv"])
SSP_OUTPUT_PATH = build_path([OUTPUT_PATH, "ssp"])
MISC_FILES_PATH = build_path([SRC_FILE_PATH, 'misc_files'])
OPT_CONFIG_FILES_PATH = build_path([SRC_FILE_PATH, 'config_opt'])
OPT_OUTPUT_PATH = build_path([SRC_FILE_PATH,"..", "opt_output"])

# Get important params from the YAML file

try:
    yaml_file = sys.argv[1]
except IndexError:
    raise ValueError("YAML configuration file must be provided as a command-line argument.")

param_dict = helper_functions.get_parameters_from_opt_yaml(build_path([OPT_CONFIG_FILES_PATH, yaml_file]))

target_country = param_dict['target_country']
iso_code3 = param_dict['iso_code3']
detailed_diff_report_flag = param_dict['detailed_diff_report_flag'] 
unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

logging.info(f"Starting optimization for {target_country} (ISO code: {iso_code3}). Detailed diff report: {'enabled' if detailed_diff_report_flag else 'disabled'}.")

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
        logging.warning("Emissions DataFrame is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    return emissions_df

# Objective function
def objective_function(scaling_vector: np.ndarray) -> float:
    """
    Evaluates the error between simulated and ground truth outputs.
    """
    logging.info("Executing objective function...")

    stressed_df = df_input.copy()
    stressed_df[cols_to_stress] = df_input[cols_to_stress] * scaling_vector

    # Normalize fractional columns
    stressed_df_with_norm = helper_functions.normalize_frac_vars(stressed_df, cols_to_avoid)

    # Simulate outputs
    simulated_outputs = simulation_model(stressed_df_with_norm)
    
    # Handle empty simulation outputs
    if simulated_outputs.empty:
        high_mse = 1e6  # Assign a high MSE for garbage outputs
        logging.warning("Simulation returned an empty DataFrame. Setting MSE to a high value.")
        log_to_csv(scaling_vector, high_mse)
        return high_mse

    # Generate diff reports to calculate MSE
    detailed_diff_report, normal_diff_report = diff_report_helpers.generate_diff_reports(simulated_outputs, iso_code3, MISC_FILES_PATH)

    # Calculate error (Weighted Mean Squared Error): Subsectors with Edgar value == 0.0 are not considered
    if detailed_diff_report_flag:
        mse = helper_functions.weighted_mse(detailed_diff_report)
    else:
        mse = helper_functions.weighted_mse(normal_diff_report)

    logging.info("=" * 30)
    logging.info(f"Current MSE: {mse:.6f}")
    logging.info("=" * 30)

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
    log_file = build_path([OPT_OUTPUT_PATH, f"opt_results_{target_country}_{unique_id}.csv"])
    try:
        log_df.to_csv(log_file, mode='a', header=not pd.io.common.file_exists(log_file), index=False)
        logging.info(f"Logged current MSE and scaling vector to {log_file}")
    except Exception as e:
        logging.error(f"Error logging data to CSV: {e}")

ssp_model = SSPModelForCalibartion(SSP_OUTPUT_PATH=SSP_OUTPUT_PATH , target_country=target_country)
diff_report_helpers = SectoralDiffReport()

# Run PSO to find optimal scaling vector
best_scaling_vector, best_error = pso(
    objective_function,  # Objective function
    lb,  # Lower bounds
    ub,  # Upper bounds
    swarmsize = 50,  # Number of particles in the swarm
    maxiter = 100,  # Maximum iterations
    debug=True  # Display progress
)

helper_functions.print_elapsed_time(start_time)

# Save the best scaling vector and its MSE
results = np.append(best_scaling_vector, best_error)  # Append the MSE to the scaling vector
header = ','.join([f"scale_{i}" for i in range(len(best_scaling_vector))]) + ',mse'  # Create a header
output_file = build_path([OPT_OUTPUT_PATH, "best_scaling_vector.csv"])  # Save under opt_output
np.savetxt(output_file, [results], delimiter=",", header=header, comments="")  # Save with header
logging.info(f"Best scaling vector: {best_scaling_vector}")
logging.info(f"Best error (MSE): {best_error}")
