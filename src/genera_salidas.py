"""
TODO:
- Now we can take the special cases we added to info_grupos back to frac_vars.xlsx since we have them in cols to avoid.
- The HelperFunctions can raise Errors or warning that help us identify null values, mismatching vars and things like that.
- We can create the sim_inputs, sim_outputs folder automatically if they do not exist.
- Add some safety guards to the argv params son the code raises an error if they aren't passed.
"""

import copy
import datetime as dt
import importlib # needed so that we can reload packages
import matplotlib.pyplot as plt
import os, os.path
import numpy as np
import pandas as pd
import pathlib
import sys
import time
import pickle
from typing import Union
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
from info_grupos import empirical_vars_to_avoid, frac_vars_special_cases_list
from genera_muestra import GenerateLHS
from utils import HelperFunctions

##  IMPORT SISEPUEDE EXAMPLES AND TRANSFORMERS

from sisepuede.manager.sisepuede_examples import SISEPUEDEExamples
from sisepuede.manager.sisepuede_file_structure import SISEPUEDEFileStructure
import sisepuede.core.support_classes as sc
import sisepuede.transformers as trf
import sisepuede.utilities._plotting as spu
import sisepuede.utilities._toolbox as sf
import sisepuede as si

# Record the start time
start_time = time.time()

# Import helper functiosns
helper_functions = HelperFunctions()

# Defining some important parameters
experiment_id = int(sys.argv[1])
param_file_name = sys.argv[2]

param_dict = helper_functions.get_parameters_from_yaml(os.path.join('config_files', param_file_name)) 

target_country = param_dict['target_country']
batch_id = param_dict['batch_id']

print(f"Executing Python Script for {target_country} with experiment id {experiment_id} for batch id {batch_id}")

# Defining paths
FILE_PATH = os.getcwd()
build_path = lambda PATH : os.path.abspath(os.path.join(*PATH))

DATA_PATH = build_path([FILE_PATH, "..", "data"])
OUTPUT_PATH = build_path([FILE_PATH, "..", "output"])

SSP_OUTPUT_PATH = build_path([OUTPUT_PATH, "ssp"])

REAL_DATA_FILE_PATH = build_path([DATA_PATH, "real_data.csv"]) 

SALIDAS_EXPERIMENTOS_PATH = build_path([OUTPUT_PATH, f"experiments_batch_{target_country}_{batch_id}"]) 

INPUTS_ESTRESADOS_PATH = build_path([SALIDAS_EXPERIMENTOS_PATH, "sim_inputs"])
OUTPUTS_ESTRESADOS_PATH = build_path([SALIDAS_EXPERIMENTOS_PATH, "sim_outputs"])


helper_functions.ensure_directory_exists(SALIDAS_EXPERIMENTOS_PATH)
helper_functions.ensure_directory_exists(INPUTS_ESTRESADOS_PATH)
helper_functions.ensure_directory_exists(OUTPUTS_ESTRESADOS_PATH)

### Load Costa Rica Example df to fill out data gaps in our input df

examples = SISEPUEDEExamples()
cr = examples("input_data_frame")

# Adding missing cols and setting correct format in our input df
df_input = pd.read_csv(REAL_DATA_FILE_PATH)
df_input = df_input.rename(columns={'period': 'time_period'})
df_input = helper_functions.add_missing_cols(cr, df_input.copy())
df_input = df_input.drop(columns='iso_code3')

# Ensure 'lndu_reallocation_factor' column exists and set all its values to 0
if 'lndu_reallocation_factor' not in df_input.columns:
    print('Adding lndu_reallocation_factor to the df')
    df_input['lndu_reallocation_factor'] = 0
else:
    print("Setting lndu_reallocation_factor to 0")
    df_input['lndu_reallocation_factor'] = 0

# Obtain the column names that we are going to sample 
columns_all_999 = df_input.columns[(df_input == -999).any()].tolist()
pij_cols = [col for col in df_input.columns if col.startswith('pij')]
cols_to_avoid = pij_cols + frac_vars_special_cases_list + columns_all_999 + empirical_vars_to_avoid
cols_to_stress = helper_functions.get_indicators_col_names(df_input, cols_with_issue=cols_to_avoid)

# Defines upper bound to pass to GenerateLHS
u_bound = param_dict['u_bound']

# Defines number of sample vectors that GenerateLHS will create
n_arrays = param_dict['n_arrays']
sampling_file_path = os.path.join('sampling_files', f'sample_scaled_{n_arrays}_{u_bound}.pickle') 

# Generates sampling matrix
if not os.path.exists(sampling_file_path):
    # Generates sampling matrix if it does not exist
    generate_sample = GenerateLHS(n_arrays, n_var=len(cols_to_stress), u_bound=u_bound)
    generate_sample.generate_sample()

# Load the sampling matrix
with open(sampling_file_path, 'rb') as handle:
    sample_scaled = pickle.load(handle)

# Creating new df with the sampled data
stressed_df = df_input.copy()

# Apply sampling to create stressed data
stressed_df[cols_to_stress] = (df_input[cols_to_stress] * sample_scaled[experiment_id]).to_numpy()

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



# Load SSP objects with input data and transformation folders
transformers = trf.transformers.Transformers(
    {},
    df_input = stressed_df,
)

##  SETUP SOME SISEPUEDE STUFF

file_struct = SISEPUEDEFileStructure()

matt = file_struct.model_attributes
regions = sc.Regions(matt)
time_periods = sc.TimePeriods(matt)

# set an ouput path and instantiate

trf.instantiate_default_strategy_directory(
        transformers,
        SSP_OUTPUT_PATH,
    )

# then, you can load this back in after modifying (play around with it)
transformations = trf.Transformations(
        SSP_OUTPUT_PATH,
        transformers = transformers,
    )

strategies = trf.Strategies(
        transformations,
        export_path = "transformations",
        prebuild = True,
    )

df_vargroups = examples("variable_trajectory_group_specification")

strategies.build_strategies_to_templates(
        df_trajgroup = df_vargroups,
        include_simplex_group_as_trajgroup = True,
        strategies = [0, 1000],
    )

ssp = si.SISEPUEDE(
        "calibrated",
        initialize_as_dummy = False, # no connection to Julia is initialized if set to True
        regions = [target_country],
        db_type = "csv",
        strategies = strategies,
        try_exogenous_xl_types_in_variable_specification = True,
    )

# Checks if the land use reallocation factor is set to 0.0
helper_functions.check_land_use_factor(ssp_object=ssp, target_country=target_country)

# Create parameters dict for the model to run
dict_run = {
        ssp.key_future: [0],
        ssp.key_design: [0],
        ssp.key_strategy: [
            0,
            1000,
        ],
    }

# we'll save inputs since we're doing a small set of runs
ssp.project_scenarios(
        dict_run,
        save_inputs = True,
    )


# Saves input and output
INPUTS_ESTRESADOS_FILE_PATH = build_path([INPUTS_ESTRESADOS_PATH, f"sim_input_{experiment_id}.csv"])
OUTPUTS_ESTRESADOS_FILE_PATH = build_path([OUTPUTS_ESTRESADOS_PATH, f"sim_output_{experiment_id}.csv"])


df_out = ssp.read_output(None)

sample_id = f'{batch_id}-{experiment_id}'

df_out['sample_id'] = sample_id
stressed_df['sample_id'] = sample_id

df_out.to_csv(OUTPUTS_ESTRESADOS_FILE_PATH, index=False)
stressed_df.to_csv(INPUTS_ESTRESADOS_FILE_PATH, index=False)

helper_functions.print_elapsed_time(start_time)
