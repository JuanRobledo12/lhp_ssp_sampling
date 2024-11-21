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
from info_grupos import empirial_vars_to_avoid, frac_vars_special_cases_list
from genera_muestra import GenerateLHCSample
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



## Define paths and global variables
target_country = sys.argv[1]
id_experimento = int(sys.argv[2])

print(f"Executing Python Script for {target_country} with experiment id {id_experimento}")

FILE_PATH = os.getcwd()
build_path = lambda PATH : os.path.abspath(os.path.join(*PATH))

DATA_PATH = build_path([FILE_PATH, "..", "data"])
OUTPUT_PATH = build_path([FILE_PATH, "..", "output"])

SSP_OUTPUT_PATH = build_path([OUTPUT_PATH, "ssp"])

REAL_DATA_FILE_PATH = build_path([DATA_PATH, "real_data.csv"]) 

SALIDAS_EXPERIMENTOS_PATH = build_path([OUTPUT_PATH, "experiments"]) 

INPUTS_ESTRESADOS_PATH = build_path([SALIDAS_EXPERIMENTOS_PATH, "sim_inputs"])
OUTPUTS_ESTRESADOS_PATH = build_path([SALIDAS_EXPERIMENTOS_PATH, "sim_outputs"])
helper_functions = HelperFunctions()

helper_functions.ensure_directory_exists(INPUTS_ESTRESADOS_PATH)
helper_functions.ensure_directory_exists(OUTPUTS_ESTRESADOS_PATH)

### Load Costa Rica Example df to fill out data gaps in our input df

examples = SISEPUEDEExamples()
cr = examples("input_data_frame")

# Adding missing cols and setting correct format in our input df
df_input = pd.read_csv(REAL_DATA_FILE_PATH)
df_input = df_input.rename(columns={'period':'time_period'})
df_input = helper_functions.add_missing_cols(cr, df_input.copy())
df_input = df_input.drop(columns='iso_code3')

# Obtain the column names that we are going to sample 
columns_all_999 = df_input.columns[(df_input == -999).any()].tolist()
pij_cols = [col for col in df_input.columns if col.startswith('pij')]
cols_to_avoid = pij_cols + frac_vars_special_cases_list + columns_all_999 + empirial_vars_to_avoid
campos_estresar = helper_functions.get_indicators_col_names(df_input, cols_with_issue=cols_to_avoid)

# Defines upper bound to pass to GenerateLHCSample
u_bound = 2

# Defines number of sample vectors that GenerateLHCSample will create
n_vectors = 100
sampling_file_path = os.path.join('sampling_files', f'sample_scaled_{n_vectors}_{u_bound}.pickle') 

# Generates sampling matrix
if not os.path.exists(sampling_file_path):
    # Generates sampling matrix if it does not exist
    generate_sample = GenerateLHCSample(n_vectors, n_var=len(campos_estresar), u_bound=u_bound)
    generate_sample.generate_sample()

# Load the sampling matrix
with open(sampling_file_path, 'rb') as handle:
    sample_scaled = pickle.load(handle)


# Creating new df with the sampled data
df_estresado = df_input.copy()
df_estresado[campos_estresar]  = (df_input[campos_estresar]*sample_scaled[id_experimento]).to_numpy()

# Normalizing frac_ var groups to make sure they sum to 1
df_frac_vars = pd.read_excel('frac_vars.xlsx', sheet_name='frac_vars_no_special_cases')
need_norm_prefix = df_frac_vars.frac_var_name_prefix.unique()

for grupo in need_norm_prefix:
    vars_grupo = [i for i in df_estresado.columns if grupo in i]
    
    # Skip normalization for columns in cols_to_avoid
    if any(col in cols_to_avoid for col in vars_grupo):
        continue

    # Apply conditional log transformation
    df_estresado[vars_grupo] = df_estresado[vars_grupo].applymap(lambda y: -np.log(y) if y != 0 else 0)
    
    # Check if the sum is zero before normalizing
    sum_values = df_estresado[vars_grupo].sum(axis=1)
    df_estresado[vars_grupo] = df_estresado[vars_grupo].div(sum_values, axis=0).fillna(0)


# This is also an special case
ce_problematic = ['frac_waso_biogas_food',
                  'frac_waso_biogas_sludge',
                  'frac_waso_biogas_yard',
                  'frac_waso_compost_food',
                  'frac_waso_compost_methane_flared',
                  'frac_waso_compost_sludge',
                  'frac_waso_compost_yard']

# Apply conditional log transformation
df_estresado[ce_problematic] = df_estresado[ce_problematic].applymap(lambda y: -np.log(y) if y != 0 else 0)
# Check if the sum is zero before normalizing
sum_values = df_estresado[ce_problematic].sum(axis=1)
df_estresado[ce_problematic] = df_estresado[ce_problematic].div(sum_values, axis=0).fillna(0)

# Load SSP objects with input data and transformation folders
transformers = trf.transformers.Transformers(
    {},
    df_input = df_estresado,
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
INPUTS_ESTRESADOS_FILE_PATH = build_path([INPUTS_ESTRESADOS_PATH, f"sim_input_{id_experimento}.csv"])
OUTPUTS_ESTRESADOS_FILE_PATH = build_path([OUTPUTS_ESTRESADOS_PATH, f"sim_output_{id_experimento}.csv"])


df_out = ssp.read_output(None)
df_out.to_csv(OUTPUTS_ESTRESADOS_FILE_PATH, index=False)
df_estresado[campos_estresar].to_csv(INPUTS_ESTRESADOS_FILE_PATH, index=False)

helper_functions.print_elapsed_time(start_time)
