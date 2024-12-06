import pandas as pd 
import os 
import glob
import sys
from utils import HelperFunctions

helper_functions = HelperFunctions()

# Ensure the user has provided a command-line argument
if len(sys.argv) < 3:
    raise ValueError("No country name or batch id provided. Please pass a country name and  batch_id as a command-line argument.")


country_name = sys.argv[1]
batch_id = sys.argv[2]
print(f'Creating feature variables and target variables csv files for {country_name} in batch id {batch_id}')

## Define paths

FILE_PATH = os.getcwd()
build_path = lambda PATH : os.path.abspath(os.path.join(*PATH))

OUTPUT_PATH = build_path([FILE_PATH, "..", "output"])
SALIDAS_EXPERIMENTOS_PATH = build_path([OUTPUT_PATH, f"experiments_batch_{country_name}_{batch_id}"])
COMPLETE_DF_OUTPUT_PATH = build_path([SALIDAS_EXPERIMENTOS_PATH, "grouped_data"]) 

INPUTS_ESTRESADOS_PATH = build_path([SALIDAS_EXPERIMENTOS_PATH, "sim_inputs"])
OUTPUTS_ESTRESADOS_PATH = build_path([SALIDAS_EXPERIMENTOS_PATH, "sim_outputs"])

PATHS_OUTPUT_CSV_FILES = glob.glob(OUTPUTS_ESTRESADOS_PATH+"/*.csv")
PATHS_INPUT_CSV_FILES = glob.glob(INPUTS_ESTRESADOS_PATH+"/*.csv")


PATHS_OUTPUT_CSV_FILES.sort()
PATHS_INPUT_CSV_FILES.sort()


PATHS_OUTPUT_CSV_FILES.sort()
PATHS_INPUT_CSV_FILES.sort()

if len(PATHS_OUTPUT_CSV_FILES) != len(PATHS_INPUT_CSV_FILES):
    raise ValueError("The amount of input files does not match the amount of output files.")

df_targets = pd.concat([pd.read_csv(i).iloc[[0]] for i in glob.glob(OUTPUTS_ESTRESADOS_PATH+"/*.csv")], ignore_index = True)
df_features = pd.concat([pd.read_csv(i).iloc[[0]] for i in glob.glob(INPUTS_ESTRESADOS_PATH+"/*.csv")], ignore_index = True)

df_features = helper_functions.create_id_column(df_features.copy())
df_targets = helper_functions.create_id_column(df_targets.copy())

df_targets_only_emission = df_targets[[col for col in df_targets.columns if (col.startswith('emission_co2e_subsector') or col =='sample_id')]]

# Imputing nan values
df_targets_only_emission_imputed = df_targets_only_emission.apply(lambda col: col.fillna(col.median()) if col.dtype in ['float64', 'int64'] else col)

# Create the directory if it does not exist
if not os.path.exists(COMPLETE_DF_OUTPUT_PATH):
    os.makedirs(COMPLETE_DF_OUTPUT_PATH)

print(f"Saved a dataframes with shapes: {df_features.shape}, {df_targets_only_emission_imputed.shape}")

df_features.to_csv(os.path.join(COMPLETE_DF_OUTPUT_PATH, f'lhs_sampled_{country_name}_features.csv'), index = False)
df_targets_only_emission_imputed.to_csv(os.path.join(COMPLETE_DF_OUTPUT_PATH, f'lhs_sampled_{country_name}_targets.csv'), index = False)