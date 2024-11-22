import pandas as pd 
import os 
import glob
import sys
from utils import HelperFunctions

helper_functions = HelperFunctions()

# Ensure the user has provided a command-line argument
if len(sys.argv) < 2:
    raise ValueError("No country name provided. Please pass a country name as a command-line argument.")


country_name = sys.argv[1]
print('Creating attributes and target variables csv files for ', country_name)

## Define paths

FILE_PATH = os.getcwd()
build_path = lambda PATH : os.path.abspath(os.path.join(*PATH))

OUTPUT_PATH = build_path([FILE_PATH, "..", "output"])
SALIDAS_EXPERIMENTOS_PATH = build_path([OUTPUT_PATH, "experiments"])
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
df_attributes = pd.concat([pd.read_csv(i).iloc[[0]] for i in glob.glob(INPUTS_ESTRESADOS_PATH+"/*.csv")], ignore_index = True)

df_attributes = helper_functions.create_id_column(df_attributes.copy())
df_targets = helper_functions.create_id_column(df_targets.copy())

# Imputing nan values
# This can be changed to see if the RF performance is better
df_targets_imputed = df_targets.apply(lambda col: col.fillna(col.median()) if col.dtype in ['float64', 'int64'] else col)
df_targets_only_emission = df_targets_imputed[[col for col in df_targets_imputed.columns if col.startwith('emissions_co2e')]]
# Before contatanating we have to create the target variables using the mapping_2.csv
# Load mapping table
mapping = pd.read_csv("mapping_2.csv")

mapping['Edgar_Class'] = mapping['Edgar_Class'].str.replace(' ', '', regex=False)
mapping['Edgar_Class'] = mapping['Edgar_Class'].str.replace(' ', '_').str.replace('-', '_')
mapping['Edgar_Class'] = mapping['Edgar_Class'].str.replace(':', '_')
mapping['Edgar_Class'] = mapping['Edgar_Class'].str.replace('/', '')

for i in range(len(mapping)):
    vars_ = mapping.loc[i, 'Vars'].split(":")
    df_targets_imputed[mapping.loc[i, 'Edgar_Class']] = df_targets_imputed[vars_].sum(axis=1)
    # print(mapping.loc[i, 'Edgar_Class'])

target_vars = list(mapping['Edgar_Class'].unique())
new_vars = ['id'] + target_vars
df_targets_final = df_targets_imputed[new_vars]

# Create the directory if it does not exist
if not os.path.exists(COMPLETE_DF_OUTPUT_PATH):
    os.makedirs(COMPLETE_DF_OUTPUT_PATH)

print(f"Saved a dataframes with shapes: {df_attributes.shape} and {df_targets_final.shape}")

df_attributes.to_csv(os.path.join(COMPLETE_DF_OUTPUT_PATH, f'lhp_sampled_{country_name}_attributes.csv'), index = False)
df_targets_final.to_csv(os.path.join(COMPLETE_DF_OUTPUT_PATH, f'lhp_sampled_{country_name}_targets.csv'), index = False)
df_targets_only_emission.to_csv(os.path.join(COMPLETE_DF_OUTPUT_PATH, f'lhp_sampled_{country_name}_features_co2e.csv'), index = False)