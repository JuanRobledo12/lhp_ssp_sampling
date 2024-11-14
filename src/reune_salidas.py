import pandas as pd 
import os 
import glob
import sys

# Ensure the user has provided a command-line argument
if len(sys.argv) < 2:
    raise ValueError("No country name provided. Please pass a country name as a command-line argument.")


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

df_output = pd.concat([pd.read_csv(i).iloc[[0]] for i in glob.glob(OUTPUTS_ESTRESADOS_PATH+"/*.csv")], ignore_index = True)
df_input = pd.concat([pd.read_csv(i).iloc[[0]] for i in glob.glob(INPUTS_ESTRESADOS_PATH+"/*.csv")], ignore_index = True)

df_complete = pd.concat([df_input, df_output], axis = 1)


country_name = sys.argv[1]

# Create the directory if it does not exist
if not os.path.exists(COMPLETE_DF_OUTPUT_PATH):
    os.makedirs(COMPLETE_DF_OUTPUT_PATH)

DF_COMPLETE_PATH = build_path([COMPLETE_DF_OUTPUT_PATH, f"lhp_sampled_{country_name}_data.csv"])

print("Saved a dataframe with shape: ", df_complete.shape)

df_complete.to_csv(DF_COMPLETE_PATH, index = False)