import pandas as pd 
import os 
import glob

## Define paths

FILE_PATH = os.getcwd()
build_path = lambda PATH : os.path.abspath(os.path.join(*PATH))

OUTPUT_PATH = build_path([FILE_PATH, "..", "output"])
SALIDAS_EXPERIMENTOS_PATH = build_path([OUTPUT_PATH, "experimentos"]) 

INPUTS_ESTRESADOS_PATH = build_path([SALIDAS_EXPERIMENTOS_PATH, "input_estresados"])
OUTPUTS_ESTRESADOS_PATH = build_path([SALIDAS_EXPERIMENTOS_PATH, "output_estresados"])

PATHS_OUTPUT_CSV_FILES = glob.glob(OUTPUTS_ESTRESADOS_PATH+"/*.csv")
PATHS_INPUT_CSV_FILES = glob.glob(INPUTS_ESTRESADOS_PATH+"/*.csv")

PATHS_OUTPUT_CSV_FILES.sort()
PATHS_INPUT_CSV_FILES.sort()

df_output = pd.concat([pd.read_csv(i).iloc[[0]] for i in glob.glob(OUTPUTS_ESTRESADOS_PATH+"/*.csv")], ignore_index = True)
df_input = pd.concat([pd.read_csv(i).iloc[[0]] for i in glob.glob(INPUTS_ESTRESADOS_PATH+"/*.csv")], ignore_index = True)

df_complete = pd.concat([df_input, df_output], axis = 1)

DF_COMPLETE_PATH = build_path([SALIDAS_EXPERIMENTOS_PATH, "ssp_iran_sensibilidad.csv"])

df_complete.to_csv(DF_COMPLETE_PATH, index = False)