import time
import os
import yaml

##  IMPORT SISEPUEDE EXAMPLES AND TRANSFORMERS

from sisepuede.manager.sisepuede_examples import SISEPUEDEExamples
from sisepuede.manager.sisepuede_file_structure import SISEPUEDEFileStructure
import sisepuede.core.support_classes as sc
import sisepuede.transformers as trf
import sisepuede.utilities._plotting as spu
import sisepuede.utilities._toolbox as sf
import sisepuede as si

class HelperFunctions:
    
    def __init__(self) -> None:
        pass

    
    def print_elapsed_time(self, start_time):

        # Record the end time
        end_time = time.time()

        # Calculate and print the execution time
        execution_time = end_time - start_time
        print(f"------------------------ EXECUTION TIME: {execution_time} seconds ------------------------")

    def check_land_use_factor(self, ssp_object, target_country):
        try:
            dict_scendata = ssp_object.generate_scenario_database_from_primary_key(0)
            df_inputs_check = dict_scendata.get(target_country) # Change the name of the country if running a different one
            lndu_realloc_fact_df = ssp_object.model_attributes.extract_model_variable(df_inputs_check, "Land Use Yield Reallocation Factor")
        except:
            print("Error in lndu factor...")

        if lndu_realloc_fact_df['lndu_reallocation_factor'].sum() > 0:
            raise ValueError(" --------------- The sum of 'lndu_reallocation_factor' is greater than 0. Script terminated. -----------------")
        

    def compare_dfs(self, df1, df2):
        # Assuming your DataFrames are df1 and df2
        columns_df1 = set(df1.columns)
        columns_df2 = set(df2.columns)

        # Columns present in df1 but not in df2
        diff_in_df1 = columns_df1 - columns_df2

        # Columns present in df2 but not in df1
        diff_in_df2 = columns_df2 - columns_df1

        # Columns shared in both df1 and df2
        shared_columns = columns_df1 & columns_df2

        print("Columns in df1 but not in df2:", diff_in_df1)
        print("Columns in df2 but not in df1:", diff_in_df2)
        print("Columns shared in both df1 and df2:", shared_columns)


    def add_missing_cols(self, df1, df2):
        # Identify columns in df1 but not in df2
        columns_to_add = [col for col in df1.columns if col not in df2.columns]

        # Add missing columns to df2 with their values from df1
        for col in columns_to_add:
            df2[col] = df1[col]
        
        return df2

    def get_indicators_col_names(self, df, cols_with_issue = []):

        cols_to_avoid = ['time_period', 'region'] + cols_with_issue
        col_names = [col for col in df.columns if col not in cols_to_avoid]

        # # Check if the length of col_names is as expected
        # expected_length = len(df.columns) - len(cols_to_avoid)
        # print(f"Expected length after removal: {expected_length}")
        # print(f"Actual length of col_names: {len(col_names)}")

        # # Verify if all cols_to_avoid were removed from col_names
        # removed_successfully = all(col not in col_names for col in cols_to_avoid)
        # if removed_successfully:
        #     print("All columns in cols_to_avoid were successfully removed.")
        # else:
        #     print("Some columns in cols_to_avoid are still present in col_names.")
        #     # Optionally, print the columns that were not removed
        #     remaining_cols = [col for col in cols_to_avoid if col in col_names]
        #     print("Columns not removed:", remaining_cols)

        return col_names
    
    def get_cols_with_nans(self, df):

        # Checking if there are any columns with null values in it
        columns_with_na = df.columns[df.isna().any()].tolist()

        print(columns_with_na)

        return columns_with_na
    
    def create_id_column(self, df):

        df['id'] = range(1, len(df) + 1)
        # Assuming 'df' is your DataFrame and 'id' is the column you want to move to the front
        cols = ['id'] + [col for col in df if col != 'id']
        df = df[cols]

        return df
    
    def ensure_directory_exists(self, path):
        """Creates a directory if it does not exist."""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

    # Load configuration from a YAML file
    def get_parameters_from_yaml(self, file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        param_dict = {
            'target_country': config['target_country'],
            'batch_id': config['batch_id'],
            'u_bound': config['u_bound'],
            'n_arrays': config['n_arrays']
        }
        return param_dict
    

class SSPModelForCalibartion:

    def __init__(self, SSP_OUTPUT_PATH, target_country):
        
        ##  SETUP SOME SISEPUEDE STUFF
        self.file_struct = SISEPUEDEFileStructure()

        self.matt = self.file_struct.model_attributes
        self.regions = sc.Regions(self.matt)
        self.time_periods = sc.TimePeriods(self.matt)
        self.examples = SISEPUEDEExamples()

        # Set up other important vars
        self.SSP_OUTPUT_PATH = SSP_OUTPUT_PATH

        self.target_country = target_country



    def run_ssp_simulation(self, stressed_df):
        # Load SSP objects with input data and transformation folders
        transformers = trf.transformers.Transformers(
            {},
            df_input = stressed_df,
        )

        # set an ouput path and instantiate

        trf.instantiate_default_strategy_directory(
                transformers,
                self.SSP_OUTPUT_PATH,
            )

        # then, you can load this back in after modifying (play around with it)
        transformations = trf.Transformations(
                self.SSP_OUTPUT_PATH,
                transformers = transformers,
            )

        strategies = trf.Strategies(
                transformations,
                export_path = "transformations",
                prebuild = True,
            )

        df_vargroups = self.examples("variable_trajectory_group_specification")

        strategies.build_strategies_to_templates(
                df_trajgroup = df_vargroups,
                include_simplex_group_as_trajgroup = True,
                strategies = [0, 1000],
            )

        ssp = si.SISEPUEDE(
                "calibrated",
                initialize_as_dummy = False, # no connection to Julia is initialized if set to True
                regions = [self.target_country],
                db_type = "csv",
                strategies = strategies,
                try_exogenous_xl_types_in_variable_specification = True,
            )

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


        # Returns only subsector outputs
        df_out = ssp.read_output(None)
        df_target = df_out[[col for col in df_out.columns if col.startswith('emission_co2e_subsector')]]

        return df_target

