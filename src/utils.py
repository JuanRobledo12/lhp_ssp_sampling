import time
import os
import yaml
import numpy as np
import pandas as pd

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
    
    def normalize_frac_vars(self, stressed_df, cols_to_avoid):

        df = stressed_df.copy()

        # Normalizing frac_ var groups using softmax
        df_frac_vars = pd.read_excel('frac_vars.xlsx', sheet_name='frac_vars_no_special_cases')
        need_norm_prefix = df_frac_vars.frac_var_name_prefix.unique()

        random_scale = 1e-2  # Scale for random noise
        epsilon = 1e-6

        for subgroup in need_norm_prefix:
            subgroup_cols = [i for i in df.columns if subgroup in i]
            
            # Skip normalization for columns in cols_to_avoid
            if any(col in cols_to_avoid for col in subgroup_cols):
                continue

            # Check if the sum of the group is zero or too small
            group_sum = df[subgroup_cols].sum(axis=1)
            is_zero_sum = group_sum < epsilon

            # Add random variability for zero-sum groups
            if is_zero_sum.any():
                noise = np.random.uniform(0, random_scale, size=(is_zero_sum.sum(), len(subgroup_cols)))
                df.loc[is_zero_sum, subgroup_cols] = noise

            # Apply softmax normalization
            df[subgroup_cols] = df[subgroup_cols].apply(
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
        df[ce_problematic] = df[ce_problematic].apply(
            lambda row: np.exp(row) / np.exp(row).sum(), axis=1
        )

        return df
    

    def weighted_mse(self, dataframe):
        # Ensure diff is absolute for weights
        dataframe['weight'] = dataframe['diff'].abs()
        
        # Calculate squared difference between simulation and Edgar_value
        dataframe['squared_error'] = (dataframe['simulation'] - dataframe['Edgar_value']) ** 2
        
        # Weighted MSE calculation
        weighted_mse_value = (dataframe['squared_error'] * dataframe['weight']).sum() / dataframe['weight'].sum()
        return weighted_mse_value

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


        # Returns df_out
        try:
            df_out = ssp.read_output(None)
            if df_out is None or df_out.empty:
                raise ValueError("The output DataFrame is None or empty. Returning an empty DataFrame.")
        except Exception as e:
            print(f"Warning: {e}")
            df_out = pd.DataFrame()

        return df_out
    

class SectoralDiffReport:
    
    def __init__(self):
        pass


    def generate_diff_reports(self, df_out, iso_code3, refy=2015, ref_primary_id=0):

        # Base directory paths
        base_path = os.getcwd()  # Set to the current working directory or customize
        misc_files_path = os.path.join(base_path,"src", "misc_files")


        # Load mapping table
        mapping = pd.read_csv(os.path.join(misc_files_path, "mapping.csv"))

        # Load raw simulation data
        slt = df_out.copy()

        # Estimate emission totals for the initial year
        slt['Year'] = slt['time_period'] + 2015
        # print("SLT DataFrame sample after adding 'Year' column:")
        # print(slt.head())

        for i in range(len(mapping)):
            vars_ = mapping.loc[i, 'Vars'].split(":")
            try:
                if len(vars_) > 1:
                    mapping.loc[i, 'simulation'] = slt[(slt['primary_id'] == ref_primary_id) & (slt['Year'] == refy)][vars_].sum(axis=1).sum()
                else:
                    mapping.loc[i, 'simulation'] = slt[(slt['primary_id'] == ref_primary_id) & (slt['Year'] == refy)][vars_[0]].sum()
            except KeyError as e:
                print(f"Warning: Column(s) {vars_} not found in simulation data. Error: {e}")

        # Debugging the mapping DataFrame after adding 'simulation' values
        # print("Mapping DataFrame sample after simulation calculations:")
        # print(mapping.head())

        # Load edgar data and filter by iso_code3
        edgar = pd.read_csv(os.path.join(misc_files_path, "CSC-GHG_emissions-April2024_to_calibrate.csv"), encoding='latin1')
        # print("Edgar DataFrame sample before filtering:")
        # print(edgar.head())

        edgar = edgar[edgar['Code'] == iso_code3]  # Filter by iso_code3
        # print(f"Edgar DataFrame after filtering for iso_code3 = {iso_code3}:")
        # print(edgar.head())

        edgar['Edgar_Class'] = edgar['CSC Subsector'] + ":" + edgar['Gas']
        # print("Unique Edgar_Class values in edgar after adding 'Edgar_Class':")
        # print(edgar['Edgar_Class'].unique())

        # Melt edgar data
        id_varsEd = ["Edgar_Class"]
        measure_vars_Ed = [col for col in edgar.columns if col.isdigit()]  # Select year columns
        edgar = pd.melt(edgar, id_vars=id_varsEd, value_vars=measure_vars_Ed, var_name="Year", value_name="Edgar_value")
        edgar['Year'] = edgar['Year'].astype(int)
        edgar = edgar[edgar['Year'] == refy][["Edgar_Class", "Edgar_value"]]
        # print(f"Edgar DataFrame after melting and filtering for Year = {refy}:")
        # print(edgar.head())

        # # Debugging unique Edgar_Class values before merging
        # print("Unique Edgar_Class values in mapping before merging:")
        # print(mapping['Edgar_Class'].unique())

        # Merge both and generate reports
        report_1 = mapping.groupby(['Subsector', 'Edgar_Class'])['simulation'].sum().reset_index()
        report_1 = pd.merge(report_1, edgar, on="Edgar_Class", how="left", indicator=True)
        # print("Report 1 after merging with edgar:")
        # print(report_1.head())
        # print("Merge indicator value counts in report_1:")
        # print(report_1['_merge'].value_counts())

        # Calculate differences and save reports
        report_1['diff'] = (report_1['simulation'] - report_1['Edgar_value']) / report_1['Edgar_value']
        report_1['Year'] = refy
        report_1.to_csv(os.path.join(misc_files_path, "detailed_diff_report.csv"), index=False)

        report_2 = report_1.groupby('Subsector').agg({'simulation': 'sum', 'Edgar_value': 'sum'}).reset_index()
        report_2['diff'] = (report_2['simulation'] - report_2['Edgar_value']) / report_2['Edgar_value']
        report_2['Year'] = refy
        report_2.to_csv(os.path.join(misc_files_path, "sector_diff_report.csv"), index=False)
        # print("Report generation completed.")

        return report_1, report_2

