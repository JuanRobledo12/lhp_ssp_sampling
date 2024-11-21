import time
import os

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
        dict_scendata = ssp_object.generate_scenario_database_from_primary_key(0)
        df_inputs_check = dict_scendata.get(target_country) # Change the name of the country if running a different one
        lndu_realloc_fact_df = ssp_object.model_attributes.extract_model_variable(df_inputs_check, "Land Use Yield Reallocation Factor")

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
