#28DiasPythonParaMinería

#The Bull Miner GitHub repository
#Follow me on Linkedin for more content like this:    https://www.linkedin.com/in/mikemine/



# === 1. IMPORT LIBRARIES ===
import pandas as pd # Imports the pandas library and gives it the common alias 'pd' for working with data tables (DataFrames).
import numpy as np # Imports the numpy library with the alias 'np' for fast mathematical and numerical operations.
import os # Imports the 'os' library, which lets our script interact with the operating system (like finding file paths).

# Defines the main function that will contain all of our script's logic.
def desurvey_drillholes():
    """
    Loads collar and sample data, calculates XYZ coordinates for each sample,
    and saves a clean result with X, Y, Z, Ag, Pb, and Zn columns to a new CSV file.
    """
    # Prints a message to the user's console to show that the script has started running.
    print("--- Drillhole Desurvey Script Started ---")

    # --- SETUP FILE PATHS ---
    # Gets the absolute path of the directory where this exact Python script is located.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Defines the name of the drillhole collar data file as a string variable.
    collar_filename = 'MPA_Collar_20240227.csv'
    # Defines the name of the drillhole samples data file as a string variable.
    samples_filename = 'MPA_Samples_BD_20240227.csv'
    # Defines the desired name for our final output file.
    output_filename = 'clean_xyz_ag_pb_zn_data.csv' # Name to reflect new metals.

    # Creates the full, operating-system-agnostic path to the collar file by joining the script's directory and the filename.
    collar_file_path = os.path.join(script_dir, collar_filename)
    # Creates the full path to the samples file.
    samples_file_path = os.path.join(script_dir, samples_filename)
    # Creates the full path for our final output file.
    output_file_path = os.path.join(script_dir, output_filename)
    
    # --- 2. LOAD DATA ---
    try: # Starts a 'try' block, which lets us attempt code that might cause an error (like a file not being found).
        print(f"Loading collar data from '{collar_file_path}'...") # Informs the user what file is being loaded.
        collar_df = pd.read_csv(collar_file_path) # Reads the collar CSV file into a pandas DataFrame called 'collar_df'.
        print(f"Loading samples data from '{samples_file_path}'...") # Informs the user about the next file being loaded.
        samples_df = pd.read_csv(samples_file_path) # Reads the samples CSV file into a pandas DataFrame called 'samples_df'.
        print("Files loaded successfully.") # Confirms that both files were found and loaded without errors.
    except FileNotFoundError as e: # If a 'FileNotFoundError' occurs in the 'try' block, this code will run.
        print(f"\nERROR: Could not find a file! - {e}") # Prints a helpful error message, including the system error 'e'.
        print("Please make sure your CSV files are in the same 'mina' folder as 'main.py'.") # Gives the user instructions to fix the error.
        return # Exits the function immediately because the script cannot continue without the data.

    # --- 3. MERGE DATA ---
    print("\nMerging collar and sample data...") # Informs the user about the current step.
    required_collar_cols = ['HoleID', 'Easting', 'Northing', 'Elevation', 'Dip', 'Azimuth'] # Creates a list of required columns for the collar file.
    required_sample_cols = ['HoleID', 'From_m', 'To_m', 'Ag_ppm', 'Pb_pct', 'Zn_pct'] # Checks for Ag, Pb and Zn columns.
    
    # This 'if' statement checks if all required columns are present in their respective DataFrames.
    if not all(col in collar_df.columns for col in required_collar_cols) or \
       not all(col in samples_df.columns for col in required_sample_cols):
        print("ERROR: One of your files is missing a required column.") # If a column is missing, prints an error message.
        return # Exits the function because the script can't run without these specific columns.
        
    # Combines the two DataFrames into one called 'merged_df'.
    # It uses 'HoleID' as the key and a 'left' merge to keep all sample rows.
    merged_df = pd.merge(samples_df, collar_df, on='HoleID', how='left')
    
    # Checks if any row in the 'Easting' column is empty (null/NaN), which indicates a failed merge for that sample.
    if merged_df['Easting'].isnull().any():
        print("WARNING: Some samples did not have a matching HoleID in the collar file. These rows will be dropped.") # Warns the user.
        merged_df.dropna(subset=['Easting'], inplace=True) # Removes any rows where 'Easting' is null, as they can't be processed.

    # --- 4. CALCULATE SAMPLE XYZ COORDINATES ---
    print("Calculating XYZ coordinates for each sample midpoint...") # Informs the user about the calculation step.
    merged_df['Midpoint_Depth'] = (merged_df['From_m'] + merged_df['To_m']) / 2 # Calculates the midpoint depth for each sample and creates a new column.
    merged_df['Azimuth_rad'] = np.radians(merged_df['Azimuth']) # Converts the 'Azimuth' from degrees to radians for trig functions and creates a new column.
    merged_df['Dip_rad'] = np.radians(merged_df['Dip']) # Converts the 'Dip' from degrees to radians and creates a new column.
    
    # Calculates the change in X (Easting) based on the depth, azimuth, and dip.
    merged_df['delta_X'] = merged_df['Midpoint_Depth'] * np.sin(merged_df['Azimuth_rad']) * np.cos(merged_df['Dip_rad'])
    # Calculates the change in Y (Northing).
    merged_df['delta_Y'] = merged_df['Midpoint_Depth'] * np.cos(merged_df['Azimuth_rad']) * np.cos(merged_df['Dip_rad'])
    # Calculates the change in Z (Elevation). A negative dip correctly results in a negative delta_Z.
    merged_df['delta_Z'] = merged_df['Midpoint_Depth'] * np.sin(merged_df['Dip_rad'])
    
    # Calculates the final X coordinate of the sample by adding the change in X to the collar's Easting.
    merged_df['Sample_X'] = merged_df['Easting'] + merged_df['delta_X']
    # Calculates the final Y coordinate of the sample.
    merged_df['Sample_Y'] = merged_df['Northing'] + merged_df['delta_Y']
    # Calculates the final Z coordinate of the sample.
    merged_df['Sample_Z'] = merged_df['Elevation'] + merged_df['delta_Z']
    print("Calculations complete.") # Confirms that the calculations are finished.

    # --- 5. CREATE AND SAVE CLEAN FINAL OUTPUT ---
    print("\nCreating final, clean output file with X, Y, Z, Ag, Pb, and Zn...")

    # Step A: Selects only the final desired columns and creates a new, clean DataFrame called 'final_df'.
    # Includes Pb_pct and Zn_pct in the final selection.
    final_df = merged_df[['Sample_X', 'Sample_Y', 'Sample_Z', 'Ag_ppm', 'Pb_pct', 'Zn_pct']].copy()

    # Step B: Stores the number of rows before cleaning the data.
    rows_before_cleaning = len(final_df)
    
    # Defines the list of metal columns to check for missing values.
    metal_columns = ['Ag_ppm', 'Pb_pct', 'Zn_pct']
    # Removes a row ONLY IF all of the specified metal columns are empty (NaN) for that row.
    final_df.dropna(subset=metal_columns, how='all', inplace=True)
    
    # Stores the number of rows after the cleaning process.
    rows_after_cleaning = len(final_df)
    
    # Calculates how many rows were removed during cleaning.
    rows_removed = rows_before_cleaning - rows_after_cleaning
    # Checks if any rows were actually removed.
    if rows_removed > 0:
        #Message now explains the new cleaning logic.
        print(f"Cleaned data: Removed {rows_removed} rows that had no values for Ag, Pb, or Zn.")
    else: # If no rows were removed.
        # This confirms that the data was already clean.
        print("Data is already clean. No rows with missing metal values were found.")

    # Step C: Saves the 'final_df' DataFrame to a CSV file at the specified path.
    # index=False prevents writing row numbers, and float_format rounds numbers to 3 decimal places.
    final_df.to_csv(output_file_path, index=False, float_format='%.3f')

    print("\n--- All Done! ---") # Prints a final success message.
    print(f"Successfully created '{output_filename}' in your 'mina' folder.") # Tells the user where the file was saved.
    print(f"It contains {rows_after_cleaning} data points with XYZ and metal grades, ready for kriging.") # UPDATED: More general success message.

# This is a standard Python entry point.
# The code inside this 'if' statement will only run when the script is executed directly.
if __name__ == "__main__":
    # Calls the main function to start the entire process.
    desurvey_drillholes()