import pandas as pd
import os

def search_parquet_files(directory, search_string, column_to_search):
    """
    Searches for a string in a specific column across all Parquet files in a directory.

    Args:
        directory (str): The path to the folder containing the .parquet files.
        search_string (str): The text you want to find.
        column_to_search (str): The name of the column to search within.
    """
    print(f"Searching for '{search_string}' in column '{column_to_search}'...")

    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    # Find all Parquet files in the directory
    parquet_files = [f for f in os.listdir(directory) if f.endswith('.parquet')]

    if not parquet_files:
        print(f"No .parquet files found in '{directory}'.")
        return

    found_something = False
    for filename in parquet_files:
        file_path = os.path.join(directory, filename)
        try:
            # Read the Parquet file
            df = pd.read_parquet(file_path)

            # Ensure the column exists in this file
            if column_to_search not in df.columns:
                continue
            
            # Search for the string and filter the DataFrame
            results = df[df[column_to_search].str.contains(search_string, case=False, na=False)]

            if not results.empty:
                print(f"\n--- Found results in file: {filename} ---")
                print(results)
                found_something = True

        except Exception as e:
            print(f"Could not process file {filename}: {e}")
            
    if not found_something:
        print("\nNo matching results found in any file.")

# --- Main part of the script ---
if __name__ == "__main__":
    # --- Configure your search here ---

    # Path from your root directory to the data folder
    data_directory = 'coco_val2014_blip2_processed/data'
    
    # String you want to find
    string_to_find = 'This wire metal rack holds several pairs'

    # Column you want to search in
    search_column = 'text'

    search_parquet_files(data_directory, string_to_find, search_column)