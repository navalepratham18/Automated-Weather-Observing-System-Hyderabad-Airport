import os
import glob
import pandas as pd
import numpy as np

def normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strips whitespace from string columns and maps empty or invalid missing tokens
    to genuine np.nan values to ensure accurate observation flags.
    """
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            # Replace invalid/missing representations with np.nan
            invalid_tokens = ["", " ", "[' ']", "NA", "na", "null", "NULL", "None", "nan", "NaN"]
            df[col] = df[col].replace(invalid_tokens, np.nan)
    return df

def merge_interim_files(interim_dir: str = "data/03_processed", output_path: str = "data/04_master/MASTER_DATASET.csv") -> pd.DataFrame:
    """
    Merges all processed interim CSV files, normalizes missing tokens, generates 
    unambiguous observation flags (is_observed_*), and saves the authentic master reference.
    """
    print(f"INFO: Scanning for interim CSV files in '{interim_dir}'...")
    csv_files = glob.glob(os.path.join(interim_dir, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {interim_dir}")

    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file, low_memory=False)
        df_list.append(temp_df)

    merged_df = pd.concat(df_list, ignore_index=True)

    # Convert timestamp and sort chronologically
    if 'timestamp' in merged_df.columns:
        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)

    print("INFO: Normalizing missing tokens and whitespace across all columns...")
    merged_df = normalize_missing_tokens(merged_df)

    # Define key weather fields to create provenance observation masks
    value_cols = [col for col in merged_df.columns if not col.startswith('is_observed_') and col != 'timestamp']

    print("INFO: Generating precise is_observed_* provenance masks...")
    for col in value_cols:
        merged_df[f'is_observed_{col}'] = merged_df[col].notna().astype(int)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"INFO: Authentic master dataset saved to '{output_path}' with shape {merged_df.shape}.")
    
    return merged_df

if __name__ == "__main__":
    merge_interim_files()