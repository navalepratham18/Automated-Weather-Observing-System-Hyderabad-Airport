import os
import pandas as pd
import numpy as np

def standardize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects any timestamp column variant ('Datetime', 'Timestamp', etc.)
    and standardizes it to 'timestamp' in pd.Timestamp format.
    """
    possible_names = ['Datetime', 'datetime', 'timestamp', 'Timestamp', 'DATETIME', 'time', 'TIME']
    found_col = None
    for col in possible_names:
        if col in df.columns:
            found_col = col
            break

    if found_col:
        print(f"INFO: Found timestamp column '{found_col}'. Standardizing to 'timestamp'...")
        df['timestamp'] = pd.to_datetime(df[found_col], errors='coerce')
        if found_col != 'timestamp':
            df = df.drop(columns=[found_col])
    else:
        print("WARNING: No recognized timestamp column found in dataset!")

    return df

def filter_continuous_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataset to keep only the continuous 24/7 observation window 
    (March 1, 2025 to January 31, 2026), dropping early outage/fragmented blocks.
    """
    if 'timestamp' not in df.columns:
        print("WARNING: 'timestamp' column not present. Skipping period filtering.")
        return df

    initial_rows = len(df)
    
    # Drop invalid/unparseable timestamps
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)

    # Keep only records from March 1, 2025 00:00:00 onwards
    start_date = pd.Timestamp('2025-03-01 00:00:00')
    continuous_mask = df['timestamp'] >= start_date
    
    df = df[continuous_mask].reset_index(drop=True)
    
    dropped_rows = initial_rows - len(df)
    print(f"INFO: Dropped {dropped_rows} fragmented/outage rows prior to March 2025. Remaining continuous rows: {len(df)}.")
    
    return df

def apply_time_aware_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies time-aware linear interpolation and short-window forward fills, 
    and generates explicit is_imputed_* indicator flags for key predictors.
    """
    print("INFO: Applying time-aware interpolation and creating is_imputed_* flags...")
    
    target_vars = [col for col in df.columns if not col.startswith('is_observed_') 
                   and not col.startswith('is_imputed_') 
                   and col not in ['timestamp', 'Intensity_Value', 'Is_Clear', 'Is_Cloudy', 'Is_Drizzle', 'Is_Rain']]

    # Step A: Create binary is_imputed_* indicators based on missingness BEFORE filling
    for col in target_vars:
        if col in df.columns:
            df[f'is_imputed_{col}'] = df[col].isna().astype(int)

    # Step B: Perform time-indexed short interpolation
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
        
        for col in target_vars:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].interpolate(method='time', limit=15)
                df[col] = df[col].ffill(limit=15)

        df = df.reset_index()
    else:
        for col in target_vars:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].interpolate(method='linear', limit=15).ffill(limit=15)

    # Step C: Fill remaining long-term missing cloud bases with clear-sky ceiling default (25,000 ft)
    cloud_cols = [col for col in df.columns if 'Cloud_Base' in col]
    for col in cloud_cols:
        df[col] = df[col].fillna(25000.0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df

def run_imputation_pipeline(input_path: str = "data/04_master/MASTER_DATASET.csv",
                            output_path: str = "data/04_master/MASTER_DATASET_IMPUTED.csv") -> pd.DataFrame:
    """
    Executes Phase 2 imputation pipeline: Standardizes timestamps, filters continuous period,
    applies time-aware fills, and constructs explicit imputation provenance masks.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Master file not found: {input_path}")

    print(f"INFO: Loading authentic master dataset from '{input_path}'...")
    df = pd.read_csv(input_path, low_memory=False)

    # 1. Standardize timestamp column name and data type
    df = standardize_timestamp_column(df)

    # 2. Filter for continuous observation period (March 2025 – Jan 2026)
    df = filter_continuous_period(df)

    # 3. Sort chronologically
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    # 4. Time-aware imputation and indicator flag generation
    df = apply_time_aware_imputation(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"INFO: Imputed master saved to '{output_path}' with shape {df.shape}.")

    return df

if __name__ == "__main__":
    run_imputation_pipeline()