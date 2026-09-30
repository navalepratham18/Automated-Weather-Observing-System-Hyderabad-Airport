import os
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "data/04_master/MASTER_DATASET_IMPUTED.csv"
OUTPUT_FILE = "data/04_master/MASTER_DATASET_CLEANED_V3.csv"
TARGET_COLS = ["Cloud_Base_1", "Visibility_MOR"]

def clean_and_feature_engineer():
    print(f"INFO: Reading imputed dataset from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found at {INPUT_FILE}. Please verify file location.")

    df = pd.read_csv(INPUT_FILE)
    initial_rows, initial_cols = df.shape
    print(f"INFO: Initial dataset shape: {initial_rows} rows x {initial_cols} columns")

    # 1. Parse Datetime and Sort Chronologically
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.sort_values('Datetime').reset_index(drop=True)

    # 2. Add Engineered Physical Features
    print("INFO: Engineering physical domain signals...")
    
    # DewPoint Depression (Temperature - DewPoint)
    if 'Temperature' in df.columns and 'DewPoint' in df.columns:
        df['DewPoint_Depression'] = df['Temperature'] - df['DewPoint']
        print("  -> Created 'DewPoint_Depression' (Temp - DewPoint)")
    else:
        print("  -> WARNING: 'Temperature' or 'DewPoint' missing. Skipping DewPoint_Depression.")

    # Cyclic Time Features (Hour Sin/Cos)
    if 'Datetime' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['Datetime'].dt.hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['Datetime'].dt.hour / 24.0)
        print("  -> Created cyclic time encodings ('hour_sin', 'hour_cos')")

    # 3. Ensure Observation Masks Exist
    for target in TARGET_COLS:
        mask_col = f'is_observed_{target}'
        if target in df.columns and mask_col not in df.columns:
            df[mask_col] = df[target].notnull().astype(int)
            print(f"  -> Created default observation mask column '{mask_col}'")

    # 4. Save Final Phase 3 Asset (V3)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    final_rows, final_cols = df.shape
    print("\n" + "="*60)
    print("  PHASE 3 MASTER CLEANED DATASET (V3) READY")
    print("="*60)
    print(f"Output Path: {OUTPUT_FILE}")
    print(f"Final Shape: {final_rows} rows x {final_cols} columns")
    print("="*60)

if __name__ == "__main__":
    clean_and_feature_engineer()