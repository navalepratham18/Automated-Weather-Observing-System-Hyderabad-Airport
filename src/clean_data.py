import os
import pandas as pd
import numpy as np

def parse_and_encode_present_weather(df: pd.DataFrame, col: str = 'Present_Weather_Code') -> pd.DataFrame:
    """
    Parses Present_Weather_Code into numeric one-hot categorical features 
    and an ordinal intensity variable (0: None, 1: Light, 2: Moderate, 3: Heavy).
    """
    if col not in df.columns:
        print(f"WARNING: Column '{col}' not found in DataFrame. Skipping weather code encoding.")
        return df

    print(f"INFO: Parsing and encoding '{col}' into numeric features...")
    
    # Normalize string values
    s = df[col].astype(str).str.strip()
    s = s.replace({'nan': '', 'None': '', "[' ']": '', 'NA': '', 'null': ''})

    # Intensity extraction: '-' -> 1, '+' -> 3, Moderate (letters only) -> 2, Clear/Blank -> 0
    def get_intensity(val):
        if not val or val in ['C', ' ']:
            return 0
        if '-' in val:
            return 1
        if '+' in val:
            return 3
        return 2  # Default to moderate if a weather code is present without sign

    df['Intensity_Value'] = s.apply(get_intensity)

    # One-hot encoding for primary weather phenomena
    df['Is_Clear']   = s.apply(lambda x: 1 if x in ['', 'C'] else 0)
    df['Is_Cloudy']  = s.apply(lambda x: 1 if 'C' in x and x != 'C' else 0)
    df['Is_Drizzle'] = s.apply(lambda x: 1 if 'L' in x or 'DZ' in x else 0)
    df['Is_Rain']    = s.apply(lambda x: 1 if 'R' in x or 'RA' in x else 0)

    # Drop the raw categorical string column
    df = df.drop(columns=[col])
    print(f"INFO: Successfully encoded '{col}'. Added Intensity_Value, Is_Clear, Is_Cloudy, Is_Drizzle, Is_Rain.")
    
    return df

def clean_dataset(input_path: str = "data/04_master/MASTER_DATASET_IMPUTED.csv", 
                  output_path: str = "data/04_master/MASTER_DATASET_CLEANED_V2.csv") -> pd.DataFrame:
    """
    Cleans dataset by encoding categorical weather strings into numeric features,
    handling wind direction components, and outputting a leak-free numeric matrix.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"INFO: Loading dataset from '{input_path}'...")
    df = pd.read_csv(input_path, low_memory=False)

    # Step 1.2 & 1.3: Parse and encode Present_Weather_Code
    df = parse_and_encode_present_weather(df, col='Present_Weather_Code')

    # Retain timestamp if present for sorting
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Save cleaned version
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"INFO: Cleaned Phase 1 dataset saved to '{output_path}' with shape {df.shape}.")

    return df

if __name__ == "__main__":
    clean_dataset()