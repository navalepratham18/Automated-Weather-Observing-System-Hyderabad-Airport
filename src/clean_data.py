# src/clean_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def cap_outliers_iqr(df, column, multiplier=3.0):
    """Caps extreme outliers using the IQR method (multiplier=3.0 for extreme outliers)."""
    if column not in df.columns or df[column].isnull().all():
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    
    # Cap the values
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

def clean_master_dataset():
    input_path = Path("data/04_master/MASTER_DATASET_IMPUTED.csv")
    output_path = Path("data/04_master/MASTER_DATASET_CLEANED.csv")
    
    if not input_path.exists():
        logging.error(f"Cannot find {input_path}")
        return

    logging.info("Loading imputed master dataset...")
    df = pd.read_csv(input_path, parse_dates=['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

    # DROP RAW WIND_DIR: We already have the sin/cos vectors
    if 'Wind_Dir' in df.columns:
        df = df.drop(columns=['Wind_Dir'])

    # 1. Continuous Variables: IQR Capping & Rolling Median for micro-gaps
    continuous_cols = ['Pressure_QNH', 'Temperature', 'Humidity', 'DewPoint', 'Visibility_MOR', 'RVR']
    logging.info("Cleaning continuous variables (IQR Capping & Rolling Median)...")
    for col in continuous_cols:
        if col in df.columns:
            df = cap_outliers_iqr(df, col)
            # Fill remaining small gaps with a 30-minute rolling median
            rolling_median = df[col].rolling(window=30, min_periods=1, center=True).median()
            df[col] = df[col].fillna(rolling_median)
            # If large gaps still exist, forward fill up to 3 hours, then global median as absolute last resort
            df[col] = df[col].ffill(limit=180).fillna(df[col].median())

    # 2. Wind Variables: Extreme outlier capping and Zero-fill (Relying on observation masks)
    wind_cols = ['Wind_Speed', 'Wind_Gust', 'Wind_Dir_sin', 'Wind_Dir_cos', 'Wind_Dir_Imputed']
    logging.info("Handling missing wind data (Zero-fill for masked features)...")
    for col in wind_cols:
        if col in df.columns:
            if col in ['Wind_Speed', 'Wind_Gust']:
                df = cap_outliers_iqr(df, col)
            df[col] = df[col].fillna(0)

    # 3. Categorical/State Variables: Forward fill large persistent gaps
    state_cols = ['Present_Weather_Code', 'OCTA1', 'OCTA2', 'OCTA3']
    logging.info("Cleaning state variables (Forward Fill)...")
    for col in state_cols:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0) # 0 as fallback if start of dataset is NaN

    # 4. Precipitation & Cloud Bases: Logical defaults
    logging.info("Enforcing logical defaults for Clouds and Rain...")
    for col in ['Cloud_Base_1', 'Cloud_Base_2', 'Cloud_Base_3']:
        if col in df.columns:
            df[col] = df[col].fillna(25000) # Clear skies
            
    for col in ['Rain_Intensity', 'Rain_Sum']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    logging.info(f"Dataset cleaned. Saving to {output_path.name}...")
    df.to_csv(output_path, index=False)
    
    # Verification check
    missing_counts = df.isnull().sum()
    remaining_cols = missing_counts[missing_counts > 0]
    
    if not remaining_cols.empty:
        logging.warning(f"Total remaining NaNs in dataset: {remaining_cols.sum()}")
        logging.warning(f"Columns with NaNs:\n{remaining_cols}")
    else:
        logging.info("Dataset completely clean. 0 NaNs.")

if __name__ == "__main__":
    clean_master_dataset()