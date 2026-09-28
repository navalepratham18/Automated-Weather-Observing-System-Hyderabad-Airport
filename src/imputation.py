# src/imputation.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def apply_physics_imputation(master_path: Path, output_path: Path):
    logging.info(f"Loading master dataset from {master_path}...")
    try:
        df = pd.read_csv(master_path)
    except FileNotFoundError:
        logging.error(f"Master dataset not found at {master_path}. Run Phase 3 first.")
        return

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    # 1. Structural Missingness (Clear Skies & No Rain)
    cloud_cols = ['Cloud_Base_1', 'Cloud_Base_2', 'Cloud_Base_3']
    for col in cloud_cols:
        if col in df.columns:
            # 25,000 feet is a standard aviation ceiling for "Clear/No Clouds detected"
            df[col] = df[col].fillna(25000)
            
    rain_cols = ['Rain_Intensity', 'Rain_Sum']
    for col in rain_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # 2. Categorical Variables (Forward Fill, max 15 minutes)
    cat_cols = ['Present_Weather_Code', 'OCTA1', 'OCTA2', 'OCTA3']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].ffill(limit=15)

    # 3. Wind Direction (Sine/Cosine Transformation)
    if 'Wind_Dir' in df.columns:
        # BULLETPROOFING: Force column to numeric. Any lingering text junk becomes NaN.
        df['Wind_Dir'] = pd.to_numeric(df['Wind_Dir'], errors='coerce')
        
        # Convert degrees to radians and extract vector components
        rad = df['Wind_Dir'] * np.pi / 180
        df['Wind_Dir_sin'] = np.sin(rad)
        df['Wind_Dir_cos'] = np.cos(rad)
        
        # Interpolate the vectors (max 15 minutes)
        df['Wind_Dir_sin'] = df['Wind_Dir_sin'].interpolate(method='linear', limit=15)
        df['Wind_Dir_cos'] = df['Wind_Dir_cos'].interpolate(method='linear', limit=15)
        
        # Reconstruct the angle for human readability (optional, LSTM will use vectors)
        df['Wind_Dir_Imputed'] = (np.arctan2(df['Wind_Dir_sin'], df['Wind_Dir_cos']) * 180 / np.pi) % 360

    # 4. Continuous Variables (Linear Interpolation, max 15 minutes)
    continuous_cols = [
        'Pressure_QNH', 'Temperature', 'Humidity', 'DewPoint', 
        'Visibility_MOR', 'RVR', 'Wind_Speed', 'Wind_Gust'
    ]
    for col in continuous_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=15)

    # Save Imputed Master Dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logging.info(f"Physics-aware imputation complete. Saved to {output_path.name}")
    
    # Print diagnostics for genuine long-term sensor outages
    missing_after = df.isnull().sum()
    logging.info("Remaining missing values (Gaps > 15 mins):")
    for col, count in missing_after.items():
        if count > 0 and not col.startswith('is_observed_') and col != 'Wind_Dir':
            logging.info(f"  {col}: {count} rows")