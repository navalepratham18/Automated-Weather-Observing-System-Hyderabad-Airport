# src/merging.py
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

AGGREGATION_RULES = {
    'Pressure_QNH': 'mean',
    'Temperature': 'mean',
    'Humidity': 'mean',
    'DewPoint': 'mean',
    'Visibility_MOR': 'mean',
    'RVR': 'mean',
    'Wind_Speed': 'mean',
    'Cloud_Base_1': 'mean',
    'Cloud_Base_2': 'mean',
    'Cloud_Base_3': 'mean',
    'Wind_Gust': 'max',
    'Rain_Intensity': 'max', 
    'Rain_Sum': 'max',
    'Present_Weather_Code': 'last',
    'OCTA1': 'last',
    'OCTA2': 'last',
    'OCTA3': 'last',
    'Wind_Dir': 'last' 
}

def extract_date_from_path(file_path: Path) -> str:
    """Extracts the exact YYYY-MM-DD from the folder structure and filename."""
    try:
        # Example: data/02_interim/2025/Oct/PTU_RWY09L_01.csv
        month_str = file_path.parent.name
        year_str = file_path.parent.parent.name
        day_str = file_path.stem.split('_')[-1]
        
        # Map 3-letter month names to standard numeric strings
        month_map = {
            "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", 
            "May": "05", "Jun": "06", "Jul": "07", "Aug": "08", 
            "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
        }
        
        # If it's already a number like '10', it falls back to that. Otherwise uses the map.
        month_num = month_map.get(month_str, month_str)
        
        if len(year_str) == 4 and len(day_str) == 2:
            return f"{year_str}-{month_num}-{day_str}"
        return ""
    except Exception:
        return ""

def align_and_merge_daily_files(daily_files: list[Path], output_dir: Path, date_str: str):
    if not daily_files:
        return

    dataframes = []
    
    for file in daily_files:
        try:
            df = pd.read_csv(file)
            if 'Datetime' not in df.columns or df.empty:
                continue
                
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
            df = df.dropna(subset=['Datetime'])
            if df.empty:
                continue
            
            # Fix: Strip hidden spaces and bad characters before numeric conversion
            for col in df.columns:
                if col in AGGREGATION_RULES and AGGREGATION_RULES[col] in ['mean', 'max']:
                    # Clean out everything except numbers, decimals, and negative signs
                    df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['Datetime'] = df['Datetime'].dt.floor('min')
            
            agg_dict = {col: AGGREGATION_RULES[col] for col in df.columns if col in AGGREGATION_RULES}
            if not agg_dict: 
                agg_dict = {col: 'last' for col in df.columns if col != 'Datetime'}
                
            df = df.groupby('Datetime').agg(agg_dict).reset_index()
            df.set_index('Datetime', inplace=True)
            dataframes.append(df)
            
        except Exception as e:
            logging.warning(f"Skipping {file.name}: {e}")

    if not dataframes:
        return

    # Safely merge without column duplication
    merged_df = pd.concat(dataframes, axis=1, join='outer')
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].reset_index()
    merged_df = merged_df.sort_values('Datetime')
    
    for col in merged_df.columns:
        if col != 'Datetime' and not col.startswith('is_observed_'):
            merged_df[f'is_observed_{col}'] = merged_df[col].notna().astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"MERGED_{date_str}.csv"
    merged_df.to_csv(output_file, index=False)
    logging.info(f"Created physics-aware daily master: {output_file.name} ({len(merged_df)} rows)")