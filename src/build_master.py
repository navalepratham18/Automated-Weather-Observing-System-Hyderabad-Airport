# src/build_master.py
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_master_dataset(processed_dir: Path, output_dir: Path):
    """Binds all daily aligned CSVs into a single chronological master dataset."""
    daily_files = list(processed_dir.glob("MERGED_*.csv"))
    
    if not daily_files:
        logging.error(f"No daily files found in {processed_dir}")
        return

    logging.info(f"Binding {len(daily_files)} daily files into master dataset...")
    
    dataframes = []
    for file in daily_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            logging.warning(f"Failed to read {file.name}: {e}")

    # Concatenate all days
    master_df = pd.concat(dataframes, ignore_index=True)
    
    # Ensure Datetime is proper and sort chronologically
    master_df['Datetime'] = pd.to_datetime(master_df['Datetime'])
    master_df = master_df.sort_values('Datetime').reset_index(drop=True)
    
    # Drop completely empty columns (e.g., if a sensor never fired once all year)
    master_df = master_df.dropna(axis=1, how='all')

    output_dir.mkdir(parents=True, exist_ok=True)
    master_file = output_dir / "MASTER_DATASET.csv"
    master_df.to_csv(master_file, index=False)
    
    logging.info(f"Master Dataset created at {master_file.name}")
    logging.info(f"Total Rows: {len(master_df)}")
    logging.info(f"Total Columns: {len(master_df.columns)}")