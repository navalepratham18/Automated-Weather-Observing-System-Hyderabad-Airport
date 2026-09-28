# run_pipeline.py
from pathlib import Path
from src.ingestion import load_config, convert_his_to_csv
from src.merging import extract_date_from_path, align_and_merge_daily_files
from src.build_master import create_master_dataset
from src.imputation import apply_physics_imputation
import logging
from collections import defaultdict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the IMD-HYD data pipeline.")
    parser.add_argument('--skip-phase1', action='store_true', help="Skip converting .his to .csv")
    parser.add_argument('--skip-phase2', action='store_true', help="Skip merging daily files")
    parser.add_argument('--skip-phase3', action='store_true', help="Skip building master dataset")
    args = parser.parse_args()

    config = load_config()
    raw_dir = Path(config['paths']['raw_data'])
    interim_dir = Path(config['paths']['interim_data'])
    processed_dir = Path(config['paths']['processed_data'])
    master_dir = Path(config['paths']['master_data'])
    
    # Phase 1: Ingestion
    if not args.skip_phase1:
        logging.info("Starting Phase 1: Ingestion")
        his_files = list(raw_dir.rglob("*.his"))
        if not his_files:
            logging.warning(f"No .his files found in {raw_dir}.")
        else:
            for file in his_files:
                convert_his_to_csv(file, interim_dir, config, raw_dir)
    else:
        logging.info("Skipping Phase 1.")
        
    # Phase 2: Time-Alignment and Merging
    if not args.skip_phase2:
        logging.info("Starting Phase 2: Time-Alignment and Merging")
        csv_files = list(interim_dir.rglob("*.csv"))
        if not csv_files:
            logging.error(f"No CSVs found in {interim_dir}. Run Phase 1 first.")
        else:
            files_by_date = defaultdict(list)
            for file in csv_files:
                date_str = extract_date_from_path(file)
                if date_str:
                    files_by_date[date_str].append(file)
                    
            for date_str, daily_files in files_by_date.items():
                align_and_merge_daily_files(daily_files, processed_dir, date_str)
    else:
        logging.info("Skipping Phase 2.")
        
    # Phase 3: Build Master Dataset
    if not args.skip_phase3:
        logging.info("Starting Phase 3: Building Master Dataset")
        create_master_dataset(processed_dir, master_dir)
    else:
        logging.info("Skipping Phase 3.")

    # Phase 4: Physics-Aware Imputation
    logging.info("Starting Phase 4: Physics-Aware Imputation")
    raw_master = master_dir / "MASTER_DATASET.csv"
    imputed_master = master_dir / "MASTER_DATASET_IMPUTED.csv"
    apply_physics_imputation(raw_master, imputed_master)
        
    logging.info("Pipeline complete.")