import pandas as pd
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_config(config_path="config.yaml"):
    # Force UTF-8 so Windows doesn't corrupt the degree symbol
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def convert_his_to_csv(
    file_path: Path,
    output_dir: Path,
    config: dict,
    raw_dir: Path | None = None,
):
    """Converts a single .his file to .csv securely and normalizes the schema."""
    try:
        df = pd.read_csv(
            file_path, 
            sep=config['ingestion']['delimiter'], 
            skiprows=config['ingestion']['skip_rows'],
            on_bad_lines='error',
            encoding='utf-8'
        )
        
        mapping = config.get('schema_mapping', {})
        columns_to_keep = [col for col in df.columns if col in mapping]
        
        if not columns_to_keep:
            logging.warning(f"File {file_path.name} contains no recognized columns. Skipping.")
            return

        df = df[columns_to_keep]
        df = df.rename(columns=mapping)
        
        relative_dir = file_path.parent.relative_to(raw_dir) if raw_dir else Path()
        target_dir = output_dir / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file = target_dir / f"{file_path.stem}.csv"
        
        df.to_csv(output_file, index=False)
        logging.info(f"Successfully converted and normalized: {file_path.name}")
        
    except Exception as e:
        logging.error(f"Failed to process {file_path.name}: {e}")
        raise