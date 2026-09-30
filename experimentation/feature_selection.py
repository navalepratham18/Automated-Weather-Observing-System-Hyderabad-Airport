import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/04_master/MASTER_DATASET_CLEANED_V2.csv"
TARGETS = ["Cloud_Base_1", "Visibility_MOR"]

def run_feature_selection():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print("INFO: Loading MASTER_DATASET_CLEANED_V2.csv for feature ranking...")
    df = pd.read_csv(DATA_PATH)

    # Calculate DewPoint Depression (Temperature - DewPoint) as a physical domain signal
    if 'Temperature' in df.columns and 'DewPoint' in df.columns:
        df['DewPoint_Depression'] = df['Temperature'] - df['DewPoint']
        print("INFO: Engineered physical feature 'DewPoint_Depression' (Temp - DewPoint)...")

    # Cyclic time encoding
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24.0)
        df = df.drop(columns=['timestamp'])

    # Drop target columns and observation masks from candidates
    ignore_cols = TARGETS + [f'is_observed_{t}' for t in TARGETS] + ['is_observed_Datetime']
    candidate_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore_cols]

    print(f"INFO: Screening {len(candidate_cols)} candidate features against targets...")

    # Sample dataset (100k rows) for rapid, robust forest estimation
    sample_df = df.sample(n=min(100000, len(df)), random_state=42).dropna()
    X = sample_df[candidate_cols]

    feature_scores = {col: 0.0 for col in candidate_cols}

    for target in TARGETS:
        if target not in df.columns:
            continue
        print(f"\n--- Ranking Features for Target: '{target}' ---")
        y = sample_df[target]

        rf = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        print(f"Top 10 Features for {target}:")
        for i in range(10):
            idx = sorted_idx[i]
            col_name = candidate_cols[idx]
            score = importances[idx]
            feature_scores[col_name] += score
            print(f"  {i+1:02d}. {col_name:<30} | Importance Score: {score:.4f}")

    # Aggregate scores across both targets
    sorted_combined = sorted(feature_scores.items(), key=lambda item: item[2] if len(item) > 2 else item[1], reverse=True)

    print("\n" + "="*65)
    print("  RECOMMENDED TOP 15 CRITICAL METEOROLOGICAL SIGNALS")
    print("="*65)
    selected_features = []
    for rank, (col, score) in enumerate(sorted_combined[:15], 1):
        selected_features.append(col)
        print(f"  {rank:02d}. {col:<35} | Combined Score: {score:.4f}")
    print("="*65)

    return selected_features

if __name__ == "__main__":
    run_feature_selection()