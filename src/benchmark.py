# src/benchmark.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def enforce_gpu():
    """Forces a strict check for CUDA availability before allowing the pipeline to run."""
    logging.info("Verifying CUDA GPU availability for XGBoost...")
    try:
        # Create a microscopic dummy dataset to test VRAM allocation
        X_dummy = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_dummy = np.array([1.0, 0.0])
        dtrain_dummy = xgb.DMatrix(X_dummy, label=y_dummy)
        
        # Force the device to CUDA
        params = {'tree_method': 'hist', 'device': 'cuda'}
        _ = xgb.train(params, dtrain_dummy, num_boost_round=1)
        logging.info("SUCCESS: RTX 4050 GPU is active and locked in for training.")
    except Exception as e:
        logging.error(f"FATAL: GPU not detected or CUDA failed to initialize. Error details: {e}")
        logging.error("Halting execution. Ensure NVIDIA drivers and CUDA toolkit are installed.")
        sys.exit(1)

def run_benchmark():
    # 1. Enforce GPU presence before doing any data lifting
    enforce_gpu()

    master_path = Path("data/04_master/MASTER_DATASET_CLEANED.csv")
    if not master_path.exists():
        logging.error("Cleaned master dataset not found. Run clean_data.py first.")
        return

    logging.info("Loading cleaned master dataset for advanced feature engineering...")
    df = pd.read_csv(master_path, parse_dates=['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

    target_col = 'Visibility_MOR'
    
    # 2. Feature Engineering: Momentum, Rates of Change, and Lags
    logging.info("Engineering momentum and atmospheric delta features...")
    features = []
    
    cols_to_engineer = ['Pressure_QNH', 'Temperature', 'Humidity', 'DewPoint', 'Visibility_MOR']
    
    for col in cols_to_engineer:
        if col in df.columns:
            # Standard Lags
            for lag in [1, 15, 60]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                features.append(f'{col}_lag_{lag}')
                
            # Rolling Means
            df[f'{col}_roll_mean_60'] = df[col].shift(1).rolling(window=60).mean()
            features.append(f'{col}_roll_mean_60')
            
            # Atmospheric Momentum / Rate of Change (Delta over 60 mins)
            df[f'{col}_delta_60'] = df[col] - df[col].shift(60)
            features.append(f'{col}_delta_60')

    # Target shift: 60-minute (1-hour) prediction horizon
    horizon = 60
    df['Target'] = df[target_col].shift(-horizon)
    df['Persistence_Target'] = df[target_col] # Current visibility as baseline

    # Add temporal features
    df['Hour'] = df['Datetime'].dt.hour
    df['Month'] = df['Datetime'].dt.month
    features.extend(['Hour', 'Month'])

    # Drop rows with NaNs across features, target, and persistence baseline
    ml_df = df.dropna(subset=features + ['Target', 'Persistence_Target']).copy()

    # 3. Train/Test Split (Time-based split: 80% train, 20% test)
    split_idx = int(len(ml_df) * 0.8)
    train_df = ml_df.iloc[:split_idx]
    test_df = ml_df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df['Target']
    X_test, y_test = test_df[features], test_df['Target']

    logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    results = {}

    # --- BASELINE: Persistence Model (60m Horizon) ---
    pers_preds = test_df['Persistence_Target'].values
    results['Persistence'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, pers_preds)),
        'MAE': mean_absolute_error(y_test, pers_preds)
    }

    # --- MODEL 1: Ridge Regression ---
    logging.info("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    results['Ridge'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, ridge_preds)),
        'MAE': mean_absolute_error(y_test, ridge_preds)
    }

    # --- MODEL 2: LightGBM ---
    logging.info("Training LightGBM on CPU cores...")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    results['LightGBM'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, lgb_preds)),
        'MAE': mean_absolute_error(y_test, lgb_preds)
    }

    # --- MODEL 3: XGBoost (RTX 4050 GPU Acceleration) ---
    logging.info("Training XGBoost strictly on RTX 4050 GPU...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 6,
        'tree_method': 'hist',
        'device': 'cuda'
    }
    
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)
    xgb_preds = xgb_model.predict(dtest)
    results['XGBoost (GPU)'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, xgb_preds)),
        'MAE': mean_absolute_error(y_test, xgb_preds)
    }

    # --- Print Benchmark Summary ---
    print("\n" + "="*50)
    print(" 60-MINUTE HORIZON BENCHMARK RESULTS")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"{model_name:<16} | RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()