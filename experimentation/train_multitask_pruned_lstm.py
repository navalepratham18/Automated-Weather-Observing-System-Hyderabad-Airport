import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
DATA_PATH = "data/04_master/MASTER_DATASET_CLEANED_V3.csv"
TARGET_COLS = ["Cloud_Base_1", "Visibility_MOR"]
LOOKBACK = 120
HORIZON = 120
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TOP 14 PRUNED FEATURES FROM SCREENING
PRUNED_FEATURES = [
    'OCTA1', 'Temperature', 'RVR', 'DewPoint_Depression', 'Pressure_QNH',
    'Cloud_Base_2', 'Rain_Sum', 'DewPoint', 'Humidity', 'Wind_Speed',
    'hour_cos', 'Rain_Intensity', 'hour_sin', 'Wind_Gust'
]

# --- 2. MULTI-TASK ASYMMETRIC LOSS FUNCTION ---
class MultiTaskAsymmetricLoss(nn.Module):
    def __init__(self, high_penalty_weight=3.0):
        super(MultiTaskAsymmetricLoss, self).__init__()
        self.high_penalty_weight = high_penalty_weight

    def forward(self, pred, target):
        # pred and target shape: (batch_size, 2)
        error = pred - target
        weights = torch.where(error > 0, self.high_penalty_weight, 1.0)
        loss = weights * (error ** 2)
        return torch.mean(loss)

# --- 3. MULTI-TASK DATASET CLASS ---
class MultiTaskWeatherDataset(Dataset):
    def __init__(self, X, y, mask, lookback, horizon):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.lookback]
        target_val = self.y[idx + self.lookback + self.horizon - 1]
        mask_val = self.mask[idx + self.lookback + self.horizon - 1]
        return x_window, target_val, mask_val

# --- 4. MULTI-TASK LSTM ARCHITECTURE ---
class MultiTaskWeatherLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(MultiTaskWeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        # Shared representations feed into joint predictions
        self.fc_cloud = nn.Linear(hidden_dim, 1)
        self.fc_vis = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        
        cloud_pred = self.fc_cloud(last_hidden)
        vis_pred = self.fc_vis(last_hidden)
        
        # Output shape: (batch_size, 2) -> [Cloud_Base_1, Visibility_MOR]
        return torch.cat([cloud_pred, vis_pred], dim=1)

def main():
    print("INFO: Loading cleaned dataset for MULTI-TASK PRUNED FEATURE experiment...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
    
    df = pd.read_csv(DATA_PATH)
    
    if 'Temperature' in df.columns and 'DewPoint' in df.columns and 'DewPoint_Depression' not in df.columns:
        df['DewPoint_Depression'] = df['Temperature'] - df['DewPoint']
        
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24.0)

    # Observation masks for evaluation
    obs_mask_cloud = df['is_observed_Cloud_Base_1'].values if 'is_observed_Cloud_Base_1' in df.columns else np.ones(len(df))
    obs_mask_vis = df['is_observed_Visibility_MOR'].values if 'is_observed_Visibility_MOR' in df.columns else np.ones(len(df))
    obs_masks = np.column_stack([obs_mask_cloud, obs_mask_vis])

    # Select strictly the 14 pruned features and 2 targets
    features = df[PRUNED_FEATURES].values
    raw_targets = df[TARGET_COLS].values

    print(f"INFO: Matrix dimensions reduced from 60 features down to {features.shape[1]} PRUNED features!")

    # Chronological Split (70% train, 15% val, 15% test)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train_raw, y_train_raw = features[:train_end], raw_targets[:train_end]
    X_val_raw, y_val_raw     = features[train_end:val_end], raw_targets[train_end:val_end]
    X_test_raw, y_test_raw   = features[val_end:], raw_targets[val_end:]

    mask_train = obs_masks[:train_end]
    mask_val   = obs_masks[train_end:val_end]
    mask_test  = obs_masks[val_end:]

    print("INFO: Fitting scalers STRICTLY on training split...")
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train = feature_scaler.fit_transform(X_train_raw)
    y_train = target_scaler.fit_transform(y_train_raw)

    X_val = feature_scaler.transform(X_val_raw)
    y_val = target_scaler.transform(y_val_raw)

    X_test = feature_scaler.transform(X_test_raw)
    y_test = target_scaler.transform(y_test_raw)

    train_dataset = MultiTaskWeatherDataset(X_train, y_train, mask_train, LOOKBACK, HORIZON)
    val_dataset   = MultiTaskWeatherDataset(X_val, y_val, mask_val, LOOKBACK, HORIZON)
    test_dataset  = MultiTaskWeatherDataset(X_test, y_test, mask_test, LOOKBACK, HORIZON)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"INFO: Building Multi-Task LSTM Model on {DEVICE}...")
    model = MultiTaskWeatherLSTM(input_dim=features.shape[1]).to(DEVICE)
    
    criterion = MultiTaskAsymmetricLoss(high_penalty_weight=3.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/multitask_pruned_lstm.pth"

    print(f"INFO: Starting Training on {DEVICE} (Max Epochs: {EPOCHS})...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"INFO: Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best Multi-Task model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("WARNING: Early stopping triggered. No improvement for 10 epochs.")
                break

    print("\nINFO: Evaluating Multi-Task Predictions on Genuine Observed Test Targets...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    preds_list, trues_list, mask_list = [], [], []
    with torch.no_grad():
        for X_batch, y_batch, m_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch)
            preds_list.append(preds.cpu().numpy())
            trues_list.append(y_batch.numpy())
            mask_list.append(m_batch.numpy())

    preds_scaled = np.vstack(preds_list)
    trues_scaled = np.vstack(trues_list)
    masks = np.vstack(mask_list)

    preds_orig = target_scaler.inverse_transform(preds_scaled)
    trues_orig = target_scaler.inverse_transform(trues_scaled)

    # Filter targets individually based on observation masks
    cloud_obs_idx = np.where(masks[:, 0] == 1)[0]
    vis_obs_idx   = np.where(masks[:, 1] == 1)[0]

    cloud_preds = preds_orig[cloud_obs_idx, 0]
    cloud_trues = trues_orig[cloud_obs_idx, 0]

    vis_preds = preds_orig[vis_obs_idx, 1]
    vis_trues = trues_orig[vis_obs_idx, 1]

    cloud_rmse = np.sqrt(np.mean((cloud_preds - cloud_trues) ** 2))
    cloud_mae  = np.mean(np.abs(cloud_preds - cloud_trues))

    vis_rmse = np.sqrt(np.mean((vis_preds - vis_trues) ** 2))
    vis_mae  = np.mean(np.abs(vis_preds - vis_trues))

    print("\n" + "="*65)
    print("  MULTI-TASK PRUNED LSTM TEST RESULTS")
    print("="*65)
    print(f"Cloud_Base_1  -> Test RMSE: {cloud_rmse:.4f} ft  | Test MAE: {cloud_mae:.4f} ft")
    print(f"Visibility_MOR -> Test RMSE: {vis_rmse:.4f} m   | Test MAE: {vis_mae:.4f} m")
    print("="*65)

if __name__ == "__main__":
    main()