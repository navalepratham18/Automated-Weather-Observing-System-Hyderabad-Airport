import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
DATA_PATH = "data/04_master/MASTER_DATASET_CLEANED_V2.csv"
TARGET_COL = "Cloud_Base_1"
LOOKBACK = 120
HORIZON = 120
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. VALUE-BASED WEIGHTED MSE LOSS ---
class WeightedMSELoss(nn.Module):
    def __init__(self, target_scaler):
        super(WeightedMSELoss, self).__init__()
        self.target_scaler = target_scaler

    def forward(self, pred, target):
        # Inverse transform scaled target back to real feet to determine weight tier
        target_np = target.detach().cpu().numpy().reshape(-1, 1)
        target_feet = self.target_scaler.inverse_transform(target_np).flatten()

        # Construct weight tensor based on target altitude thresholds
        weights_list = []
        for val in target_feet:
            if val < 2500.0:
                weights_list.append(5.0)   # High priority for low cloud bases
            elif val < 5000.0:
                weights_list.append(3.0)   # Medium priority for marginal bases
            else:
                weights_list.append(1.0)   # Standard priority for clear/high skies

        weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=pred.device).view(-1, 1)
        
        squared_errors = (pred - target) ** 2
        weighted_loss = weights_tensor * squared_errors
        return torch.mean(weighted_loss)

# --- 3. DATASET CLASS ---
class WeatherDataset(Dataset):
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

# --- 4. MODEL ARCHITECTURE ---
class WeatherLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def main():
    print("INFO: Loading Phase 2 cleaned dataset for FIX 3 (Sample Weighting Experiment)...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
    
    df = pd.read_csv(DATA_PATH)
    
    # Cyclic time features
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24.0)
        df = df.drop(columns=['timestamp'])
    
    obs_mask_col = f'is_observed_{TARGET_COL}'
    obs_mask = df[obs_mask_col].values if obs_mask_col in df.columns else np.ones(len(df))

    df = df.select_dtypes(include=[np.number])
    target_col = TARGET_COL if TARGET_COL in df.columns else df.columns[0]

    features = df.drop(columns=[target_col]).values
    raw_targets = df[[target_col]].values

    # Chronological Split (70% train, 15% val, 15% test)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train_raw, y_train_raw = features[:train_end], raw_targets[:train_end]
    X_val_raw, y_val_raw     = features[train_end:val_end], raw_targets[train_end:val_end]
    X_test_raw, y_test_raw   = features[val_end:], raw_targets[val_end:]

    mask_train = obs_mask[:train_end]
    mask_val   = obs_mask[train_end:val_end]
    mask_test  = obs_mask[val_end:]

    print("INFO: Fitting scalers STRICTLY on training split...")
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train = feature_scaler.fit_transform(X_train_raw)
    y_train = target_scaler.fit_transform(y_train_raw)

    X_val = feature_scaler.transform(X_val_raw)
    y_val = target_scaler.transform(y_val_raw)

    X_test = feature_scaler.transform(X_test_raw)
    y_test = target_scaler.transform(y_test_raw)

    train_dataset = WeatherDataset(X_train, y_train, mask_train, LOOKBACK, HORIZON)
    val_dataset   = WeatherDataset(X_val, y_val, mask_val, LOOKBACK, HORIZON)
    test_dataset  = WeatherDataset(X_test, y_test, mask_test, LOOKBACK, HORIZON)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"INFO: Building model on {DEVICE} for Fix 3 with Stratified Sample Weighting...")
    input_dim = features.shape[1]
    model = WeatherLSTM(input_dim=input_dim).to(DEVICE)
    
    # Custom loss initialized with target scaler to map thresholds in real feet
    criterion = WeightedMSELoss(target_scaler=target_scaler)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/v8_fix3_weighted_lstm.pth"

    print(f"INFO: Starting Training on {DEVICE} (Max Epochs: {EPOCHS})...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(X_batch).view(-1, 1)
            y_batch = y_batch.view(-1, 1)
            
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
                preds = model(X_batch).view(-1, 1)
                y_batch = y_batch.view(-1, 1)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"INFO: Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best Fix 3 model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("WARNING: Early stopping triggered. No improvement for 10 epochs.")
                break

    print("INFO: Evaluating Fix 3 Predictions on Genuine Observed Test Targets...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    preds_list, trues_list, mask_list = [], [], []
    with torch.no_grad():
        for X_batch, y_batch, m_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch).view(-1, 1)
            preds_list.append(preds.cpu().numpy())
            trues_list.append(y_batch.numpy())
            mask_list.append(m_batch.numpy())

    preds_scaled = np.vstack(preds_list)
    trues_scaled = np.vstack(trues_list)
    masks = np.concatenate(mask_list)

    preds_orig = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    trues_orig = target_scaler.inverse_transform(trues_scaled.reshape(-1, 1)).flatten()

    obs_indices = np.where(masks == 1)[0]
    eval_preds = preds_orig[obs_indices] if len(obs_indices) > 0 else preds_orig
    eval_trues = trues_orig[obs_indices] if len(obs_indices) > 0 else trues_orig

    mse = np.mean((eval_preds - eval_trues) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(eval_preds - eval_trues))

    persistence_preds = eval_trues[:-HORIZON] if len(eval_trues) > HORIZON else eval_trues
    persistence_actuals = eval_trues[HORIZON:] if len(eval_trues) > HORIZON else eval_trues
    baseline_rmse = np.sqrt(np.mean((persistence_preds - persistence_actuals) ** 2))

    print("\n" + "="*60)
    print("  FIX 3: SAMPLE WEIGHTING EXPERIMENT RESULTS")
    print("="*60)
    print(f"Test RMSE: {rmse:.4f} ft")
    print(f"Test MAE:  {mae:.4f} ft")
    print(f"Baseline Persistence RMSE: {baseline_rmse:.2f} ft")
    print("="*60)

if __name__ == "__main__":
    main()