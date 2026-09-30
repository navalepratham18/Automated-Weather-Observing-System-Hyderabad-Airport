import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
DATA_PATH = "data/04_master/MASTER_DATASET_CLEANED.csv"  # Adjust if your path differs
TARGET_COL = "Cloud_Base_1"                      # Target column to predict
LOOKBACK = 120                                   # 120-minute lookback window
HORIZON = 120                                    # 120-minute forecast horizon
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATASET CLASS ---
class WeatherDataset(Dataset):
    def __init__(self, X, y, lookback, horizon):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.lookbatch] if hasattr(self, 'lookbatch') else self.X[idx : idx + self.lookback]
        target_val = self.y[idx + self.lookback + self.horizon - 1]
        return x_window, target_val

# --- 3. MODEL ARCHITECTURE ---
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
        return out  # Shape: [batch_size, 1]

def main():
    print("INFO: Loading dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Check your file path.")
    
    df = pd.read_csv(DATA_PATH)
    
    # --- FEATURE CLEANING & DATETIME HANDLING ---
    # 1. Extract cyclic time features if timestamp exists, then DROP the raw datetime column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24.0)
        df = df.drop(columns=['timestamp'])
    
    # Also drop common non-predictive or leaky ID/string columns if they exist
    cols_to_drop = [col for col in ['id', 'station_id', 'Unnamed: 0'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Coerce everything else to strict numeric types and handle missing values
    df = df.select_dtypes(include=[np.number]).ffill().bfill().fillna(0)
    
    if TARGET_COL not in df.columns:
        target_col = df.columns[0]
        print(f"WARNING: Target '{TARGET_COL}' not found. Using '{target_col}' instead.")
    else:
        target_col = TARGET_COL

    # Clip extreme clear-sky artificial fill values (e.g., 25,000 ft caps)
    if target_col in df.columns:
        upper_cap = df[target_col].quantile(0.98)
        df[target_col] = df[target_col].clip(upper=upper_cap)

    features = df.drop(columns=[target_col]).values
    raw_targets = df[[target_col]].values

    print(f"INFO: Cleaned features shape: {features.shape}. Training without raw timestamps or unnecessary columns.")

    print("INFO: Scaling features and targets...")
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = target_scaler.fit_transform(raw_targets)

    # Chronological Split (70% train, 15% val, 15% test)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = features_scaled[:train_end], targets_scaled[:train_end]
    X_val, y_val = features_scaled[train_end:val_end], targets_scaled[train_end:val_end]
    X_test, y_test = features_scaled[val_end:], targets_scaled[val_end:]

    train_dataset = WeatherDataset(X_train, y_train, LOOKBACK, HORIZON)
    val_dataset = WeatherDataset(X_val, y_val, LOOKBACK, HORIZON)
    test_dataset = WeatherDataset(X_test, y_test, LOOKBACK, HORIZON)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"INFO: Building model on {DEVICE}...")
    input_dim = features_scaled.shape[1]
    model = WeatherLSTM(input_dim=input_dim).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_path = "models/v3_cleaned_features.pth"

    print(f"INFO: Starting Training on {DEVICE} (Max Epochs: {EPOCHS})...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(X_batch).view(-1, 1)
            y_batch = y_batch.view(-1, 1)
            
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
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
            print(f"  -> New best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("WARNING: Early stopping triggered. No improvement for 10 epochs.")
                break

    print("INFO: Evaluating Predictions on Test Set...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    preds_list = []
    trues_list = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch).view(-1, 1)
            preds_list.append(preds.cpu().numpy())
            trues_list.append(y_batch.numpy())

    preds_scaled = np.vstack(preds_list)
    trues_scaled = np.vstack(trues_list)

    # Inverse transform back to original units (feet)
    preds_orig = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1))
    trues_orig = target_scaler.inverse_transform(trues_scaled.reshape(-1, 1))

    mse = np.mean((preds_orig - trues_orig) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_orig - trues_orig))

    # Compute Persistence Baseline on the test split
    persistence_preds = trues_orig[:-HORIZON]
    persistence_actuals = trues_orig[HORIZON:]
    baseline_rmse = np.sqrt(np.mean((persistence_preds - persistence_actuals) ** 2))

    print("\n" + "="*50)
    print("  LSTM + CLEANED FEATURES RESULTS")
    print("="*50)
    print(f"Test RMSE: {rmse:.4f} ft")
    print(f"Test MAE:  {mae:.4f} ft")
    print(f"Baseline Persistence RMSE to beat: {baseline_rmse:.2f} ft")
    print("="*50)

if __name__ == "__main__":
    main()