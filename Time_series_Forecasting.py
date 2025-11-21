"""
Advanced Time Series Forecasting with Deep Learning and Explainability
Project: LSTM-based multi-step forecasting with SHAP explainability and baseline comparison

Deliverables included in this single script:
- Programmatic multivariate dataset generation (two seasonalities, trend, external regressors)
- Data preprocessing and sequence creation (look-back windows)
- Walk-forward (rolling-origin) cross-validation
- LSTM model implemented in PyTorch for multi-step forecasting
- Baseline: SARIMAX (statsmodels) and Prophet (if installed)
- Evaluation metrics: RMSE, MAE, MAPE
- Explainability: SHAP DeepExplainer applied to the trained PyTorch model (per-lag & per-regressor importance)
- Plots and model saving

Notes:
- This script is modular and intended to be run as a single file.
- Dependencies: numpy, pandas, scipy, matplotlib, scikit-learn, torch, shap, statsmodels, prophet (optional)

Run example:
    python advanced_time_series_forecasting.py --seed 42 --model lstm

"""

import os
import argparse
import math
import json
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy import signal

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Explainability
import shap

# Baseline
import statsmodels.api as sm
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# -------------------------------
# Utilities: metrics
# -------------------------------

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


# -------------------------------
# 1. Programmatically generate dataset
# -------------------------------

def generate_multivariate_series(n_steps: int = 2000,
                                 freq: float = 1.0,
                                 seasonal_periods: Tuple[int, int] = (24, 168),
                                 trend_type: str = 'linear',
                                 noise_std: float = 0.5,
                                 n_regressors: int = 2,
                                 seed: int = 0) -> pd.DataFrame:
    """
    Create a dataframe with columns: ['y', 'reg_0', 'reg_1', ...]
    y is composed of two seasonalities, a trend, and noise.
    Regressors are simulated external signals correlated with y (with lags).

    seasonal_periods: e.g., (24, 168) for daily and weekly cycles when freq=1 hr.
    """
    np.random.seed(seed)

    t = np.arange(n_steps)

    # Trend
    if trend_type == 'linear':
        trend = 0.005 * t
    elif trend_type == 'quadratic':
        trend = 1e-6 * (t ** 2)
    else:
        trend = np.zeros_like(t)

    # Two seasonal components (sinusoids with different amplitudes & harmonics)
    s1_period, s2_period = seasonal_periods
    s1 = 2.0 * np.sin(2 * np.pi * t / s1_period) + 0.5 * np.sin(2 * np.pi * t / (s1_period / 2))
    s2 = 1.5 * np.sin(2 * np.pi * t / s2_period) + 0.3 * np.sin(2 * np.pi * t / (s2_period / 3))

    # Nonlinear seasonal modulation (interaction)
    seasonal_interaction = 0.2 * s1 * np.cos(2 * np.pi * t / (s2_period * 0.9))

    base_signal = trend + s1 + s2 + seasonal_interaction

    # Heteroscedastic noise and occasional spikes
    noise = np.random.normal(scale=noise_std * (1 + 0.5 * np.sin(2 * np.pi * t / (s2_period * 2))), size=n_steps)
    spikes = np.zeros_like(t, dtype=float)
    spike_indices = np.random.choice(n_steps, size=max(1, n_steps // 200), replace=False)
    spikes[spike_indices] = np.random.normal(loc=5.0, scale=2.0, size=spike_indices.shape[0])

    y = base_signal + noise + spikes

    df = pd.DataFrame({'y': y})

    # External regressors: create correlated signals with lags
    for r in range(n_regressors):
        phase = np.random.uniform(0, 2 * np.pi)
        freq_r = np.random.uniform(0.8, 1.2)
        reg_signal = (np.sin(2 * np.pi * freq_r * t / s1_period + phase) * (1.0 + 0.5 * np.cos(2 * np.pi * t / s2_period)))
        # add some trend and noise
        reg_signal = reg_signal * (1 + 0.002 * t) + np.random.normal(scale=noise_std * 0.7, size=n_steps)
        df[f'reg_{r}'] = reg_signal

    # Add timestamps
    start = pd.Timestamp('2018-01-01')
    # Use hourly frequency for granularity
    df['ds'] = pd.date_range(start, periods=n_steps, freq='H')
    df = df.set_index('ds')

    return df


# -------------------------------
# Sequence Dataset for PyTorch
# -------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        """data: array shape (n_samples, n_features), where first column is target y"""
        self.X = []
        self.y = []
        n = data.shape[0]
        for i in range(lookback, n - horizon + 1):
            seq_x = data[i - lookback:i, :]
            seq_y = data[i:i + horizon, 0]  # predict multi-step for y only
            self.X.append(seq_x)
            self.y.append(seq_y)
        self.X = np.stack(self.X).astype(np.float32)
        self.y = np.stack(self.y).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------------
# LSTM Model in PyTorch
# -------------------------------

class LSTMForecast(nn.Module):
    def __init__(self, n_features, hidden_size=64, n_layers=2, dropout=0.1, horizon=24):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, horizon)
        )

    def forward(self, x):
        # x: batch, seq_len, n_features
        out, (hn, cn) = self.lstm(x)
        # use last hidden state
        h_last = out[:, -1, :]
        return self.fc(h_last)


# -------------------------------
# Training utilities
# -------------------------------

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                n_epochs: int = 30, lr: float = 1e-3, device: str = 'cpu') -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses) if train_losses else float('nan')
        avg_val = np.mean(val_losses) if val_losses else float('nan')
        print(f"Epoch {epoch}/{n_epochs} - train_loss: {avg_train:.6f}, val_loss: {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -------------------------------
# Walk-forward evaluation / rolling-origin cross-validation
# -------------------------------

def walk_forward_evaluation(df: pd.DataFrame, features: List[str], target: str = 'y',
                            lookback: int = 168, horizon: int = 24,
                            initial_train_steps: int = 1000, step: int = 168,
                            model_params: dict = None,
                            batch_size: int = 64,
                            n_epochs: int = 30,
                            device: str = 'cpu') -> Dict:
    """
    Perform walk-forward validation. For each fold, train LSTM on the train windows and evaluate on the next horizon block.
    Returns results summary and saves models.
    """
    if model_params is None:
        model_params = {}

    n = df.shape[0]
    folds = []
    train_start = 0
    train_end = initial_train_steps

    fold_idx = 0
    results = []

    scaler = StandardScaler()
    data_all = scaler.fit_transform(df[features].values)

    while train_end + horizon <= n:
        fold_idx += 1
        # define ranges
        train_range = slice(train_start, train_end)
        val_range = slice(train_end, train_end + horizon)  # one block validation

        train_data = data_all[train_range]
        val_data = data_all[val_range]

        # Prepare datasets (we'll create a combined dataset but ensure sequences do not mix beyond bounds)
        combined = np.vstack([train_data, val_data])
        dataset = TimeSeriesDataset(combined, lookback=lookback, horizon=horizon)

        # split dataset into train and val by index
        n_train_seq = max(0, len(dataset.X) - (horizon))
        train_ds = torch.utils.data.Subset(dataset, range(0, n_train_seq))
        val_ds = torch.utils.data.Subset(dataset, range(n_train_seq, len(dataset)))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # model
        n_features = len(features)
        model = LSTMForecast(n_features=n_features, horizon=horizon, **model_params)

        print(f"Training fold {fold_idx}: train indices {train_range.start}:{train_range.stop}, val indices {val_range.start}:{val_range.stop}")
        model = train_model(model, train_loader, val_loader, n_epochs=n_epochs, device=device)

        # Evaluate on the holdout block using scaler inverse
        # Build test sequences that start within original index range so we predict the horizon beginning at train_end
        # We'll create sequences that correspond to the first available prediction point
        test_start_idx = train_end  # index in data_all
        # We need lookback window ending at test_start_idx
        if test_start_idx - lookback < 0:
            print("Not enough history for lookback; skipping fold")
            break

        seq_x = data_all[test_start_idx - lookback:test_start_idx, :]
        seq_x = seq_x[np.newaxis, :, :].astype(np.float32)
        model.eval()
        with torch.no_grad():
            inp = torch.from_numpy(seq_x).to(device)
            pred = model(inp).cpu().numpy()[0]

        # Inverse transform pred and true
        # To inverse transform, we need to reconstruct full features array; replace y column with pred and then inverse
        # But scaler was fit on full feature set; so to inverse transform target only, compute using scaler mean/std for y column
        y_mean = scaler.mean_[features.index(target)]
        y_std = np.sqrt(scaler.var_[features.index(target)])
        pred_unscaled = pred * y_std + y_mean

        true_block = df[target].values[test_start_idx:test_start_idx + horizon]

        fold_metrics = {
            'fold': fold_idx,
            'start': df.index[test_start_idx],
            'rmse': rmse(true_block, pred_unscaled),
            'mae': mae(true_block, pred_unscaled),
            'mape': mape(true_block, pred_unscaled)
        }
        print(f"Fold {fold_idx} metrics: RMSE {fold_metrics['rmse']:.4f}, MAE {fold_metrics['mae']:.4f}, MAPE {fold_metrics['mape']:.2f}%")

        results.append({**fold_metrics, 'y_true': true_block, 'y_pred': pred_unscaled})

        # advance the window
        train_end += step

    return {
        'results': results,
        'scaler': scaler,
        'features': features,
        'lookback': lookback,
        'horizon': horizon
    }


# -------------------------------
# Baseline Models
# -------------------------------

def baseline_sarimax_forecast(train_series: pd.Series, steps: int, exog_train: pd.DataFrame = None, exog_forecast: pd.DataFrame = None):
    # use simple SARIMAX with order selection skipped for speed; in production use auto_arima
    # We'll fit SARIMAX(1,0,1)(1,0,1,seasonal_period) with seasonal_period inferred from a common periodicity
    seasonal_period = 24 if len(train_series) > 48 else 0
    order = (1, 0, 1)
    seasonal_order = (1, 0, 1, seasonal_period) if seasonal_period else (0, 0, 0, 0)
    model = sm.tsa.SARIMAX(train_series, exog=exog_train, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    forecast = res.get_forecast(steps=steps, exog=exog_forecast)
    return forecast.predicted_mean


def baseline_prophet_forecast(df: pd.DataFrame, horizon: int):
    if not PROPHET_AVAILABLE:
        raise RuntimeError('Prophet not available in environment')
    # df should have columns ds and y
    model = Prophet()
    model.fit(df.reset_index()[['ds', 'y']].rename(columns={'ds': 'ds', 'y': 'y'}))
    future = model.make_future_dataframe(periods=horizon, freq='H')
    forecast = model.predict(future)
    return forecast.set_index('ds')['yhat'].iloc[-horizon:]


# -------------------------------
# SHAP explainability for PyTorch LSTM
# -------------------------------

def shap_explain_lstm(model: nn.Module, background: np.ndarray, X_sample: np.ndarray, feature_names: List[str]):
    """
    Use SHAP DeepExplainer (works with PyTorch) to explain predictions of the LSTM model.
    background: ndarray shape (n_background, seq_len, n_features)
    X_sample: ndarray shape (n_explain, seq_len, n_features)
    Returns: shap_values (n_explain, seq_len, n_features)
    """
    # Wrap model so SHAP can call it: input -> output (predict horizon); we'll explain mean of horizon predictions to get single value
    class WrapperModel(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            # x: numpy array passed via shap -> torch
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32))
            if x.device != next(self.base.parameters()).device:
                x = x.to(next(self.base.parameters()).device)
            out = self.base(x)
            # condense multi-step output into a scalar per sample: mean across horizon
            return torch.mean(out, dim=1, keepdim=True)

    wrapper = WrapperModel(model)
    wrapper.eval()

    # shap expects background and X_sample as arrays
    explainer = shap.DeepExplainer(wrapper, torch.from_numpy(background.astype(np.float32)))
    shap_values = explainer.shap_values(torch.from_numpy(X_sample.astype(np.float32)))
    # shap_values is a list for each output; our wrapper returns single output -> list length 1
    shap_vals = shap_values[0]  # shape (n_explain, seq_len, n_features)

    # Aggregate by feature across the sequence or by lag
    # We'll return raw shap array and let calling code aggregate
    return shap_vals


# -------------------------------
# Plot utilities
# -------------------------------

def plot_preds(idx_datetime, y_true, y_pred, title='Forecast vs True', savepath=None):
    plt.figure(figsize=(10, 4))
    plt.plot(idx_datetime, y_true, label='True')
    plt.plot(idx_datetime, y_pred, label='Pred')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()


def plot_shap_time_lag(shap_vals, feature_names, savepath=None):
    # shap_vals shape: (n_samples, seq_len, n_features)
    # average absolute SHAP across samples -> shape (seq_len, n_features)
    avg_abs = np.mean(np.abs(shap_vals), axis=0)
    seq_len, n_features = avg_abs.shape
    plt.figure(figsize=(12, 6))
    for i in range(n_features):
        plt.plot(range(-seq_len, 0), avg_abs[:, i], label=feature_names[i])
    plt.xlabel('Lag (negative -> older)')
    plt.ylabel('Mean |SHAP value|')
    plt.title('Per-lag feature importance (SHAP)')
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()


# -------------------------------
# Main orchestration
# -------------------------------

def main(args):
    # 1) Generate dataset
    df = generate_multivariate_series(n_steps=args.n_steps, seasonal_periods=(24, 168), noise_std=args.noise,
                                      n_regressors=args.n_regressors, seed=args.seed)

    print("Generated data preview:")
    print(df.head())

    features = ['y'] + [f'reg_{i}' for i in range(args.n_regressors)]

    # 2) Walk-forward evaluation training
    wf = walk_forward_evaluation(df=df, features=features, target='y', lookback=args.lookback,
                                 horizon=args.horizon, initial_train_steps=args.initial_train,
                                 step=args.step, model_params={'hidden_size': args.hidden, 'n_layers': args.n_layers},
                                 batch_size=args.batch_size, n_epochs=args.epochs, device=args.device)

    # summarize results
    rmses = [r['rmse'] for r in wf['results']]
    maes = [r['mae'] for r in wf['results']]
    mapes = [r['mape'] for r in wf['results']]
    print('\n=== Walk-forward summary ===')
    print(f'Folds: {len(wf['results'])}, RMSE mean: {np.mean(rmses):.4f}, MAE mean: {np.mean(maes):.4f}, MAPE mean: {np.mean(mapes):.2f}%')

    # Plot last fold predictions
    last = wf['results'][-1]
    last_idx = pd.date_range(start=last['start'], periods=args.horizon, freq='H')
    plot_preds(last_idx, last['y_true'], last['y_pred'], title='Last fold forecast vs true', savepath='last_fold_forecast.png')

    # 3) Explainability on last trained model: rebuild scaler and model to explain
    # Recreate final training range data
    scaler = wf['scaler']
    features = wf['features']
    lookback = wf['lookback']
    horizon = wf['horizon']

    # For explanation we need the trained model. For simplicity, re-train on full data up to last fold end.
    # Train on all data up to the last fold's start index
    last_fold_start = pd.to_datetime(last['start'])
    last_fold_pos = df.index.get_loc(last_fold_start)
    train_up_to = last_fold_pos
    data_all = scaler.transform(df[features].values)

    # Build dataset for training full model
    combined = data_all[:train_up_to]
    ds_full = TimeSeriesDataset(combined, lookback=lookback, horizon=horizon)
    train_loader_full = DataLoader(ds_full, batch_size=args.batch_size, shuffle=True)
    # small validation split from last part of training
    val_loader_full = DataLoader(torch.utils.data.Subset(ds_full, range(max(0, len(ds_full) - 200), len(ds_full))), batch_size=args.batch_size)

    n_features = len(features)
    model_full = LSTMForecast(n_features=n_features, horizon=horizon, hidden_size=args.hidden, n_layers=args.n_layers)
    model_full = train_model(model_full, train_loader_full, val_loader_full, n_epochs=args.epochs, device=args.device)

    # Prepare background and samples for SHAP
    # background: select 50 random sequences from training
    seqs = []
    for i in range(lookback, combined.shape[0] - horizon + 1):
        seqs.append(combined[i - lookback:i, :])
    seqs = np.stack(seqs).astype(np.float32)
    n_bg = min(50, max(1, seqs.shape[0] // 10))
    bg_idx = np.random.choice(seqs.shape[0], size=n_bg, replace=False)
    background = seqs[bg_idx]

    # sample few recent sequences to explain
    X_sample = seqs[-min(20, seqs.shape[0]):]

    print('Running SHAP DeepExplainer (this may take a while)')
    shap_vals = shap_explain_lstm(model_full, background, X_sample, feature_names=features)

    # Aggregate & plot
    plot_shap_time_lag(shap_vals, feature_names=features, savepath='shap_time_lag.png')

    # 4) Baseline comparison on the last fold
    # Use SARIMAX trained on same data as our final model to forecast last horizon
    train_series = df['y'].iloc[:train_up_to]
    exog_train = df[[c for c in features if c != 'y']].iloc[:train_up_to]
    exog_forecast = df[[c for c in features if c != 'y']].iloc[train_up_to:train_up_to + horizon]

    try:
        sarimax_pred = baseline_sarimax_forecast(train_series, steps=horizon, exog_train=exog_train, exog_forecast=exog_forecast)
        sarimax_pred = np.array(sarimax_pred)
        print('SARIMAX done')
        print(f'SARIMAX RMSE: {rmse(last['y_true'], sarimax_pred):.4f}')
    except Exception as e:
        print('SARIMAX failed:', e)

    if PROPHET_AVAILABLE:
        try:
            prophet_pred = baseline_prophet_forecast(df.iloc[:train_up_to], horizon=horizon)
            prophet_pred = np.array(prophet_pred)
            print('Prophet RMSE:', rmse(last['y_true'], prophet_pred))
        except Exception as e:
            print('Prophet failed:', e)

    # Save key outputs
    os.makedirs('outputs', exist_ok=True)
    pd.DataFrame({
        'y_true': last['y_true'],
        'y_pred_dl': last['y_pred'],
        'index': pd.date_range(start=last['start'], periods=horizon, freq='H')
    }).set_index('index').to_csv('outputs/last_fold_preds.csv')

    print('\nSaved plots: last_fold_forecast.png, shap_time_lag.png and outputs/last_fold_preds.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Time Series Forecasting - LSTM + SHAP')
    parser.add_argument('--n_steps', type=int, default=2500)
    parser.add_argument('--n_regressors', type=int, default=2)
    parser.add_argument('--noise', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lookback', type=int, default=168)
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--initial_train', type=int, default=1200)
    parser.add_argument('--step', type=int, default=168)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    main(args)
