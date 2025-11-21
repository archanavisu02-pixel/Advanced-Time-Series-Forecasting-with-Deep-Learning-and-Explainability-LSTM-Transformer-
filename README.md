# Advanced Time Series Forecasting with Deep Learning and Explainability

This project implements a **complete, production-quality pipeline** for advanced multivariate time series forecasting using **LSTM deep learning models**, **SHAP explainability**, and **benchmarking against statistical baselines** such as SARIMAX and Prophet.

It is designed for advanced students or practitioners who want to:

* Build realistic synthetic datasets mimicking real-world seasonal/trend dynamics.
* Train and evaluate deep learning models for **multi-step forecasting**.
* Apply rigorous **time-series cross-validation** (rolling/walk-forward).
* Integrate **model interpretability** using SHAP.
* Compare performance against established statistical models.

This README explains how the project is structured, how the dataset is generated, how to run the code, and how to interpret the results.

---

## 📁 Project Structure

The project consists of a single Python script:

```
advanced_time_series_forecasting.py
```

This script performs all tasks:

* Dataset generation
* Model training & evaluation
* Explainability
* Baseline benchmarking
* Plot generation
* Saving outputs

Plots and results are saved automatically:

```
last_fold_forecast.png
shap_time_lag.png
outputs/last_fold_preds.csv
```

---

## 🔧 1. Dataset Generation

The dataset is **programmatically generated** using NumPy/SciPy and includes:

### ✔ Multiple Seasonalities

Two sinusoidal signals:

* **Daily cycle** (period = 24)
* **Weekly cycle** (period = 168)

Each includes harmonics and nonlinear interactions.

### ✔ Trend Component

Either linear or quadratic (configurable).

### ✔ Heteroscedastic Noise

Noise variance changes over time using sinusoidal modulation.

### ✔ Random Spikes

Rare shocks represent anomalies in real-world time series.

### ✔ External Regressors

Simulated features that are:

* Correlated with the target
* Contain their own seasonality
* Contain mild trend & noise

### ✔ Timestamps

Data indexed using hourly frequency starting from 2018-01-01.

The result is a rich, non-stationary multivariate time series suitable for deep learning forecasting.

---

## 🤖 2. Deep Learning Model (LSTM)

The script implements a **multi-layer LSTM forecaster** using PyTorch.

### Model Inputs

* Look-back window (default: 168 hours)
* Multivariate features (target + regressors)

### Model Outputs

* **Multi-step forecast** (default horizon: 24 hours)

### Architecture

* Configurable number of layers
* Configurable hidden size
* ReLU feed-forward head
* Dropout for regularization

---

## 🔁 3. Time-Series Cross-Validation (Walk-Forward)

Standard k-fold CV is invalid for time series.
This project uses **rolling-origin evaluation**, where each fold consists of:

1. Train on an expanding window
2. Predict the next horizon
3. Slide the window forward
4. Repeat

Each fold reports RMSE, MAE, MAPE.

A summary of all folds is printed after training.

---

## 🧠 4. Model Explainability (SHAP)

The project uses **SHAP DeepExplainer** to interpret the LSTM model.

### Explanation Method

* Uses background sequences sampled from training data
* Explains the *mean* output across the forecast horizon
* Produces SHAP values for each:

  * Feature
  * Time lag

### Output

A plot showing **importance of each lag per feature**, helping answer:

* Which lags matter most?
* Are regressors helpful?
* Which temporal patterns influence predictions?

Saved as:

```
shap_time_lag.png
```

---

## 📊 5. Baseline Models

Two baselines are included:

### **SARIMAX**

Automatically configured with seasonal structure.
Used when Prophet is not available.

### **Prophet (Optional)**

Used if installed.
Predicts horizon steps ahead.

Each baseline forecast is compared to LSTM using RMSE.

---

## 📈 6. Evaluation Metrics

The script reports standard forecasting metrics:

* **RMSE** – Root Mean Squared Error
* **MAE** – Mean Absolute Error
* **MAPE** – Mean Absolute Percentage Error

Each fold reports these values individually.
The final fold's predictions are saved to CSV.

---

## ▶️ Running the Script

### Install Dependencies

```
pip install numpy pandas scipy matplotlib scikit-learn torch shap statsmodels prophet
```

(Prophet is optional.)

### Run

```
python advanced_time_series_forecasting.py \
  --seed 42 \
  --device cpu \
  --epochs 20
```

### Useful Options

| Argument          | Meaning                  | Default |
| ----------------- | ------------------------ | ------- |
| `--n_steps`       | Length of dataset        | 2500    |
| `--lookback`      | Input sequence length    | 168     |
| `--horizon`       | Forecast steps           | 24      |
| `--hidden`        | LSTM hidden size         | 128     |
| `--n_layers`      | LSTM layers              | 2       |
| `--initial_train` | First fold training size | 1200    |
| `--step`          | Fold sliding step        | 168     |

---

## 📁 Output
# Advanced-Time-Series-Forecasting-with-Deep-Learning-and-Explainability-LSTM-Transformer-
