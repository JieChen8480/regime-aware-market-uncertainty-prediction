# ============================================================
# 1. Imports and Global Configuration
# ============================================================

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# GitHub-friendly settings
USE_SYNTHETIC_IF_NO_DATA = True
SAVE_OUTPUTS = True
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Candidate data paths. Edit this list if your data is stored elsewhere.
DATA_PATHS = [
    Path("feature_futures_5min.csv"),
    Path("data/feature_futures_5min.csv"),
    Path("market_data.csv"),
    Path("data/market_data.csv"),
]

print("Notebook initialized.")
print(f"Output directory: {OUTPUT_DIR.resolve()}")

# ============================================================
# 2. Data Loading Utilities
# ============================================================

def generate_synthetic_market_data(n=3000, seed=RANDOM_STATE):
    """
    Generate market-like intraday data with volatility clustering.
    This is only a fallback so the notebook can run without proprietary data.
    Replace it with real market data for actual analysis.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04 09:30", periods=n, freq="5min")

    # Three hidden volatility states to mimic regime changes
    regimes = rng.choice([0, 1, 2], size=n, p=[0.55, 0.30, 0.15])
    sigma = np.select([regimes == 0, regimes == 1, regimes == 2], [0.0006, 0.0012, 0.0024])

    # Mild autocorrelation plus noise
    ret = np.zeros(n)
    noise = rng.normal(0, sigma)
    for t in range(1, n):
        ret[t] = 0.08 * ret[t - 1] + noise[t]

    close = 4500 * np.exp(np.cumsum(ret))
    volume = rng.lognormal(mean=10.5, sigma=0.45, size=n).astype(int)
    high = close * (1 + np.abs(rng.normal(0, sigma * 0.8)))
    low = close * (1 - np.abs(rng.normal(0, sigma * 0.8)))
    open_ = close / (1 + ret)
    vwap = (open_ + high + low + close) / 4

    # Cross-asset proxy features
    vix_ret = np.abs(ret) * 2 + rng.normal(0, 0.001, n)
    crude_ret = 0.20 * ret + rng.normal(0, 0.0012, n)
    gold_ret = -0.10 * ret + rng.normal(0, 0.0009, n)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "Vwap": vwap,
        "ret": ret,
        "VIX_ret": vix_ret,
        "Crude_ret": crude_ret,
        "Gold_ret": gold_ret,
    }, index=idx)
    df.index.name = "DateTime"
    return df


def load_market_data(paths=DATA_PATHS, use_synthetic=USE_SYNTHETIC_IF_NO_DATA):
    """
    Load market data from local CSV files. If no file is found, optionally use synthetic data.
    """
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            print(f"Loaded data from: {path}")

            # Try to infer datetime index
            datetime_candidates = ["DateTime", "Datetime", "datetime", "Date", "date", "Time", "time"]
            for col in datetime_candidates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    if df[col].notna().mean() > 0.80:
                        df = df.set_index(col).sort_index()
                    break
            return df

    if use_synthetic:
        print("No local CSV found. Using synthetic market-like data for reproducibility demo.")
        return generate_synthetic_market_data()

    raise FileNotFoundError("No data file found. Please add a CSV file or enable synthetic fallback.")

raw_data = load_market_data()
raw_data.head()

# ============================================================
# 3. Feature Engineering
# ============================================================

def infer_return_column(df):
    """
    Ensure the dataset has a return column named 'ret'.
    """
    out = df.copy()
    if "ret" in out.columns:
        out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
        return out

    price_candidates = ["Close", "close", "Price", "price", "Last", "last"]
    for col in price_candidates:
        if col in out.columns:
            out["ret"] = pd.to_numeric(out[col], errors="coerce").pct_change()
            return out

    raise ValueError("Dataset must contain either a 'ret' column or a price column such as 'Close' or 'Price'.")


def engineer_features(df):
    """
    Create a clean modeling dataset.
    The target is next-period directional movement.
    """
    out = infer_return_column(df)
    out = out.replace([np.inf, -np.inf], np.nan).sort_index()

    # Core lagged return features
    for lag in [1, 2, 3, 6, 12, 24]:
        out[f"ret_lag_{lag}"] = out["ret"].shift(lag)

    # Rolling statistical features
    for window in [6, 12, 24, 72]:
        out[f"rolling_mean_ret_{window}"] = out["ret"].rolling(window).mean()
        out[f"rolling_std_ret_{window}"] = out["ret"].rolling(window).std()
        out[f"rolling_skew_ret_{window}"] = out["ret"].rolling(window).skew()
        out[f"rolling_abs_ret_{window}"] = out["ret"].abs().rolling(window).mean()

    # Price-based features, when available
    if "Close" in out.columns:
        close = pd.to_numeric(out["Close"], errors="coerce")
        for window in [6, 12, 24, 72]:
            out[f"close_momentum_{window}"] = close.pct_change(window)
            out[f"close_ma_ratio_{window}"] = close / close.rolling(window).mean() - 1

    # Range and volume features, when available
    if {"High", "Low", "Close"}.issubset(out.columns):
        out["range_pct"] = (out["High"] - out["Low"]) / out["Close"]
        out["range_pct_lag_1"] = out["range_pct"].shift(1)

    if "Volume" in out.columns:
        volume = pd.to_numeric(out["Volume"], errors="coerce")
        out["volume_log"] = np.log1p(volume)
        out["volume_z_72"] = (volume - volume.rolling(72).mean()) / volume.rolling(72).std()
        out["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)

    # Target: next-period direction
    out["target_return"] = out["ret"].shift(-1)
    out["target_direction"] = (out["target_return"] > 0).astype(int)

    return out

feature_data = engineer_features(raw_data)
feature_data.head()

# ============================================================
# 4. LSTM Volatility Proxy with EWMA Fallback
# ============================================================

def add_lstm_volatility_proxy(df, ret_col="ret", lookback=24, epochs=6, batch_size=64):
    """
    Add a sequence-based volatility proxy.

    Primary approach:
        LSTM uses the past `lookback` returns to predict next-period absolute return.

    Fallback approach:
        If TensorFlow is unavailable or the sample is too small, use EWMA volatility.
    """
    out = df.copy()
    out["ewma_volatility"] = out[ret_col].ewm(span=lookback, adjust=False).std()
    out["rolling_volatility"] = out[ret_col].rolling(lookback).std()
    out["lstm_volatility_proxy"] = np.nan

    returns = out[ret_col].astype(float).to_numpy()
    target_abs_ret = np.abs(out[ret_col].shift(-1).astype(float).to_numpy())

    X_seq, y_seq, idx_seq = [], [], []
    for i in range(lookback, len(out) - 1):
        window = returns[i - lookback:i]
        if np.any(pd.isna(window)) or pd.isna(target_abs_ret[i]):
            continue
        X_seq.append(window.reshape(-1, 1))
        y_seq.append(target_abs_ret[i])
        idx_seq.append(out.index[i])

    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping

        X_seq = np.asarray(X_seq)
        y_seq = np.asarray(y_seq)

        if len(X_seq) < 250:
            raise ValueError("Not enough observations for LSTM training.")

        split = int(len(X_seq) * 0.70)
        X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
        y_train_seq = y_seq[:split]
        idx_test_seq = idx_seq[split:]

        scaler = StandardScaler()
        scaler.fit(X_train_seq.reshape(-1, 1))
        X_train_scaled = scaler.transform(X_train_seq.reshape(-1, 1)).reshape(X_train_seq.shape)
        X_test_scaled = scaler.transform(X_test_seq.reshape(-1, 1)).reshape(X_test_seq.shape)

        model = Sequential([
            LSTM(16, input_shape=(lookback, 1), return_sequences=False),
            Dropout(0.20),
            Dense(8, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

        model.fit(
            X_train_scaled,
            y_train_seq,
            validation_split=0.20,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )

        pred_vol = model.predict(X_test_scaled, verbose=0).reshape(-1)
        out.loc[idx_test_seq, "lstm_volatility_proxy"] = pred_vol
        out["volatility_proxy_source"] = "LSTM_with_EWMA_fill"
        print("LSTM volatility proxy created. Missing early values will be filled with EWMA volatility.")

    except Exception as exc:
        print(f"TensorFlow/LSTM unavailable or skipped: {exc}")
        print("Using EWMA volatility as the volatility proxy.")
        out["volatility_proxy_source"] = "EWMA_fallback"

    out["lstm_volatility_proxy"] = out["lstm_volatility_proxy"].fillna(out["ewma_volatility"])
    return out

feature_data = add_lstm_volatility_proxy(feature_data)
feature_data[["ret", "ewma_volatility", "rolling_volatility", "lstm_volatility_proxy", "volatility_proxy_source"]].tail()

# ============================================================
# 5. Regime Construction
# ============================================================

def add_market_regimes(df, vol_col="lstm_volatility_proxy", n_components=3):
    """
    Add transparent volatility regimes and unsupervised Gaussian mixture regimes.
    """
    out = df.copy()

    # Transparent quantile-based regimes
    valid_vol = out[vol_col].replace([np.inf, -np.inf], np.nan)
    out["volatility_regime"] = pd.qcut(
        valid_vol.rank(method="first"),
        q=3,
        labels=["Low volatility", "Medium volatility", "High volatility"],
    )

    # GMM regimes based on return and volatility-related variables
    gmm_cols = ["ret", vol_col, "rolling_std_ret_24", "rolling_abs_ret_24"]
    gmm_cols = [col for col in gmm_cols if col in out.columns]
    gmm_input = out[gmm_cols].replace([np.inf, -np.inf], np.nan).dropna()

    out["gmm_regime"] = np.nan
    if len(gmm_input) >= 200 and len(gmm_cols) >= 2:
        scaler = StandardScaler()
        X_gmm = scaler.fit_transform(gmm_input)
        gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE)
        out.loc[gmm_input.index, "gmm_regime"] = gmm.fit_predict(X_gmm)

    return out

feature_data = add_market_regimes(feature_data)
feature_data[["ret", "lstm_volatility_proxy", "volatility_regime", "gmm_regime"]].dropna().head()

# Quick visualization of uncertainty regimes
plot_df = feature_data[["lstm_volatility_proxy", "volatility_regime"]].dropna().copy()

plt.figure(figsize=(10, 4))
plt.plot(plot_df.index, plot_df["lstm_volatility_proxy"])
plt.title("Volatility Proxy over Time")
plt.xlabel("Time")
plt.ylabel("Volatility proxy")
plt.tight_layout()
plt.show()

regime_summary = feature_data.groupby("volatility_regime", observed=False).agg(
    observations=("ret", "count"),
    avg_return=("ret", "mean"),
    return_std=("ret", "std"),
    avg_volatility_proxy=("lstm_volatility_proxy", "mean"),
)
regime_summary

# ============================================================
# 6. Modeling Dataset
# ============================================================

def build_modeling_dataset(df):
    """
    Select numeric features, remove leakage columns, and return clean modeling data.
    """
    out = df.copy().replace([np.inf, -np.inf], np.nan)

    leakage_cols = {
        "target_return",
        "target_direction",
        "volatility_proxy_source",
    }

    # Exclude raw future target and non-numeric variables. Keep regimes separately for analysis.
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        col for col in numeric_cols
        if col not in leakage_cols
        and not col.lower().startswith("target")
        and col not in ["gmm_regime"]
    ]

    # Avoid using contemporaneous raw Close-like price level if available; derived features are more stable.
    for raw_price_col in ["Open", "High", "Low", "Close", "Price", "Vwap"]:
        if raw_price_col in feature_cols:
            feature_cols.remove(raw_price_col)

    required_cols = list(dict.fromkeys(feature_cols + ["target_direction", "ret", "volatility_regime"]))
    model_data = out[required_cols].dropna().copy()

    return model_data, feature_cols

model_data, feature_cols = build_modeling_dataset(feature_data)

print(f"Modeling observations: {len(model_data):,}")
print(f"Number of model features: {len(feature_cols)}")
print("First 15 features:", feature_cols[:15])
model_data.head()

# ============================================================
# 7. Chronological Train-Test Split
# ============================================================

def chronological_train_test_split(df, test_size=0.30):
    split = int(len(df) * (1 - test_size))
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test

train_df, test_df = chronological_train_test_split(model_data, test_size=0.30)

print(f"Train period: {train_df.index.min()} to {train_df.index.max()} | n={len(train_df):,}")
print(f"Test period:  {test_df.index.min()} to {test_df.index.max()} | n={len(test_df):,}")
print("Target class balance in test set:")
print(test_df["target_direction"].value_counts(normalize=True).rename("share"))

# ============================================================
# 8. Metrics and Benchmarks
# ============================================================

def classification_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def evaluate_baselines(train, test):
    majority_class = int(train["target_direction"].mode().iloc[0])
    majority_pred = np.repeat(majority_class, len(test))

    # Persistence/random-walk baseline: next direction equals current direction.
    # Since target_direction at t is direction of ret at t+1, we use observed ret at t.
    persistence_pred = (test["ret"] > 0).astype(int).to_numpy()

    rows = []
    for name, pred in [
        ("Majority Class Benchmark", majority_pred),
        ("Random-Walk Persistence Benchmark", persistence_pred),
    ]:
        row = {"model": name}
        row.update(classification_metrics(test["target_direction"], pred))
        rows.append(row)
    return pd.DataFrame(rows)

baseline_results = evaluate_baselines(train_df, test_df)
baseline_results

# ============================================================
# 9. Model Training and Evaluation
# ============================================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=50,
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


def fit_and_evaluate_models(train, test, feature_cols, models):
    X_train = train[feature_cols]
    y_train = train["target_direction"]
    X_test = test[feature_cols]
    y_test = test["target_direction"]

    rows = []
    fitted_models = {}

    for name, model in models.items():
        clf = clone(model)
        clf.fit(X_train, y_train)
        fitted_models[name] = clf

        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        test_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        row = {"model": name}
        row["train_accuracy"] = accuracy_score(y_train, train_pred)
        test_metrics = classification_metrics(y_test, test_pred, test_score)
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        row["overfitting_gap"] = row["train_accuracy"] - row["test_accuracy"]
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("test_balanced_accuracy", ascending=False)
    return results, fitted_models

model_results, fitted_models = fit_and_evaluate_models(train_df, test_df, feature_cols, models)

combined_results = pd.concat([
    baseline_results.assign(train_accuracy=np.nan, overfitting_gap=np.nan).rename(columns={
        "accuracy": "test_accuracy",
        "balanced_accuracy": "test_balanced_accuracy",
        "f1": "test_f1",
        "roc_auc": "test_roc_auc",
    }),
    model_results,
], ignore_index=True)

combined_results = combined_results[[
    "model",
    "train_accuracy",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_f1",
    "test_roc_auc",
    "overfitting_gap",
]]
combined_results

if SAVE_OUTPUTS:
    combined_results.to_csv(OUTPUT_DIR / "model_performance_summary.csv", index=False)

plt.figure(figsize=(9, 4))
plt.bar(combined_results["model"], combined_results["test_balanced_accuracy"])
plt.title("Out-of-Sample Balanced Accuracy by Model")
plt.ylabel("Balanced accuracy")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# ============================================================
# 10. Overfitting Detection
# ============================================================

overfitting_report = model_results[[
    "model",
    "train_accuracy",
    "test_accuracy",
    "overfitting_gap",
]].copy()

overfitting_report["interpretation"] = np.where(
    overfitting_report["overfitting_gap"] > 0.10,
    "Potential overfitting: large train-test gap",
    "No severe overfitting detected by this threshold",
)

overfitting_report

plt.figure(figsize=(9, 4))
plt.bar(overfitting_report["model"], overfitting_report["overfitting_gap"])
plt.axhline(0.10, linestyle="--")
plt.title("Overfitting Gap: Train Accuracy minus Test Accuracy")
plt.ylabel("Accuracy gap")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# ============================================================
# 11. Regime-Specific Performance Comparison
# ============================================================

def regime_specific_performance(test, feature_cols, fitted_models):
    rows = []
    for regime, subset in test.groupby("volatility_regime", observed=False):
        if len(subset) < 20:
            continue

        # Benchmarks within each regime
        majority_pred = np.repeat(int(train_df["target_direction"].mode().iloc[0]), len(subset))
        persistence_pred = (subset["ret"] > 0).astype(int).to_numpy()

        for name, pred in [
            ("Majority Class Benchmark", majority_pred),
            ("Random-Walk Persistence Benchmark", persistence_pred),
        ]:
            row = {"regime": regime, "model": name, "observations": len(subset)}
            row.update(classification_metrics(subset["target_direction"], pred))
            rows.append(row)

        # ML models within each regime
        X_regime = subset[feature_cols]
        y_regime = subset["target_direction"]
        for name, clf in fitted_models.items():
            pred = clf.predict(X_regime)
            score = clf.predict_proba(X_regime)[:, 1] if hasattr(clf, "predict_proba") else None
            row = {"regime": regime, "model": name, "observations": len(subset)}
            row.update(classification_metrics(y_regime, pred, score))
            rows.append(row)

    return pd.DataFrame(rows)

regime_results = regime_specific_performance(test_df, feature_cols, fitted_models)
regime_results.sort_values(["regime", "balanced_accuracy"], ascending=[True, False])

if SAVE_OUTPUTS:
    regime_results.to_csv(OUTPUT_DIR / "regime_specific_performance.csv", index=False)

pivot_regime = regime_results.pivot_table(
    index="model",
    columns="regime",
    values="balanced_accuracy",
    aggfunc="mean",
)
pivot_regime

# Plot regime-specific balanced accuracy for each model
for model_name, subset in regime_results.groupby("model"):
    plt.figure(figsize=(7, 4))
    plt.bar(subset["regime"].astype(str), subset["balanced_accuracy"])
    plt.title(f"Regime-Specific Balanced Accuracy: {model_name}")
    plt.ylabel("Balanced accuracy")
    plt.xlabel("Volatility regime")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

# ============================================================
# 12. Rolling-Window Validation
# ============================================================

def rolling_window_validation(df, feature_cols, models, train_window=None, test_window=None, step=None):
    n = len(df)
    if train_window is None:
        train_window = min(1500, max(300, int(n * 0.50)))
    if test_window is None:
        test_window = min(300, max(100, int(n * 0.10)))
    if step is None:
        step = test_window

    rows = []
    fold = 0

    for start in range(0, n - train_window - test_window + 1, step):
        fold += 1
        train = df.iloc[start:start + train_window]
        test = df.iloc[start + train_window:start + train_window + test_window]

        # Benchmarks
        base = evaluate_baselines(train, test)
        for _, r in base.iterrows():
            rows.append({
                "fold": fold,
                "model": r["model"],
                "accuracy": r["accuracy"],
                "balanced_accuracy": r["balanced_accuracy"],
                "f1": r["f1"],
                "roc_auc": r["roc_auc"],
                "test_start": test.index.min(),
                "test_end": test.index.max(),
            })

        X_train, y_train = train[feature_cols], train["target_direction"]
        X_test, y_test = test[feature_cols], test["target_direction"]

        for name, model in models.items():
            clf = clone(model)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
            m = classification_metrics(y_test, pred, score)
            rows.append({
                "fold": fold,
                "model": name,
                **m,
                "test_start": test.index.min(),
                "test_end": test.index.max(),
            })

    return pd.DataFrame(rows)

rolling_results = rolling_window_validation(model_data, feature_cols, models)
rolling_results.head()

if rolling_results.empty:
    print("Not enough observations for rolling-window validation.")
else:
    rolling_summary = rolling_results.groupby("model").agg(
        folds=("fold", "nunique"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        min_balanced_accuracy=("balanced_accuracy", "min"),
        max_balanced_accuracy=("balanced_accuracy", "max"),
    ).sort_values("mean_balanced_accuracy", ascending=False)

    if SAVE_OUTPUTS:
        rolling_results.to_csv(OUTPUT_DIR / "rolling_window_results.csv", index=False)
        rolling_summary.to_csv(OUTPUT_DIR / "rolling_window_summary.csv")

    display(rolling_summary)

if not rolling_results.empty:
    for model_name, subset in rolling_results.groupby("model"):
        plt.figure(figsize=(9, 4))
        plt.plot(subset["fold"], subset["balanced_accuracy"], marker="o")
        plt.title(f"Rolling-Window Balanced Accuracy: {model_name}")
        plt.xlabel("Rolling fold")
        plt.ylabel("Balanced accuracy")
        plt.tight_layout()
        plt.show()

# ============================================================
# 13. Feature Importance
# ============================================================

def extract_feature_importance(fitted_model, feature_cols, top_n=15):
    """
    Extract feature importance for models that expose feature_importances_.
    """
    model = fitted_model
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        model = model.named_steps["clf"]

    if not hasattr(model, "feature_importances_"):
        return None

    importance = pd.Series(model.feature_importances_, index=feature_cols)
    return importance.sort_values(ascending=False).head(top_n)

for model_name in ["Decision Tree", "Random Forest"]:
    importance = extract_feature_importance(fitted_models[model_name], feature_cols)
    if importance is not None:
        display(pd.DataFrame({"feature": importance.index, "importance": importance.values}))
        plt.figure(figsize=(8, 5))
        plt.barh(importance.sort_values().index, importance.sort_values().values)
        plt.title(f"Top Feature Importances: {model_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

# ============================================================
# 14. Final Research Summary Table
# ============================================================

summary_items = {
    "Research question": "How stable are directional prediction models across uncertainty regimes?",
    "Primary uncertainty measure": feature_data["volatility_proxy_source"].dropna().iloc[-1] if "volatility_proxy_source" in feature_data else "Not available",
    "Number of observations": len(model_data),
    "Number of features": len(feature_cols),
    "Best model by test balanced accuracy": combined_results.sort_values("test_balanced_accuracy", ascending=False).iloc[0]["model"],
    "Best test balanced accuracy": round(combined_results["test_balanced_accuracy"].max(), 4),
    "Rolling validation folds": 0 if rolling_results.empty else int(rolling_results["fold"].nunique()),
}

research_summary = pd.DataFrame(summary_items.items(), columns=["Item", "Value"])
research_summary
