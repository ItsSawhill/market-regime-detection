import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ----------------------------
# CONFIG
# ----------------------------
TICKER = "SPY"
START_DATE = "2015-01-01"
END_DATE = None
RANDOM_STATE = 42
CLUSTER_CANDIDATES = [3, 4, 5]

OUTPUT_FIGURES_DIR = "outputs/figures"
OUTPUT_METRICS_DIR = "outputs/metrics"

os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
print(f"Downloading {TICKER} data...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)

if df.empty:
    raise ValueError("No data downloaded. Check ticker or internet connection.")

# Fix possible MultiIndex columns returned by yfinance
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Keep only required columns
df = df[["Close", "Volume"]].copy()
df.dropna(inplace=True)

print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
df["return_1d"] = df["Close"].pct_change()
df["return_5d"] = df["Close"].pct_change(5)

df["momentum_10d"] = df["Close"].pct_change(10)
df["momentum_20d"] = df["Close"].pct_change(20)

df["volatility_10d"] = df["return_1d"].rolling(10).std()
df["volatility_20d"] = df["return_1d"].rolling(20).std()

df["downside_vol_20d"] = (
    df["return_1d"].where(df["return_1d"] < 0, 0).rolling(20).std()
)

df["ma_20"] = df["Close"].rolling(20).mean()
df["ma_50"] = df["Close"].rolling(50).mean()
df["ma_gap_20"] = (df["Close"] - df["ma_20"]) / df["ma_20"]
df["ma_gap_50"] = (df["Close"] - df["ma_50"]) / df["ma_50"]

df["vol_mean_20"] = df["Volume"].rolling(20).mean()
df["vol_std_20"] = df["Volume"].rolling(20).std()
df["volume_zscore"] = (df["Volume"] - df["vol_mean_20"]) / (df["vol_std_20"] + 1e-8)

df["rolling_sharpe_20"] = (
    df["return_1d"].rolling(20).mean() /
    (df["return_1d"].rolling(20).std() + 1e-8)
)

df.dropna(inplace=True)

feature_cols = [
    "return_1d",
    "return_5d",
    "momentum_10d",
    "momentum_20d",
    "volatility_10d",
    "volatility_20d",
    "downside_vol_20d",
    "ma_gap_20",
    "ma_gap_50",
    "volume_zscore",
    "rolling_sharpe_20",
]

X = df[feature_cols].copy()

print("\nFeature sample:")
print(X.head())

# ----------------------------
# SCALE FEATURES
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# SELECT NUMBER OF CLUSTERS
# ----------------------------
k_results = []

for k in CLUSTER_CANDIDATES:
    temp_model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    temp_clusters = temp_model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, temp_clusters)
    k_results.append({"k": k, "silhouette_score": score})

k_results_df = pd.DataFrame(k_results)
print("\nCluster Selection Results:")
print(k_results_df)

best_k = int(
    k_results_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"]
)
print(f"\nSelected number of clusters: {best_k}")

k_results_df.to_csv(
    os.path.join(OUTPUT_METRICS_DIR, "cluster_selection.csv"),
    index=False
)

# ----------------------------
# FINAL KMEANS CLUSTERING
# ----------------------------
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
df["cluster"] = kmeans.fit_predict(X_scaled)

# ----------------------------
# INTERPRET CLUSTERS AS REGIMES
# ----------------------------
cluster_summary = df.groupby("cluster")[feature_cols].mean().copy()

print("\nCluster Summary:")
print(cluster_summary)

vol_rank = cluster_summary["volatility_20d"].rank(ascending=False, method="dense")
mom_rank = cluster_summary["momentum_20d"].rank(ascending=False, method="dense")

regime_map = {}
for cluster in cluster_summary.index:
    if vol_rank[cluster] == 1:
        regime_map[cluster] = "Volatile"
    elif mom_rank[cluster] == 1:
        regime_map[cluster] = "Bullish"
    elif mom_rank[cluster] == len(cluster_summary):
        regime_map[cluster] = "Bearish"
    else:
        regime_map[cluster] = "Neutral"

df["regime"] = df["cluster"].map(regime_map)

print("\nRegime Mapping:")
print(regime_map)

print("\nRegime Counts:")
print(df["regime"].value_counts())

# ----------------------------
# REGIME TRANSITION MATRIX
# ----------------------------
transition = pd.crosstab(
    df["regime"].shift(1),
    df["regime"],
    normalize="index"
)

print("\nRegime Transition Matrix:")
print(transition)

transition.to_csv(
    os.path.join(OUTPUT_METRICS_DIR, "regime_transition_matrix.csv")
)

# ----------------------------
# STRATEGY RULES
# ----------------------------
# Bullish -> full exposure
# Neutral -> half exposure
# Bearish -> cash
# Volatile -> quarter exposure
exposure_map = {
    "Bullish": 1.0,
    "Neutral": 0.5,
    "Bearish": 0.0,
    "Volatile": 0.25,
}

df["exposure"] = df["regime"].map(exposure_map)

# Use previous day's regime to avoid look-ahead bias
df["strategy_return"] = df["exposure"].shift(1) * df["return_1d"]
df["buyhold_return"] = df["return_1d"]

df.dropna(inplace=True)

# ----------------------------
# EQUITY CURVES
# ----------------------------
df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
df["buyhold_equity"] = (1 + df["buyhold_return"]).cumprod()

# ----------------------------
# PERFORMANCE METRICS
# ----------------------------
def annualized_return(returns, periods_per_year=252):
    mean_daily = returns.mean()
    return (1 + mean_daily) ** periods_per_year - 1


def annualized_volatility(returns, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0.0):
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return np.nan
    return (ann_ret - risk_free_rate) / ann_vol


def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


metrics = pd.DataFrame({
    "Strategy": ["Regime Strategy", "Buy and Hold"],
    "Annualized Return": [
        annualized_return(df["strategy_return"]),
        annualized_return(df["buyhold_return"]),
    ],
    "Annualized Volatility": [
        annualized_volatility(df["strategy_return"]),
        annualized_volatility(df["buyhold_return"]),
    ],
    "Sharpe Ratio": [
        sharpe_ratio(df["strategy_return"]),
        sharpe_ratio(df["buyhold_return"]),
    ],
    "Max Drawdown": [
        max_drawdown(df["strategy_equity"]),
        max_drawdown(df["buyhold_equity"]),
    ],
})

print("\nPerformance Metrics:")
print(metrics)

metrics.to_csv(
    os.path.join(OUTPUT_METRICS_DIR, "performance_metrics.csv"),
    index=False
)
cluster_summary.to_csv(
    os.path.join(OUTPUT_METRICS_DIR, "cluster_summary.csv")
)

# Save regime counts
regime_counts = df["regime"].value_counts().reset_index()
regime_counts.columns = ["regime", "count"]
regime_counts.to_csv(
    os.path.join(OUTPUT_METRICS_DIR, "regime_counts.csv"),
    index=False
)

# Save mapping
with open(os.path.join(OUTPUT_METRICS_DIR, "regime_mapping.json"), "w") as f:
    json.dump({str(k): v for k, v in regime_map.items()}, f, indent=2)

# ----------------------------
# PLOT 1: PRICE WITH REGIMES
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="SPY Close", linewidth=1.2)

for regime, color in [
    ("Bullish", "green"),
    ("Bearish", "red"),
    ("Volatile", "orange"),
    ("Neutral", "blue"),
]:
    if regime in df["regime"].unique():
        mask = df["regime"] == regime
        plt.scatter(df.index[mask], df.loc[mask, "Close"], s=8, label=regime)

plt.title("SPY Price with Detected Market Regimes")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, "price_regimes.png"))
plt.close()

# ----------------------------
# PLOT 2: EQUITY CURVES
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["strategy_equity"], label="Regime Strategy", linewidth=1.5)
plt.plot(df.index, df["buyhold_equity"], label="Buy and Hold", linewidth=1.5)
plt.title("Strategy vs Buy-and-Hold Equity Curve")
plt.xlabel("Date")
plt.ylabel("Cumulative Growth")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, "equity_curve.png"))
plt.close()

# ----------------------------
# PLOT 3: CLUSTER SELECTION
# ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(k_results_df["k"], k_results_df["silhouette_score"], marker="o")
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, "cluster_selection.png"))
plt.close()

# ----------------------------
# SAVE FULL DATASET
# ----------------------------
df.to_csv(os.path.join("data", "spy_regime_data.csv"))

print("\nDone. Files saved to outputs/figures and outputs/metrics.")