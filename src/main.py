import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# Optional HMM
HMM_AVAILABLE = True
try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    HMM_AVAILABLE = False


# ----------------------------
# CONFIG
# ----------------------------
ASSETS = ["SPY", "QQQ", "IWM"]
MACRO_TICKERS = {
    "VIX": "^VIX",
    "TNX": "^TNX",          # 10Y Treasury yield proxy
    "DXY": "DX-Y.NYB",      # Dollar index
}
START_DATE = "2012-01-01"
END_DATE = None

TRAIN_WINDOW = 252 * 3      # 3 years rolling train window
REBALANCE_FREQ = 21         # roughly monthly re-fit
CLUSTER_CANDIDATES = [3, 4, 5]
RANDOM_STATE = 42
USE_HMM = True              # if hmmlearn installed, also run HMM
N_HMM_STATES = 3

OUTPUT_FIGURES_DIR = "outputs/figures"
OUTPUT_METRICS_DIR = "outputs/metrics"
DATA_DIR = "data"

os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ----------------------------
# HELPERS
# ----------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


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


def feature_engineering(price_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()

    # Base market features
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

    # Macro / cross-asset proxies
    df = df.join(macro_df, how="left")

    # Transform macro signals into returns / changes
    for col in macro_df.columns:
        df[f"{col}_ret_5d"] = df[col].pct_change(5)
        df[f"{col}_z_20"] = (
            (df[col] - df[col].rolling(20).mean()) /
            (df[col].rolling(20).std() + 1e-8)
        )

    df.dropna(inplace=True)
    return df


def build_feature_columns(df: pd.DataFrame):
    base_cols = [
        "return_1d", "return_5d",
        "momentum_10d", "momentum_20d",
        "volatility_10d", "volatility_20d",
        "downside_vol_20d",
        "ma_gap_20", "ma_gap_50",
        "volume_zscore", "rolling_sharpe_20"
    ]
    macro_cols = [c for c in df.columns if c.endswith("_ret_5d") or c.endswith("_z_20")]
    return base_cols + macro_cols


def choose_kmeans_k(X_train_scaled: np.ndarray, candidates):
    results = []
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_train_scaled)
        score = silhouette_score(X_train_scaled, labels)
        results.append({"k": k, "silhouette_score": score})
    result_df = pd.DataFrame(results)
    best_k = int(result_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
    return best_k, result_df


def map_kmeans_regimes(train_df_with_cluster: pd.DataFrame):
    feature_summary = train_df_with_cluster.groupby("cluster")[
        ["momentum_20d", "volatility_20d", "rolling_sharpe_20"]
    ].mean()

    vol_rank = feature_summary["volatility_20d"].rank(ascending=False, method="dense")
    mom_rank = feature_summary["momentum_20d"].rank(ascending=False, method="dense")
    sharpe_rank = feature_summary["rolling_sharpe_20"].rank(ascending=False, method="dense")

    regime_map = {}
    n = len(feature_summary)

    for cluster in feature_summary.index:
        if vol_rank[cluster] == 1:
            regime_map[cluster] = "Volatile"
        elif mom_rank[cluster] == 1 and sharpe_rank[cluster] <= 2:
            regime_map[cluster] = "Bullish"
        elif mom_rank[cluster] == n:
            regime_map[cluster] = "Bearish"
        else:
            regime_map[cluster] = "Neutral"
    return regime_map, feature_summary


def map_hmm_regimes(hidden_states: np.ndarray, returns: pd.Series, vol: pd.Series):
    summary = pd.DataFrame({
        "state": hidden_states,
        "momentum": returns,
        "vol": vol
    }).groupby("state").mean()

    vol_rank = summary["vol"].rank(ascending=False, method="dense")
    mom_rank = summary["momentum"].rank(ascending=False, method="dense")

    state_map = {}
    n = len(summary)
    for state in summary.index:
        if vol_rank[state] == 1:
            state_map[state] = "Volatile"
        elif mom_rank[state] == 1:
            state_map[state] = "Bullish"
        elif mom_rank[state] == n:
            state_map[state] = "Bearish"
        else:
            state_map[state] = "Neutral"
    return state_map, summary


def exposure_map():
    return {
        "Bullish": 1.0,
        "Neutral": 0.5,
        "Bearish": 0.0,
        "Volatile": 0.25,
    }


def compute_strategy_returns(df: pd.DataFrame, regime_col: str, return_col="return_1d"):
    exp_map = exposure_map()
    exposure = df[regime_col].map(exp_map)
    return exposure.shift(1) * df[return_col]


def compute_momentum_strategy(df: pd.DataFrame):
    signal = (df["momentum_20d"] > 0).astype(float)
    return signal.shift(1) * df["return_1d"]


def compute_trend_strategy(df: pd.DataFrame):
    signal = (df["Close"] > df["ma_50"]).astype(float)
    return signal.shift(1) * df["return_1d"]


def evaluate_strategies(df: pd.DataFrame, asset_name: str):
    strategy_cols = {
        "KMeans Regime": "kmeans_strategy_return",
        "HMM Regime": "hmm_strategy_return" if "hmm_strategy_return" in df.columns else None,
        "Momentum": "momentum_strategy_return",
        "Trend Following": "trend_strategy_return",
        "Buy and Hold": "buyhold_return",
    }

    rows = []
    for label, col in strategy_cols.items():
        if col is None or col not in df.columns:
            continue

        ret = df[col].dropna()
        equity = (1 + ret).cumprod()

        rows.append({
            "Asset": asset_name,
            "Strategy": label,
            "Annualized Return": annualized_return(ret),
            "Annualized Volatility": annualized_volatility(ret),
            "Sharpe Ratio": sharpe_ratio(ret),
            "Max Drawdown": max_drawdown(equity),
        })

    return pd.DataFrame(rows)


# ----------------------------
# DOWNLOAD MACRO DATA
# ----------------------------
print("Downloading macro proxies...")
macro_frames = []
for name, ticker in MACRO_TICKERS.items():
    temp = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    temp = flatten_columns(temp)
    temp = temp[["Close"]].rename(columns={"Close": name})
    macro_frames.append(temp)

macro_df = pd.concat(macro_frames, axis=1).sort_index()
macro_df.ffill(inplace=True)

all_metrics = []
all_cluster_results = []

# ----------------------------
# MAIN LOOP OVER ASSETS
# ----------------------------
for asset in ASSETS:
    print(f"\n{'='*70}")
    print(f"Processing asset: {asset}")
    print(f"{'='*70}")

    price_df = yf.download(asset, start=START_DATE, end=END_DATE, auto_adjust=True)
    price_df = flatten_columns(price_df)

    if price_df.empty:
        print(f"Skipping {asset}: no data")
        continue

    price_df = price_df[["Close", "Volume"]].copy()
    price_df.dropna(inplace=True)

    df = feature_engineering(price_df, macro_df)
    feature_cols = build_feature_columns(df)

    X = df[feature_cols].copy()

    # Containers for walk-forward outputs
    kmeans_regimes = pd.Series(index=df.index, dtype="object")
    hmm_regimes = pd.Series(index=df.index, dtype="object") if HMM_AVAILABLE and USE_HMM else None

    selected_k_records = []

    # ----------------------------
    # WALK-FORWARD REGIME RE-TRAINING
    # ----------------------------
    for i in range(TRAIN_WINDOW, len(df), REBALANCE_FREQ):
        train_slice = df.iloc[i - TRAIN_WINDOW:i].copy()
        test_end = min(i + REBALANCE_FREQ, len(df))
        test_slice = df.iloc[i:test_end].copy()

        X_train = train_slice[feature_cols]
        X_test = test_slice[feature_cols]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # KMeans cluster selection on rolling window
        best_k, k_result_df = choose_kmeans_k(X_train_scaled, CLUSTER_CANDIDATES)
        selected_k_records.append({
            "date": df.index[i],
            "selected_k": best_k,
            "best_silhouette": k_result_df["silhouette_score"].max(),
        })

        km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
        train_clusters = km.fit_predict(X_train_scaled)
        test_clusters = km.predict(X_test_scaled)

        train_with_cluster = train_slice.copy()
        train_with_cluster["cluster"] = train_clusters

        k_regime_map, _ = map_kmeans_regimes(train_with_cluster)
        test_regimes = pd.Series(test_clusters, index=test_slice.index).map(k_regime_map)
        kmeans_regimes.loc[test_slice.index] = test_regimes

        # Optional HMM
        if HMM_AVAILABLE and USE_HMM:
            hmm_features = train_slice[["return_1d", "volatility_20d", "momentum_20d"]].values
            hmm_test_features = test_slice[["return_1d", "volatility_20d", "momentum_20d"]].values

            try:
                hmm = GaussianHMM(
                    n_components=N_HMM_STATES,
                    covariance_type="full",
                    n_iter=200,
                    random_state=RANDOM_STATE
                )
                hmm.fit(hmm_features)

                train_states = hmm.predict(hmm_features)
                test_states = hmm.predict(hmm_test_features)

                hmm_map, _ = map_hmm_regimes(
                    train_states,
                    train_slice["momentum_20d"],
                    train_slice["volatility_20d"]
                )

                hmm_regimes.loc[test_slice.index] = pd.Series(
                    test_states, index=test_slice.index
                ).map(hmm_map)
            except Exception:
                pass

    # Drop pre-window region with no prediction
    df = df.copy()
    df["kmeans_regime"] = kmeans_regimes
    if hmm_regimes is not None:
        df["hmm_regime"] = hmm_regimes

    df = df.dropna(subset=["kmeans_regime"]).copy()

    # ----------------------------
    # REGIME TRANSITION MATRICES
    # ----------------------------
    kmeans_transition = pd.crosstab(
        df["kmeans_regime"].shift(1),
        df["kmeans_regime"],
        normalize="index"
    )
    kmeans_transition.to_csv(
        os.path.join(OUTPUT_METRICS_DIR, f"{asset}_kmeans_transition_matrix.csv")
    )

    if hmm_regimes is not None and "hmm_regime" in df.columns:
        hmm_transition = pd.crosstab(
            df["hmm_regime"].shift(1),
            df["hmm_regime"],
            normalize="index"
        )
        hmm_transition.to_csv(
            os.path.join(OUTPUT_METRICS_DIR, f"{asset}_hmm_transition_matrix.csv")
        )

    # ----------------------------
    # STRATEGIES
    # ----------------------------
    df["buyhold_return"] = df["return_1d"]
    df["kmeans_strategy_return"] = compute_strategy_returns(df, "kmeans_regime")
    df["momentum_strategy_return"] = compute_momentum_strategy(df)
    df["trend_strategy_return"] = compute_trend_strategy(df)

    if hmm_regimes is not None and "hmm_regime" in df.columns:
        df["hmm_strategy_return"] = compute_strategy_returns(df, "hmm_regime")

    # Drop first signal lag NaNs
    strategy_cols = [
        "buyhold_return", "kmeans_strategy_return",
        "momentum_strategy_return", "trend_strategy_return"
    ]
    if "hmm_strategy_return" in df.columns:
        strategy_cols.append("hmm_strategy_return")

    df = df.dropna(subset=strategy_cols).copy()

    # Equity curves
    for col in strategy_cols:
        eq_name = col.replace("_return", "_equity")
        df[eq_name] = (1 + df[col]).cumprod()

    # ----------------------------
    # METRICS
    # ----------------------------
    metrics_df = evaluate_strategies(df, asset)
    all_metrics.append(metrics_df)

    selected_k_df = pd.DataFrame(selected_k_records)
    selected_k_df.to_csv(
        os.path.join(OUTPUT_METRICS_DIR, f"{asset}_rolling_selected_k.csv"),
        index=False
    )

    # Save regime counts
    df["kmeans_regime"].value_counts().rename_axis("regime").reset_index(name="count").to_csv(
        os.path.join(OUTPUT_METRICS_DIR, f"{asset}_kmeans_regime_counts.csv"),
        index=False
    )

    if "hmm_regime" in df.columns:
        df["hmm_regime"].value_counts().rename_axis("regime").reset_index(name="count").to_csv(
            os.path.join(OUTPUT_METRICS_DIR, f"{asset}_hmm_regime_counts.csv"),
            index=False
        )

    # ----------------------------
    # PLOTS
    # ----------------------------
    # Price with KMeans regimes
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label=f"{asset} Close", linewidth=1.2)

    for regime, color in [
        ("Bullish", "green"),
        ("Neutral", "blue"),
        ("Bearish", "red"),
        ("Volatile", "orange"),
    ]:
        if regime in df["kmeans_regime"].unique():
            mask = df["kmeans_regime"] == regime
            plt.scatter(df.index[mask], df.loc[mask, "Close"], s=8, label=regime)

    plt.title(f"{asset} Price with Walk-Forward KMeans Regimes")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, f"{asset}_price_regimes.png"))
    plt.close()

    # Equity curves
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["kmeans_strategy_equity"], label="KMeans Regime", linewidth=1.5)
    if "hmm_strategy_equity" in df.columns:
        plt.plot(df.index, df["hmm_strategy_equity"], label="HMM Regime", linewidth=1.5)
    plt.plot(df.index, df["momentum_strategy_equity"], label="Momentum", linewidth=1.5)
    plt.plot(df.index, df["trend_strategy_equity"], label="Trend Following", linewidth=1.5)
    plt.plot(df.index, df["buyhold_equity"], label="Buy and Hold", linewidth=1.5)

    plt.title(f"{asset}: Strategy Comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, f"{asset}_equity_curve.png"))
    plt.close()

    # Selected k over time
    plt.figure(figsize=(10, 4))
    plt.plot(selected_k_df["date"], selected_k_df["selected_k"], marker="o", linewidth=1)
    plt.title(f"{asset}: Walk-Forward Selected Number of Clusters")
    plt.xlabel("Date")
    plt.ylabel("Selected k")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES_DIR, f"{asset}_selected_k_over_time.png"))
    plt.close()

    # Save asset-level enriched data
    df.to_csv(os.path.join(DATA_DIR, f"{asset.lower()}_walkforward_regime_data.csv"))

# ----------------------------
# SAVE COMBINED RESULTS
# ----------------------------
if all_metrics:
    final_metrics = pd.concat(all_metrics, ignore_index=True)
    final_metrics.to_csv(
        os.path.join(OUTPUT_METRICS_DIR, "all_assets_strategy_metrics.csv"),
        index=False
    )

    print("\nFinal Combined Metrics:")
    print(final_metrics)

    # Summary table by strategy
    summary = final_metrics.groupby("Strategy")[
        ["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown"]
    ].mean().sort_values("Sharpe Ratio", ascending=False)

    summary.to_csv(
        os.path.join(OUTPUT_METRICS_DIR, "strategy_summary_across_assets.csv")
    )

    print("\nAverage Strategy Summary Across Assets:")
    print(summary)

print("\nDone. Files saved to outputs/figures, outputs/metrics, and data.")