# 📈 Market Regime Detection & Systematic Trading Framework

## 📌 Overview

This project builds a **multi-asset market regime detection framework** using both clustering and probabilistic models, and evaluates **regime-aware trading strategies** under realistic conditions.

The system integrates:
- Walk-forward (rolling) model retraining
- Multi-asset evaluation (SPY, QQQ, IWM)
- Macroeconomic proxies (VIX, Treasury yields, USD index)
- Strategy benchmarking against traditional approaches

The objective is to assess whether **regime-based models improve risk-adjusted performance** compared to standard strategies.

---

## 🎯 Objectives

- Detect hidden market regimes using:
  - KMeans clustering (static)
  - Hidden Markov Models (time-dependent)
- Engineer meaningful financial features across:
  - price, volatility, momentum, volume, macro signals
- Select optimal clustering structure using **silhouette analysis**
- Apply **walk-forward validation** to simulate real-world deployment
- Analyze **regime persistence and transitions**
- Compare multiple strategies:
  - Regime-based (KMeans & HMM)
  - Momentum strategy
  - Trend-following strategy
  - Buy-and-hold benchmark

---

## 📊 Data

### Assets
- SPY (S&P 500)
- QQQ (NASDAQ 100)
- IWM (Russell 2000)

### Macro Proxies
- VIX (volatility index)
- TNX (10Y Treasury yield proxy)
- DXY (US Dollar index)

### Source
- `yfinance`

### Period
- 2012 – Present

---

## 🧠 Feature Engineering

The model incorporates a comprehensive set of financial signals:

### Returns & Momentum
- 1-day return
- 5-day return
- 10-day momentum
- 20-day momentum

### Volatility
- 10-day volatility
- 20-day volatility
- Downside volatility

### Trend Signals
- 20-day moving average gap
- 50-day moving average gap

### Volume Signal
- Volume z-score

### Risk-Adjusted Signal
- Rolling Sharpe ratio (20-day)

### Macro Features
- VIX, TNX, DXY transformations (returns & z-scores)

---

## 🤖 Models

### 1. KMeans Clustering
- Unsupervised clustering
- Optimal number of clusters selected via **silhouette score**

### 2. Hidden Markov Model (HMM)
- Time-dependent probabilistic model
- Captures **sequential market regimes**
- Better suited for dynamic market transitions

---

## 🔁 Walk-Forward Framework

- Training window: 3 years
- Re-training frequency: monthly (~21 days)
- Models are re-fitted on rolling windows to avoid look-ahead bias

This simulates **real-world trading conditions**.

---

## 📌 Regime Interpretation

Regimes are identified based on:
- Momentum ranking
- Volatility ranking
- Risk-adjusted signals

Typical regimes:
- 🟢 Bullish → strong momentum, low risk
- ⚪ Neutral → mixed signals
- 🔴 Bearish → negative trend
- 🟠 Volatile → high uncertainty / crisis periods

---

## 🔁 Regime Transition Analysis

Transition matrices show strong persistence:

- Bullish regimes persist ~90%+
- Neutral regimes persist ~80%
- Volatile regimes are rare but highly persistent (~90%+)

This confirms:
> Markets operate in **state-dependent regimes**, not random behavior.

---

## 💼 Trading Strategies

### Regime-Based Allocation

| Regime   | Exposure |
|---------|---------|
| Bullish | 100%    |
| Neutral | 50%     |
| Bearish | 0%      |
| Volatile| 25%     |

### Additional Benchmarks
- Momentum strategy (20-day signal)
- Trend-following (price vs MA50)
- Buy-and-hold

---

## 📈 Results

### 📊 Average Performance Across Assets

| Strategy | Avg Annual Return | Sharpe Ratio | Max Drawdown |
|---------|------------------:|-------------:|-------------:|
| **HMM Regime** | ~12.6% | 🔥 **Best** | 🔥 **Lowest (~-18%)** |
| Momentum | ~10.0% | Strong | Moderate |
| Buy & Hold | ~15.3% | Moderate | ❌ Worst (~-36%) |
| KMeans Regime | ~8.1% | Weaker | Moderate |
| Trend Following | ~7.8% | Weaker | Moderate |

---

## 🔍 Key Insights

- HMM-based regime detection outperforms static clustering (KMeans)
- Time-dependent models better capture market structure
- Regime-based strategies significantly reduce drawdowns
- Momentum strategies remain strong in trending markets
- Buy-and-hold maximizes return but carries high risk

---

## ⚠️ Limitations

- No transaction costs included
- Model assumes fixed feature set
- HMM convergence can be unstable
- Does not include full macroeconomic datasets (FRED, etc.)
- Strategy is rule-based, not optimized

---

## 🚀 Future Improvements

- Transaction cost modeling
- Portfolio optimization (mean-variance / risk parity)
- Deep learning models for regime detection
- Integration with macroeconomic data (FRED)
- Live trading pipeline / signal deployment

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/main.py