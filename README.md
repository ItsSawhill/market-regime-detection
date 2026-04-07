# 📈 Market Regime Detection & Regime-Aware Trading Strategy

## 📌 Overview

This project builds a **market regime detection framework** using unsupervised learning on S&P 500 (SPY) time-series data and develops a **regime-aware trading strategy**.

The objective is to identify different market states (e.g., bullish, neutral, volatile) and adjust portfolio exposure accordingly to improve **risk-adjusted performance**.

---

## 🎯 Objectives

- Detect hidden market regimes using clustering techniques
- Engineer meaningful time-series features (momentum, volatility, volume)
- Select optimal number of regimes using **silhouette analysis**
- Analyze **regime persistence and transitions**
- Build a **rule-based trading strategy conditioned on regimes**
- Compare performance vs **buy-and-hold benchmark**

---

## 📊 Data

- Asset: SPY (S&P 500 ETF)
- Source: `yfinance`
- Period: 2015 – Present
- Frequency: Daily

---

## 🧠 Feature Engineering

The model uses a rich set of financial features:

### Returns & Momentum
- 1-day return
- 5-day return
- 10-day momentum
- 20-day momentum

### Volatility
- 10-day volatility
- 20-day volatility
- Downside volatility (negative returns only)

### Trend Indicators
- 20-day moving average gap
- 50-day moving average gap

### Volume Signal
- Volume z-score

### Risk-Adjusted Signal
- Rolling Sharpe ratio (20-day)

---

## 🤖 Model

### Clustering Algorithm
- KMeans (unsupervised learning)

### Model Selection
- Tested k = 3, 4, 5
- Selected optimal k using **silhouette score**

---

## 📌 Regime Interpretation

Clusters are interpreted using volatility and momentum ranking:

- 🟢 **Bullish** → strong momentum, positive trend  
- ⚪ **Neutral** → weak/mixed signals  
- 🟠 **Volatile** → high volatility, unstable conditions  

---

## 🔁 Regime Transition Analysis

A transition matrix is computed to analyze regime persistence:

- Bullish regime is highly persistent (~92%)
- Neutral regime shows moderate persistence (~80%)
- Volatile regime, although rare, is extremely stable once entered (~94%)

This confirms that markets exhibit **state-dependent behavior** rather than random transitions.

---

## 💼 Trading Strategy

A simple regime-based allocation strategy:

| Regime   | Exposure |
|---------|---------|
| Bullish | 100%    |
| Neutral | 50%     |
| Bearish | 0%      |
| Volatile| 25%     |

- Strategy uses **previous day's regime** to avoid look-ahead bias
- Compared against buy-and-hold benchmark

---

## 📈 Results

| Metric | Strategy | Buy & Hold |
|------|--------|------------|
| Annual Return | ~10.97% | ~14.70% |
| Sharpe Ratio | **0.91** | 0.83 |
| Max Drawdown | **-19.7%** | -33.7% |

---

## 🔍 Key Insights

- Regime detection successfully captures underlying market structure
- Volatility regimes are rare but highly persistent
- Strategy significantly reduces drawdowns
- Risk-adjusted performance improves (higher Sharpe)
- Regime-based approaches are more effective for **risk management than return maximization**

---

## ⚠️ Limitations

- Uses single asset (SPY)
- KMeans assumes spherical clusters
- No transaction costs included
- Static clustering (no rolling retraining)
- No macroeconomic variables included

---

## 🚀 Future Improvements

- Walk-forward (rolling) regime re-training
- Apply to multiple assets (QQQ, IWM)
- Compare with momentum or trend-following strategies
- Use Hidden Markov Models (HMM) for regime detection
- Incorporate macroeconomic indicators

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/main.py
