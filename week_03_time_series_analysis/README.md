# Week 3: Time Series Analysis

Core idea: financial data is sequential. The order of observations matters because trends, autocorrelation, volatility clustering, and regimes can change how a signal behaves through time.

## Days

- [x] Day 15: Stationarity - The Foundation of Time Series
- [x] Day 16: Autocorrelation - Does the Past Predict the Future?
- [x] Day 17: Moving Averages & Trend Following
- [x] Day 18: GARCH Volatility Forecaster
- [x] Day 19: Cointegration
- [x] Day 20: Regime Detection
- [x] Day 21: Week 3 Capstone - HSI Time-Series Strategy Diagnostic

## Week 3 Capstone

The capstone combines the Week 3 tools into one practical diagnostic project:

- Stationarity checks to separate price-level behavior from return behavior
- Autocorrelation checks to see whether returns show useful memory
- Moving-average trend-following rules
- Volatility regime detection using rolling volatility
- Strategy diagnostics such as return, volatility, Sharpe ratio, drawdown, and regime-based performance

Notebook:

```text
day_21_week_3_capstone/HSI_Time_Series_Strategy_Diagnostic.ipynb
```

## Main Lesson

A strategy should not be judged from one backtest alone. Week 3 asks whether the strategy works across time-series diagnostics, whether it behaves differently in high-volatility and low-volatility regimes, and whether risk controls make the result more robust.
