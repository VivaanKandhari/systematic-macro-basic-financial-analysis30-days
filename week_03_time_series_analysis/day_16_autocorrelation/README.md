# Day 16: Autocorrelation

## Concept

Autocorrelation measures whether a time series is related to its own past values. Positive autocorrelation suggests momentum, while negative autocorrelation suggests mean reversion.

## Market Connection

If returns have autocorrelation, past returns may contain information about future returns. The mini-project tests whether that statistical pattern survives as a simple trading rule after transaction costs.

## Mini Project

`day_16_momentum_mean_reversion_detector.py`:

- Downloads HSI, gold, US 10Y yield, and oil data.
- Calculates autocorrelation across several lags.
- Runs the Ljung-Box test for autocorrelation up to lag 10.
- Plots ACF and PACF for HSI returns.
- Tests a simple 1-day momentum strategy against buy-and-hold after transaction costs.

