# Day 15: Stationarity

## Concept

Stationarity means a time series has a reasonably stable statistical behavior over time: mean, variance, and autocorrelation structure do not drift wildly. Raw market prices are usually non-stationary because they trend or wander. Returns and log returns are usually closer to stationary, which is why quant models often work with returns instead of price levels.

## Market Connection

Regressing one non-stationary price series on another can produce a convincing-looking but fake relationship, known as spurious regression.

## Mini Project

Stationarity Audit:

- Download 10 years of Hang Seng Index data.
- Test HSI price, simple returns, and log returns with the ADF test.
- Compare price and return ACF plots.
- Demonstrate spurious regression with a fake price built from shuffled HSI returns.

Run:

```bash
python week_03_time_series_analysis/day_15_stationarity/day_15_stationarity_audit.py
```
