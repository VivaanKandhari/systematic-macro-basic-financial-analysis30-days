# Day 17: Moving Averages & Trend Following

## Concept

Moving averages smooth noisy prices so we can compare short-term and long-term trend. A 50-day simple moving average above a 200-day simple moving average is treated as an uptrend signal; below it is treated as a weak-trend or downtrend signal.

## Market Connection

Trend following is common in systematic macro because macro regimes can persist for months. The strategy does not predict the exact turning point; it tries to participate after an uptrend is visible and avoid exposure when the trend weakens.

## Mini Project

The project tests a 50-day / 200-day moving average crossover strategy on the Hang Seng Index (`^HSI`) from 2010 onward.

Rules:

- Go long HSI when the 50-day SMA is above the 200-day SMA.
- Stay in cash when the 50-day SMA is below or equal to the 200-day SMA.
- Shift the signal by one day to avoid look-ahead bias.
- Subtract 0.10% transaction cost whenever the position changes.

Outputs:

- Strategy vs buy-and-hold performance table.
- HSI price chart with 50-day and 200-day SMAs.
- Cumulative growth of $1 for the strategy and buy-and-hold.
- Interpretation of Sharpe ratio, drawdown, and whipsaw risk.

## Run

```bash
python week_03_time_series_analysis/day_17_moving_averages_trend_following/day_17_trend_following_system.py
```

## Research Takeaway

Trend following can reduce exposure during weak markets, but it can also enter late, exit late, and get whipsawed in sideways regimes. A trend rule should be judged against buy-and-hold using return, volatility, Sharpe ratio, max drawdown, transaction costs, and trade count.
