# Day 18: GARCH Volatility Forecaster

## Concept

GARCH models changing volatility through time. Instead of assuming one fixed risk level for the whole sample, it estimates conditional volatility, meaning the model's risk estimate for each day using information available up to that day.

## Market Connection

Financial markets often show volatility clustering. Calm days tend to appear near other calm days, while crisis periods often have many large moves close together. This matters for position sizing because the same trade can be much riskier in a high-volatility regime than in a calm regime.

## Mini Project

The project fits a GARCH(1,1) model to Hang Seng Index (`^HSI`) daily returns over the last 10 years.

Outputs:

- GARCH model summary with mean and volatility parameters.
- Estimated historical conditional volatility.
- Next 5-day volatility forecast.
- Latest volatility versus long-run volatility.
- Simple volatility-targeting position size.

## Run

```bash
python week_03_time_series_analysis/day_18_garch_volatility_forecaster/day_18_garch_volatility_forecaster.py
```

## Research Takeaway

This project does not predict whether HSI will rise or fall. It estimates how risky the market currently is and gives a short-horizon forecast of volatility. That makes it useful for risk management, exposure scaling, and understanding whether current market conditions are calm or stressed.
