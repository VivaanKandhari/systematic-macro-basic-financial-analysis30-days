# Week 4 Final Capstone: Does the US Yield Curve Predict Hang Seng Returns?

This capstone is the final project in my 30-day systematic macro basic financial analysis series.

The project tests whether the US Treasury yield curve can be used as a systematic macro signal for future Hang Seng Index returns. The workflow moves from an economic hypothesis to cleaned data, feature engineering, statistical testing, signal construction, backtesting, robustness checks, and final research conclusions.

## Research Question

Does the US yield curve predict future Hang Seng Index returns?

More specifically:

- Does the US 10-year minus 2-year Treasury yield spread help forecast future HSI returns?
- Does an inverted or flatter yield curve signal weaker future HSI performance?
- Can the yield curve be turned into a practical trading or risk-management signal?
- Does the result survive robustness tests?

## Economic Hypothesis

The hypothesis is that an inverted or flatter US yield curve may predict weaker Hang Seng returns over the following months because it can reflect tighter monetary policy, recession risk, weaker global risk appetite, and pressure on Asia-linked equity markets.

Hong Kong equities are exposed to global liquidity conditions, China-related growth expectations, and international risk appetite. If the US yield curve signals stress or a future slowdown, it may contain information relevant for future HSI returns.

## Notebook

Main notebook:

```text
Does_US_Yield_Curve_Predict_HSI_Returns.ipynb
```

## Data

The project uses publicly available market and macro data:

- Hang Seng Index price data from Yahoo Finance (`^HSI`)
- US 10-year Treasury yield from FRED (`DGS10`)
- US 2-year Treasury yield from FRED (`DGS2`)
- Yield curve spread calculated as `10Y yield - 2Y yield`
- Alternative macro robustness signals:
  - US high-yield credit spreads
  - VIX
  - Fed funds rate
- Cross-market robustness checks:
  - Hang Seng Index
  - Nikkei 225
  - KOSPI

## Methodology

The project follows a research-style systematic macro workflow:

1. Build a clean monthly macro-financial dataset.
2. Calculate HSI monthly returns and forward returns.
3. Engineer the yield curve spread and related macro features.
4. Test predictive relationships using correlation and regression.
5. Build a simple signal rule based on the yield curve spread.
6. Backtest the signal against buy-and-hold HSI.
7. Evaluate performance using total return, annualized return, volatility, Sharpe ratio, max drawdown, and Calmar ratio.
8. Run robustness checks across thresholds, alternative macro signals, other Asian markets, overlapping-return adjustments, and non-overlapping return windows.

## Key Result

The yield curve signal showed some usefulness as a risk-management filter, but it was not strong enough to be treated as a standalone alpha signal.

In the threshold sensitivity test, the best-performing rule was to be long HSI when the US 10Y-2Y yield spread was above `0.50` percentage points.

That rule produced:

- Total return: `48.15%`
- Annualized return: `2.42%`
- Annualized volatility: `13.93%`
- Sharpe ratio: `0.17`
- Max drawdown: `-32.07%`
- Calmar ratio: `0.08`

Buy-and-hold HSI produced:

- Total return: `23.71%`
- Annualized return: `1.30%`
- Annualized volatility: `19.97%`
- Sharpe ratio: `0.06`
- Max drawdown: `-55.34%`
- Calmar ratio: `0.02`

The yield curve rule improved drawdown and volatility compared with buy-and-hold, but the absolute Sharpe ratio remained low.

## Robustness Checks

The project tested whether the signal survived multiple variations:

- Threshold sensitivity from `-1.00` to `+1.00`
- Alternative macro signals such as credit spreads, VIX, and Fed funds variables
- Cross-market tests on HSI, Nikkei, and KOSPI
- Newey-West/HAC standard errors to adjust for overlapping forward returns
- Non-overlapping forward return tests
- Benchmark comparison against buy-and-hold

The robustness tests supported a cautious interpretation. The yield curve signal was not a strong standalone predictor for HSI. It performed better as a risk filter than as a pure return-forecasting signal.

One interesting cross-market finding was that Fed funds level showed a stronger negative relationship with future KOSPI returns than with HSI returns. However, this was treated as a separate robustness insight rather than the main HSI result.

## Final Conclusion

The US yield curve has limited predictive value for Hang Seng returns in this sample.

The signal appears more useful as a macro risk-management filter than as a standalone trading signal. It helped reduce volatility and max drawdown compared with buy-and-hold, but it did not generate a high-quality Sharpe ratio by itself.

A practical strategy would likely need to combine the yield curve with other indicators, such as China-specific macro data, liquidity indicators, valuation, trend, or volatility regime filters.

## Limitations

This project uses simple linear tests, monthly data, and free public data sources. The results should not be treated as a live trading strategy.

Important limitations:

- The sample size is limited for monthly macro testing.
- Forward return horizons can create overlapping observations.
- Free macro data may not capture the most relevant China/Hong Kong-specific conditions.
- The trading rule is intentionally simple.
- Transaction costs and implementation constraints are simplified.

## Next Steps

Future improvements could include:

- Adding China PMI or China credit data
- Combining yield curve, volatility, and trend into an ensemble signal
- Running stricter walk-forward out-of-sample backtests
- Testing position sizing based on signal strength
- Comparing results across more Asian markets
- Building a one-page research report for internship or professor outreach

