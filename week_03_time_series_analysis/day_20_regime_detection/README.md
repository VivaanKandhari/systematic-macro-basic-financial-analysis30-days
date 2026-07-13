# Day 20: Regime Detection Mini Project

## Concept

Regime detection means splitting market history into different environments.

In this project, the regimes are based on volatility:

- **High Vol**: the market is moving around more than usual.
- **Low Vol**: the market is calmer than usual.

This matters because the same strategy can behave very differently in calm markets versus stressed markets.

## Market Connection

A strategy should not only be judged by its average return.

We also want to know:

- Does it work better in high-volatility periods?
- Does it work better in low-volatility periods?
- Does risk increase when the market becomes unstable?
- Is the strategy secretly dependent on one specific market environment?

Regime analysis helps answer those questions.

## Mini Project

This project downloads Hang Seng Index (`^HSI`) data from Yahoo Finance starting in 2010.

It then:

- Calculates daily returns.
- Calculates 60-day rolling annualized volatility.
- Defines high-volatility and low-volatility regimes.
- Compares HSI returns across regimes.
- Tests a simple momentum strategy across regimes.
- Plots HSI price with high-volatility periods marked.
- Plots rolling volatility with the regime threshold.

## Key Terms

### Daily Return

Daily return measures the percentage change from one closing price to the next.

```text
daily return = today's close / yesterday's close - 1
```

### Rolling Window

A rolling window looks at the most recent fixed number of observations.

In this project, a 60-day rolling window means each volatility estimate uses the most recent 60 trading days.

### Volatility

Volatility measures how much returns move around.

Higher volatility means larger day-to-day swings. Lower volatility means calmer movement.

### Annualized Volatility

Daily volatility is converted into yearly volatility by multiplying by `sqrt(252)`.

We use `252` because there are roughly 252 trading days in a year.

### Median Threshold

The median is the middle value.

This project uses median rolling volatility as the cutoff:

- Above median volatility = `High Vol`
- Below median volatility = `Low Vol`

### Sharpe Ratio

Sharpe ratio measures return per unit of risk.

```text
Sharpe ratio = annualized return / annualized volatility
```

Higher is better, but it should not be read alone. A strategy can have a decent Sharpe in-sample and still fail out-of-sample.

### Look-Ahead Bias

Look-ahead bias happens when a strategy accidentally uses information from the future.

The script avoids this by using yesterday's return to decide today's signal.

## Strategy Logic

The strategy is intentionally simple:

- If yesterday's HSI return was positive, hold HSI today.
- If yesterday's HSI return was negative, stay in cash today.

This is not meant to be a finished trading strategy. It is a clean test case for learning whether strategy behavior changes across regimes.

## Outputs

The script prints:

- Volatility threshold.
- Latest rolling volatility.
- Latest market regime.
- Number of days in each regime.
- Return behavior by regime.
- Momentum strategy performance by regime.

It also plots:

- HSI price with high-volatility periods marked in red.
- 60-day rolling volatility with the median threshold line.

## How To Interpret The Results

If high-volatility periods have worse returns and higher risk, the market is more dangerous during stress.

If the strategy performs better in low-volatility periods, it may need a volatility filter.

If the strategy performs worse in high-volatility periods, the strategy may be vulnerable when risk rises.

If performance is similar across regimes, the strategy may be less regime-dependent.

## Run

```bash
python week_03_time_series_analysis/day_20_regime_detection/day_20_regime_detection.py
```

## Research Takeaway

Regime detection is not about predicting the future perfectly.

It is about asking whether market behavior changes under different conditions.

The big lesson:

> A strategy can look fine on average but still fail badly in the wrong regime.

