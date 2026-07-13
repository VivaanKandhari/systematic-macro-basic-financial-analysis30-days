# Day 19: HK Cointegration Explorer

## Concept

Cointegration asks whether two non-stationary price series move together over the long run.

This is different from normal correlation.

Correlation usually compares short-term returns. Cointegration compares price levels and asks whether the distance between two assets stays mean-reverting over time.

If two assets are cointegrated, they can drift up and down individually, but their spread may keep returning toward a long-run average.

## Market Connection

Pairs trading and relative-value trading often start with this idea.

Instead of asking:

> Will HSI go up tomorrow?

we ask:

> Has HSI moved unusually far away from a related asset, and is that gap likely to close?

That gap is called the spread.

If the spread is stationary, it behaves more like a mean-reverting variable than a random-walking price. That makes it more useful for systematic trading research.

## Mini Project

This project tests whether selected HK/global market pairs share a long-run relationship.

The script downloads 5 years of price data from Yahoo Finance and tests these candidate pairs:

- Hang Seng Index (`HSI`) vs S&P 500 (`S&P500`)
- Hang Seng Index (`HSI`) vs Hang Seng Tech Index (`HSTECH`)
- USD/CNH (`USD_CNH`) vs USD/HKD (`USD_HKD`)

The script automatically drops assets with too little usable data so one broken Yahoo ticker does not kill the whole analysis.

## Key Terms

### Cointegration

Cointegration means two price series may not be stationary by themselves, but some weighted combination of them is stationary.

In simple language: the two prices may wander, but the gap between them does not wander forever.

### Engle-Granger Test

The Engle-Granger test checks whether two series are cointegrated.

Null hypothesis:

> The pair is not cointegrated.

If the p-value is below `0.05`, we reject that null and say there is evidence of cointegration at the 5% level.

### Hedge Ratio

The hedge ratio tells us how much of asset B is used to explain or hedge asset A.

The project estimates it with a regression:

```text
asset_a = alpha + hedge_ratio * asset_b + error
```

The spread is the leftover error:

```text
spread = asset_a - (alpha + hedge_ratio * asset_b)
```

### Spread

The spread is the distance between the two assets after adjusting for the hedge ratio.

If the spread is high, asset A may be expensive relative to asset B.

If the spread is low, asset A may be cheap relative to asset B.

### Spread ADF Test

The ADF test checks whether the spread is stationary.

Null hypothesis:

> The spread is non-stationary.

If the spread ADF p-value is below `0.05`, the spread looks stationary, which supports the mean-reversion idea.

### Z-Score

The z-score tells us how extreme the current spread is compared with its own history.

```text
z-score = (spread - average spread) / spread standard deviation
```

A z-score near `0` means the spread is normal.

A z-score above `+2` means the spread is unusually high.

A z-score below `-2` means the spread is unusually low.

## Outputs

The project prints:

- Downloaded columns and non-missing data counts.
- Pairs that can actually be tested.
- Cointegration p-value for each pair.
- Hedge ratio for each pair.
- Spread ADF p-value for each pair.
- Whether each pair is cointegrated at the 5% level.
- Number of long-spread and short-spread signals.

It also plots:

- The estimated spread through time.
- The spread z-score with `+2`, `0`, and `-2` reference lines.

## How To Interpret The Summary Table

### Cointegration p-value

Lower is stronger.

If this is below `0.05`, the pair passes the main cointegration test.

If this is above `0.05`, the pair does not have strong enough evidence of cointegration.

### Hedge Ratio

This is the model's estimated relationship between the two log price series.

Example:

```text
Hedge Ratio = 1.50
```

This means the model estimates that asset A moves with roughly `1.50` units of asset B in log-price space.

Do not treat this as a final trade size yet. It is a research estimate, not a production portfolio weight.

### Spread ADF p-value

This checks whether the spread itself is mean-reverting.

For a pairs-trading idea, this is important because the strategy depends on the spread coming back toward normal.

### Cointegrated at 5%

This is a simple `True` or `False` based on whether the cointegration p-value is below `0.05`.

`True` means the pair deserves further research.

`False` means the evidence is weak, so we should not force a trading story onto it.

## Signal Logic

The signal count section is not a full backtest.

It only counts how often the spread became extreme.

- `z-score < -2`: spread is unusually low, possible long-spread setup.
- `z-score > +2`: spread is unusually high, possible short-spread setup.

A signal is only a research flag. It does not include transaction costs, slippage, stop-loss rules, position sizing, or out-of-sample validation.

## Run

```bash
python week_03_time_series_analysis/day_19_cointegration_explorer/day_19_cointegration_explorer.py
```

## Research Takeaway

Cointegration is useful because it gives a more serious reason to study a pair than simple correlation.

But a pair passing a cointegration test does not automatically mean it is tradeable.

Before using it as a strategy, we would still need:

- Out-of-sample testing.
- Transaction cost assumptions.
- Slippage assumptions.
- Entry and exit rules.
- Risk controls.
- Checks for whether the relationship breaks during market stress.

The correct mindset is:

> Cointegration can identify a possible long-run relationship, but the trading edge still has to be proven.
