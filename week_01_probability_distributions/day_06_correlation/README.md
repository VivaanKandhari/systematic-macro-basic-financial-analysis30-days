# Day 6 — Correlation: The Most Important Stat in Macro

## Overview

Correlation measures how two return series move together. Values near `+1` indicate the series tend to move in the same direction, values near `-1` indicate they tend to move in opposite directions, and values near `0` indicate little linear relationship. In macro research, correlation is central to understanding diversification, crisis behaviour, and risk concentration.

## Why it matters (Market connection)

Macro investors rely on correlations to:

- Build diversified portfolios and identify hidden risk concentrations.
- Monitor how relationships between assets change during stress events (e.g., crashes).
- Inform hedging and allocation decisions: an asset that looks uncorrelated in calm markets may become highly correlated in crises.

A small change in correlations can produce large changes in portfolio risk, so tracking them through time matters.

## Mini‑project

This folder contains `day_06_macro_correlation_dashboard.py`, a self-contained script that builds a compact macro correlation dashboard for a set of core assets:

- Hang Seng Index (HSI)
- S&P 500
- US 10Y Yield
- Gold
- USD/CNH
- Oil

What the script does:

- Fetches historical prices/levels for each asset (see the script for exact sources).
- Computes returns and reports data coverage per series.
- Builds a correlation matrix for a chosen sample period.
- Saves a heatmap image of the correlation matrix.
- Plots a rolling 60‑day correlation series (example: HSI vs S&P 500).
- Prints clean tables to the console for quick inspection.

Files in this folder:

- `day_06_macro_correlation_dashboard.py` — main script

## Run

From the repository root run:

```bash
python week_01_probability_distributions/day_06_correlation/day_06_macro_correlation_dashboard.py
```

To run the rolling study only:

```bash
python week_01_probability_distributions/day_06_correlation/day_06_rolling_correlation.py
```

If you prefer to run interactively, open the script in a notebook and run the cells step by step.

## Output

The script prints coverage and correlation tables to the console and saves visual outputs (heatmap and rolling correlation plot) to the working directory. Check the script header for exact output filenames.

## Interpretation / Research questions

Use the dashboard to explore questions such as:

- Which macro assets tend to move together, and which provide diversification?
- Which assets help diversify HSI exposure?
- How did HSI–S&P 500 correlation evolve during the COVID‑19 crash (or other crisis periods)?
- Do correlations increase during stress periods, and how quickly?

## Quick reference: correlation strength

- +0.70 to +1.00 — strong positive relationship
- +0.30 to +0.70 — moderate positive relationship
- -0.30 to +0.30 — weak or little linear relationship
- -0.70 to -0.30 — moderate negative relationship
- -1.00 to -0.70 — strong negative relationship

## Next steps / experiments

- Try different rolling window lengths (30, 60, 120 days) to test sensitivity.
- Subsample by regime (pre‑crisis, crisis, post‑crisis) and compare cross‑sectional correlations.
- Expand the asset list to include currencies, global bond indices, or volatility measures.
- Add statistical tests or significance bands to rolling correlation plots.

---

If you'd like, I can also tidy the scripts in this folder (improve docstrings, add argparse, or include direct data source checks). Reply with the change you'd like and I'll update the code.