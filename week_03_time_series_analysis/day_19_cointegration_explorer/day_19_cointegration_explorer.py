import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.stattools import coint, adfuller

# ------------------------------------------------------------
# Day 19 Mini Project: HK Cointegration Explorer
# ------------------------------------------------------------
# Research question:
# Do related HK/global market pairs share a long-run relationship?
#
# We test cointegration between:
# - HSI vs S&P 500
# - HSI vs Hang Seng Tech Index
# - USD/CNH vs USD/HKD, if data works
#
# Cointegration is useful because a cointegrated spread may mean-revert.
# ------------------------------------------------------------

tickers = {
    "HSI": "^HSI",
    "S&P500": "^GSPC",
    "HSTECH": "^HSTECH",
    "USD_HKD": "HKD=X",
    "USD_CNH": "CNH=X",
}

data = yf.download(
    list(tickers.values()),
    period="5y",
    auto_adjust=True,
    progress=False
)

prices = data["Close"]

reverse_names = {symbol: name for name, symbol in tickers.items()}
prices = prices.rename(columns=reverse_names)

print("Downloaded columns:")
print(prices.columns.tolist())

print("\nNon-missing counts:")
print(prices.count())

# Drop columns with too little data.
# This prevents one broken Yahoo ticker from killing the project.
min_required_prices = 500
prices = prices.loc[:, prices.count() >= min_required_prices]

print("\nColumns kept:")
print(prices.columns.tolist())

# ------------------------------------------------------------
# Use log prices
# ------------------------------------------------------------
# Cointegration tests are often run on price levels or log price levels.
#
# Log prices are useful because differences in logs approximate returns,
# and log spreads are easier to interpret proportionally.
# ------------------------------------------------------------

log_prices = np.log(prices).dropna(how="all")

# ------------------------------------------------------------
# Pair list
# ------------------------------------------------------------

candidate_pairs = [
    ("HSI", "S&P500"),
    ("HSI", "HSTECH"),
    ("USD_CNH", "USD_HKD"),
]

available_pairs = [
    pair for pair in candidate_pairs
    if pair[0] in log_prices.columns and pair[1] in log_prices.columns
]

print("\nPairs tested:")
print(available_pairs)

# ------------------------------------------------------------
# Helper: run cointegration analysis
# ------------------------------------------------------------
# Engle-Granger cointegration test:
#
# Null hypothesis:
# No cointegration.
#
# Alternative:
# Cointegration exists.
#
# p-value < 0.05:
# Evidence of cointegration.
#
# Hedge ratio:
# Estimated by regressing Asset A on Asset B.
#
# spread = Asset A - hedge_ratio * Asset B
#
# If spread is stationary, it may mean-revert.
# ------------------------------------------------------------

def analyze_pair(asset_a, asset_b):
    pair_data = log_prices[[asset_a, asset_b]].dropna()

    y = pair_data[asset_a]
    x = pair_data[asset_b]

    # Engle-Granger test
    score, p_value, critical_values = coint(y, x)

    # Estimate hedge ratio with OLS
    x_const = sm.add_constant(x)
    hedge_model = sm.OLS(y, x_const).fit()

    alpha = hedge_model.params["const"]
    hedge_ratio = hedge_model.params[asset_b]

    spread = y - (alpha + hedge_ratio * x)

    # Test whether spread is stationary
    adf_result = adfuller(spread.dropna())
    spread_adf_p = adf_result[1]

    # Spread z-score
    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std

    return {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "observations": len(pair_data),
        "coint_p_value": p_value,
        "hedge_ratio": hedge_ratio,
        "spread_adf_p_value": spread_adf_p,
        "spread": spread,
        "zscore": zscore,
    }

results = []

for asset_a, asset_b in available_pairs:
    result = analyze_pair(asset_a, asset_b)
    results.append(result)

# ------------------------------------------------------------
# Summary table
# ------------------------------------------------------------

summary_rows = []

for result in results:
    summary_rows.append({
        "Pair": f"{result['asset_a']} vs {result['asset_b']}",
        "Observations": result["observations"],
        "Cointegration p-value": result["coint_p_value"],
        "Hedge Ratio": result["hedge_ratio"],
        "Spread ADF p-value": result["spread_adf_p_value"],
        "Cointegrated at 5%": result["coint_p_value"] < 0.05,
    })

summary = pd.DataFrame(summary_rows)

display_summary = summary.copy()
display_summary["Cointegration p-value"] = display_summary["Cointegration p-value"].map(lambda x: f"{x:.4f}")
display_summary["Hedge Ratio"] = display_summary["Hedge Ratio"].map(lambda x: f"{x:.4f}")
display_summary["Spread ADF p-value"] = display_summary["Spread ADF p-value"].map(lambda x: f"{x:.4f}")

print("\nCointegration Summary")
print(display_summary.to_string(index=False))

# ------------------------------------------------------------
# Plot spreads and z-scores
# ------------------------------------------------------------

for result in results:
    pair_name = f"{result['asset_a']} vs {result['asset_b']}"
    spread = result["spread"]
    zscore = result["zscore"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(spread)
    axes[0].axhline(spread.mean(), color="black", linewidth=1, label="Mean")
    axes[0].set_title(f"Spread: {pair_name}")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(zscore, label="Spread z-score")
    axes[1].axhline(2, color="red", linestyle="--", label="+2")
    axes[1].axhline(-2, color="green", linestyle="--", label="-2")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title(f"Z-Score: {pair_name}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Simple mean-reversion signal count
# ------------------------------------------------------------
# If z-score > +2:
# spread is unusually high.
# Potential signal: short spread.
#
# If z-score < -2:
# spread is unusually low.
# Potential signal: long spread.
#
# This does NOT execute a full backtest yet.
# It just counts signal opportunities.
# ------------------------------------------------------------

signal_rows = []

for result in results:
    zscore = result["zscore"]

    long_spread_signals = (zscore < -2).sum()
    short_spread_signals = (zscore > 2).sum()

    signal_rows.append({
        "Pair": f"{result['asset_a']} vs {result['asset_b']}",
        "Long spread signals z < -2": long_spread_signals,
        "Short spread signals z > +2": short_spread_signals,
        "Total signals": long_spread_signals + short_spread_signals,
    })

signals = pd.DataFrame(signal_rows)

print("\nMean-Reversion Signal Counts")
print(signals.to_string(index=False))

# ------------------------------------------------------------
# Interpretation helper
# ------------------------------------------------------------

print("\nInterpretation Guide")
print("Cointegration p-value < 0.05 means evidence of a long-run relationship.")
print("Spread ADF p-value < 0.05 means the spread appears stationary.")
print("A stationary spread may support mean-reversion trading.")
print("But cointegration can break, so this is only a research starting point.")

