import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.stattools import coint, adfuller

# Install tabulate for beautiful table output
# pip install tabulate

from tabulate import tabulate

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

print("=" * 80)
print("DAY 19: HK COINTEGRATION EXPLORER")
print("=" * 80)
print("\n📊 Downloaded Data Summary")
print("-" * 80)

download_summary = pd.DataFrame({
    "Ticker": prices.columns,
    "Symbol": [tickers[name] for name in prices.columns],
    "Observations": prices.count().values,
    "Start Date": [prices[col].first_valid_index().strftime('%Y-%m-%d') for col in prices.columns],
    "End Date": [prices[col].last_valid_index().strftime('%Y-%m-%d') for col in prices.columns],
})

print(tabulate(download_summary, headers="keys", tablefmt="grid", showindex=False))

# Drop columns with too little data.
# This prevents one broken Yahoo ticker from killing the project.
min_required_prices = 500
prices = prices.loc[:, prices.count() >= min_required_prices]

print("\n✓ Columns retained (≥500 observations):", prices.columns.tolist())

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

print("\n🔗 Pairs to Test:")
for i, (a, b) in enumerate(available_pairs, 1):
    print(f"   {i}. {a} ↔ {b}")

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
        "coint_score": score,
        "hedge_ratio": hedge_ratio,
        "alpha": alpha,
        "r_squared": hedge_model.rsquared,
        "spread_adf_p_value": spread_adf_p,
        "spread": spread,
        "zscore": zscore,
    }

results = []

for asset_a, asset_b in available_pairs:
    result = analyze_pair(asset_a, asset_b)
    results.append(result)

# ============================================================
# MAIN RESULTS TABLE
# ============================================================

print("\n" + "=" * 80)
print("📈 COINTEGRATION ANALYSIS RESULTS")
print("=" * 80)

summary_rows = []

for result in results:
    is_cointegrated = result["coint_p_value"] < 0.05
    is_spread_stationary = result["spread_adf_p_value"] < 0.05
    
    summary_rows.append({
        "Pair": f"{result['asset_a']} ↔ {result['asset_b']}",
        "Obs": result["observations"],
        "Coint p-val": f"{result['coint_p_value']:.4f}",
        "Cointegrated?": "✓ YES" if is_cointegrated else "✗ NO",
        "Hedge Ratio": f"{result['hedge_ratio']:.4f}",
        "α (intercept)": f"{result['alpha']:.6f}",
        "R² (fit)": f"{result['r_squared']:.4f}",
        "Spread ADF p": f"{result['spread_adf_p_value']:.4f}",
        "Spread Stat?": "✓ YES" if is_spread_stationary else "✗ NO",
    })

summary = pd.DataFrame(summary_rows)
print("\n" + tabulate(summary, headers="keys", tablefmt="grid", showindex=False))

# ============================================================
# DETAILED PAIR BREAKDOWN
# ============================================================

print("\n" + "=" * 80)
print("🔬 DETAILED PAIR ANALYSIS")
print("=" * 80)

for idx, result in enumerate(results, 1):
    pair_name = f"{result['asset_a']} ↔ {result['asset_b']}"
    print(f"\n{'─' * 80}")
    print(f"Pair {idx}: {pair_name}")
    print(f"{'─' * 80}")
    
    detail_rows = [
        ["Observations", result["observations"]],
        ["Sample Period", f"{log_prices.index[0].strftime('%Y-%m-%d')} to {log_prices.index[-1].strftime('%Y-%m-%d')}"],
        ["", ""],
        ["COINTEGRATION TEST (Engle-Granger)", ""],
        ["  Test Statistic", f"{result['coint_score']:.6f}"],
        ["  p-value", f"{result['coint_p_value']:.4f}"],
        ["  Interpretation", "Evidence of cointegration (reject H0)" if result['coint_p_value'] < 0.05 else "No evidence of cointegration (fail to reject H0)"],
        ["", ""],
        ["HEDGE RATIO REGRESSION", ""],
        ["  Formula", f"{result['asset_a']} = α + β×{result['asset_b']}"],
        ["  Intercept (α)", f"{result['alpha']:.6f}"],
        ["  Beta (β)", f"{result['hedge_ratio']:.6f}"],
        ["  R² (model fit)", f"{result['r_squared']:.4f}"],
        ["", ""],
        ["SPREAD STATIONARITY TEST (ADF on spread)", ""],
        ["  Spread ADF p-value", f"{result['spread_adf_p_value']:.4f}"],
        ["  Interpretation", "Spread is stationary (likely mean-reverting)" if result['spread_adf_p_value'] < 0.05 else "Spread is non-stationary (not mean-reverting)"],
    ]
    
    print(tabulate(detail_rows, headers=["Metric", "Value"], tablefmt="simple", showindex=False))

# ============================================================
# SPREAD STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("📊 SPREAD STATISTICS")
print("=" * 80)

spread_stats_rows = []

for result in results:
    spread = result["spread"]
    zscore = result["zscore"]
    
    spread_stats_rows.append({
        "Pair": f"{result['asset_a']} ↔ {result['asset_b']}",
        "Mean": f"{spread.mean():.6f}",
        "Std Dev": f"{spread.std():.6f}",
        "Min": f"{spread.min():.6f}",
        "Max": f"{spread.max():.6f}",
        "Skew": f"{spread.skew():.4f}",
        "Kurt": f"{spread.kurtosis():.4f}",
    })

spread_df = pd.DataFrame(spread_stats_rows)
print("\n" + tabulate(spread_df, headers="keys", tablefmt="grid", showindex=False))

# ============================================================
# MEAN-REVERSION SIGNAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("🎯 MEAN-REVERSION SIGNAL OPPORTUNITIES")
print("=" * 80)

signal_rows = []

for result in results:
    zscore = result["zscore"]

    long_spread_signals = (zscore < -2).sum()
    short_spread_signals = (zscore > 2).sum()
    neutral_signals = ((zscore >= -2) & (zscore <= 2)).sum()

    signal_rows.append({
        "Pair": f"{result['asset_a']} ↔ {result['asset_b']}",
        "Long (z < -2)": long_spread_signals,
        "Neutral (-2≤z≤2)": neutral_signals,
        "Short (z > +2)": short_spread_signals,
        "Total Trading Days": len(zscore),
        "Signal %": f"{((long_spread_signals + short_spread_signals) / len(zscore) * 100):.2f}%",
    })

signals_df = pd.DataFrame(signal_rows)
print("\n" + tabulate(signals_df, headers="keys", tablefmt="grid", showindex=False))

print("\nSignal Interpretation:")
print("  • Long signal (z < -2): Spread unusually LOW → consider long spread")
print("  • Short signal (z > +2): Spread unusually HIGH → consider short spread")
print("  • Neutral: Spread near mean, no extreme signal")

# ============================================================
# KEY FINDINGS & RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 80)
print("🔍 KEY FINDINGS & INTERPRETATION")
print("=" * 80)

print("""
1. COINTEGRATION TEST (p-value < 0.05):
   ✓ Pairs with low p-values show evidence of a long-run equilibrium relationship.
   
2. HEDGE RATIO:
   ✓ The coefficient β represents the equilibrium hedge ratio.
   ✓ Spread = Asset A - β × Asset B should be mean-reverting if cointegrated.

3. SPREAD STATIONARITY (ADF p-value < 0.05):
   ✓ If the spread itself passes ADF, it is stationary = suitable for mean-reversion.
   ✓ This reinforces the cointegration signal.

4. SIGNAL FREQUENCY:
   ✓ Higher signal percentage = more frequent mean-reversion opportunities.
   ✓ But higher frequency ≠ higher profitability (requires PnL testing).

5. CAVEATS:
   ⚠ Cointegration can break or shift over time (structural break).
   ⚠ Past cointegration does not guarantee future mean-reversion.
   ⚠ Transaction costs and slippage matter in real execution.
   ⚠ This analysis is research-stage, NOT a trading strategy.
""")

# Plot spreads and z-scores
print("\nGenerating plots...")

for result in results:
    pair_name = f"{result['asset_a']} ↔ {result['asset_b']}"
    spread = result["spread"]
    zscore = result["zscore"]
    coint_p = result["coint_p_value"]
    adf_p = result["spread_adf_p_value"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: Spread with mean
    axes[0].plot(spread, linewidth=1.5, label="Spread (log price diff)", color="steelblue")
    axes[0].axhline(spread.mean(), color="red", linewidth=2, linestyle="--", label=f"Mean = {spread.mean():.4f}")
    axes[0].fill_between(spread.index, spread.mean() - spread.std(), spread.mean() + spread.std(), 
                         alpha=0.2, color="red", label="±1 Std Dev")
    axes[0].set_title(f"Spread: {pair_name} (Coint p={coint_p:.4f})", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Spread (log units)", fontsize=11)
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Z-score with bands
    axes[1].plot(zscore, label="Spread z-score", color="darkblue", linewidth=1.5)
    axes[1].axhline(2, color="red", linestyle="--", linewidth=2, label="Short signal (z=+2)")
    axes[1].axhline(-2, color="green", linestyle="--", linewidth=2, label="Long signal (z=-2)")
    axes[1].axhline(0, color="black", linewidth=1, alpha=0.5)
    axes[1].fill_between(zscore.index, -2, 2, alpha=0.1, color="gray", label="Neutral zone")
    axes[1].set_title(f"Mean-Reversion Z-Score (ADF p={adf_p:.4f})", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Z-Score", fontsize=11)
    axes[1].set_xlabel("Date", fontsize=11)
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

print("\n" + "=" * 80)
print("✓ Analysis complete!")
print("=" * 80)
