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
# Do related HK / China / global market assets share a long-run relationship?
#
# Cointegration idea:
# Two price series can each be non-stationary, but if:
#
#     spread = Y - beta * X
#
# is stationary / mean-reverting, then the two series are cointegrated.
#
# Trading intuition:
# If spread is unusually high -> short spread
# If spread is unusually low  -> long spread
# ------------------------------------------------------------

tickers = {
    "HSI": "^HSI",
    "S&P500": "^GSPC",

    # More reliable Yahoo ETF proxies:
    # 3033.HK = Hang Seng TECH ETF proxy
    # 2822.HK = China A50 ETF proxy
    "HSTECH_ETF": "3033.HK",
    "China_A50_ETF": "2822.HK",

    # USD/HKD usually downloads more reliably than USD/CNH
    "USD_HKD": "HKD=X",

    # Try USD/CNH alternatives, but these may fail
    "USD_CNH": "USDCNH=X",
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

print("\nNon-missing counts before filtering:")
print(prices.count())

# ------------------------------------------------------------
# Drop broken / low-data columns
# ------------------------------------------------------------
# If a ticker has only 0, 1, or very few observations, it is a data issue.
# We require at least 500 usable prices.
# ------------------------------------------------------------

min_required_prices = 500
prices = prices.loc[:, prices.count() >= min_required_prices]

print("\nColumns kept after filtering:")
print(prices.columns.tolist())

print("\nNon-missing counts after filtering:")
print(prices.count())

# ------------------------------------------------------------
# Candidate pairs
# ------------------------------------------------------------
# We only keep pairs where both assets survived the data filter.
# ------------------------------------------------------------

candidate_pairs = [
    ("HSI", "S&P500"),
    ("HSI", "HSTECH_ETF"),
    ("HSI", "China_A50_ETF"),
    ("HSTECH_ETF", "China_A50_ETF"),
    ("USD_HKD", "HSI"),
    ("USD_CNH", "USD_HKD"),
]

available_pairs = [
    pair for pair in candidate_pairs
    if pair[0] in prices.columns and pair[1] in prices.columns
]

print("\nAvailable pairs to test:")
for pair in available_pairs:
    print(pair)

if len(available_pairs) == 0:
    raise ValueError("No usable pairs available. Check ticker downloads.")

# ------------------------------------------------------------
# Helper: ADF stationarity test
# ------------------------------------------------------------
# ADF null hypothesis:
# The series is non-stationary.
#
# p-value < 0.05:
# Reject non-stationarity -> likely stationary.
# ------------------------------------------------------------

def adf_pvalue(series):
    return adfuller(series.dropna())[1]

# ------------------------------------------------------------
# Helper: estimate spread
# ------------------------------------------------------------
# We regress:
#
#     Y = alpha + beta * X + error
#
# Then:
#
#     spread = Y - beta * X
#
# If spread is stationary, X and Y may be cointegrated.
# ------------------------------------------------------------

def estimate_spread(y, x):
    aligned = pd.concat([y, x], axis=1).dropna()
    aligned.columns = ["Y", "X"]

    Y = aligned["Y"]
    X = sm.add_constant(aligned["X"])

    model = sm.OLS(Y, X).fit()

    alpha = model.params["const"]
    beta = model.params["X"]

    spread = Y - beta * aligned["X"]

    return spread, alpha, beta, model.rsquared

# ------------------------------------------------------------
# Cointegration tests
# ------------------------------------------------------------
# Engle-Granger cointegration test:
#
# Null hypothesis:
# No cointegration.
#
# p-value < 0.05:
# Evidence that the pair is cointegrated.
# ------------------------------------------------------------

results = []

spreads = {}

for y_name, x_name in available_pairs:
    pair_prices = prices[[y_name, x_name]].dropna()

    y = pair_prices[y_name]
    x = pair_prices[x_name]

    coint_stat, coint_pvalue, critical_values = coint(y, x)

    spread, alpha, beta, regression_r2 = estimate_spread(y, x)
    spread_adf_pvalue = adf_pvalue(spread)

    results.append({
        "Y": y_name,
        "X": x_name,
        "Cointegration p-value": coint_pvalue,
        "Spread ADF p-value": spread_adf_pvalue,
        "Beta / hedge ratio": beta,
        "Regression R2": regression_r2,
        "Cointegrated at 5%": coint_pvalue < 0.05,
        "Spread stationary at 5%": spread_adf_pvalue < 0.05,
    })

    spreads[(y_name, x_name)] = spread

results = pd.DataFrame(results)

display_results = results.copy()
display_results["Cointegration p-value"] = display_results["Cointegration p-value"].map(lambda x: f"{x:.4f}")
display_results["Spread ADF p-value"] = display_results["Spread ADF p-value"].map(lambda x: f"{x:.4f}")
display_results["Beta / hedge ratio"] = display_results["Beta / hedge ratio"].map(lambda x: f"{x:.4f}")
display_results["Regression R2"] = display_results["Regression R2"].map(lambda x: f"{x:.2%}")

print("\nCointegration Test Results")
print(display_results.to_string(index=False))

# ------------------------------------------------------------
# Pick best pair
# ------------------------------------------------------------
# We choose the pair with the lowest cointegration p-value.
# This does NOT automatically mean trade it.
# It just gives us the most promising pair from this set.
# ------------------------------------------------------------

best_row = results.sort_values("Cointegration p-value").iloc[0]

best_y = best_row["Y"]
best_x = best_row["X"]
best_pair = (best_y, best_x)

print("\nMost promising pair by cointegration p-value:")
print(f"Y: {best_y}")
print(f"X: {best_x}")
print(f"Cointegration p-value: {best_row['Cointegration p-value']:.4f}")
print(f"Beta / hedge ratio: {best_row['Beta / hedge ratio']:.4f}")

spread = spreads[best_pair].dropna()

# ------------------------------------------------------------
# Spread z-score
# ------------------------------------------------------------
# z-score tells us how abnormal the spread is.
#
# z = (current spread - average spread) / spread standard deviation
#
# z-score = 0:
# spread is normal.
#
# z-score > +2:
# spread is unusually high.
#
# z-score < -2:
# spread is unusually low.
# ------------------------------------------------------------

spread_mean = spread.mean()
spread_std = spread.std()

zscore = (spread - spread_mean) / spread_std

# ------------------------------------------------------------
# Trading signals
# ------------------------------------------------------------
# spread = Y - beta * X
#
# If z-score > +2:
# spread is too high.
# Y is expensive relative to X.
# Short spread = short Y, long X.
#
# If z-score < -2:
# spread is too low.
# Y is cheap relative to X.
# Long spread = long Y, short X.
#
# If z-score moves back near 0:
# spread normalized.
# Exit.
# ------------------------------------------------------------

signals = pd.DataFrame(index=zscore.index)
signals["spread"] = spread
signals["zscore"] = zscore

signals["signal"] = 0

signals.loc[signals["zscore"] > 2, "signal"] = -1
signals.loc[signals["zscore"] < -2, "signal"] = 1

print("\nSignal counts:")
print(signals["signal"].value_counts().sort_index())

print("\nLatest spread signal:")
latest_z = signals["zscore"].iloc[-1]
latest_signal = signals["signal"].iloc[-1]

print(f"Latest z-score: {latest_z:.2f}")

if latest_signal == -1:
    print(f"Signal: SHORT spread -> short {best_y}, long {best_x}")
elif latest_signal == 1:
    print(f"Signal: LONG spread -> long {best_y}, short {best_x}")
else:
    print("Signal: No trade. Spread is not extreme.")

# ------------------------------------------------------------
# Plot prices
# ------------------------------------------------------------

pair_prices = prices[[best_y, best_x]].dropna()

normalized_prices = pair_prices / pair_prices.iloc[0]

plt.figure(figsize=(14, 6))
plt.plot(normalized_prices[best_y], label=best_y)
plt.plot(normalized_prices[best_x], label=best_x)

plt.title(f"Normalized Prices: {best_y} vs {best_x}")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Plot spread
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))
plt.plot(spread, label="Spread")
plt.axhline(spread_mean, color="black", linewidth=1, label="Mean")
plt.axhline(spread_mean + 2 * spread_std, color="red", linestyle="--", label="+2 std")
plt.axhline(spread_mean - 2 * spread_std, color="green", linestyle="--", label="-2 std")

plt.title(f"Spread: {best_y} - beta * {best_x}")
plt.ylabel("Spread")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Plot z-score
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))
plt.plot(zscore, label="Spread z-score")
plt.axhline(0, color="black", linewidth=1)
plt.axhline(2, color="red", linestyle="--", label="+2")
plt.axhline(-2, color="green", linestyle="--", label="-2")

plt.title(f"Spread Z-Score: {best_y} vs {best_x}")
plt.ylabel("Z-score")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Final interpretation helper
# ------------------------------------------------------------

print("\nInterpretation Guide")

if best_row["Cointegrated at 5%"]:
    print("Best pair passes the Engle-Granger cointegration test at 5%.")
else:
    print("Best pair does NOT pass the Engle-Granger cointegration test at 5%.")

if best_row["Spread stationary at 5%"]:
    print("The estimated spread appears stationary by ADF test.")
else:
    print("The estimated spread does NOT appear stationary by ADF test.")

print(
    "\nImportant: A cointegration signal is not automatically a trade. "
    "You still need transaction costs, liquidity checks, out-of-sample testing, "
    "and risk controls."
)
