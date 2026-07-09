import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# ------------------------------------------------------------
# Day 16 Mini Project: Momentum vs Mean Reversion Detector
# ------------------------------------------------------------
# Research question:
# Do past returns help predict future returns?
#
# We test autocorrelation in:
# - HSI
# - Gold
# - US 10Y yield changes
# - Oil
#
# Then we test a simple rule:
# If yesterday's return was positive, buy today.
# ------------------------------------------------------------

tickers = {
    "HSI": "^HSI",
    "Gold": "GLD",
    "US_10Y_Yield": "^TNX",
    "Oil": "CL=F",
}

data = yf.download(
    list(tickers.values()),
    period="10y",
    auto_adjust=True,
    progress=False,
)

prices = data["Close"]

reverse_names = {symbol: name for name, symbol in tickers.items()}
prices = prices.rename(columns=reverse_names)

print("Downloaded columns:")
print(prices.columns.tolist())

print("\nNon-missing price counts:")
print(prices.count())

# Drop assets with insufficient data.
prices = prices.loc[:, prices.count() >= 500]

print("\nColumns kept:")
print(prices.columns.tolist())

# ------------------------------------------------------------
# Build returns / changes
# ------------------------------------------------------------
# For HSI, Gold, Oil:
# returns = percentage price changes.
#
# For US 10Y Yield:
# ^TNX is a yield index, so pct_change can be awkward.
# For yields, we use simple daily changes in yield level instead.
# ------------------------------------------------------------

returns = pd.DataFrame(index=prices.index)

for col in prices.columns:
    if col == "US_10Y_Yield":
        returns[col] = prices[col].diff()
    else:
        returns[col] = prices[col].pct_change(fill_method=None)

returns = returns.dropna()

# ------------------------------------------------------------
# Autocorrelation table
# ------------------------------------------------------------
# Autocorrelation:
# correlation of a series with its own past values.
#
# lag 1:
# correlation between today's return and yesterday's return.
#
# Positive autocorrelation:
# momentum tendency.
#
# Negative autocorrelation:
# mean-reversion tendency.
# ------------------------------------------------------------

lags_to_check = [1, 2, 3, 5, 10, 21]

autocorr_rows = []

for asset in returns.columns:
    r = returns[asset].dropna()

    row = {"Asset": asset}

    for lag in lags_to_check:
        row[f"Lag {lag}"] = r.autocorr(lag=lag)

    autocorr_rows.append(row)

autocorr_table = pd.DataFrame(autocorr_rows)

display_autocorr = autocorr_table.copy()
for col in display_autocorr.columns:
    if col != "Asset":
        display_autocorr[col] = display_autocorr[col].map(lambda x: f"{x:.4f}")

print("\nAutocorrelation Table")
print(display_autocorr.to_string(index=False))

# ------------------------------------------------------------
# Ljung-Box test
# ------------------------------------------------------------
# Ljung-Box test checks whether autocorrelations up to a given lag
# are jointly different from zero.
#
# Null hypothesis:
# No autocorrelation up to the tested lag.
#
# p-value < 0.05:
# Evidence that the series has autocorrelation.
#
# p-value >= 0.05:
# No strong evidence of autocorrelation.
# ------------------------------------------------------------

ljung_rows = []

for asset in returns.columns:
    r = returns[asset].dropna()

    lb = acorr_ljungbox(
        r,
        lags=[10],
        return_df=True,
    )

    p_value = lb["lb_pvalue"].iloc[0]

    ljung_rows.append(
        {
            "Asset": asset,
            "Ljung-Box p-value lag 10": p_value,
            "Autocorrelation evidence at 5%": p_value < 0.05,
        }
    )

ljung_table = pd.DataFrame(ljung_rows)

display_ljung = ljung_table.copy()
display_ljung["Ljung-Box p-value lag 10"] = display_ljung[
    "Ljung-Box p-value lag 10"
].map(lambda x: f"{x:.4f}")

print("\nLjung-Box Test Results")
print(display_ljung.to_string(index=False))

# ------------------------------------------------------------
# Plot ACF and PACF for HSI returns
# ------------------------------------------------------------
# ACF:
# Shows raw autocorrelation at each lag.
#
# PACF:
# Shows autocorrelation at a lag after controlling for shorter lags.
# ------------------------------------------------------------

hsi_returns = returns["HSI"].dropna()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(hsi_returns, lags=40, ax=axes[0])
axes[0].set_title("ACF: HSI Returns")

plot_pacf(hsi_returns, lags=40, ax=axes[1], method="ywm")
axes[1].set_title("PACF: HSI Returns")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Simple momentum strategy
# ------------------------------------------------------------
# Rule:
# If yesterday's return was positive, be long today.
# If yesterday's return was negative, stay in cash today.
#
# This tests whether positive short-term autocorrelation translates into
# a simple trading rule.
#
# Important:
# signal is shifted by 1 day, so we only use information known before today.
# This avoids look-ahead bias.
# ------------------------------------------------------------

strategy_rows = []


def annualized_return(x):
    return (1 + x.mean()) ** 252 - 1


def annualized_volatility(x):
    return x.std() * np.sqrt(252)


def sharpe_ratio(x):
    vol = annualized_volatility(x)
    if vol == 0:
        return np.nan
    return annualized_return(x) / vol


for asset in returns.columns:
    r = returns[asset].dropna()

    signal = (r.shift(1) > 0).astype(int)

    strategy_return = signal * r

    # Transaction cost: 0.10% when the position changes.
    transaction_cost = 0.001
    trades = signal.diff().abs().fillna(0)
    strategy_return_after_cost = strategy_return - trades * transaction_cost

    buy_hold_return = r

    strategy_rows.append(
        {
            "Asset": asset,
            "Strategy Ann Return": annualized_return(strategy_return_after_cost),
            "Strategy Ann Vol": annualized_volatility(strategy_return_after_cost),
            "Strategy Sharpe": sharpe_ratio(strategy_return_after_cost),
            "BuyHold Ann Return": annualized_return(buy_hold_return),
            "BuyHold Ann Vol": annualized_volatility(buy_hold_return),
            "BuyHold Sharpe": sharpe_ratio(buy_hold_return),
            "Number of Trades": int(trades.sum()),
        }
    )

strategy_table = pd.DataFrame(strategy_rows)

display_strategy = strategy_table.copy()
for col in [
    "Strategy Ann Return",
    "Strategy Ann Vol",
    "BuyHold Ann Return",
    "BuyHold Ann Vol",
]:
    display_strategy[col] = display_strategy[col].map(lambda x: f"{x:.2%}")

for col in ["Strategy Sharpe", "BuyHold Sharpe"]:
    display_strategy[col] = display_strategy[col].map(lambda x: f"{x:.2f}")

print("\nSimple Momentum Strategy Results")
print(display_strategy.to_string(index=False))

# ------------------------------------------------------------
# Interpretation helper
# ------------------------------------------------------------

print("\nInterpretation Guide")
print("Positive autocorrelation suggests momentum.")
print("Negative autocorrelation suggests mean reversion.")
print("Ljung-Box p < 0.05 suggests autocorrelation is statistically detectable.")
print("A strategy must still beat buy-and-hold after transaction costs to be useful.")

# In the observed run, the simple 1-day momentum strategy performed poorly
# across all tested assets. Annualized strategy returns and Sharpe ratios were
# negative, and the strategy failed to beat buy-and-hold for Gold, HSI, and
# US 10Y Yield. This shows that weak short-term autocorrelation does not
# automatically become a tradeable signal.
