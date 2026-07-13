# Day 20 - Regime Detection Mini Project
# Goal:
# Detect whether HSI is in a high-volatility or low-volatility regime,
# then test whether returns and strategy performance differ by regime.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Download HSI price data
# -----------------------------

ticker = "^HSI"

prices = yf.download(ticker, start="2010-01-01", auto_adjust=True)["Close"]

# Sometimes yfinance returns a DataFrame instead of a Series.
# This converts it into one clean price series.
if isinstance(prices, pd.DataFrame):
    prices = prices.iloc[:, 0]

prices = prices.dropna()

# -----------------------------
# 2. Calculate daily returns
# -----------------------------

# Daily return = percentage change from yesterday's closing price to today's closing price.
# Example:
# If HSI goes from 20000 to 20200, return = 20200 / 20000 - 1 = 0.01 = 1%.
returns = prices.pct_change().dropna()

# -----------------------------
# 3. Calculate rolling volatility
# -----------------------------

# A rolling window means we keep looking at the most recent fixed number of days.
# Here, window = 60 means "look at the last 60 trading days."
window = 60

# Volatility means standard deviation of returns.
# Standard deviation measures how much returns move around their average.
#
# We multiply by sqrt(252) to annualize daily volatility.
# 252 is used because there are roughly 252 trading days in a year.
rolling_vol = returns.rolling(window).std() * np.sqrt(252)

# Build one clean dataframe with price, return, and rolling volatility.
df = pd.DataFrame({
    "price": prices,
    "return": returns,
    "rolling_vol": rolling_vol
}).dropna()

# -----------------------------
# 4. Define market regimes
# -----------------------------

# Median = middle value.
# Half the observations are above it, half are below it.
#
# If current rolling volatility is above the median, we call it High Vol.
# If current rolling volatility is below the median, we call it Low Vol.
vol_threshold = df["rolling_vol"].median()

df["regime"] = np.where(
    df["rolling_vol"] > vol_threshold,
    "High Vol",
    "Low Vol"
)

# -----------------------------
# 5. Regime summary
# -----------------------------

print("REGIME DETECTION SUMMARY")
print("-" * 40)

print(f"Volatility threshold: {vol_threshold:.2%}")
print(f"Latest rolling volatility: {df['rolling_vol'].iloc[-1]:.2%}")
print(f"Latest regime: {df['regime'].iloc[-1]}")

print("\nNumber of days in each regime:")
print(df["regime"].value_counts())

# -----------------------------
# 6. Return behavior by regime
# -----------------------------

# Groupby means we calculate separate statistics for each regime.
# So we are asking:
# What were returns like during High Vol periods?
# What were returns like during Low Vol periods?
regime_stats = df.groupby("regime")["return"].agg(
    mean_daily_return="mean",
    daily_volatility="std",
    observations="count"
)

# Annualized return:
# Converts average daily return into an approximate yearly return.
regime_stats["annualized_return"] = (
    (1 + regime_stats["mean_daily_return"]) ** 252 - 1
)

# Annualized volatility:
# Converts daily volatility into yearly volatility.
regime_stats["annualized_volatility"] = (
    regime_stats["daily_volatility"] * np.sqrt(252)
)

# Sharpe ratio:
# Measures return per unit of risk.
# Higher is better.
# Here we assume risk-free rate = 0 to keep the lesson simple.
regime_stats["sharpe_ratio"] = (
    regime_stats["annualized_return"] / regime_stats["annualized_volatility"]
)

# Display table:
# Only percentage columns are multiplied by 100.
# Observations should NOT be multiplied by 100.
# Sharpe ratio should NOT be multiplied by 100.
regime_display = regime_stats.copy()

percentage_columns = [
    "mean_daily_return",
    "daily_volatility",
    "annualized_return",
    "annualized_volatility"
]

for col in percentage_columns:
    regime_display[col] = regime_display[col] * 100

regime_display = regime_display.round({
    "mean_daily_return": 2,
    "daily_volatility": 2,
    "observations": 0,
    "annualized_return": 2,
    "annualized_volatility": 2,
    "sharpe_ratio": 2
})

print("\nRETURN BEHAVIOR BY REGIME")
print("-" * 40)
print(regime_display)

# -----------------------------
# 7. Simple momentum strategy
# -----------------------------

# Strategy idea:
# If yesterday's return was positive, be invested in HSI today.
# If yesterday's return was negative, stay in cash today.
#
# This is intentionally simple, not a final trading strategy.
#
# We use shift(1) to avoid look-ahead bias.
# Look-ahead bias means accidentally using future information that would not
# have been available at the time of the trade.
df["momentum_signal"] = np.where(df["return"].shift(1) > 0, 1, 0)

# Strategy return:
# If signal = 1, we earn the HSI return.
# If signal = 0, we earn 0 because we are in cash.
df["strategy_return"] = df["momentum_signal"] * df["return"]

# -----------------------------
# 8. Strategy performance by regime
# -----------------------------

strategy_stats = df.groupby("regime")["strategy_return"].agg(
    mean_daily_strategy_return="mean",
    daily_strategy_volatility="std",
    observations="count"
)

strategy_stats["annualized_strategy_return"] = (
    (1 + strategy_stats["mean_daily_strategy_return"]) ** 252 - 1
)

strategy_stats["annualized_strategy_volatility"] = (
    strategy_stats["daily_strategy_volatility"] * np.sqrt(252)
)

strategy_stats["strategy_sharpe"] = (
    strategy_stats["annualized_strategy_return"] /
    strategy_stats["annualized_strategy_volatility"]
)

# Again, only percentage columns are multiplied by 100.
# Do not multiply observations or Sharpe ratio.
strategy_display = strategy_stats.copy()

strategy_percentage_columns = [
    "mean_daily_strategy_return",
    "daily_strategy_volatility",
    "annualized_strategy_return",
    "annualized_strategy_volatility"
]

for col in strategy_percentage_columns:
    strategy_display[col] = strategy_display[col] * 100

strategy_display = strategy_display.round({
    "mean_daily_strategy_return": 2,
    "daily_strategy_volatility": 2,
    "observations": 0,
    "annualized_strategy_return": 2,
    "annualized_strategy_volatility": 2,
    "strategy_sharpe": 2
})

print("\nMOMENTUM STRATEGY PERFORMANCE BY REGIME")
print("-" * 40)
print(strategy_display)

# -----------------------------
# 9. Plot HSI price and regimes
# -----------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df["price"],
    label="HSI Price",
    color="black",
    linewidth=1
)

# Red dots show dates where the market was in a high-volatility regime.
high_vol_dates = df[df["regime"] == "High Vol"].index

plt.scatter(
    high_vol_dates,
    df.loc[high_vol_dates, "price"],
    color="red",
    s=8,
    alpha=0.5,
    label="High Vol Regime"
)

plt.title("HSI Price with High-Volatility Regimes")
plt.xlabel("Date")
plt.ylabel("HSI Price")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 10. Plot rolling volatility
# -----------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    df.index,
    df["rolling_vol"],
    label="60-Day Rolling Volatility",
    color="blue"
)

plt.axhline(
    vol_threshold,
    color="red",
    linestyle="--",
    label="Median Volatility Threshold"
)

plt.title("HSI 60-Day Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Annualized Volatility")
plt.legend()
plt.grid(True)
plt.show()

