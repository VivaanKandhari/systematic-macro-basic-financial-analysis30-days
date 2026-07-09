import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Day 17 Mini Project: Trend Following System
# ------------------------------------------------------------
# Research question:
# Does a 50-day / 200-day moving average crossover strategy improve
# Hang Seng Index (HSI) performance versus buy-and-hold?
#
# Strategy:
# If 50-day SMA > 200-day SMA, hold HSI.
# Otherwise, stay in cash.
#
# This is a long-or-cash trend-following strategy.
# It does not short HSI.
# ------------------------------------------------------------

ticker = "^HSI"

data = yf.download(
    ticker,
    start="2010-01-01",
    auto_adjust=True,
    progress=False,
)

prices = data["Close"]

if isinstance(prices, pd.DataFrame):
    prices = prices[ticker]

prices = prices.dropna()

df = pd.DataFrame({
    "price": prices,
})

# ------------------------------------------------------------
# Moving averages
# ------------------------------------------------------------
# SMA = Simple Moving Average.
#
# 50-day SMA:
# Average closing price over the last 50 trading days.
#
# 200-day SMA:
# Average closing price over the last 200 trading days.
#
# If 50d SMA > 200d SMA:
# recent/medium-term price trend is stronger than the long-term trend.
# ------------------------------------------------------------

df["sma_50"] = df["price"].rolling(50).mean()
df["sma_200"] = df["price"].rolling(200).mean()

# ------------------------------------------------------------
# Signal generation
# ------------------------------------------------------------
# signal = 1 means long HSI.
# signal = 0 means stay in cash.
#
# Important:
# The signal uses moving averages computed with data up to today's close.
# To avoid look-ahead bias, we shift the position by 1 day.
#
# This means:
# today's signal is applied to tomorrow's return.
# ------------------------------------------------------------

df["signal"] = np.where(df["sma_50"] > df["sma_200"], 1, 0)
df["position"] = df["signal"].shift(1)

# ------------------------------------------------------------
# Returns
# ------------------------------------------------------------
# buy_hold_return:
# Return from simply holding HSI every day.
#
# strategy_return_before_cost:
# HSI return only on days when position = 1.
# If position = 0, strategy is in cash and earns 0% here.
# ------------------------------------------------------------

df["buy_hold_return"] = df["price"].pct_change(fill_method=None)
df["strategy_return_before_cost"] = df["position"] * df["buy_hold_return"]

# ------------------------------------------------------------
# Transaction costs
# ------------------------------------------------------------
# We assume 0.10% cost whenever the strategy changes position.
#
# Examples:
# cash -> long = trade
# long -> cash = trade
#
# position.diff().abs() equals 1 when position changes.
# ------------------------------------------------------------

transaction_cost = 0.001

df["trade"] = df["position"].diff().abs().fillna(0)

df["strategy_return"] = (
    df["strategy_return_before_cost"]
    - df["trade"] * transaction_cost
)

df = df.dropna()

# ------------------------------------------------------------
# Performance metric functions
# ------------------------------------------------------------

def cumulative_return(return_series):
    """Convert daily returns into a compounded growth-of-$1 equity curve."""
    return (1 + return_series).cumprod()


def total_return(return_series):
    """Calculate total compounded return over the full test period."""
    return (1 + return_series).prod() - 1


def annualized_return(return_series):
    """Annualize average daily return using 252 trading days."""
    return (1 + return_series.mean()) ** 252 - 1


def annualized_volatility(return_series):
    """Annualize daily return volatility using the square-root-of-time rule."""
    return return_series.std() * np.sqrt(252)


def sharpe_ratio(return_series):
    """Measure return per unit of volatility, assuming 0% risk-free rate."""
    vol = annualized_volatility(return_series)
    if vol == 0:
        return np.nan
    return annualized_return(return_series) / vol


def max_drawdown(return_series):
    """Calculate the worst peak-to-trough loss in the equity curve."""
    equity_curve = cumulative_return(return_series)
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1
    return drawdown.min()


# ------------------------------------------------------------
# Calculate metrics
# ------------------------------------------------------------

strategy_returns = df["strategy_return"]
buy_hold_returns = df["buy_hold_return"]

metrics = pd.DataFrame({
    "Strategy": [
        total_return(strategy_returns),
        annualized_return(strategy_returns),
        annualized_volatility(strategy_returns),
        sharpe_ratio(strategy_returns),
        max_drawdown(strategy_returns),
        int(df["trade"].sum()),
    ],
    "Buy and Hold": [
        total_return(buy_hold_returns),
        annualized_return(buy_hold_returns),
        annualized_volatility(buy_hold_returns),
        sharpe_ratio(buy_hold_returns),
        max_drawdown(buy_hold_returns),
        1,
    ],
}, index=[
    "Total Return",
    "Annualized Return",
    "Annualized Volatility",
    "Sharpe Ratio",
    "Max Drawdown",
    "Number of Trades",
])

display_metrics = metrics.copy()

for row in [
    "Total Return",
    "Annualized Return",
    "Annualized Volatility",
    "Max Drawdown",
]:
    display_metrics.loc[row] = display_metrics.loc[row].map(lambda x: f"{x:.2%}")

display_metrics.loc["Sharpe Ratio"] = display_metrics.loc["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
display_metrics.loc["Number of Trades"] = display_metrics.loc["Number of Trades"].map(lambda x: f"{int(x)}")

print("Trend Following Strategy Performance")
print(display_metrics.to_string())

# ------------------------------------------------------------
# Plot price and moving averages
# ------------------------------------------------------------

plt.figure(figsize=(14, 7))
plt.plot(df["price"], label="HSI Price", alpha=0.7)
plt.plot(df["sma_50"], label="50-day SMA", linewidth=1.5)
plt.plot(df["sma_200"], label="200-day SMA", linewidth=1.5)

plt.title("HSI Price With 50-Day and 200-Day Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Plot cumulative returns
# ------------------------------------------------------------

df["strategy_equity"] = cumulative_return(df["strategy_return"])
df["buy_hold_equity"] = cumulative_return(df["buy_hold_return"])

plt.figure(figsize=(14, 7))
plt.plot(df["strategy_equity"], label="50/200 SMA Strategy")
plt.plot(df["buy_hold_equity"], label="Buy and Hold HSI")

plt.title("Trend Following Strategy vs Buy-and-Hold")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Interpretation helper
# ------------------------------------------------------------

print("\nInterpretation Guide")

if metrics.loc["Sharpe Ratio", "Strategy"] > metrics.loc["Sharpe Ratio", "Buy and Hold"]:
    print("Strategy had better risk-adjusted return than buy-and-hold.")
else:
    print("Strategy did NOT have better risk-adjusted return than buy-and-hold.")

if metrics.loc["Max Drawdown", "Strategy"] > metrics.loc["Max Drawdown", "Buy and Hold"]:
    print("Strategy had a smaller max drawdown than buy-and-hold.")
else:
    print("Strategy did NOT reduce max drawdown.")

print(
    "Check the plots for whipsaws: periods where the moving averages cross repeatedly "
    "and the strategy gets in and out without capturing a strong trend."
)
