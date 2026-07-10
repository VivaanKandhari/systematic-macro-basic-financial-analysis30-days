import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from arch import arch_model


# ------------------------------------------------------------
# Day 18 Mini Project: GARCH Volatility Forecaster
# ------------------------------------------------------------
# Research question:
# Can a GARCH(1,1) model estimate and forecast Hang Seng Index volatility?
#
# GARCH is used because market volatility is usually clustered:
# large moves tend to be followed by more large moves, and calm periods
# tend to be followed by more calm periods.
#
# Important:
# This project forecasts risk/volatility, not market direction.
# ------------------------------------------------------------

ticker = "^HSI"

data = yf.download(
    ticker,
    period="10y",
    auto_adjust=True,
    progress=False,
)

prices = data["Close"]

# yfinance can sometimes return a DataFrame even when one ticker is requested.
# This keeps the script robust by converting it into a single price Series.
if isinstance(prices, pd.DataFrame):
    prices = prices[ticker]

prices = prices.dropna()


# ------------------------------------------------------------
# Returns
# ------------------------------------------------------------
# pct_change calculates daily percentage changes:
# today's return = today's price / yesterday's price - 1
#
# We multiply by 100 because the arch package estimates GARCH models more
# reliably when returns are expressed in percentage units instead of tiny
# decimal units.
#
# Example:
# 0.01 decimal return becomes 1.0 percent return.
# ------------------------------------------------------------

returns = prices.pct_change(fill_method=None).dropna() * 100


# ------------------------------------------------------------
# Fit GARCH(1,1)
# ------------------------------------------------------------
# mean="Constant":
# The model estimates one constant average daily return.
#
# vol="GARCH":
# The model uses GARCH volatility dynamics.
#
# p=1:
# Uses 1 lag of past squared shocks. This is the ARCH effect.
# It measures how strongly yesterday's surprise move affects today's volatility.
#
# q=1:
# Uses 1 lag of past conditional variance. This is the GARCH effect.
# It measures how persistent volatility is from one day to the next.
#
# dist="normal":
# Assumes standardized residuals are normally distributed.
# This is a simple starting assumption; Student-t can be better for fat tails.
# ------------------------------------------------------------

model = arch_model(
    returns,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal",
)

result = model.fit(disp="off")

print(result.summary())


# ------------------------------------------------------------
# Conditional volatility
# ------------------------------------------------------------
# Conditional volatility means the model's estimated volatility for each day,
# using information available up to that point.
#
# This is different from one full-sample volatility number because it changes
# through time as the market becomes calmer or more stressed.
# ------------------------------------------------------------

conditional_vol = result.conditional_volatility

# 252 is the common approximation for the number of trading days in a year.
# Daily volatility is annualized by multiplying by sqrt(252).
annualized_conditional_vol = conditional_vol * np.sqrt(252)


# ------------------------------------------------------------
# Forecast next 5 days
# ------------------------------------------------------------
# GARCH forecasts variance first.
#
# Variance = volatility squared.
# Volatility = square root of variance.
# ------------------------------------------------------------

forecast = result.forecast(horizon=5)

variance_forecast = forecast.variance.iloc[-1]
daily_vol_forecast = np.sqrt(variance_forecast)
annualized_vol_forecast = daily_vol_forecast * np.sqrt(252)

print("\nNext 5-Day Volatility Forecast")
for day, vol in enumerate(daily_vol_forecast, start=1):
    print(f"Day {day} daily volatility forecast: {vol:.2f}%")

print("\nNext 5-Day Annualized Volatility Forecast")
for day, vol in enumerate(annualized_vol_forecast, start=1):
    print(f"Day {day} annualized volatility forecast: {vol:.2f}%")


# ------------------------------------------------------------
# Plot returns and conditional volatility
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

axes[0].plot(returns, label="HSI daily returns (%)", alpha=0.8)
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_title("HSI Daily Returns")
axes[0].set_ylabel("Return (%)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    annualized_conditional_vol,
    label="GARCH annualized conditional volatility",
    color="red",
)
axes[1].set_title("GARCH(1,1) Estimated Conditional Volatility")
axes[1].set_ylabel("Annualized volatility (%)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Compare latest volatility to long-run volatility
# ------------------------------------------------------------

long_run_daily_vol = returns.std()
long_run_annualized_vol = long_run_daily_vol * np.sqrt(252)

latest_daily_vol = conditional_vol.iloc[-1]
latest_annualized_vol = latest_daily_vol * np.sqrt(252)

print("\nVolatility Comparison")
print(f"Long-run annualized volatility: {long_run_annualized_vol:.2f}%")
print(f"Latest GARCH annualized volatility: {latest_annualized_vol:.2f}%")

if latest_annualized_vol > long_run_annualized_vol:
    print("Latest GARCH volatility is above long-run volatility: current risk is elevated.")
else:
    print("Latest GARCH volatility is below long-run volatility: current risk is calmer than average.")


# ------------------------------------------------------------
# Position sizing implication
# ------------------------------------------------------------
# Simple volatility targeting:
#
# position size = target volatility / forecast volatility
#
# If forecast volatility rises, suggested position size falls.
# If forecast volatility falls, suggested position size rises.
# ------------------------------------------------------------

target_annual_vol = 10.0

position_size = target_annual_vol / latest_annualized_vol

print("\nPosition Sizing Implication")
print(f"Target annual volatility: {target_annual_vol:.2f}%")
print(f"Latest forecast annual volatility: {latest_annualized_vol:.2f}%")
print(f"Suggested position size: {position_size:.2f}x capital")

if position_size < 1:
    print("Risk is high relative to target, so reduce exposure.")
else:
    print("Risk is low relative to target, so full or larger exposure may fit the risk target.")
