import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm


# ------------------------------------------------------------
# Day 15 Mini Project: Stationarity Audit
# ------------------------------------------------------------
# Research question:
# Is HSI price stationary?
# Are HSI returns stationary?
# Are HSI log returns stationary?
#
# Why it matters:
# Many time-series models assume stationarity.
# If we model non-stationary price levels directly, we can get misleading
# results and fake relationships.
# ------------------------------------------------------------

ticker = "^HSI"

data = yf.download(
    ticker,
    period="10y",
    auto_adjust=True,
    progress=False,
)

if data.empty:
    raise ValueError(
        "No data downloaded for ^HSI. This is likely a temporary Yahoo/yfinance issue."
    )

prices = data["Close"]

if isinstance(prices, pd.DataFrame):
    prices = prices.iloc[:, 0]

prices = prices.dropna()

if len(prices) < 30:
    raise ValueError("Not enough HSI price observations to run the stationarity audit.")

# ------------------------------------------------------------
# Build transformations
# ------------------------------------------------------------
# Simple return:
# percentage change from yesterday to today.
#
# return_t = price_t / price_{t-1} - 1
#
# Log return:
# log(price_t / price_{t-1})
#
# Log returns are popular because they are time-additive:
# multi-period log return = sum of daily log returns.
# ------------------------------------------------------------

simple_returns = prices.pct_change(fill_method=None).dropna()
log_prices = np.log(prices)
log_returns = log_prices.diff().dropna()

series_dict = {
    "HSI Price": prices,
    "HSI Simple Returns": simple_returns,
    "HSI Log Returns": log_returns,
}


# ------------------------------------------------------------
# ADF test function
# ------------------------------------------------------------
# ADF = Augmented Dickey-Fuller test.
#
# It tests whether a time series has a unit root.
#
# Unit root:
# A sign that the series is non-stationary and behaves like a random walk.
#
# Null hypothesis:
# The series is non-stationary.
#
# Alternative hypothesis:
# The series is stationary.
#
# Interpretation:
# p-value < 0.05:
#     reject the null -> evidence the series is stationary
#
# p-value >= 0.05:
#     cannot reject the null -> series may be non-stationary
# ------------------------------------------------------------

def run_adf_test(series, name):
    series = series.dropna()

    if len(series) < 30:
        print(f"\nADF Test skipped for {name}: not enough observations.")
        return

    result = adfuller(series)

    adf_stat = result[0]
    p_value = result[1]
    used_lags = result[2]
    observations = result[3]
    critical_values = result[4]

    print(f"\nADF Test: {name}")
    print(f"ADF statistic: {adf_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Used lags: {used_lags}")
    print(f"Observations: {observations}")

    print("Critical values:")
    for level, value in critical_values.items():
        print(f"  {level}: {value:.4f}")

    if p_value < 0.05:
        print("Conclusion: Reject non-stationarity. Series is likely stationary.")
    else:
        print("Conclusion: Cannot reject non-stationarity. Series may be non-stationary.")


# ------------------------------------------------------------
# Run ADF tests
# ------------------------------------------------------------

for name, series in series_dict.items():
    run_adf_test(series, name)

# ------------------------------------------------------------
# Plot series
# ------------------------------------------------------------
# Visual test:
# A non-stationary series often trends or changes level over time.
#
# A stationary series usually fluctuates around a relatively stable mean.
# ------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

axes[0].plot(prices)
axes[0].set_title("HSI Price Level")
axes[0].set_ylabel("Price")

axes[1].plot(simple_returns)
axes[1].set_title("HSI Simple Returns")
axes[1].set_ylabel("Return")

axes[2].plot(log_returns)
axes[2].set_title("HSI Log Returns")
axes[2].set_ylabel("Log Return")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# ACF plots
# ------------------------------------------------------------
# ACF = Autocorrelation Function.
#
# Autocorrelation means correlation of a series with its own past values.
#
# Example:
# lag 1 autocorrelation asks:
# Is today's value related to yesterday's value?
#
# Non-stationary price levels often show very high autocorrelation.
# Returns usually show much lower autocorrelation.
# ------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

plot_acf(prices, lags=40, ax=axes[0])
axes[0].set_title("ACF: HSI Price Level")

plot_acf(simple_returns, lags=40, ax=axes[1])
axes[1].set_title("ACF: HSI Simple Returns")

plot_acf(log_returns, lags=40, ax=axes[2])
axes[2].set_title("ACF: HSI Log Returns")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Spurious regression demonstration
# ------------------------------------------------------------
# Spurious regression:
# A regression that looks statistically strong only because two series trend,
# not because they have a real economic relationship.
#
# Here we compare HSI price to an unrelated cumulative series built from
# shuffled HSI returns.
#
# The shuffled series has no true time relationship to HSI, but because it is
# cumulative/trending, it can still produce misleading regression results.
# ------------------------------------------------------------

shuffled_returns = simple_returns.sample(frac=1, random_state=42).reset_index(drop=True)
fake_price = 100 * (1 + shuffled_returns).cumprod()
fake_price.index = prices.iloc[1:].index[:len(fake_price)]

aligned = pd.DataFrame({
    "HSI_price": prices.iloc[1:].iloc[:len(fake_price)],
    "Fake_price": fake_price,
}).dropna()

Y = aligned["HSI_price"]
X = sm.add_constant(aligned["Fake_price"])

spurious_model = sm.OLS(Y, X).fit()

print("\nSpurious Regression Demonstration")
print(f"R-squared: {spurious_model.rsquared:.2%}")
print(f"P-value for fake price coefficient: {spurious_model.pvalues['Fake_price']:.4f}")
print(
    "Interpretation: A high R-squared or low p-value here would be misleading because "
    "the fake price was built from shuffled returns and has no real time relationship to HSI."
)
