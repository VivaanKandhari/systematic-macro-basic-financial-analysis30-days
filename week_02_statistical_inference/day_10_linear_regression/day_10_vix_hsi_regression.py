import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ------------------------------------------------------------
# Day 10 Mini Project: Macro Predictor Test
# ------------------------------------------------------------
# Research question:
# Does the VIX, often called the "fear index", explain HSI daily returns?
#
# Model:
# HSI_return = alpha + beta * VIX_change + error
#
# Important:
# This is a same-day explanatory regression.
# It tests relationship, not necessarily a tradable prediction.
# For prediction, we would need lagged variables.
# ------------------------------------------------------------

tickers = {
    "HSI": "^HSI",
    "VIX": "^VIX",
}

data = yf.download(
    list(tickers.values()),
    period="10y",
    auto_adjust=True
)

prices = data["Close"]

reverse_names = {
    "^HSI": "HSI",
    "^VIX": "VIX",
}

prices = prices.rename(columns=reverse_names)

# ------------------------------------------------------------
# Build variables
# ------------------------------------------------------------
# HSI_return:
# Daily percentage return of the Hang Seng Index.
#
# VIX_change:
# Daily percentage change in VIX.
#
# We use percentage changes for both so that the variables are more comparable
# and stationary than raw price/index levels.
# ------------------------------------------------------------

hsi_return = prices["HSI"].pct_change()
vix_change = prices["VIX"].pct_change()

df = pd.DataFrame({
    "HSI_return": hsi_return,
    "VIX_change": vix_change,
}).dropna()

# ------------------------------------------------------------
# Regression setup
# ------------------------------------------------------------
# Y is the dependent variable:
# The thing we are trying to explain.
#
# X is the independent variable:
# The variable we use to explain Y.
#
# Here:
# Y = HSI_return
# X = VIX_change
# ------------------------------------------------------------

Y = df["HSI_return"]
X = df["VIX_change"]

# statsmodels does not automatically add an intercept.
# sm.add_constant(X) adds alpha to the regression.
#
# Without this, the regression line would be forced through zero,
# which is usually not what we want.
X_with_constant = sm.add_constant(X)

# ------------------------------------------------------------
# OLS regression
# ------------------------------------------------------------
# OLS = Ordinary Least Squares.
#
# It fits the line that minimizes the sum of squared residuals.
#
# Residual = actual Y - predicted Y
#
# Squaring residuals penalizes large errors more heavily and prevents
# positive and negative errors from cancelling out.
# ------------------------------------------------------------

model = sm.OLS(Y, X_with_constant).fit()

print(model.summary())

# ------------------------------------------------------------
# Extract key regression results
# ------------------------------------------------------------
# alpha:
# Predicted HSI return when VIX_change = 0.
#
# beta:
# Sensitivity of HSI returns to VIX changes.
# If beta is negative, HSI tends to fall when VIX rises.
#
# p-value for beta:
# Tests whether beta is statistically different from zero.
#
# R-squared:
# Fraction of HSI return variation explained by VIX changes.
# Example: R² = 0.05 means VIX explains 5% of daily HSI variation.
# ------------------------------------------------------------

alpha = model.params["const"]
beta = model.params["VIX_change"]
beta_p_value = model.pvalues["VIX_change"]
r_squared = model.rsquared

print("\nKey Regression Interpretation")
print(f"Alpha: {alpha:.4%}")
print(f"Beta: {beta:.4f}")
print(f"Beta p-value: {beta_p_value:.4f}")
print(f"R-squared: {r_squared:.2%}")

if beta < 0:
    print("Beta is negative: HSI tends to fall when VIX rises.")
else:
    print("Beta is positive: HSI tends to rise when VIX rises.")

if beta_p_value < 0.05:
    print("Beta is statistically significant at the 5% level.")
else:
    print("Beta is NOT statistically significant at the 5% level.")

print(f"VIX changes explain about {r_squared:.2%} of HSI daily return variation.")

# ------------------------------------------------------------
# Plot 1: Scatter plot with regression line
# ------------------------------------------------------------
# Each dot is one trading day.
#
# x-axis = VIX daily change
# y-axis = HSI daily return
#
# The red line is the regression line:
# predicted HSI_return = alpha + beta * VIX_change
# ------------------------------------------------------------

df["predicted_HSI_return"] = model.predict(X_with_constant)
df["residual"] = df["HSI_return"] - df["predicted_HSI_return"]

plt.figure(figsize=(10, 6))
plt.scatter(df["VIX_change"], df["HSI_return"], alpha=0.4, label="Daily observations")
plt.plot(
    df["VIX_change"],
    df["predicted_HSI_return"],
    color="red",
    linewidth=2,
    label="Regression line"
)
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.title("HSI Returns vs VIX Changes")
plt.xlabel("VIX daily change")
plt.ylabel("HSI daily return")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Plot 2: Residuals over time
# ------------------------------------------------------------
# Residuals show what the model failed to explain.
#
# Large residuals mean HSI moved much more or less than the VIX model predicted.
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["residual"], label="Regression residuals")
plt.axhline(0, color="black", linewidth=1)
plt.title("Regression Residuals Over Time")
plt.xlabel("Date")
plt.ylabel("Actual HSI return - predicted HSI return")
plt.legend()
plt.grid(True)
plt.show()
