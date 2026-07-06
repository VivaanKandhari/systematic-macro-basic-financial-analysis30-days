import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ------------------------------------------------------------
# Day 11 Mini Project: Multi-Factor HSI Model
# ------------------------------------------------------------
# Research question:
# Can multiple macro variables explain HSI daily returns better than VIX alone?
#
# Model:
# HSI_return = alpha
#            + beta1 * VIX_change
#            + beta2 * Oil_return
#            + beta3 * SPX_return
#            + optional beta4 * USD_CNH_change
#            + error
#
# If USD/CNH has insufficient Yahoo data, we skip it.
# ------------------------------------------------------------

tickers = {
    "HSI": "^HSI",
    "VIX": "^VIX",
    "Oil": "CL=F",
    "S&P500": "^GSPC",
    "USD_CNH": "CNH=X",
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

print("Downloaded price columns:")
print(prices.columns.tolist())

print("\nNon-missing price counts before filtering:")
print(prices.count())

# ------------------------------------------------------------
# Drop columns with too little usable data
# ------------------------------------------------------------
# A column with only a few prices produces almost all missing returns.
# Then dropna() can delete the whole regression dataset.
#
# We require at least 500 usable prices for an asset to be included.
# ------------------------------------------------------------

min_required_prices = 500
prices = prices.loc[:, prices.count() >= min_required_prices]

print("\nColumns kept after dropping low-data columns:")
print(prices.columns.tolist())

print("\nNon-missing price counts after filtering:")
print(prices.count())

# ------------------------------------------------------------
# Required columns check
# ------------------------------------------------------------
# HSI is the target variable.
# VIX, Oil, and S&P500 are core predictors for this lesson.
# USD_CNH is optional because Yahoo data may fail.
# ------------------------------------------------------------

required_core_columns = ["HSI", "VIX", "Oil", "S&P500"]

missing_core = [
    col for col in required_core_columns
    if col not in prices.columns
]

if missing_core:
    raise ValueError(f"Missing required core columns: {missing_core}")

# ------------------------------------------------------------
# Build returns/changes
# ------------------------------------------------------------
# pct_change(fill_method=None) avoids a pandas FutureWarning.
#
# We use returns/changes rather than raw levels because regression on
# raw market levels can create misleading relationships.
# ------------------------------------------------------------

returns = prices.pct_change(fill_method=None)

df = pd.DataFrame(index=returns.index)

df["HSI_return"] = returns["HSI"]
df["VIX_change"] = returns["VIX"]
df["Oil_return"] = returns["Oil"]
df["SPX_return"] = returns["S&P500"]

if "USD_CNH" in returns.columns:
    df["USD_CNH_change"] = returns["USD_CNH"]
else:
    print("\nUSD_CNH skipped because it has insufficient usable data.")

print("\nRegression columns before dropna:")
print(df.columns.tolist())

print("\nNon-missing counts before dropna:")
print(df.count())

print("\nMissing counts before dropna:")
print(df.isna().sum())

# Drop rows where any regression variable is missing.
df = df.dropna()

print("\nRegression dataset shape after dropna:")
print(df.shape)

if df.empty:
    raise ValueError(
        "Regression dataset is still empty. Try reducing the asset list or checking ticker downloads."
    )

# ------------------------------------------------------------
# In-sample multiple regression
# ------------------------------------------------------------
# Y = dependent variable
# X = predictors
#
# Multiple regression estimates each predictor's relationship with HSI
# while controlling for the others.
# ------------------------------------------------------------

Y = df["HSI_return"]

predictor_columns = [col for col in df.columns if col != "HSI_return"]
X = df[predictor_columns]

# Add intercept/alpha.
X_with_constant = sm.add_constant(X)

model = sm.OLS(Y, X_with_constant).fit()

print("\nMULTI-FACTOR HSI MODEL")
print(model.summary())

# ------------------------------------------------------------
# Key coefficient table
# ------------------------------------------------------------

summary_table = pd.DataFrame({
    "Coefficient": model.params,
    "P-value": model.pvalues,
})

summary_table["Significant at 5%"] = summary_table["P-value"] < 0.05

print("\nKey Coefficient Table")
print(summary_table.to_string())

print("\nModel Fit")
print(f"R-squared: {model.rsquared:.2%}")
print(f"Adjusted R-squared: {model.rsquared_adj:.2%}")

# ------------------------------------------------------------
# VIF multicollinearity check
# ------------------------------------------------------------
# VIF = Variance Inflation Factor.
#
# It measures whether predictors overlap too much with each other.
#
# Rough guide:
# VIF = 1      no multicollinearity
# VIF 1-5      usually okay
# VIF > 5      concerning
# VIF > 10     serious
# ------------------------------------------------------------

X_for_vif = sm.add_constant(X)

vif_table = pd.DataFrame({
    "Variable": X_for_vif.columns,
    "VIF": [
        variance_inflation_factor(X_for_vif.values, i)
        for i in range(X_for_vif.shape[1])
    ],
})

print("\nVIF Multicollinearity Check")
print(vif_table.to_string(index=False))

# ------------------------------------------------------------
# Out-of-sample test
# ------------------------------------------------------------
# Train before 2023, test from 2023 onward.
#
# This checks whether the model generalizes to unseen data.
# ------------------------------------------------------------

train = df[df.index < "2023-01-01"]
test = df[df.index >= "2023-01-01"]

if len(train) == 0 or len(test) == 0:
    print("\nOUT-OF-SAMPLE TEST SKIPPED")
    print("Not enough train/test data after cleaning.")
else:
    Y_train = train["HSI_return"]
    X_train = train[predictor_columns]
    X_train = sm.add_constant(X_train)

    Y_test = test["HSI_return"]
    X_test = test[predictor_columns]
    X_test = sm.add_constant(X_test)

    oos_model = sm.OLS(Y_train, X_train).fit()

    test_predictions = oos_model.predict(X_test)

    # Out-of-sample R-squared:
    # Can be negative if the model predicts worse than a simple average.
    oos_r2 = r2_score(Y_test, test_predictions)

    # RMSE = Root Mean Squared Error.
    # It measures typical prediction error size.
    rmse = np.sqrt(mean_squared_error(Y_test, test_predictions))

    print("\nOUT-OF-SAMPLE TEST")
    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")
    print(f"Out-of-sample R-squared: {oos_r2:.2%}")
    print(f"RMSE: {rmse:.4%}")

# ------------------------------------------------------------
# Practical interpretation helper
# ------------------------------------------------------------

print("\nInterpretation Prompts")

significant_predictors = summary_table[
    (summary_table.index != "const") &
    (summary_table["Significant at 5%"])
].index.tolist()

if significant_predictors:
    print(f"Statistically significant predictors: {significant_predictors}")
else:
    print("No predictors are statistically significant at the 5% level.")

high_vif = vif_table[
    (vif_table["Variable"] != "const") &
    (vif_table["VIF"] > 5)
]

if len(high_vif) > 0:
    print("Potential multicollinearity concern:")
    print(high_vif.to_string(index=False))
else:
    print("No major VIF-based multicollinearity concern.")

if len(train) > 0 and len(test) > 0:
    if oos_r2 > 0:
        print("The model has positive out-of-sample explanatory power.")
    else:
        print("The model has weak or negative out-of-sample explanatory power.")
