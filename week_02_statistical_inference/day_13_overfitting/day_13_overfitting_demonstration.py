import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------------------------------------
# Day 13 Mini Project: Overfitting Demonstration
# ------------------------------------------------------------
# Research question:
# Does a more complex model predict HSI returns better, or does it just
# overfit historical noise?
#
# We compare polynomial regression models with different degrees:
# degree 1, 3, 5, 10, 20
#
# Higher degree = more flexible model.
# More flexible models can fit the past better, but may fail on new data.
# ------------------------------------------------------------

ticker = "^HSI"

data = yf.download(
    ticker,
    period="10y",
    auto_adjust=True,
    progress=False
)

prices = data["Close"]

if isinstance(prices, pd.DataFrame):
    prices = prices[ticker]

returns = prices.pct_change(fill_method=None).dropna()

df = pd.DataFrame({
    "HSI_return": returns
})

# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------
# We create simple lagged features.
#
# lag_1_return:
# Yesterday's return.
#
# lag_2_return:
# Return from two days ago.
#
# lag_5_return:
# Return from five days ago.
#
# rolling_5_return:
# Average return over the last 5 days.
#
# rolling_20_return:
# Average return over the last 20 days.
#
# Important:
# These features use only past information.
# That avoids look-ahead bias.
# ------------------------------------------------------------

df["lag_1_return"] = df["HSI_return"].shift(1)
df["lag_2_return"] = df["HSI_return"].shift(2)
df["lag_5_return"] = df["HSI_return"].shift(5)

df["rolling_5_return"] = df["HSI_return"].shift(1).rolling(5).mean()
df["rolling_20_return"] = df["HSI_return"].shift(1).rolling(20).mean()

df = df.dropna()

feature_columns = [
    "lag_1_return",
    "lag_2_return",
    "lag_5_return",
    "rolling_5_return",
    "rolling_20_return",
]

X = df[feature_columns]
Y = df["HSI_return"]

# ------------------------------------------------------------
# Train/test split
# ------------------------------------------------------------
# In-sample / training data:
# Used to fit the model.
#
# Out-of-sample / testing data:
# Held back until after training.
# This checks whether the model generalizes to unseen data.
#
# We use a time-based split, not random split.
# Why?
# Financial data is time-ordered. Random splitting can leak future
# information into the training set.
# ------------------------------------------------------------

train = df[df.index < "2023-01-01"]
test = df[df.index >= "2023-01-01"]

X_train = train[feature_columns]
Y_train = train["HSI_return"]

X_test = test[feature_columns]
Y_test = test["HSI_return"]

degrees = [1, 3, 5, 10, 20]

results = []

for degree in degrees:
    # --------------------------------------------------------
    # Polynomial regression
    # --------------------------------------------------------
    # PolynomialFeatures creates nonlinear combinations of features.
    #
    # degree 1:
    # Simple linear regression.
    #
    # degree 3, 5, 10, 20:
    # Increasingly flexible models.
    #
    # StandardScaler rescales features so polynomial terms behave better.
    # --------------------------------------------------------

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    model.fit(X_train, Y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(Y_train, train_pred)
    test_r2 = r2_score(Y_test, test_pred)

    train_rmse = np.sqrt(mean_squared_error(Y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(Y_test, test_pred))

    results.append({
        "degree": degree,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
    })

results = pd.DataFrame(results)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

display_results = results.copy()
display_results["train_r2"] = display_results["train_r2"].map(lambda x: f"{x:.2%}")
display_results["test_r2"] = display_results["test_r2"].map(lambda x: f"{x:.2%}")
display_results["train_rmse"] = display_results["train_rmse"].map(lambda x: f"{x:.4%}")
display_results["test_rmse"] = display_results["test_rmse"].map(lambda x: f"{x:.4%}")

print("Overfitting Demonstration Results")
print(display_results.to_string(index=False))

# ------------------------------------------------------------
# Plot in-sample vs out-of-sample R²
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(results["degree"], results["train_r2"], marker="o", label="In-sample R²")
plt.plot(results["degree"], results["test_r2"], marker="o", label="Out-of-sample R²")

plt.axhline(0, color="black", linewidth=1)
plt.title("Overfitting: In-Sample vs Out-of-Sample R²")
plt.xlabel("Polynomial Degree")
plt.ylabel("R²")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Plot train vs test RMSE
# ------------------------------------------------------------
# RMSE = Root Mean Squared Error.
# It measures typical prediction error size.
#
# Lower RMSE is better.
#
# Overfitting often shows up as:
# train RMSE decreases
# test RMSE increases
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(results["degree"], results["train_rmse"], marker="o", label="Train RMSE")
plt.plot(results["degree"], results["test_rmse"], marker="o", label="Test RMSE")

plt.title("Overfitting: Train vs Test Prediction Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Interpretation helper
# ------------------------------------------------------------

best_train = results.loc[results["train_r2"].idxmax()]
best_test = results.loc[results["test_r2"].idxmax()]

print("\nInterpretation")
print(f"Best in-sample degree: {int(best_train['degree'])}, train R² = {best_train['train_r2']:.2%}")
print(f"Best out-of-sample degree: {int(best_test['degree'])}, test R² = {best_test['test_r2']:.2%}")

if best_train["degree"] != best_test["degree"]:
    print(
        "The model that fit the training data best was not the model that performed best out-of-sample. "
        "This is evidence of overfitting."
    )
else:
    print(
        "The same model performed best in-sample and out-of-sample in this run, but still check whether "
        "test performance is economically meaningful."
    )

print(
    "\n lesson: Better historical fit does not guarantee better future performance."
)

#The results show severe overfitting. Higher-degree polynomial models fit the training data much better, with degree 10 reaching nearly 100% training R², but out-of-sample performance collapsed. 
#Test R² became strongly negative and test RMSE exploded, meaning the complex models memorized historical noise rather than learning a stable predictive relationship. 
#The simple model was weak, but the complex models were actively dangerous.