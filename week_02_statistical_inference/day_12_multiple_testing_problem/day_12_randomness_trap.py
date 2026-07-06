import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


# ------------------------------------------------------------
# Day 12 Mini Project: The Randomness Trap
# ------------------------------------------------------------
# Research question:
# If we test many completely random trading signals, how many will look
# statistically significant just by luck?
#
# Core lesson:
# p < 0.05 does not mean much if you tested many ideas.
# ------------------------------------------------------------

ticker = "^HSI"

data = yf.download(
    ticker,
    period="10y",
    auto_adjust=True,
    progress=False,
)

prices = data["Close"]

if isinstance(prices, pd.DataFrame):
    prices = prices[ticker]

returns = prices.pct_change(fill_method=None).dropna()

# ------------------------------------------------------------
# Generate random signals
# ------------------------------------------------------------
# A "signal" usually tells us whether to buy, sell, or stay out.
#
# Here we intentionally create fake/random signals.
# They have no real economic relationship to HSI.
#
# Signal values:
#   +1 = pretend buy/long signal
#   -1 = pretend sell/short signal
#
# Since these are random, any significant result should be false.
# ------------------------------------------------------------

np.random.seed(42)

num_signals = 100

results = []

for i in range(num_signals):
    random_signal = np.random.choice(
        [-1, 1],
        size=len(returns),
    )

    df = pd.DataFrame({
        "return": returns.values,
        "signal": random_signal,
    }, index=returns.index)

    long_returns = df[df["signal"] == 1]["return"]
    short_returns = df[df["signal"] == -1]["return"]

    # --------------------------------------------------------
    # T-test
    # --------------------------------------------------------
    # Null hypothesis:
    # Long-signal days and short-signal days have the same mean return.
    #
    # Alternative hypothesis:
    # Long-signal days and short-signal days have different mean returns.
    #
    # Since the signal is random, the null hypothesis should be true.
    #
    # But if we test many random signals, some may still get p < 0.05
    # just by chance.
    # --------------------------------------------------------

    t_stat, p_value = stats.ttest_ind(
        long_returns,
        short_returns,
        equal_var=False,
        nan_policy="omit",
    )

    effect = long_returns.mean() - short_returns.mean()

    results.append({
        "signal_id": i + 1,
        "long_mean": long_returns.mean(),
        "short_mean": short_returns.mean(),
        "effect": effect,
        "t_stat": t_stat,
        "p_value": p_value,
    })

results = pd.DataFrame(results)

# ------------------------------------------------------------
# Significance at normal 5% level
# ------------------------------------------------------------
# alpha = 0.05 means we accept a 5% false-positive risk per test.
#
# If we test 100 random signals, we expect around:
# 100 * 0.05 = 5 false positives
# ------------------------------------------------------------

alpha = 0.05

results["significant_5pct"] = results["p_value"] < alpha

num_significant = results["significant_5pct"].sum()

# ------------------------------------------------------------
# Bonferroni correction
# ------------------------------------------------------------
# Bonferroni adjusts for the fact that we tested many signals.
#
# adjusted_alpha = alpha / number_of_tests
#
# If alpha = 0.05 and tests = 100:
# adjusted_alpha = 0.0005
#
# This means a signal must have p < 0.0005 to survive correction.
# ------------------------------------------------------------

bonferroni_alpha = alpha / num_signals

results["significant_bonferroni"] = results["p_value"] < bonferroni_alpha

num_bonferroni = results["significant_bonferroni"].sum()

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

print("Randomness Trap Results")
print(f"Number of random signals tested: {num_signals}")
print(f"Normal significance threshold: {alpha}")
print(f"Expected false positives at 5%: {num_signals * alpha:.1f}")
print(f"Actual signals significant at 5%: {num_significant}")
print(f"Bonferroni-adjusted threshold: {bonferroni_alpha:.5f}")
print(f"Signals significant after Bonferroni correction: {num_bonferroni}")

print("\nTop 10 lowest p-values")
top_results = results.sort_values("p_value").head(10).copy()

for col in ["long_mean", "short_mean", "effect"]:
    top_results[col] = top_results[col].map(lambda x: f"{x:.4%}")

top_results["p_value"] = top_results["p_value"].map(lambda x: f"{x:.5f}")
top_results["t_stat"] = top_results["t_stat"].map(lambda x: f"{x:.3f}")

print(top_results.to_string(index=False))

# ------------------------------------------------------------
# Interpretation helper
# ------------------------------------------------------------

print("\nInterpretation")

if num_significant > 0:
    print(
        "Some random signals looked significant at p < 0.05. "
        "These are false positives because the signals were random."
    )
else:
    print(
        "No random signals were significant this run, but with many repeated experiments, "
        "false positives will appear."
    )

if num_bonferroni == 0:
    print(
        "After Bonferroni correction, no random signals survived. "
        "The correction helped remove false discoveries."
    )
else:
    print(
        "Some random signals survived even Bonferroni correction. "
        "This can happen rarely, especially with noisy data."
    )

print(
    "\nQuant lesson: If you test many strategies, p < 0.05 is not enough. "
    "You must account for how many ideas you tried."
)
