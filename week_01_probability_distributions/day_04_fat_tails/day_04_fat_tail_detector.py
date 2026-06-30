# Day 4: Fat Tails & Why Normal Distribution Fails Markets
# Mini-project: Fat Tail Detector
#
# Research question:
# Do HSI and S&P 500 daily returns produce more 3-sigma events than a normal
# distribution predicts?
#
# Summary:
# This program compares actual extreme return days against the number of
# extreme days predicted by a fitted normal distribution.
#
# Conclusion to look for:
# If actual 3-sigma days are much higher than expected normal-model 3-sigma days,
# the market has fat tails. Extreme events happen more often than the naive
# bell curve suggests.

!pip install yfinance -q

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from scipy.stats import kurtosis, norm, probplot

try:
    from IPython.display import display
except ImportError:
    display = None


TICKERS = {
    "HSI": "^HSI",
    "S&P 500": "^GSPC",
}

PERIOD = "5y"
SIGMA_THRESHOLD = 3


data = yf.download(
    list(TICKERS.values()),
    period=PERIOD,
    auto_adjust=True,
    progress=False
)

prices = data["Close"]

reverse_names = {
    "^HSI": "HSI",
    "^GSPC": "S&P 500",
}

prices = prices.rename(columns=reverse_names)

returns = prices.pct_change(fill_method=None).dropna()

results = []

for asset in returns.columns:
    r = returns[asset].dropna()

    mean = r.mean()
    std = r.std()

    raw_kurtosis = kurtosis(r, fisher=False)
    excess_kurtosis = kurtosis(r, fisher=True)

    upper_3sigma = mean + SIGMA_THRESHOLD * std
    lower_3sigma = mean - SIGMA_THRESHOLD * std

    extreme_days = (r > upper_3sigma) | (r < lower_3sigma)

    actual_3sigma_days = extreme_days.sum()
    actual_3sigma_frequency = extreme_days.mean()

    # A standard normal distribution has equal left and right tails.
    # 1 - norm.cdf(3) is the probability of being above +3 sigma.
    # Multiplying by 2 counts both tails: below -3 sigma and above +3 sigma.
    normal_3sigma_frequency = 2 * (1 - norm.cdf(SIGMA_THRESHOLD))

    expected_3sigma_days = normal_3sigma_frequency * len(r)

    danger_ratio = actual_3sigma_days / expected_3sigma_days

    results.append({
        "Asset": asset,
        "Kurtosis": raw_kurtosis,
        "Excess Kurtosis": excess_kurtosis,
        "Actual 3-sigma days": actual_3sigma_days,
        "Expected 3-sigma days under normal": expected_3sigma_days,
        "Actual 3-sigma frequency": actual_3sigma_frequency,
        "Normal 3-sigma frequency": normal_3sigma_frequency,
        "Danger ratio": danger_ratio,
    })

results = pd.DataFrame(results)

display_results = pd.DataFrame({
    "Asset": results["Asset"],
    "Kurtosis": results["Kurtosis"].map(lambda x: f"{x:.2f}"),
    "Excess Kurtosis": results["Excess Kurtosis"].map(lambda x: f"{x:.2f}"),
    "Actual 3σ Days": results["Actual 3-sigma days"].map(lambda x: f"{x:.0f}"),
    "Expected 3σ Days": results["Expected 3-sigma days under normal"].map(lambda x: f"{x:.1f}"),
    "Actual Freq": results["Actual 3-sigma frequency"].map(lambda x: f"{x:.2%}"),
    "Normal Freq": results["Normal 3-sigma frequency"].map(lambda x: f"{x:.2%}"),
    "Danger Ratio": results["Danger ratio"].map(lambda x: f"{x:.1f}x"),
})

print("Fat Tail Detector")
print("3-sigma events = returns below -3 sigma or above +3 sigma")

if display is not None:
    display(display_results)
else:
    print(display_results.to_string(index=False))

print("\nConclusion Summary")

for _, row in results.iterrows():
    print()
    print(
        f"{row['Asset']}: normal distribution predicted "
        f"{row['Expected 3-sigma days under normal']:.1f} extreme days. "
        f"Actual: {row['Actual 3-sigma days']:.0f} days. "
        f"The market was {row['Danger ratio']:.1f}x more dangerous than "
        "the naive normal model suggested."
    )

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, asset in zip(axes, returns.columns):
    r = returns[asset].dropna()
    probplot(r, dist="norm", plot=ax)
    ax.set_title(f"{asset} QQ-Plot vs Normal")
    ax.grid(True)

plt.tight_layout()
plt.show()

# Final interpretation:
# A straight QQ-plot line would suggest normally distributed returns.
# When the tail points bend away from the line, the market has more extreme
# observations than the normal model expects.

# Macro quant takeaway:
# Do not rely blindly on normal-distribution risk models. They often underestimate
# crash risk, which can lead to oversized positions and excessive leverage.
