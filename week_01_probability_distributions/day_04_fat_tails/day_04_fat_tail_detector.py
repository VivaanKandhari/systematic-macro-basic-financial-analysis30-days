import yfinance as yf
import pandas as pd
from scipy.stats import kurtosis, norm
from pathlib import Path
from tabulate import tabulate

tickers = {
    "HSI": "^HSI",
    "S&P 500": "^GSPC",
}

data = yf.download(
    list(tickers.values()),
    period="5y",
    auto_adjust=True
)

prices = data["Close"]

reverse_names = {
    "^HSI": "HSI",
    "^GSPC": "S&P 500",
}

prices = prices.rename(columns=reverse_names)
returns = prices.pct_change().dropna()

results = []

for asset in returns.columns:
    r = returns[asset].dropna()

    mean = r.mean()
    std = r.std()

    # scipy kurtosis with fisher=False gives normal distribution kurtosis = 3
    raw_kurtosis = kurtosis(r, fisher=False)
    excess_kurtosis = kurtosis(r, fisher=True)

    upper_3sigma = mean + 3 * std
    lower_3sigma = mean - 3 * std

    actual_3sigma_days = ((r > upper_3sigma) | (r < lower_3sigma)).sum()
    actual_3sigma_frequency = ((r > upper_3sigma) | (r < lower_3sigma)).mean()

    normal_3sigma_frequency = 2 * (1 - norm.cdf(3))
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

results_df = pd.DataFrame(results)

# Create clean display table with tabulate
display_data = []
for _, row in results_df.iterrows():
    display_data.append([
        row["Asset"],
        f"{row['Kurtosis']:.2f}",
        f"{row['Excess Kurtosis']:.2f}",
        f"{row['Actual 3-sigma days']:.0f}",
        f"{row['Expected 3-sigma days under normal']:.1f}",
        f"{row['Actual 3-sigma frequency']:.2%}",
        f"{row['Normal 3-sigma frequency']:.2%}",
        f"{row['Danger ratio']:.1f}x",
    ])

headers = [
    "Asset",
    "Kurtosis",
    "Excess Kurtosis",
    "Actual 3σ Days",
    "Expected 3σ Days",
    "Actual Freq",
    "Normal Freq",
    "Danger Ratio"
]

print("\n" + "="*110)
print("FAT TAIL DETECTOR: Do Markets Have Fatter Tails Than Normal Distribution Predicts?")
print("="*110 + "\n")

print(tabulate(display_data, headers=headers, tablefmt="grid"))

print("\n" + "="*110)
print("INTERPRETATION")
print("="*110 + "\n")

for _, row in results_df.iterrows():
    asset = row["Asset"]
    expected = row["Expected 3-sigma days under normal"]
    actual = row["Actual 3-sigma days"]
    danger = row["Danger ratio"]
    
    print(f"{asset}:")
    print(f"  • Normal model predicted:  {expected:.1f} extreme days over 5 years")
    print(f"  • Actually observed:       {actual:.0f} extreme days")
    print(f"  • Reality vs. Normal:      {danger:.1f}x MORE dangerous than expected")
    print()

print("="*110)
print("CONCLUSION")
print("="*110)
print("""
Markets exhibit FAT TAILS. Extreme moves happen much more frequently than
a normal distribution would predict. This means:

✓ Risk models assuming normal distribution significantly underestimate crash risk
✓ Leverage and oversized positions are riskier than naive statistics suggest
✓ Quant strategies must account for tail events and correlation breakdowns
""")
print("="*110 + "\n")

# Generate visualization
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt
from scipy.stats import probplot

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, asset in zip(axes, returns.columns):
    r = returns[asset].dropna()
    probplot(r, dist="norm", plot=ax)
    ax.set_title(f"{asset}: QQ-Plot vs Normal Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "qq_plots_vs_normal.png", dpi=150, bbox_inches="tight")
print(f"✓ Chart saved to: {OUTPUT_DIR / 'qq_plots_vs_normal.png'}")
plt.show()
