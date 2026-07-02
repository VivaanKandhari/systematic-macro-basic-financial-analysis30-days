"""
Day 9 Mini Project: Signal Strength Analyser

Research question:
    Is HSI's average return estimate precise enough to trust, and is the
    January effect statistically and practically meaningful?

Day 8 focused on whether a pattern was statistically significant. Day 9 adds
two more research questions:
    1. How uncertain is the estimate?
    2. Is the effect large enough to matter relative to volatility and costs?
"""

import yfinance as yf
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind


TICKER = "^HSI"
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05
TRANSACTION_COST = 0.001  # 0.10% = 10 basis points = 10 bps


def download_hsi_returns() -> pd.Series:
    """Download HSI prices and convert them into daily returns."""
    data = yf.download(TICKER, period="10y", auto_adjust=True)

    prices = data["Close"]

    # yfinance sometimes returns a DataFrame even for one ticker.
    # This converts it into a clean one-dimensional Series.
    if isinstance(prices, pd.DataFrame):
        prices = prices[TICKER]

    # Daily return = percentage change from one closing price to the next.
    return prices.pct_change().dropna()


def calculate_mean_confidence_interval(returns: pd.Series) -> tuple[float, float, float, float]:
    """Calculate a 95% confidence interval for the mean daily return."""
    sample_size = len(returns)
    mean_return = returns.mean()
    daily_volatility = returns.std()

    # Standard error measures uncertainty in the estimated mean.
    #
    # Formula:
    #     standard error = standard deviation / sqrt(sample size)
    #
    # More volatile returns make the mean estimate less precise.
    # More observations make the mean estimate more precise.
    standard_error = daily_volatility / (sample_size ** 0.5)

    # A confidence interval gives a plausible range for the true mean return.
    #
    # For a rough 95% confidence interval, we use:
    #     mean +/- z_critical * standard_error
    #
    # z_critical is about 1.96 for a 95% interval.
    # ppf means percent point function, which is the inverse of the normal CDF.
    # stats.norm.ppf(0.975) asks:
    #     What z-score has 97.5% of the normal distribution to its left?
    z_critical = stats.norm.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2)

    ci_lower = mean_return - z_critical * standard_error
    ci_upper = mean_return + z_critical * standard_error

    print("Mean Return Confidence Interval")
    print(f"Sample size: {sample_size}")
    print(f"Mean daily return: {mean_return:.4%}")
    print(f"Daily volatility: {daily_volatility:.2%}")
    print(f"Standard error: {standard_error:.4%}")
    print(f"95% confidence interval: [{ci_lower:.4%}, {ci_upper:.4%}]")

    if ci_lower <= 0 <= ci_upper:
        print("Interpretation: The confidence interval includes 0, so the true mean daily return may be zero.")
    else:
        print("Interpretation: The confidence interval does not include 0, so the mean return estimate is more convincing.")

    return mean_return, standard_error, ci_lower, ci_upper


def pooled_standard_deviation(group_1: pd.Series, group_2: pd.Series) -> float:
    """Calculate pooled standard deviation for Cohen's d."""
    # Pooled standard deviation is one combined volatility estimate for two groups.
    #
    # Cohen's d needs this because it asks:
    #     How large is the difference in means compared with typical noise?
    #
    # The formula weights each group's variance by its sample size, so the
    # group with more observations has more influence on the combined estimate.
    n1 = len(group_1)
    n2 = len(group_2)
    std1 = group_1.std()
    std2 = group_2.std()

    pooled_variance = ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
    return pooled_variance ** 0.5


def run_january_effect_test(returns: pd.Series) -> tuple[float, float, float]:
    """Test whether January returns differ from returns in other months."""
    df = pd.DataFrame({"return": returns})
    df["month"] = df.index.month

    january_returns = df[df["month"] == 1]["return"]
    other_month_returns = df[df["month"] != 1]["return"]

    # Null hypothesis (H0):
    #     January returns are not different from other-month returns.
    #
    # Alternative hypothesis (H1):
    #     January returns are different from other-month returns.
    #
    # A t-test compares the average of two groups while accounting for noise.
    # equal_var=False uses Welch's t-test, which does not assume both groups
    # have identical volatility.
    t_stat, p_value = ttest_ind(
        january_returns,
        other_month_returns,
        equal_var=False,
        nan_policy="omit",
    )

    january_mean = january_returns.mean()
    other_month_mean = other_month_returns.mean()
    difference = january_mean - other_month_mean

    # Cohen's d is an effect size measure.
    #
    # Formula:
    #     Cohen's d = difference in means / pooled standard deviation
    #
    # Market interpretation:
    #     How large is the January effect compared with normal daily volatility?
    pooled_std = pooled_standard_deviation(january_returns, other_month_returns)
    cohens_d = difference / pooled_std

    print("\nJanuary Effect Test")
    print(f"January observations: {len(january_returns)}")
    print(f"Other month observations: {len(other_month_returns)}")
    print(f"January mean return: {january_mean:.4%}")
    print(f"Other months mean return: {other_month_mean:.4%}")
    print(f"Difference: {difference:.4%}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.4f}")

    return difference, p_value, cohens_d


def print_signal_strength_verdict(difference: float, p_value: float, cohens_d: float) -> None:
    """Judge statistical and practical significance together."""
    # Statistical significance asks:
    #     Is the result unlikely to be pure chance?
    #
    # Practical significance asks:
    #     Is the effect large enough to matter after costs, slippage, and risk?
    #
    # A signal should ideally pass both tests before being taken seriously.
    statistically_significant = p_value < ALPHA
    practically_significant = abs(difference) > TRANSACTION_COST

    print("\nSignal Strength Verdict")

    if statistically_significant:
        print("Statistical test: Significant at the 5% level.")
    else:
        print("Statistical test: NOT significant at the 5% level.")

    if practically_significant:
        print("Practical test: Effect is larger than assumed transaction cost.")
    else:
        print("Practical test: Effect is smaller than assumed transaction cost.")

    # Cohen's d guide:
    #     around 0.2 = small effect
    #     around 0.5 = medium effect
    #     around 0.8 = large effect
    #
    # In daily market returns, even "small" effects can matter if they are
    # robust, cheap to trade, and persistent. But a small d with a high p-value
    # should be treated cautiously.
    if statistically_significant and practically_significant:
        verdict = "Potentially interesting, but still needs out-of-sample testing."
    elif practically_significant and not statistically_significant:
        verdict = "Large-looking effect, but not statistically reliable."
    elif statistically_significant and not practically_significant:
        verdict = "Statistically visible, but likely too small to trade."
    else:
        verdict = "Not statistically reliable and not economically attractive."

    print(f"Effect size note: Cohen's d = {cohens_d:.4f}")
    print(f"Final verdict: {verdict}")


def main() -> None:
    returns = download_hsi_returns()

    calculate_mean_confidence_interval(returns)
    difference, p_value, cohens_d = run_january_effect_test(returns)
    print_signal_strength_verdict(difference, p_value, cohens_d)


if __name__ == "__main__":
    main()
