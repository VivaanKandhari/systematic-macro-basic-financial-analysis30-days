"""
Day 8 Mini Project: Day-of-Week Effect Test

Research question:
    Does the Hang Seng Index (^HSI) perform differently on different weekdays?

In quant research, we do not trust a pattern just because the average return
looks different. Markets are noisy, so we use statistical tests to ask whether
the observed difference is likely real or just luck.
"""

import yfinance as yf
import pandas as pd
from scipy.stats import f_oneway, ttest_ind


TICKER = "^HSI"
ALPHA = 0.05
TRANSACTION_COST = 0.001  # 0.10% = 10 basis points = 10 bps
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def download_hsi_returns() -> pd.Series:
    """Download HSI prices and convert them into daily returns."""
    data = yf.download(TICKER, period="10y", auto_adjust=True)

    prices = data["Close"]

    # yfinance sometimes returns a DataFrame even for one ticker.
    # This converts it into a clean one-dimensional Series.
    if isinstance(prices, pd.DataFrame):
        prices = prices[TICKER]

    # Daily return = percentage change from one closing price to the next.
    # This is the basic unit of analysis for most market statistics.
    return prices.pct_change().dropna()


def build_weekday_frame(returns: pd.Series) -> pd.DataFrame:
    """Create a DataFrame with returns and weekday names."""
    df = pd.DataFrame({"return": returns})

    # Extract weekday name from the date index:
    # Monday, Tuesday, Wednesday, Thursday, Friday.
    df["weekday"] = df.index.day_name()

    return df


def print_weekday_summary(df: pd.DataFrame) -> None:
    """Print descriptive statistics by weekday."""
    # count  = number of observations for that weekday
    # mean   = average daily return
    # median = middle daily return after sorting
    # std    = standard deviation, also called volatility in finance
    weekday_summary = (
        df.groupby("weekday")["return"]
        .agg(["count", "mean", "median", "std"])
        .reindex(WEEKDAY_ORDER)
    )

    display_summary = weekday_summary.copy()

    # Convert decimals into readable percentages.
    for col in ["mean", "median", "std"]:
        display_summary[col] = display_summary[col].map(lambda x: f"{x:.4%}")

    print("Average HSI Returns By Weekday")
    print(display_summary.to_string())


def run_monday_t_test(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, float, float]:
    """Run Welch's t-test comparing Monday returns with all other days."""
    # Null hypothesis (H0):
    #     Monday returns are not different from returns on other days.
    #
    # Alternative hypothesis (H1):
    #     Monday returns are different from returns on other days.
    #
    # This is a two-sample t-test.
    #
    # A t-test compares the average of two groups while accounting for noise.
    # Here the two groups are:
    #     Group 1: Monday returns
    #     Group 2: non-Monday returns
    #
    # The t-statistic measures how large the difference in means is relative
    # to the variability/noise in the data.
    #
    # The p-value answers:
    #     If there were truly no Monday effect, how likely would it be to see
    #     a difference this large or larger just by random chance?
    #
    # Small p-value = result is harder to explain as pure luck.
    # Common cutoff: p < 0.05 means statistically significant at the 5% level.
    monday_returns = df[df["weekday"] == "Monday"]["return"]
    other_returns = df[df["weekday"] != "Monday"]["return"]

    t_stat, p_value = ttest_ind(
        monday_returns,
        other_returns,
        equal_var=False,  # Welch's t-test: does not assume equal variance
        nan_policy="omit",
    )

    print("\nMonday vs Other Days T-Test")
    print(f"Monday mean return: {monday_returns.mean():.4%}")
    print(f"Other days mean return: {other_returns.mean():.4%}")
    print(f"Difference: {(monday_returns.mean() - other_returns.mean()):.4%}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    return monday_returns, other_returns, t_stat, p_value


def run_weekday_anova(df: pd.DataFrame) -> tuple[float, float]:
    """Run one-way ANOVA across all weekdays."""
    # ANOVA = Analysis of Variance.
    #
    # It tests whether multiple groups have the same average.
    # Here the groups are:
    #     Monday, Tuesday, Wednesday, Thursday, Friday
    #
    # Null hypothesis (H0):
    #     All weekdays have the same average return.
    #
    # Alternative hypothesis (H1):
    #     At least one weekday has a different average return.
    #
    # ANOVA does not tell us which weekday is different.
    # It only tells us whether there is evidence that some weekday differs.
    #
    # The F-statistic compares:
    #     variation between weekday averages
    # vs
    #     variation inside each weekday group
    #
    # The p-value has the same interpretation:
    #     If weekdays truly had the same average return, how surprising is
    #     the observed difference across weekday averages?
    weekday_groups = [
        df[df["weekday"] == day]["return"]
        for day in WEEKDAY_ORDER
    ]

    anova_stat, anova_p_value = f_oneway(*weekday_groups)

    print("\nANOVA Test Across Weekdays")
    print(f"F-statistic: {anova_stat:.4f}")
    print(f"P-value: {anova_p_value:.4f}")

    return anova_stat, anova_p_value


def print_statistical_interpretation(p_value: float, anova_p_value: float) -> None:
    """Interpret p-values using the 5% significance level."""
    # alpha = significance level.
    #
    # alpha = 0.05 means we accept a 5% false-positive risk threshold.
    # False positive = we think there is an effect, but it was actually noise.
    print("\nStatistical Interpretation")

    if p_value < ALPHA:
        print("Monday returns are statistically different from other days at the 5% level.")
    else:
        print("Monday returns are NOT statistically different from other days at the 5% level.")

    if anova_p_value < ALPHA:
        print("At least one weekday appears statistically different from the others.")
    else:
        print("There is no strong evidence that weekday returns differ from each other.")


def print_tradeability_check(monday_returns: pd.Series, other_returns: pd.Series) -> None:
    """Check whether the Monday effect is larger than assumed costs."""
    # Statistical significance is not enough.
    #
    # A pattern can be statistically real but too small to trade.
    #
    # Economic significance asks:
    #     Is the effect large enough to survive transaction costs, slippage,
    #     bid-ask spread, taxes, and implementation constraints?
    #
    # Here we assume a simple round-trip transaction cost of 0.10%.
    effect_size = abs(monday_returns.mean() - other_returns.mean())

    print("\nTradeability Check")
    print(f"Absolute Monday effect size: {effect_size:.4%}")
    print(f"Assumed round-trip transaction cost: {TRANSACTION_COST:.2%}")

    if effect_size > TRANSACTION_COST:
        print("The effect is larger than the assumed transaction cost before considering risk/slippage.")
    else:
        print("The effect is smaller than the assumed transaction cost, so it is likely not tradeable.")


def main() -> None:
    returns = download_hsi_returns()
    df = build_weekday_frame(returns)

    print_weekday_summary(df)
    monday_returns, other_returns, _, p_value = run_monday_t_test(df)
    _, anova_p_value = run_weekday_anova(df)
    print_statistical_interpretation(p_value, anova_p_value)
    print_tradeability_check(monday_returns, other_returns)


if __name__ == "__main__":
    main()
