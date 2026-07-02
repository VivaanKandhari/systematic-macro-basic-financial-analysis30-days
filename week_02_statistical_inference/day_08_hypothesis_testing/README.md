# Day 8 - Hypothesis Testing: Is This Pattern Real?

## Overview

Hypothesis testing is a way to check whether an observed market pattern is strong enough to take seriously, or whether it could easily have appeared by random chance. The default position in quant research is skeptical: assume there is no effect until the data provides strong evidence otherwise.

## Market Connection

A day-of-week pattern, such as "HSI performs differently on Mondays," is not useful just because the average Monday return looks different. A systematic macro researcher needs to test whether the effect is statistically significant, economically large enough, and potentially tradeable after costs.

## Mini Project

This folder contains `day_08_day_of_week_effect_test.py`, a self-contained script that tests whether Hang Seng Index returns differ by weekday.

The script:

- Downloads 10 years of Hang Seng Index (`^HSI`) data.
- Calculates daily returns.
- Summarizes mean, median, and volatility by weekday.
- Runs a Welch two-sample t-test comparing Monday returns against all other days.
- Runs ANOVA across Monday, Tuesday, Wednesday, Thursday, and Friday returns.
- Checks whether the Monday effect is larger than an assumed 10 bps round-trip transaction cost.

## Statistical Terms

- Null hypothesis: the default assumption that there is no effect.
- Alternative hypothesis: the claim that an effect exists.
- T-test: compares the average of two groups while accounting for noise.
- T-statistic: the signal-to-noise ratio of the difference in means.
- P-value: how surprising the result would be if the null hypothesis were true.
- ANOVA: tests whether any group among several groups has a different mean.
- F-statistic: compares variation between groups against variation within groups.
- Statistical significance: evidence that a pattern is unlikely to be pure chance.
- Economic significance: evidence that a pattern is large enough to matter after trading costs.

## Run

From the repository root:

```bash
python week_02_statistical_inference/day_08_hypothesis_testing/day_08_day_of_week_effect_test.py
```

## Research Conclusion Template

```text
The Monday effect was [positive/negative] in the sample, but the t-test produced a p-value of X, so the result [is/is not] statistically significant. ANOVA produced a p-value of Y, suggesting weekday returns [do/do not] differ meaningfully overall. The absolute Monday effect size was Z, compared with an assumed 0.10% transaction cost, so the pattern [is/is not] economically attractive.
```

