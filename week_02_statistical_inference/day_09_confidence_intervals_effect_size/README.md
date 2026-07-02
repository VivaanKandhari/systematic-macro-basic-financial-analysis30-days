# Day 9 - Confidence Intervals & Effect Size

## Overview

Day 9 moves beyond p-values. A p-value helps test whether a pattern could be noise, but a quant researcher also needs to know how uncertain the estimate is and whether the effect is large enough to matter in markets.

## Market Connection

Financial returns are noisy, so a single estimate like "HSI mean daily return is 0.013%" should not be treated as exact. Confidence intervals show the uncertainty around that estimate, while effect size helps judge whether a pattern is meaningful relative to volatility.

## Mini Project

This folder contains `day_09_signal_strength_analyser.py`, a self-contained script that studies uncertainty and effect size in Hang Seng Index returns.

The script:

- Downloads 10 years of Hang Seng Index (`^HSI`) data.
- Calculates daily returns.
- Computes a 95% confidence interval for HSI's mean daily return.
- Tests whether January returns differ from other months.
- Calculates Cohen's d for the January effect.
- Checks whether the effect is statistically significant and practically meaningful after an assumed transaction cost.

## Statistical Terms

- Confidence interval: a range of plausible values for an unknown true parameter.
- Standard error: uncertainty in an estimate, such as the sample mean.
- Z-critical value: the normal-distribution cutoff used to build a confidence interval.
- T-test: compares the average of two groups while accounting for noise.
- P-value: how surprising the observed result would be if the null hypothesis were true.
- Effect size: how large a difference is in practical terms.
- Cohen's d: difference in means divided by pooled standard deviation.
- Pooled standard deviation: one combined volatility estimate for two groups.
- Statistical significance: evidence that a result is unlikely to be pure chance.
- Practical significance: evidence that a result is large enough to matter after costs and risk.

## Run

From the repository root:

```bash
python week_02_statistical_inference/day_09_confidence_intervals_effect_size/day_09_signal_strength_analyser.py
```

## Research Conclusion Template

```text
HSI's estimated mean daily return was X, with a 95% confidence interval of [Y, Z]. Because the interval [does/does not] include zero, the true mean return [may/may not] plausibly be zero. January returns differed from other months by A, with p-value B and Cohen's d of C. The signal is [statistically significant/not statistically significant] and [practically meaningful/not practically meaningful], so the final verdict is [tradeable / interesting but weak / noise].
```
