# Day 12: The Multiple Testing Problem

## Research Question

If we test many completely random trading signals, how many will look statistically significant just by luck?

## Core Lesson

`p < 0.05` is not enough if many ideas were tested. With 100 independent fake signals and a 5% false-positive threshold, roughly 5 signals may appear significant by chance.

## Mini Project

The script generates 100 random long/short signals for Hang Seng Index returns and tests whether long-signal days have different mean returns from short-signal days.

It then compares:

- normal `p < 0.05` significance
- Bonferroni-corrected significance
- the top 10 lowest p-values

## Key Interpretation

- `long`: a signal that profits if the asset rises.
- `short`: a signal that profits if the asset falls.
- `p-value`: how surprising the observed result is if the signal has no real effect.
- `false positive`: a fake signal that looks significant by luck.
- `Bonferroni correction`: a stricter threshold equal to `alpha / number_of_tests`.

## Run

```powershell
python week_02_statistical_inference/day_12_multiple_testing_problem/day_12_randomness_trap.py
```
