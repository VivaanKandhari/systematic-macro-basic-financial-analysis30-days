# Day 13: Overfitting

## Concept

Overfitting happens when a model fits historical noise instead of a stable pattern. In finance, this is especially dangerous because daily returns are noisy, regimes change, and a model that looks excellent in-sample can fail badly out-of-sample.

## Market Connection

A trading model is useful only if it generalizes to unseen data; a beautiful backtest is not proof of predictive power.

## Mini Project

The mini project demonstrates overfitting with polynomial regression on Hang Seng Index returns.

It compares polynomial models with degrees:

- 1
- 3
- 5
- 10
- 20

For each model, it reports:

- In-sample R-squared
- Out-of-sample R-squared
- Train RMSE
- Test RMSE

## Key Lesson

Higher-complexity models can fit the training data better, but they may generalize much worse. In this project, the best in-sample model is not necessarily the safest trading model.

## Run

```bash
python week_02_statistical_inference/day_13_overfitting/day_13_overfitting_demonstration.py
```
