# Day 10: Linear Regression

## Research Question

Does the VIX, often called the fear index, explain Hang Seng Index daily returns?

## Model

```text
HSI_return = alpha + beta * VIX_change + error
```

This is a same-day explanatory regression. It tests whether VIX changes and HSI returns move together, not whether VIX changes are a directly tradable forecast of future HSI returns.

## Key Interpretation

- `alpha`: predicted HSI return when VIX change is zero.
- `beta`: HSI sensitivity to VIX changes.
- `p-value`: whether beta is statistically different from zero.
- `R-squared`: share of HSI daily return variation explained by VIX changes.
- `residuals`: what the model failed to explain.

## Run

```powershell
python week_02_statistical_inference/day_10_linear_regression/day_10_vix_hsi_regression.py
```
