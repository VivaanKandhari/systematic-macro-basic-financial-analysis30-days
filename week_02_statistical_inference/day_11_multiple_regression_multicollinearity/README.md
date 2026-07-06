# Day 11: Multiple Regression & Multicollinearity

## Research Question

Can multiple macro variables explain Hang Seng Index daily returns better than VIX alone?

## Model

```text
HSI_return = alpha
           + beta1 * VIX_change
           + beta2 * Oil_return
           + beta3 * SPX_return
           + optional beta4 * USD_CNH_change
           + error
```

`USD_CNH_change` is included only when Yahoo Finance returns enough usable data. If the FX series is too sparse, the script skips it so the regression dataset does not become empty.

## Key Interpretation

- `coefficient`: estimated sensitivity of HSI returns to one predictor, holding the others constant.
- `p-value`: evidence that a coefficient is different from zero.
- `R-squared`: share of HSI daily return variation explained by the model.
- `adjusted R-squared`: R-squared with a penalty for adding extra predictors.
- `VIF`: variance inflation factor, used to detect multicollinearity between predictors.
- `out-of-sample R-squared`: whether the model explains unseen data after training on earlier observations.

## Run

```powershell
python week_02_statistical_inference/day_11_multiple_regression_multicollinearity/day_11_multi_factor_hsi_model.py
```
