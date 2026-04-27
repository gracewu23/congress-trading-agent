# AutoResearch: Capitol Gains Contract

## Objective
Maximize **Alpha** (outperformance vs S&P 500) using Congressional trade data.

## Success Criteria
1. Annualized Alpha > 5%.
2. Sharpe Ratio > 1.0.

## Rules
- You may ONLY modify `model.py`.
- Do not use future data; the `target_alpha` is calculated using disclosure-date entry.
- Strategies must generalize across different parties and committees.

## Search Ideas
- Try `GradientBoostingRegressor` with `learning_rate` tuning.
- Add `PolynomialFeatures` to see if Party + Seniority creates a multiplier effect.
- Test `RobustScaler` to handle extreme outliers in trade volume.