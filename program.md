# AutoResearch Agent Instructions

## Objective
Maximize **Alpha** (outperformance vs. S&P 500) using Congressional trade data.

## Success Criteria
1. Annualized Alpha > 5%.
2. Sharpe Ratio > 1.0.

## Rules
- You may ONLY modify `model.py`
- `prepare.py` and `run.py` are FROZEN — do not touch them
- Do not use future data; the `target_alpha` is calculated using disclosure-date entry
- Strategies must generalize across different parties and committees
- Training and evaluation must complete in under 60 seconds on CPU
- No additional data sources or external downloads

## Workflow

```
1. Read current model.py
2. Propose a modification
3. Edit model.py
4. Run: python run.py "description of change"
5. Check Alpha and Sharpe in output
6. If improved: git add model.py && git commit -m "feat: <description>"
7. If worse: git checkout model.py (revert)
8. Repeat from step 1
```

## Search Ideas
- Try `GradientBoostingRegressor` with `learning_rate` tuning.
- Add `PolynomialFeatures` to see if Party + Seniority creates a multiplier effect.
- Test `RobustScaler` to handle extreme outliers in trade volume.

- Different regressors: Ridge, Lasso, ElasticNet, SVR
- Ensemble methods: RandomForest, GradientBoosting, HistGradientBoosting
- Feature engineering: PolynomialFeatures, interaction terms
- Preprocessing: RobustScaler, QuantileTransformer
- Target transform: TransformedTargetRegressor with log
- Hyperparameter tuning within the pipeline

## What NOT To Do
- Do not modify `prepare.py` (data split, metric)
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- Do not change the function signature of `build_model()`