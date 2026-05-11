# AutoResearch Agent Instructions

## Objective
Maximize the **Information Ratio (IR)** of Congressional trades.

## Success Criteria
1. **IR** is the primary metric.
2. **KEEP** if `Current_IR > Previous_Best_IR`.
3. **DISCARD** and revert `model.py` if `Current_IR <= Previous_Best_IR`.

## Rules
- You may ONLY modify `model.py`, `results.tsv`, and `performance.png`
- `prepare.py` and `run.py` are FROZEN — do not touch them
- After completing the iterations, update `results.tsv` and `performance.png` with the results
- Do not use future data, refer to disclosure-date entry
- Strategies must generalize across different parties and committees
- Training and evaluation must complete in under 60 seconds on CPU
- No additional data sources or external downloads

## Workflow

```
1. Read current model.py
2. Propose a modification
3. Edit model.py
4. Run: python run.py "description of change"
5. Check Information Ratio in output
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