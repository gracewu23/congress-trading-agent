# Capitol Gains: An AI-Driven Analysis of Congressional Stock Performance vs. the S&P 500 (2020-2026)

## STAT 390-0 Data Science Project
This project aims to use an AutoResearch workflow to determine if Congressional stock trades maintain a statistically significant edge over the broader market (S&P 500) and to identify the specific features -- such as committee assignments or trade volume -- that predict better returns.

Data has shown that Congress members frequently outperform the stock market. In 2025, one-third of Congressional portfolios beat the S&P 500, with top traders seeing returns above 60%. This trend has persisted even after the Stop Trading on Congressional Knowledge (STOCK) Act was signed into law in 2012 requiring officials to disclose stock, bond, and commodity transactions within 30-45 days.

## Data Sources:
- Congressional trades were sourced via the Quiver Quantitative API, filtered on 2020 to 2026
- Historical adjusted closing prices and S&P 500 (SPY) benchmarks were sourced via yfinance

## Success Criteria
- **Alpha:** > 5% annualized outperformance.
- **Sharpe Ratio:** > 1.0.

The project would be considered successful if the optimized strategy produces a backtested portfolio return that exceeds the S&P 500 annualized return by at least 5% while maintaining a superior Sharpe Ratio compared to the baseline model.

## Project Structure
- `prepare.py`: FROZEN — data loading, evaluation metric, plotting
- `model.py`: EDITABLE — agent modifies only this file
- `run.py`: Run a single experiment and log result
- `program.md`: Agent instructions (the agent reads this)
- `results.tsv`: Experiment log (auto-generated)
- `performance.png`: Performance plot (auto-generated)

The key rule is that the agent may only modify `model.py`. Everything else is frozen.

## How to Run
Copy-paste this prompt into your agent:

I have initialized an AutoResearch project for Congressional trade analysis. 

1. Read program.md to understand the financial objectives and constraints.
2. Read model.py to see the current model architecture.
3. Establish the baseline: Run `python run.py "Baseline: Ridge Regression" --baseline`. Note the Alpha and Sharpe Ratio.

Now, enter the AutoResearch loop for at least 6 iterations:

1. PROPOSE one modification to model.py (e.g., try RandomForest, HistGradientBoosting, or change hyperparameters like max_depth or n_estimators).
2. EDIT model.py with your proposed change.
3. RUN the experiment: `python run.py "<short description of your change>"`.
4. EVALUATE the result:
   - If Alpha OR Sharpe Ratio improved compared to the current best: KEEP the change and commit model.py. Note the iteration in the summary table and mark as KEEP.
   - If both metrics worsened or stayed flat: REVERT model.py to the previous best version. Still note the iteration in the summary table but mark as DISCARD.
5. REPEAT until you have completed at least 6 unique experiments.

After all iterations, print a summary table of every experiment, the Alpha/Sharpe results, and whether you KEPT or DISCARDED the change.

## Plotting Results
After running experiments:

```bash
python prepare.py
# Generates performance.png from results.tsv
```

This produces a two-panel chart:
- **Top (Alpha Progression)**: Tracks the annualized Alpha per iteration (green=keep, red=discard, blue=baseline)
- **Bottom (Sharpe Ratio)**: Visualizes the risk-adjusted consistency of your strategy
- **Green line**: Best-so-far envelope