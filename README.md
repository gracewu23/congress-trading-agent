# Capitol Gains: An AI-Driven Analysis of Congressional Stock Performance vs. the S&P 500 (2020-2026)

## STAT 390-0 Data Science Project
This project aims to use an AutoResearch workflow to determine if Congressional stock trades maintain a statistically significant edge over the broader market (S&P 500) and to identify the specific features -- such as committee assignments or trade volume -- that predict better returns.

Data has shown that Congress members frequently outperform the stock market. In 2025, one-third of Congressional portfolios beat the S&P 500, with top traders seeing returns above 60%. This trend has persisted even after the Stop Trading on Congressional Knowledge (STOCK) Act was signed into law in 2012 requiring officials to disclose stock, bond, and commodity transactions within 30-45 days.

## Data Sources:
- Congressional trades were sourced via the Quiver Quantitative API, filtered on 2020 to 2026
- Historical adjusted closing prices and S&P 500 (SPY) benchmarks were sourced via yfinance

## Success Criteria
- **Information Ratio (IR)** > 0.5

The project is considered successful if the optimized strategy produces an Information Ratio that significantly exceeds the baseline model. By focusing exclusively on IR, the model is forced to maximize Alpha (outperformance) while simultaneously minimizing Tracking Error (volatility relative to the S&P 500).

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
3. Establish the baseline: Run python run.py "Baseline: ElasticNet" --baseline. Note the Information Ratio (IR).

Now, enter the AutoResearch loop for at least 6 iterations:

1. PROPOSE one modification to model.py (e.g., try RandomForestRegressor, HistGradientBoostingRegressor, or add PolynomialFeatures to capture committee/party interactions).
2. EDIT model.py with your proposed change.
3. RUN the experiment: python run.py "<short description of your change>".
4. EVALUATE the result based on the Information Ratio (IR):
   - If IR improved compared to the current best: KEEP the change and commit model.py. Mark as KEEP in the summary table.
   - If IR worsened or stayed flat: REVERT model.py to the previous best version. Mark as DISCARD in the summary table.
5. REPEAT until you have completed at least 6 unique experiments.

After all iterations, print a summary table of every experiment, the IR results, and whether you KEPT or DISCARDED the change.

## Plotting Results
After running experiments:

```bash
python prepare.py
# Generates performance.png from results.tsv
```

This produces a two-panel chart:
- **Top (Alpha Progression)**: Tracks the raw annualized Alpha per iteration for context.
- **Bottom (Information Ratio)**: The primary success metric tracking your strategy's consistency relative to the S&P 500.
- **Blue line**: Shows the "Best-so-far" envelope for the Information Ratio.