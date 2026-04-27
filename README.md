# Capitol Gains: AI Congressional Trade Analysis

An AutoResearch project investigating if Congressional stock trades outperform the S&P 500 (2020-2026).

## Success Criteria
- **Alpha:** > 5% annualized outperformance.
- **Sharpe Ratio:** > 1.0.

## Structure
- `prepare.py`: Frozen data loader and backtester.
- `model.py`: Editable ML pipeline.
- `run.py`: Experiment runner.
- `program.md`: Agent instructions.

## How to Run
1. Establish baseline: `python run.py "baseline" --baseline`
2. Run Agent Loop: Use a coding agent to iterate on `model.py` based on `program.md`.