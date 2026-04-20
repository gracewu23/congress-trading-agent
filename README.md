# Capitol Gains: An AI-Driven Analysis of Congressional Stock Performance vs. the S&P 500 (2020-2026)
## STAT 390-0 Data Science Project

This project aims to use an AutoResearch workflow to determine if Congressional stock trades maintain a statistically significant edge over the broader market (S&P 500) and to identify the specific features -- such as committee assignments or trade volume -- that predict better returns.

Data has shown that Congress members frequently outperform the stock market. In 2025, one-third of Congressional portfolios beat the S&P 500, with top traders seeing returns above 60%. This trend has persisted even after the Stop Trading on Congressional Knowledge (STOCK) Act was signed into law in 2012 requiring officials to disclose stock, bond, and commodity transactions within 30-45 days.

Data Sources:
- Congressional trades were sourced via the Quiver Quantitative API, filtered on 2020 to 2026
- Historical adjusted closing prices and S&P 500 (SPY) benchmarks were sourced via yfinance

The project would be considered successful if the optimized strategy produces a backtested portfolio return that exceeds the S&P 500 annualized return by at least 5% while maintaining a superior Sharpe Ratio compared to the baseline model.
