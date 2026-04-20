import pandas as pd
import yfinance as yf
import numpy as np

def run_baseline():
    # load dev data
    df = pd.read_csv('data/processed/dev_set.csv')
    
    # baseline: top 10 most traded tickers
    top_tickers = df['ticker'].value_counts().head(10).index.tolist()
    
    # fetch market data
    data = yf.download(top_tickers + ['SPY'], start='2024-01-01', end='2024-12-31')['Adj Close']
    returns = data.pct_change().dropna()
    
    # calculate strategy vs. benchmark
    portfolio_return = returns[top_tickers].mean(axis=1)
    spy_return = returns['SPY']
    
    # metrics
    alpha = (portfolio_return.mean() - spy_return.mean()) * 252
    sharpe = (portfolio_return.mean() / portfolio_return.std()) * np.sqrt(252)
    
    print(f"BASELINE RESULTS (2024):")
    print(f"Annualized Alpha: {alpha:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

if __name__ == "__main__":
    run_baseline()
