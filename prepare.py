import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import csv

# --- CONSTANTS ---
RESULTS_FILE = "results.tsv"

def load_data():
    """
    Simulates loading your Quiver/Yahoo joined data.
    Replace the dummy data generation with: 
    df = pd.read_csv('data/processed/congress_trades.csv')
    """
    # Dummy data generation for the structure
    np.random.seed(42)
    n_samples = 1000
    data = {
        'party_encoded': np.random.randint(0, 2, n_samples),
        'committee_seniority': np.random.randint(1, 20, n_samples),
        'trade_volume': np.random.uniform(1000, 500000, n_samples),
        'is_senate': np.random.randint(0, 2, n_samples),
        'target_alpha': np.random.normal(0.02, 0.05, n_samples) # Stock return - SPY return
    }
    df = pd.DataFrame(data)
    
    X = df.drop(columns=['target_alpha'])
    y = df['target_alpha']
    
    # Time-series split: Train on first 80%, Val on last 20%
    split = int(0.8 * len(df))
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:], X.columns

def evaluate(model, X_val, y_val):
    """
    Calculates Alpha and Sharpe-like consistency.
    The agent wants to maximize 'mean_alpha'.
    """
    preds = model.predict(X_val)
    # Success Metric 1: Annualized Excess Return (Alpha)
    mean_alpha = np.mean(preds) 
    # Success Metric 2: Sharpe (Mean / Std Dev of returns)
    sharpe = np.mean(preds) / np.std(preds) if np.std(preds) != 0 else 0
    
    return mean_alpha, sharpe

def log_result(experiment_id, alpha, sharpe, status, description):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["experiment", "alpha", "sharpe", "status", "description"])
        writer.writerow([experiment_id, f"{alpha:.6f}", f"{sharpe:.6f}", status, description])