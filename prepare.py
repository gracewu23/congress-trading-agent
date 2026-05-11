import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

RESULTS_FILE = "results.tsv"

def load_data():
    """Simulates loading Congressional trade data."""
    np.random.seed(42)
    n_samples = 1000
    data = {
        'party_encoded': np.random.randint(0, 2, n_samples),
        'committee_seniority': np.random.randint(1, 20, n_samples),
        'trade_volume': np.random.uniform(1000, 500000, n_samples),
        'is_senate': np.random.randint(0, 2, n_samples),
        'target_alpha': np.random.normal(0.02, 0.05, n_samples)
    }
    df = pd.DataFrame(data)
    X = df.drop(columns=['target_alpha'])
    y = df['target_alpha']
    split = int(0.8 * len(df))
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:], X.columns

def evaluate(model, X_val, y_val):
    """Calculates Alpha and Information Ratio (IR)."""
    preds = model.predict(X_val)
    mean_alpha = np.mean(preds) 
    
    # Information Ratio = Excess Return / Tracking Error (Std Dev of Excess Return)
    tracking_error = np.std(preds)
    ir = mean_alpha / tracking_error if tracking_error > 1e-6 else 0
    return mean_alpha, ir

def log_result(experiment_id, alpha, ir, status, description):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["experiment", "alpha", "ir", "status", "description"])
        writer.writerow([experiment_id, f"{alpha:.6f}", f"{ir:.6f}", status, description])

def plot_results(save_path="performance.png"):
    if not os.path.exists(RESULTS_FILE):
        return
    df = pd.read_csv(RESULTS_FILE, sep="\t")
    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(s, "#95a5a6") for s in df["status"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.scatter(range(len(df)), df["alpha"], c=colors, s=100, zorder=3, edgecolors="white")
    ax1.plot(range(len(df)), df["alpha"], "k--", alpha=0.2)
    ax1.set_ylabel("Annualized Alpha", fontsize=12)
    ax1.set_title("Capitol Gains: IR Optimization Progress", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(range(len(df)), df["ir"], c=colors, s=100, zorder=3, edgecolors="white")
    best_ir = df["ir"].cummax()
    ax2.plot(range(len(df)), best_ir, color="#3498db", linewidth=2, label="Best IR (Baseline)")
    ax2.set_ylabel("Information Ratio (IR)", fontsize=12)
    ax2.set_xlabel("Experiment #", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    short_labels = [d[:20] + ".." if len(d) > 22 else d for d in df["description"]]
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(short_labels, rotation=40, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)

if __name__ == "__main__":
    plot_results()