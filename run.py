import sys
import time
from prepare import load_data, evaluate, log_result

def main():
    args = sys.argv[1:]
    status = "keep"
    desc = " ".join([a for a in args if not a.startswith("--")]) or "experiment"
    
    if "--baseline" in args: status = "baseline"
    if "--discard" in args: status = "discard"

    X_train, y_train, X_val, y_val, _ = load_data()

    from model import build_model
    model = build_model()
    
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    alpha, ir = evaluate(model, X_val, y_val)
    
    print(f"\n>>> Experiment: {desc}")
    print(f"Result: Alpha={alpha:.2%}, IR={ir:.4f}, Time={duration:.2f}s")
    log_result("exp", alpha, ir, status, desc)

if __name__ == "__main__":
    main()