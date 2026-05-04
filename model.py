from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_model():
    """
    AI Agent: Modify this pipeline to improve Alpha.
    Try GradientBoosting, different depths, or feature engineering.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=10, random_state=42, n_jobs=-1))
    ])
