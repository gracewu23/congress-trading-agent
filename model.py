from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_model():
    """
    AI Agent: Modify this pipeline to improve Alpha.
    Try GradientBoosting, different depths, or feature engineering.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ExtraTreesRegressor(n_estimators=800, max_depth=1, min_samples_leaf=50, random_state=42, n_jobs=-1))
    ])
