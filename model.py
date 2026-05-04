from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

def build_model():
    """
    AI Agent: Modify this pipeline to improve Alpha.
    Try GradientBoosting, different depths, or feature engineering.
    """
    return Pipeline([
        ("model", ExtraTreesRegressor(
            n_estimators=600,
            max_depth=2,
            min_samples_leaf=100,
            random_state=42,
            n_jobs=-1,
        ))
    ])
