from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

def build_model():
    """
    AI Agent: Modify this pipeline to improve Alpha.
    Try GradientBoosting, different depths, or feature engineering.
    """
    return Pipeline([
        ("model", GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=100,
            random_state=42,
        ))
    ])
