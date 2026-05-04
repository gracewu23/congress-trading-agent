from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

def build_model():
    """
    AI Agent: Modify this pipeline to improve Alpha.
    Try GradientBoosting, different depths, or feature engineering.
    """
    return Pipeline([
        ("model", HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=8,
            l2_regularization=0.1,
            random_state=42,
        ))
    ])
