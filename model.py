from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_model():
    """
    AI Agent: Modify this pipeline to improve Alpha.
    Try GradientBoosting, different depths, or feature engineering.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000))
    ])
