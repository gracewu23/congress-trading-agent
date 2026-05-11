from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_model():
    """
    AI Agent: Modify this pipeline to improve Information Ratio (IR).
    Experiment: ElasticNet with alpha 0.005.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=10000))
    ])
