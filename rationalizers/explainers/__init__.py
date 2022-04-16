from rationalizers.explainers.attention import AttentionExplainer
from rationalizers.explainers.sparsemap import SparseMAPExplainer

available_explainers = {
    "attention": AttentionExplainer,
    "sparsemap": SparseMAPExplainer,
}
