from rationalizers.explainers.attention import AttentionExplainer
from rationalizers.explainers.sparsemap import SparseMAPExplainer
from rationalizers.explainers.hardkuma import HardKumaExplainer

available_explainers = {
    "attention": AttentionExplainer,
    "sparsemap": SparseMAPExplainer,
    "hardkuma": HardKumaExplainer,
}
