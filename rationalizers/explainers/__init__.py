from rationalizers.explainers.attention import AttentionExplainer
from rationalizers.explainers.sparsemap import SparseMAPExplainer
from rationalizers.explainers.hardkuma import HardKumaExplainer
from rationalizers.explainers.bernoulli import BernoulliExplainer
from rationalizers.explainers.sparsemap_matching import SparseMAPMatchingExplainer

available_explainers = {
    "attention": AttentionExplainer,
    "sparsemap": SparseMAPExplainer,
    "hardkuma": HardKumaExplainer,
    "bernoulli": BernoulliExplainer,
    "sparsemap_matching": SparseMAPMatchingExplainer,
}
