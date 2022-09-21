from rationalizers.explainers.attention import AttentionExplainer
from rationalizers.explainers.sparsemap import SparseMAPExplainer
from rationalizers.explainers.hardkuma import HardKumaExplainer
from rationalizers.explainers.bernoulli import BernoulliExplainer

available_explainers = {
    "attention": AttentionExplainer,
    "sparsemap": SparseMAPExplainer,
    "hardkuma": HardKumaExplainer,
    "bernoulli": BernoulliExplainer,
}
