from rationalizers.lightning_models.highlights.bernoulli import BernoulliRationalizer
from rationalizers.lightning_models.highlights.transformer import TransformerRationalizer
from rationalizers.lightning_models.highlights.transformer_counterfactual import CounterfactualTransformerRationalizer
from rationalizers.lightning_models.highlights.spectra import SPECTRARationalizer
from rationalizers.lightning_models.highlights.sparsemax import SparsemaxRationalizer
from rationalizers.lightning_models.matchings.faithful_sparsemap_matching import (
    SparseMAPFaithfulMatching,
)
from rationalizers.lightning_models.matchings.gumbel_matching import GumbelMatching
from rationalizers.lightning_models.highlights.relaxed_bernoulli import (
    RelaxedBernoulliRationalizer,
)
from rationalizers.lightning_models.highlights.hardkuma import HardKumaRationalizer
from rationalizers.lightning_models.highlights.vanilla import VanillaClassifier
from rationalizers.lightning_models.matchings.esim_matching import ESIMMatching

available_models = {
    "bernoulli": BernoulliRationalizer,
    "sparsemax": SparsemaxRationalizer,
    "spectra": SPECTRARationalizer,
    "sparsemap_faithfulmatching": SparseMAPFaithfulMatching,
    "gumbel_matching": GumbelMatching,
    "relaxed_bernoulli": RelaxedBernoulliRationalizer,
    "vanilla": VanillaClassifier,
    "esim": ESIMMatching,
    "hardkuma": HardKumaRationalizer,
    "transformer": TransformerRationalizer,
    "cf_transformer": CounterfactualTransformerRationalizer,
}
