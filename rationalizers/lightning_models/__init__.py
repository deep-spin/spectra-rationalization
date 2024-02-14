from rationalizers.lightning_models.highlights.bernoulli import BernoulliRationalizer
from rationalizers.lightning_models.highlights.spectra import SPECTRARationalizer, HSPECTRARationalizer
from rationalizers.lightning_models.highlights.mixed_spectra import MixedSPECTRARationalizer

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
    "h-spectra": HSPECTRARationalizer,
    "sparsemap_faithfulmatching": SparseMAPFaithfulMatching,
    "gumbel_matching": GumbelMatching,
    "relaxed_bernoulli": RelaxedBernoulliRationalizer,
    "vanilla": VanillaClassifier,
    "esim": ESIMMatching,
    "hardkuma": HardKumaRationalizer,
    "mixed_spectra": MixedSPECTRARationalizer,
}
