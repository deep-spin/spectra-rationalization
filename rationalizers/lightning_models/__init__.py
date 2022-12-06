from rationalizers.lightning_models.highlights.bernoulli import BernoulliRationalizer
from rationalizers.lightning_models.highlights.spectra import SPECTRARationalizer
from rationalizers.lightning_models.highlights.sparsemax import SparsemaxRationalizer
from rationalizers.lightning_models.highlights.relaxed_bernoulli import RelaxedBernoulliRationalizer
from rationalizers.lightning_models.highlights.hardkuma import HardKumaRationalizer
from rationalizers.lightning_models.highlights.transformers.info_bottleneck import TransformerInfoBottleneckRationalizer
from rationalizers.lightning_models.highlights.transformers.spectra_cf_expl import ExplSupervisedTransformerSPECTRARationalizer
from rationalizers.lightning_models.highlights.transformers.spectra_cf_semi import SemiSupervisedCfTransformerSPECTRARationalizer
from rationalizers.lightning_models.highlights.transformers.spectra_cf_supervised import SupervisedCfTransformerSPECTRARationalizer
from rationalizers.lightning_models.highlights.vanilla import VanillaClassifier
from rationalizers.lightning_models.highlights.transformers.spectra import TransformerSPECTRARationalizer
from rationalizers.lightning_models.highlights.transformers.spectra_cf import CounterfactualTransformerSPECTRARationalizer
from rationalizers.lightning_models.highlights.transformers.bernoulli import TransformerBernoulliRationalizer
from rationalizers.lightning_models.highlights.transformers.hardkuma import TransformerHardKumaRationalizer
from rationalizers.lightning_models.highlights.transformers.baseline import TransformerBaselineClassifier
from rationalizers.lightning_models.matchings.faithful_sparsemap_matching import SparseMAPFaithfulMatching
from rationalizers.lightning_models.matchings.gumbel_matching import GumbelMatching
from rationalizers.lightning_models.matchings.esim_matching import ESIMMatching
from rationalizers.lightning_models.edits.base import BaseEditor

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
    "transformer_spectra": TransformerSPECTRARationalizer,
    "transformer_spectra_cf": CounterfactualTransformerSPECTRARationalizer,
    "transformer_spectra_cf_semi_supervised": SemiSupervisedCfTransformerSPECTRARationalizer,
    "transformer_spectra_cf_expl_supervised": ExplSupervisedTransformerSPECTRARationalizer,
    "transformer_spectra_cf_supervised": SupervisedCfTransformerSPECTRARationalizer,
    "transformer_bernoulli": TransformerBernoulliRationalizer,
    "transformer_hardkuma": TransformerHardKumaRationalizer,
    "transformer_info_bottleneck": TransformerInfoBottleneckRationalizer,
    "transformer_baseline": TransformerBaselineClassifier,
    "editor": BaseEditor,
}
