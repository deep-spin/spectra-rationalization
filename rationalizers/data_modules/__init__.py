from rationalizers.data_modules.ag_news import AgNewsDataModule
from rationalizers.data_modules.amazon import AmazonDataModule
from rationalizers.data_modules.anli import AdversarialNLIDataModule
from rationalizers.data_modules.beer import BeerDataModule
from rationalizers.data_modules.bnli import BreakNLIDataModule
from rationalizers.data_modules.contrast_imdb import ContrastIMDBDataModule
from rationalizers.data_modules.contrast_imdb_cf import CounterfactualContrastIMDBDataModule
from rationalizers.data_modules.hans import HANSDataModule
from rationalizers.data_modules.hnli import HardNLIDataModule
from rationalizers.data_modules.hotel_location import HotelLocationDataModule
from rationalizers.data_modules.imdb import ImdbDataModule
from rationalizers.data_modules.imdb_aug import AugmentedImdbDataModule
from rationalizers.data_modules.imdb_syn import SyntheticImdbDataModule
from rationalizers.data_modules.imdb_syn_exp import SyntheticExplainImdbDataModule
from rationalizers.data_modules.mlqepe import MLQEPEDataModule
from rationalizers.data_modules.mnli import MultiNLIDataModule
from rationalizers.data_modules.movies import MoviesDataModule
from rationalizers.data_modules.multinli_old import OldMultiNLIDataModule
from rationalizers.data_modules.revised_imdb import RevisedIMDBDataModule
from rationalizers.data_modules.revised_imdb_cf import CounterfactualRevisedIMDBDataModule
from rationalizers.data_modules.revised_imdb_gold_z import GoldRationaleRevisedIMDBDataModule
from rationalizers.data_modules.revised_imdb_oversampled import OversampledRevisedIMDBDataModule
from rationalizers.data_modules.revised_imdb_syn import SyntheticRevisedIMDBDataModule
from rationalizers.data_modules.revised_mlqepe import RevisedMLQEPEDataModule
from rationalizers.data_modules.revised_mlqepe_cf import CounterfactualRevisedMLQEPEDataModule
from rationalizers.data_modules.revised_snli import RevisedSNLIDataModule
from rationalizers.data_modules.revised_snli_cf import CounterfactualRevisedSNLIDataModule
from rationalizers.data_modules.revised_snli_oversampled import OversampledRevisedSNLIDataModule
from rationalizers.data_modules.rottom import RotTomDataModule
from rationalizers.data_modules.snli import SNLIDataModule
from rationalizers.data_modules.snli_aug import AugmentedSNLIDataModule
from rationalizers.data_modules.snli_syn import SyntheticSNLIDataModule
from rationalizers.data_modules.snli_syn_exp import SyntheticExplainSNLIDataModule
from rationalizers.data_modules.sst import SSTDataModule
from rationalizers.data_modules.sst2 import SST2DataModule
from rationalizers.data_modules.twenty_news import TwentyNewsGroupsDataModule
from rationalizers.data_modules.wnli import WinogradNLIDataModule
from rationalizers.data_modules.yelp import YelpDataModule

available_data_modules = {
    "beer": BeerDataModule,
    "sst": SSTDataModule,
    "sst2": SST2DataModule,
    "rottom": RotTomDataModule,
    "amazon": AmazonDataModule,
    "yelp": YelpDataModule,
    "snli": SNLIDataModule,
    "snli_augmented": AugmentedSNLIDataModule,
    "snli_synthetic": SyntheticSNLIDataModule,
    "snli_synthetic_exp": SyntheticExplainSNLIDataModule,
    "ag_news": AgNewsDataModule,
    "imdb": ImdbDataModule,
    "imdb_augmented": AugmentedImdbDataModule,
    "imdb_synthetic": SyntheticImdbDataModule,
    "imdb_synthetic_exp": SyntheticExplainImdbDataModule,
    "multi_nli_old": OldMultiNLIDataModule,
    "hotel_location": HotelLocationDataModule,
    "hans": HANSDataModule,
    "mlqepe": MLQEPEDataModule,
    "revised_imdb": RevisedIMDBDataModule,
    "revised_snli": RevisedSNLIDataModule,
    "revised_mlqepe": RevisedMLQEPEDataModule,
    "contrast_imdb": ContrastIMDBDataModule,
    "revised_imdb_gold_z": GoldRationaleRevisedIMDBDataModule,
    "revised_imdb_cf": CounterfactualRevisedIMDBDataModule,
    "revised_snli_cf": CounterfactualRevisedSNLIDataModule,
    "revised_mlqepe_cf": CounterfactualRevisedMLQEPEDataModule,
    "contrast_imdb_cf": CounterfactualContrastIMDBDataModule,
    "revised_imdb_oversampled": OversampledRevisedIMDBDataModule,
    "revised_snli_oversampled": OversampledRevisedSNLIDataModule,
    "revised_imdb_synthetic": SyntheticRevisedIMDBDataModule,
    "mnli": MultiNLIDataModule,
    "anli": AdversarialNLIDataModule,
    "wnli": WinogradNLIDataModule,
    "hnli": HardNLIDataModule,
    "bnli": BreakNLIDataModule,
    "movies": MoviesDataModule,
    "20news": TwentyNewsGroupsDataModule,
}
