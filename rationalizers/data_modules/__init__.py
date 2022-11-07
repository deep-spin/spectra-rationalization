from rationalizers.data_modules.ag_news import AgNewsDataModule
from rationalizers.data_modules.amazon import AmazonDataModule
from rationalizers.data_modules.beer import BeerDataModule
from rationalizers.data_modules.contrast_imdb import ContrastIMDBDataModule
from rationalizers.data_modules.contrast_imdb_cf import CounterfactualContrastIMDBDataModule
from rationalizers.data_modules.hans import HANSDataModule
from rationalizers.data_modules.hotel_location import HotelLocationDataModule
from rationalizers.data_modules.imdb import ImdbDataModule
from rationalizers.data_modules.imdb_augmented import AugmentedImdbDataModule
from rationalizers.data_modules.mlqepe import MLQEPEDataModule
from rationalizers.data_modules.multinli import MultiNLIDataModule
from rationalizers.data_modules.revised_imdb import RevisedIMDBDataModule
from rationalizers.data_modules.revised_imdb_cf import CounterfactualRevisedIMDBDataModule
from rationalizers.data_modules.revised_imdb_oversampled import OversampledRevisedIMDBDataModule
from rationalizers.data_modules.revised_mlqepe import RevisedMLQEPEDataModule
from rationalizers.data_modules.revised_mlqepe_cf import CounterfactualRevisedMLQEPEDataModule
from rationalizers.data_modules.revised_snli import RevisedSNLIDataModule
from rationalizers.data_modules.revised_snli_cf import CounterfactualRevisedSNLIDataModule
from rationalizers.data_modules.revised_snli_oversampled import OversampledRevisedSNLIDataModule
from rationalizers.data_modules.rottom import RotTomDataModule
from rationalizers.data_modules.snli import SNLIDataModule
from rationalizers.data_modules.sst import SSTDataModule
from rationalizers.data_modules.sst2 import SST2DataModule
from rationalizers.data_modules.yelp import YelpDataModule

available_data_modules = {
    "beer": BeerDataModule,
    "sst": SSTDataModule,
    "sst2": SST2DataModule,
    "rottom": RotTomDataModule,
    "amazon": AmazonDataModule,
    "yelp": YelpDataModule,
    "snli": SNLIDataModule,
    "ag_news": AgNewsDataModule,
    "imdb": ImdbDataModule,
    "imdb_augmented": AugmentedImdbDataModule,
    "multi_nli": MultiNLIDataModule,
    "hotel_location": HotelLocationDataModule,
    "hans": HANSDataModule,
    "mlqepe": MLQEPEDataModule,
    "revised_imdb": RevisedIMDBDataModule,
    "revised_snli": RevisedSNLIDataModule,
    "revised_mlqepe": RevisedMLQEPEDataModule,
    "contrast_imdb": ContrastIMDBDataModule,
    "revised_imdb_cf": CounterfactualRevisedIMDBDataModule,
    "revised_snli_cf": CounterfactualRevisedSNLIDataModule,
    "revised_mlqepe_cf": CounterfactualRevisedMLQEPEDataModule,
    "contrast_imdb_cf": CounterfactualContrastIMDBDataModule,
    "revised_imdb_oversampled": OversampledRevisedIMDBDataModule,
    "revised_snli_oversampled": OversampledRevisedSNLIDataModule,
}
