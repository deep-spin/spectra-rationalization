from rationalizers.data_modules.beer import BeerDataModule
from rationalizers.data_modules.sst import SSTDataModule
from rationalizers.data_modules.snli import SNLIDataModule
from rationalizers.data_modules.ag_news import AgNewsDataModule
from rationalizers.data_modules.imdb import ImdbDataModule
from rationalizers.data_modules.multinli import MultiNLIDataModule
from rationalizers.data_modules.hotel_location import HotelLocationDataModule
from rationalizers.data_modules.hans import HANSDataModule
from rationalizers.data_modules.mlqepe import MLQEPEDataModule
from rationalizers.data_modules.revised_imdb import RevisedIMDBDataModule
from rationalizers.data_modules.revised_snli import RevisedSNLIDataModule
from rationalizers.data_modules.revised_mlqepe import RevisedMLQEPEDataModule
from rationalizers.data_modules.revised_imdb_cf import CounterfactualRevisedIMDBDataModule
from rationalizers.data_modules.revised_snli_cf import CounterfactualRevisedSNLIDataModule
from rationalizers.data_modules.revised_mlqepe_cf import CounterfactualRevisedMLQEPEDataModule

available_data_modules = {
    "beer": BeerDataModule,
    "sst": SSTDataModule,
    "snli": SNLIDataModule,
    "ag_news": AgNewsDataModule,
    "imdb": ImdbDataModule,
    "multi_nli": MultiNLIDataModule,
    "hotel_location": HotelLocationDataModule,
    "hans": HANSDataModule,
    "mlqepe": MLQEPEDataModule,
    "revised_imdb": RevisedIMDBDataModule,
    "revised_snli": RevisedSNLIDataModule,
    "revised_mlqepe": RevisedMLQEPEDataModule,
    "revised_imdb_cf": CounterfactualRevisedIMDBDataModule,
    "revised_snli_cf": CounterfactualRevisedSNLIDataModule,
    "revised_mlqepe_cf": CounterfactualRevisedMLQEPEDataModule,
}
