from rationalizers.data_modules.beer import BeerDataModule
from rationalizers.data_modules.sst import SSTDataModule
from rationalizers.data_modules.snli import SNLIDataModule
from rationalizers.data_modules.ag_news import AgNewsDataModule
from rationalizers.data_modules.imdb import ImdbDataModule
from rationalizers.data_modules.multinli import MultiNLIDataModule
from rationalizers.data_modules.hotel_location import HotelLocationDataModule
from rationalizers.data_modules.hans import HANSDataModule


available_data_modules = {
    "beer": BeerDataModule,
    "sst": SSTDataModule,
    "snli": SNLIDataModule,
    "ag_news": AgNewsDataModule,
    "imdb": ImdbDataModule,
    "multi_nli": MultiNLIDataModule,
    "hotel_location": HotelLocationDataModule,
    "hans": HANSDataModule,
}
