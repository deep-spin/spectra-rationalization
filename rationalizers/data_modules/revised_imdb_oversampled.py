from rationalizers.data_modules.revised_imdb import RevisedIMDBDataModule


class OversampledRevisedIMDBDataModule(RevisedIMDBDataModule):

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        self.path = "./rationalizers/custom_hf_datasets/revised_imdb_oversampled.py"
