from rationalizers.data_modules.revised_imdb_cf import CounterfactualRevisedIMDBDataModule


class CounterfactualContrastIMDBDataModule(CounterfactualRevisedIMDBDataModule):
    """DataModule for the Contrast IMDB Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer=tokenizer)
        self.path = "./rationalizers/custom_hf_datasets/contrast_imdb_cf.py"

