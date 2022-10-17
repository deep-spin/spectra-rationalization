from rationalizers.data_modules.revised_snli import RevisedSNLIDataModule


class OversampledRevisedSNLIDataModule(RevisedSNLIDataModule):

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        self.path = "./rationalizers/custom_hf_datasets/revised_snli_oversampled.py"
