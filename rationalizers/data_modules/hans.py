from rationalizers.data_modules.snli import SNLIDataModule


class HANSDataModule(SNLIDataModule):
    """DataModule for HANS Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params, tokenizer=tokenizer)
        # hard-coded stuff
        self.path = "hans"  # hf_datasets will handle everything
        self.is_multilabel = True
        self.nb_classes = 2  # entailment, and non-entailment

    def setup(self, stage: str = None):
        super().setup(stage=stage)
        self.dataset["test"] = self.dataset["validation"]
