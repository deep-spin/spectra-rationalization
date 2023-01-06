from rationalizers.data_modules.snli import SNLIDataModule


class WinogradNLIDataModule(SNLIDataModule):

    def __init__(self, d_params: dict, tokenizer: object = None):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params, tokenizer=tokenizer)
        # hard-coded stuff
        self.path = ("glue", "wnli")  # hf_datasets will handle everything
        self.is_multilabel = True
        self.nb_classes = 3  # entailment, neutral, contradiction
