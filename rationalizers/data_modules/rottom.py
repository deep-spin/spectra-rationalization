from rationalizers.data_modules.sst2 import SST2DataModule


class RotTomDataModule(SST2DataModule):
    """DataModule for RottenTomatoes Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params, tokenizer=tokenizer)
        # hard-coded stuff
        self.path = "rotten_tomatoes"
