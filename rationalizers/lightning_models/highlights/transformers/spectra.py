import logging

from rationalizers.explainers import available_explainers
from rationalizers.lightning_models.highlights.transformers.base import TransformerBaseRationalizer

shell_logger = logging.getLogger(__name__)


class TransformerSPECTRARationalizer(TransformerBaseRationalizer):

    def __init__(
        self,
        tokenizer: object,
        nb_classes: int,
        is_multilabel: bool,
        h_params: dict,
    ):
        """
        :param tokenizer (object): torchnlp tokenizer object
        :param nb_classes (int): number of classes used to create the last layer
        :param multilabel (bool): whether the problem is multilabel or not (it depends on the dataset)
        :param h_params (dict): hyperparams dict. See docs for more info.
        """
        super().__init__(tokenizer, nb_classes, is_multilabel, h_params)
        explainer_cls = available_explainers['sparsemap']
        self.explainer = explainer_cls(h_params, self.ff_gen_hidden_size)
