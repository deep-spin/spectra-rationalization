import logging
from torch import nn
from torchnlp.encoders.text import StaticTokenizerEncoder

from rationalizers import constants
from rationalizers.builders import build_embedding_weights
from rationalizers.lightning_models.matchings.base_matching import BaseMatching
from rationalizers.modules.matchings import LPSparseMAPFaithfulMatching

shell_logger = logging.getLogger(__name__)


class SparseMAPFaithfulMatching(BaseMatching):
    """Rationalizer that uses sparsemax activation function to get sparse (not necessarily contiguous) selections."""

    def __init__(
        self,
        tokenizer: StaticTokenizerEncoder,
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

        # save hyperparams to checkpoint
        self.save_hyperparameters(h_params)
        self.temperature = h_params.get("temperature", 1.0)
        self.matching_type = h_params.get("matching_type", "AtMostONE")
        self.budget = h_params.get("budget", 1.0)
        self.faithful = h_params.get("faithful", True)

        # load word embedding weights based on `emb_type` and define the embedding layer
        embedding_weights = build_embedding_weights(
            self.tokenizer.vocab, self.emb_type, self.emb_path, self.emb_size
        )
        self.emb_layer = nn.Embedding(
            self.vocab_size,
            self.emb_size,
            padding_idx=constants.PAD_ID,
            _weight=embedding_weights,
        )
        self.emb_layer.weight.requires_grad = self.emb_requires_grad

        # create generator
        # self.generator = SparsemapMatching(
        #     embed=self.emb_layer,
        #     hidden_size=self.hidden_size,
        #     dropout=self.dropout,
        #     layer=self.sentence_encoder_layer_type,
        #     temperature=self.temperature,
        # )

        # create predictor
        nonlinearity_str = "sigmoid" if not self.is_multilabel else "log_softmax"
        # self.predictor = MatchingPredictor(
        #     embed=self.emb_layer,
        #     hidden_size=self.hidden_size,
        #     output_size=self.nb_classes,
        #     dropout=self.dropout,
        #     layer=self.sentence_encoder_layer_type,
        #     nonlinearity=nonlinearity_str,
        # )

        self.matching_model = LPSparseMAPFaithfulMatching(
            temperature=self.temperature,
            embed=self.emb_layer,
            hidden_size=self.hidden_size,
            output_size=self.nb_classes,
            dropout=self.dropout,
            layer=self.sentence_encoder_layer_type,
            nonlinearity=nonlinearity_str,
            matching_type=self.matching_type,
            budget=self.budget,
            faithful=self.faithful,
        )

        # initialize params using xavier initialization for weights and zero for biases
        self.init_weights()

    def get_loss(self, y_hat, y, prefix, mask_x1, mask_x2):
        """
        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        stats = {}
        loss_vec = self.criterion(y_hat, y)  # [B] or [B,C]
        # main MSE loss for p(y | x,z)
        if not self.is_multilabel:
            loss_vec = loss_vec.mean(1)  # [B,C] -> [B]
        loss = loss_vec.mean()  # [1]
        stats[prefix + "_criterion"] = loss.item()  # [1]

        # latent selection stats
        # num_0, num_c, num_1, total = get_z_matching_stats(self.generator.z, mask)
        return loss, stats
