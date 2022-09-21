import logging

import torch
import torchmetrics
from torch import nn
from torchnlp.encoders.text import StaticTokenizerEncoder

from rationalizers import constants
from rationalizers.builders import build_embedding_weights
from rationalizers.lightning_models.highlights.base import BaseRationalizer
from rationalizers.modules.generators import SPECTRAGenerator
from rationalizers.modules.predictors import SentimentPredictor
from rationalizers.utils import get_z_stats


shell_logger = logging.getLogger(__name__)


class SPECTRARationalizer(BaseRationalizer):
    """SPECTRA Rationalizer"""

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

        # model arch:
        self.vocab_size = tokenizer.vocab_size
        self.emb_type = h_params.get("emb_type", "random")
        self.emb_path = h_params.get("emb_path", None)
        self.emb_size = h_params.get("emb_size", 300)
        self.emb_requires_grad = not h_params.get("embed_fixed", True)
        self.hidden_size = h_params.get("hidden_size", 150)
        self.dropout = h_params.get("dropout", 0.5)
        self.sentence_encoder_layer_type = h_params.get(
            "sentence_encoder_layer_type", "lstm"
        )
        self.transition = h_params.get("transition", 0.0)
        self.budget = h_params.get("budget", 0)
        self.temperature = h_params.get("temperature", 0.01)

        # save hyperparams to checkpoint
        self.save_hyperparameters(h_params)

        # define metrics
        if self.is_multilabel:
            self.train_accuracy = torchmetrics.Accuracy()
            self.val_accuracy = torchmetrics.Accuracy()
            self.test_accuracy = torchmetrics.Accuracy()
            self.train_precision = torchmetrics.Precision(
                num_classes=nb_classes, average="macro"
            )
            self.val_precision = torchmetrics.Precision(
                num_classes=nb_classes, average="macro"
            )
            self.test_precision = torchmetrics.Precision(
                num_classes=nb_classes, average="macro"
            )
            self.train_recall = torchmetrics.Recall(
                num_classes=nb_classes, average="macro"
            )
            self.val_recall = torchmetrics.Recall(num_classes=nb_classes, average="macro")
            self.test_recall = torchmetrics.Recall(
                num_classes=nb_classes, average="macro"
            )

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
        self.generator = SPECTRAGenerator(
            embed=self.emb_layer,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            layer=self.sentence_encoder_layer_type,
            transition=self.transition,
            budget=self.budget,
            temperature=self.temperature,
        )

        # create predictor
        nonlinearity_str = "sigmoid" if not self.is_multilabel else "log_softmax"
        self.predictor = SentimentPredictor(
            embed=self.emb_layer,
            hidden_size=self.hidden_size,
            output_size=self.nb_classes,
            dropout=self.dropout,
            layer=self.sentence_encoder_layer_type,
            nonlinearity=nonlinearity_str,
        )

        # initialize params using xavier initialization for weights and zero for biases
        self.init_weights()

    def forward(
        self, x: torch.LongTensor, current_epoch=None, mask: torch.BoolTensor = None
    ):
        """
        Compute forward-pass.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: the output from SentimentPredictor. Torch.Tensor of shape [B, C]
        """
        z = self.generator(x, current_epoch=self.current_epoch, mask=mask)
        y_hat = self.predictor(x, z, mask=mask)
        return z, y_hat

    def get_loss(self, y_hat, y, prefix, mask=None):
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
            stats["mse"] = float(loss_vec.item())
        else:
            loss = loss_vec.mean()  # [1]
            stats["criterion"] = float(loss.item())  # [1]

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(self.generator.z, mask)
        stats[prefix + "_p0"] = num_0 / float(total)
        stats[prefix + "_pc"] = num_c / float(total)
        stats[prefix + "_ps"] = (num_c + num_1) / float(total)

        return loss, stats
