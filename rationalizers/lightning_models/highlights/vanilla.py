import logging

import torch
import torchmetrics
from torch import nn
from torchnlp.encoders.text import StaticTokenizerEncoder

from rationalizers import constants
from rationalizers.builders import build_embedding_weights
from rationalizers.lightning_models.highlights.base import BaseRationalizer
from rationalizers.modules.predictors import SentimentPredictor

shell_logger = logging.getLogger(__name__)


class VanillaClassifier(BaseRationalizer):
    """Rationalizer for which a generator does not exist.
    It just feeds the entire text to the predictor -- full-text baselines."""

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
            "sentence_encoder_layer_type", "rcnn"
        )
        self.use_dependent_generator = h_params.get("use_dependent_generator", False)
        self.contiguous = h_params.get("contiguous", False)
        self.topk = h_params.get("topk", False)
        self.relaxed = h_params.get("relaxed", False)
        self.budget = h_params.get("budget", 10)

        # loss fn:
        self.lambda_0 = h_params.get("lambda_0", 0.0)
        self.lambda_1 = h_params.get("lambda_1", 0.0)
        self.baseline = h_params.get("baseline", False)

        if self.baseline:
            self.mean_baseline = 0
            self.n_points = 0

        # global steps for board loggers
        self.eval_global_step = {"val": 0, "test": 0}

        # save hyperparams to checkpoint
        self.save_hyperparameters(h_params)

        # define metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision(
            num_classes=nb_classes,
            average="macro",
        )
        self.val_precision = torchmetrics.Precision(
            num_classes=nb_classes,
            average="macro",
        )
        self.test_precision = torchmetrics.Precision(
            num_classes=nb_classes,
            average="macro",
        )
        self.train_recall = torchmetrics.Recall(
            num_classes=nb_classes,
            average="macro",
        )
        self.val_recall = torchmetrics.Recall(
            num_classes=nb_classes,
            average="macro",
        )
        self.test_recall = torchmetrics.Recall(
            num_classes=nb_classes,
            average="macro",
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
        z = torch.ones(x.shape)
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
        stats[prefix + "_p0"] = 0
        stats[prefix + "_pc"] = 1
        stats[prefix + "_p1"] = 1
        stats[prefix + "_ps"] = 1

        return loss, stats
