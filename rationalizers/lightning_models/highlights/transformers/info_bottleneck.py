import logging

import torch

from rationalizers.explainers import available_explainers
from rationalizers.lightning_models.highlights.transformers.spectra import TransformerSPECTRARationalizer
from rationalizers.utils import get_z_stats

shell_logger = logging.getLogger(__name__)


class TransformerInfoBottleneckRationalizer(TransformerSPECTRARationalizer):

    def __init__(self, tokenizer: object, nb_classes: int, is_multilabel: bool, h_params: dict):
        super().__init__(tokenizer, nb_classes, is_multilabel, h_params)
        # explainer
        self.prior_penalty = h_params.get('prior_penalty', 0.0)
        self.explainer_prior = h_params.get('explainer_prior', 0.5)
        explainer_cls = available_explainers['bernoulli']
        self.explainer = explainer_cls(
            h_params,
            enc_sice=self.ff_gen_hidden_size,
            relaxed=True,
        )
        self.register_buffer("prior", torch.full((1,), self.explainer_prior))

    def get_factual_loss(self, y_hat, y, z, mask, prefix):
        """
        Compute loss for the factual flow.

        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param z: latent selection vector. torch.FloatTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param prefix: prefix for loss statistics (train, val, test)
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        stats = {}
        loss_vec = self.ff_criterion(y_hat, y)  # [B] or [B,C]

        # masked average
        if loss_vec.dim() == 2:
            loss = (loss_vec * mask.float()).sum(-1) / mask.sum(-1).float()  # [1]
        else:
            loss = loss_vec.mean()

        # main loss for p(y | x, z)
        stats["mse" if not self.is_multilabel else "nll"] = loss.item()

        # compute info bottleneck loss
        q_z = self.ff_z_dist
        p_z = torch.distributions.Bernoulli(probs=self.explainer_prior.expand_as(q_z.probs))
        kl_div = torch.distributions.kl_divergence(q_z, p_z)
        info_loss = (kl_div * mask.float()).sum(-1) / mask.sum(-1).float()
        info_loss = info_loss.mean()
        stats["info_cost"] = info_loss.item()

        main_loss = loss + self.prior_penalty + info_loss
        stats["obj"] = main_loss.item()

        # pred diff doesn't do anything if only 1 aspect being trained
        pred_diff = y_hat.max(dim=1)[0] - y_hat.min(dim=1)[0]
        pred_diff = pred_diff.mean()
        stats["pred_diff"] = pred_diff.item()

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(z, mask)
        stats[prefix + "_p0"] = num_0 / float(total)
        stats[prefix + "_pc"] = num_c / float(total)
        stats[prefix + "_p1"] = num_1 / float(total)
        stats[prefix + "_ps"] = (num_c + num_1) / float(total)
        stats[prefix + "_main_loss"] = main_loss.item()

        return main_loss, stats
