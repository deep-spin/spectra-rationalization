import logging

import torch

from rationalizers.explainers import available_explainers
from rationalizers.lightning_models.highlights.transformers.spectra import TransformerSPECTRARationalizer
from rationalizers.utils import get_z_stats

shell_logger = logging.getLogger(__name__)


class TransformerBernoulliRationalizer(TransformerSPECTRARationalizer):

    def __init__(self, tokenizer: object, nb_classes: int, is_multilabel: bool, h_params: dict):
        super().__init__(tokenizer, nb_classes, is_multilabel, h_params)
        # for reinforce
        self.n_points = 0
        self.mean_baseline = 0
        self.use_baseline = h_params.get("use_baseline", False)
        self.sparsity_penalty = h_params.get('sparsity_penalty', 0.0)
        self.contiguity_penalty = h_params.get('contiguity_penalty', 0.0)
        # explainer
        self.explainer_relaxed = h_params.get('explainer_relaxed', False)
        explainer_cls = available_explainers['bernoulli']
        self.explainer = explainer_cls(
            h_params,
            enc_sice=self.ff_gen_hidden_size,
            relaxed=self.explainer_relaxed,
        )

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

        # get P(z = 0 | x) and P(z = 1 | x)
        m = self.explainer.z_dists[0]
        logp_z0 = m.log_prob(0.0).squeeze(2)  # [B, T]
        logp_z1 = m.log_prob(1.0).squeeze(2)  # [B, T]

        # compute log p(z|x) for each case (z==0 and z==1) and mask
        logpz = torch.where(z == 0, logp_z0, logp_z1)
        logpz = torch.where(mask, logpz, logpz.new_zeros([1]))

        # sparsity regularization
        zsum = z.sum(1)
        zsum_cost = self.sparsity_penalty * zsum.mean(0)
        stats["zsum_cost"] = zsum_cost.item()

        # contiguity regularization
        zdiff = (z[:, 1:] - z[:, :-1]).abs().sum(1)
        zdiff_cost = self.contiguity_penalty * zdiff.mean(0)
        stats["zdiff_cost"] = zdiff_cost.item()

        # combine both regularizations
        sparsity_cost = zsum_cost + zdiff_cost
        stats["sparsity_cost"] = sparsity_cost.item()

        # relaxed version does not use reinforce (repr. trick)
        if self.explainer_relaxed:
            main_loss = loss + zsum_cost + zdiff_cost
            stats["obj"] = main_loss.item()

        # with reinforce
        else:
            loss_vec = (loss_vec * mask.float()).sum(1) if loss_vec.dim() == 2 else loss_vec
            cost_vec = loss_vec.detach() + zsum * self.sparsity_penalty + zdiff * self.contiguity_penalty
            if self.use_baseline:
                cost_logpz = ((cost_vec - self.mean_baseline) * logpz.sum(1)).mean(0)
                self.n_points += 1.0
                self.mean_baseline += (cost_vec.detach().mean() - self.mean_baseline) / self.n_points
            else:
                cost_logpz = (cost_vec * logpz.sum(1)).mean(0)  # cost_vec is neg reward

            # MSE with regularizers = neg reward
            stats["obj"] = cost_vec.mean().item()
            # generator cost
            stats["cost_g"] = cost_logpz.item()
            # predictor cost
            stats["cost_p"] = loss.item()
            # final cost
            main_loss = loss + cost_logpz

        # pred diff doesn't do anything if only 1 aspect being trained
        if not self.is_multilabel:
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
