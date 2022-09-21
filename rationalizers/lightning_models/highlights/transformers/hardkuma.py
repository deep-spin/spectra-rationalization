import logging

import torch

from rationalizers.explainers import available_explainers
from rationalizers.lightning_models.highlights.transformers.spectra import TransformerSPECTRARationalizer
from rationalizers.utils import get_z_stats

shell_logger = logging.getLogger(__name__)


class TransformerHardKumaRationalizer(TransformerSPECTRARationalizer):

    def __init__(self, tokenizer: object, nb_classes: int, is_multilabel: bool, h_params: dict):
        super().__init__(tokenizer, nb_classes, is_multilabel, h_params)
        # for the loss
        self.sparsity_penalty = h_params.get('sparsity_penalty', 0.0)  # represents percentage of tokens to keep
        self.contiguity_penalty = h_params.get('contiguity_penalty', 0.0)

        # constrained lagrangian
        self.lagrange_lr = h_params.get("lagrange_lr", 0.01)
        self.lambda_init = h_params.get("lambda_init", 1e-4)
        self.alpha = h_params.get("alpha", 0.99)
        self.lambda_min = h_params.get("lambda_min", 1e-6)
        self.lambda_max = h_params.get("lambda_max", 1)

        # lagrange buffers
        self.register_buffer("lambda0", torch.full((1,), self.lambda_init))
        self.register_buffer("lambda1", torch.full((1,), self.lambda_init))
        self.register_buffer("c0_ma", torch.full((1,), 0.0))  # moving average
        self.register_buffer("c1_ma", torch.full((1,), 0.0))  # moving average

        # explainer
        explainer_cls = available_explainers['hardkuma']
        self.explainer = explainer_cls(
            h_params,
            enc_sice=self.ff_gen_hidden_size,
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
        batch_size = mask.size(0)
        lengths = mask.sum(1).float()  # [B]

        loss_vec = self.ff_criterion(y_hat, y)  # [B] or [B,C]

        # masked average
        if loss_vec.dim() == 2:
            loss = (loss_vec * mask.float()).sum(-1) / mask.sum(-1).float()  # [1]
        else:
            loss = loss_vec.mean()

        # L0 regularizer (sparsity constraint)
        # pre-compute for regularizers: pdf(0.)
        z_dists = self.explainer.z_dists
        if len(z_dists) == 1:
            pdf0 = z_dists[0].pdf(0.0)
        else:
            pdf0 = []
            for t in range(len(z_dists)):
                pdf_t = z_dists[t].pdf(0.0)
                pdf0.append(pdf_t)
            pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]

        pdf0 = pdf0.squeeze(-1)  # [B, T]
        pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))
        pdf_nonzero = 1.0 - pdf0  # [B, T]
        pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        l0 = l0.sum() / batch_size

        # `l0` now has the expected selection rate for this mini-batch
        # we now follow the steps Algorithm 1 (page 7) of this paper:
        # https://arxiv.org/abs/1810.00597
        # to enforce the constraint that we want l0 to be not higher
        # than `self.sparsity_penalty` (the target sparsity rate)

        # lagrange dissatisfaction, batch average of the constraint
        c0_hat = l0 - self.sparsity_penalty

        # moving average of the constraint
        self.c0_ma = self.alpha * self.c0_ma + (1 - self.alpha) * c0_hat.item()

        # compute smoothed constraint (equals moving average c0_ma)
        c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

        # update lambda
        self.lambda0 = self.lambda0 * torch.exp(self.lagrange_lr * c0.detach())
        self.lambda0 = self.lambda0.clamp(self.lambda_min, self.lambda_max)

        main_loss = loss + self.lambda0.detach() * c0

        # lagrange coherence dissatisfaction (batch average)
        if self.contiguity_penalty != 0:
            # fused lasso (coherence constraint)

            # cost z_t = 0, z_{t+1} = non-zero
            zt_zero = pdf0[:, :-1]
            ztp1_nonzero = pdf_nonzero[:, 1:]

            # cost z_t = non-zero, z_{t+1} = zero
            zt_nonzero = pdf_nonzero[:, :-1]
            ztp1_zero = pdf0[:, 1:]

            # number of transitions per sentence normalized by length
            lasso_cost = zt_zero * ztp1_nonzero + zt_nonzero * ztp1_zero
            lasso_cost = lasso_cost * mask.float()[:, :-1]
            lasso_cost = lasso_cost.sum(1) / (lengths + 1e-9)  # [B]
            lasso_cost = lasso_cost.sum() / batch_size

            # lagrange dissatisfaction, batch average of the constraint
            c1_hat = lasso_cost - self.contiguity_penalty

            # update moving average
            self.c1_ma = self.alpha * self.c1_ma + (1 - self.alpha) * c1_hat.detach()

            # compute smoothed constraint
            c1 = c1_hat + (self.c1_ma.detach() - c1_hat.detach())

            # update lambda
            self.lambda1 = self.lambda1 * torch.exp(self.lagrange_lr * c1.detach())
            self.lambda1 = self.lambda1.clamp(self.lambda_min, self.lambda_max)

            main_loss = main_loss + self.lambda1.detach() * c1

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(z, mask)
        stats[prefix + "_p0"] = num_0 / float(total)
        stats[prefix + "_pc"] = num_c / float(total)
        stats[prefix + "_p1"] = num_1 / float(total)
        stats[prefix + "_ps"] = (num_c + num_1) / float(total)
        stats[prefix + "_main_loss"] = main_loss.item()

        return main_loss, stats
