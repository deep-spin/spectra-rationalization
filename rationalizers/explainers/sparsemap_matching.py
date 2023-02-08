import torch

from rationalizers.explainers.base import BaseExplainer
from rationalizers.modules.sparsemap import (
    matching_smap,
    matching_smap_atmostone,
    matching_smap_atmostone_budget,
)


def split_pairs(t, token_type_ids, padding_value=0):
    """
    :param t: concatenated premise and hyp embeddings
    :param token_type_ids: token type ids for the concatenated premise and hyp embeddings
    :param padding_value: padding id
    """
    t1, t2 = [], []
    batch_size = t.size(0)
    for i in range(batch_size):
        pre_mask = token_type_ids[i] == 0
        hyp_mask = token_type_ids[i] == 1
        t1.append(t[i][pre_mask])
        t2.append(t[i][hyp_mask])
    t1 = torch.nn.utils.rnn.pad_sequence(t1, batch_first=True, padding_value=padding_value)
    t2 = torch.nn.utils.rnn.pad_sequence(t2, batch_first=True, padding_value=padding_value)
    return t1, t2


class SparseMAPMatchingExplainer(BaseExplainer):
    """
    The Generator takes an input text and returns samples from p(z|x)
    """

    def __init__(self, h_params: dict, enc_size):
        super().__init__()
        self.self_scorer = torch.nn.Linear(enc_size, 1)
        self.init = h_params.get('sparsemap_init', False)
        self.max_iter = h_params.get('sparsemap_max_iter', 100)
        self.transition = h_params.get('sparsemap_transition', 0)
        self.budget = h_params.get('sparsemap_budget', 0)
        self.temperature = h_params.get('sparsemap_temperature', 0.01)

    def forward(self, x1_h=None, x2_h=None, mask_x1=None, mask_x2=None, **kwargs):

        """
        :param x1_h: premise embeddings
        :param x2_h: hypothesis embeddings
        :param mask_x1: premise mask
        :param mask_x2: hypothesis mask
        """
        batch_size = x1_h.shape[0]
        lengths_x1 = mask_x1.long().sum(1)
        lengths_x2 = mask_x2.long().sum(1)

        # [B, T, T]
        h_alignments = torch.bmm(x1_h, x2_h.transpose(1, 2))

        z = []
        for k in range(batch_size):
            scores = h_alignments[k] / self.temperature

            if self.matching_type == "AtMostONE":
                if self.training:
                    z_probs = matching_smap_atmostone(scores, max_iter=10)  # [T,D]
                else:
                    z_probs = torch.zeros(scores.shape, device=scores.device)
                    z_probs_sparsemap = matching_smap_atmostone(
                        scores[: lengths_x1[k], : lengths_x2[k]] / 1e-3, max_iter=1000
                    )
                    z_probs[: lengths_x1[k], : lengths_x2[k]] = z_probs_sparsemap

            if self.matching_type == "XOR-AtMostONE":
                if self.training:
                    z_probs = matching_smap(scores, max_iter=10)  # [T,D]
                else:
                    z_probs = torch.zeros(scores.shape, device=scores.device)
                    z_probs_sparsemap = matching_smap(
                        scores[: lengths_x1[k], : lengths_x2[k]] / 1e-3, max_iter=1000
                    )
                    z_probs[: lengths_x1[k], : lengths_x2[k]] = z_probs_sparsemap

            if self.matching_type == "AtMostONE-Budget":
                if self.training:
                    z_probs = matching_smap_atmostone_budget(
                        scores, max_iter=10, budget=self.budget
                    )  # [T,D]
                else:
                    z_probs = torch.zeros(scores.shape, device=scores.device)
                    z_probs_sparsemap = matching_smap_atmostone_budget(
                        scores[: lengths_x1[k], : lengths_x2[k]] / 1e-3,
                        max_iter=1000,
                        budget=self.budget,
                    )
                    z_probs[: lengths_x1[k], : lengths_x2[k]] = z_probs_sparsemap

            z_probs = z_probs * mask_x2[k].unsqueeze(0)
            z_probs = z_probs * mask_x1[k].unsqueeze(-1)
            z.append(z_probs)

        z = torch.stack(z, dim=0).squeeze(-1)  # [B, T, D]
        z = z.to(h_alignments.device)
        self.z = z

        return z, None
