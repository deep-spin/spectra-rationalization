import torch
import numpy as np

from rationalizers.modules.gates import KumaGate
from rationalizers.explainers.base import BaseExplainer

class HardKumaExplainer(BaseExplainer):
    def __init__(
        self, h_params: dict,
        enc_size,
        budget: int = 10,
        contiguous: bool = False,
        topk: bool = False,
    ):

        super().__init__()

        self.topk = topk
        self.contiguous = contiguous
        self.budget = budget
        self.z_layer = KumaGate(enc_size)

    def forward(self, h, mask=None, **kwargs):

        self.z = None  # z samples
        self.z_dists = []  # z distribution(s)

        # encode sentence
        lengths = mask.sum(1)
        z_dist = self.z_layer(h / 0.01)

        # we sample once since the state was already repeated num_samples
        if self.training:
            if hasattr(z_dist, "rsample"):
                z = z_dist.rsample()  # use rsample() if it's there
            else:
                z = z_dist.sample()  # [B, M, 1]
        else:
            z = []
            z_probs_batch = z_dist.mean()

            if self.contiguous:
                for k in range(h.shape[0]):
                    z_probs = z_probs_batch[k]
                    z_probs[lengths[k] :] = 0
                    cumsum = torch.cumsum(z_probs, 0)
                    length = int(torch.round(self.budget / 100 * lengths[k]).item())
                    score = torch.tensor(
                        [
                            cumsum[j + length] - cumsum[j]
                            if j > 0
                            else cumsum[j + length]
                            for j in range(len(z_probs) - length)
                        ]
                    )
                    index = torch.argmax(score) + 1
                    indices = torch.tensor([index + i for i in range(length)])
                    z_probs[
                        np.setdiff1d(np.arange(z_probs.shape[0]), indices, True)
                    ] = 0
                    z_sel = z_probs > 0
                    z_sel = z_sel.long().squeeze(-1)
                    z.append(z_sel * 1.0)
                z = torch.stack(z, dim=0)

            elif self.topk:
                for k in range(h.shape[0]):
                    z_probs = z_probs_batch[k]
                    z_probs[lengths[k] :] = 0
                    length = int(torch.round(self.budget / 100 * lengths[k]).item())
                    topk, idx = torch.topk(z_probs.squeeze(-1), length)
                    z_probs[
                        np.setdiff1d(np.arange(z_probs.cpu().shape[0]), idx.cpu(), True)
                    ] = 0
                    z_sel = z_probs > 0
                    z_sel = z_sel.long().squeeze(-1)
                    z.append(z_sel * 1.0)
                z = torch.stack(z, dim=0)

            elif not self.topk and not self.contiguous:
                # deterministic strategy
                p0 = z_dist.pdf(h.new_zeros(()))
                p1 = z_dist.pdf(h.new_ones(()))
                pc = 1.0 - p0 - p1  # prob. of sampling a continuous value [B, M]
                zero_one = torch.where(p0 > p1, h.new_zeros([1]), h.new_ones([1]))
                z_sel = torch.where(
                    (pc > p0) & (pc > p1), z_dist.mean(), zero_one
                )  # [B, M]
                z = z_sel.squeeze(-1)

        # mask invalid positions
        z = z.squeeze(-1)
        z = torch.where(mask, z, z.new_zeros([1]))

        self.z = z  # [B, T]
        self.z_dists = [z_dist]

        return z, z_dist
