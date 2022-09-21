import torch
import numpy as np

from rationalizers.modules.gates import RelaxedBernoulliGate, BernoulliGate
from rationalizers.explainers.base import BaseExplainer


class BernoulliExplainer(BaseExplainer):

    def __init__(
        self,
        h_params: dict,
        enc_size,
        budget: int = 10,
        contiguous: bool = False,
        relaxed: bool = False,
        topk: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.topk = topk
        self.contiguous = contiguous
        self.budget = budget
        self.relaxed = relaxed
        if self.relaxed:
            self.z_layer = RelaxedBernoulliGate(enc_size, temperature=temperature)
        else:
            self.z_layer = BernoulliGate(enc_size)

    def forward(self, h, mask=None, **kwargs):

        # encode sentence
        lengths = mask.long().sum(1)

        # compute parameters for Bernoulli p(z|x)
        z_dist = self.z_layer(h, mask)

        if self.contiguous:
            z = []
            z_probs_batch = z_dist.probs
            if self.training:
                if self.relaxed:
                    z = z_dist.rsample()
                else:
                    z = z_dist.sample()
                z = z.squeeze(-1)
            else:
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
            z = []
            z_probs_batch = z_dist.probs
            if self.training:
                if self.relaxed:
                    z = z_dist.rsample()
                else:
                    z = z_dist.sample()
                z = z.squeeze(-1)
            else:
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

        else:
            if self.training:  # sample
                if self.relaxed:
                    z = z_dist.rsample()
                else:
                    z = z_dist.sample()
            else:  # deterministic
                z = (z_dist.probs >= 0.5).float()  # [B, T, 1]
            z = z.squeeze(-1)  # [B, T, 1]  -> [B, T]

        z = torch.where(mask, z, z.new_zeros([1]))
        z = z.view(mask.shape[0], -1)

        self.z = z
        self.z_dists = [z_dist]

        return z, z_dist
