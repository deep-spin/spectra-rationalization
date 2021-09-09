"""Copied from https://github.com/bastings/interpretable_predictions"""

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli 
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn import Linear, Sequential, Softplus

from rationalizers.modules.kuma import Kuma, HardKuma

class BernoulliGate(nn.Module):
    """
    Computes a Bernoulli Gate
    Assigns a 0 or a 1 to each input word.
    """

    def __init__(self, in_features, out_features=1):
        super(BernoulliGate, self).__init__()

        self.layer = Sequential(Linear(in_features, out_features))

    def forward(self, x, mask):
        """
        Compute Binomial gate
        :param x: word represenatations [B, T, D]
        :return: gate distribution
        """
        logits = self.layer(x)  # [B, T, 1]
        logits = logits.squeeze(-1) * mask
        logits = logits.unsqueeze(-1)
        dist = Bernoulli(logits=logits)
        return dist


class RelaxedBernoulliGate(nn.Module):
    """
    Computes a Relaxed Bernoulli Gate
    Assigns a 0 or a 1 to each input word.
    """

    def __init__(self, in_features, out_features=1):
        super(RelaxedBernoulliGate, self).__init__()

        self.layer = Sequential(Linear(in_features, out_features))

    def forward(self, x, mask):
        """
        Compute Relaxed Binomial gate
        :param x: word represenatations [B, T, D]
        :return: gate distribution
        """
        logits = self.layer(x)  # [B, T, 1]
        logits = logits.squeeze(-1) * mask
        logits = logits.unsqueeze(-1)
        dist = RelaxedBernoulli(temperature=torch.tensor([0.1], device=logits.device), logits=logits)
        return dist

class KumaGate(nn.Module):
    """
    Computes a _Hard_ Kumaraswamy Gate
    """

    def __init__(self, in_features, out_features=1, support=(-0.1, 1.1), dist_type="hardkuma"):
        super(KumaGate, self).__init__()

        self.dist_type = dist_type

        self.layer_a = Sequential(Linear(in_features, out_features), Softplus())
        self.layer_b = Sequential(Linear(in_features, out_features), Softplus())

        # support must be Tensors
        s_min = torch.Tensor([support[0]])
        s_max = torch.Tensor([support[1]])
        self.support = [s_min, s_max]

        self.a = None
        self.b = None

    def forward(self, x):
        """
        Compute latent gate
        :param x: word represenatations [B, T, D]
        :return: gate distribution
        """
        a = self.layer_a(x)
        b = self.layer_b(x)

        a = a.clamp(1e-6, 100.0)  # extreme values could result in NaNs
        b = b.clamp(1e-6, 100.0)  # extreme values could result in NaNs

        self.a = a
        self.b = b

        # we return a distribution (from which we can sample if we want)
        if self.dist_type == "kuma":
            dist = Kuma([a, b])
        elif self.dist_type == "hardkuma":
            dist = HardKuma([a, b], support=self.support)
        else:
            raise ValueError("unknown dist")

        return dist