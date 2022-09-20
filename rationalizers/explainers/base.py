import torch
from torch import nn


class BaseExplainer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.z = None  # z samples
        self.z_dists = []  # z distribution(s)

    def forward(self, x, mask=None, **kwargs):
        raise NotImplementedError

