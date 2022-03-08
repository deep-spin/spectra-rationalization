import torch
from torch import nn


class BaseExplainer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.z = None

    def forward(self, x, mask=None):
        raise NotImplementedError

