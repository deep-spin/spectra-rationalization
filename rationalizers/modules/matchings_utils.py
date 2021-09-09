import torch
import torch.nn.functional as F


def submul(x1, x2):
    mul = x1 * x2
    sub = x1 - x2
    return torch.cat([sub, mul], -1)


def apply_multiple(x):
    # input: batch_size * seq_len * (2 * hidden_size)
    p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
    p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
    # output: batch_size * (4 * hidden_size)
    return torch.cat([p1, p2], 1)
