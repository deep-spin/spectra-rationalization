import torch
import math
from torch import nn
from entmax import entmax15, sparsemax

from rationalizers.explainers.base import BaseExplainer


class SelfAdditiveScorer(nn.Module):
    """
    Simple scorer of the form:
    v^T tanh(Wx + b) / sqrt(d)
    """

    def __init__(self, vector_size, attn_hidden_size):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(attn_hidden_size, vector_size))
        self.b = nn.Parameter(torch.Tensor(attn_hidden_size))
        self.v = nn.Parameter(torch.Tensor(1, attn_hidden_size))
        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.W.shape[1])
        nn.init.uniform_(self.b, -bound, bound)
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))

    def forward(self, query, keys):
        """Computes scores for each key of size n given the queries of size m.

        Args:
            query (torch.FloatTensor): query matrix (bs, ..., target_len, d_q)
            keys (torch.FloatTensor): keys matrix (bs, ..., source_len, d_k)

        Returns:
            torch.FloatTensor: scores between source and target words: (bs, ..., target_len, source_len)
        """
        # assume query = keys
        x = torch.matmul(query, self.W.t()) + self.b
        x = self.activation(x)
        score = torch.matmul(x, self.v.t()).squeeze(-1)
        return score / math.sqrt(keys.size(-1))


class AttentionExplainer(BaseExplainer):
    def __init__(self, h_params: dict, enc_size):
        super().__init__()
        act2fn = {
            'sparsemax': sparsemax,
            'entmax': entmax15,
            'softmax': torch.softmax,
        }
        self.activation = act2fn[h_params['explainer_activation']]
        self.self_scorer = SelfAdditiveScorer(enc_size, enc_size)
        self.temperature = h_params.get('temperature', 1.)

    def forward(self, h, mask=None):
        logits = self.self_scorer(h, h)
        z = self.activation(logits / self.temperature, dim=-1)
        z = torch.where(mask, z, z.new_zeros([1]))
        self.z = z
        return z
