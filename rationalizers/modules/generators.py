"""Copied from https://github.com/bastings/interpretable_predictions"""
import math
import ipdb
import numpy as np
import torch
from entmax import sparsemax
from torch import nn

from rationalizers.builders import build_sentence_encoder
from rationalizers.modules.gates import BernoulliGate, RelaxedBernoulliGate, KumaGate
from rationalizers.modules.sparsemap import (
    seq_budget_smap,
)


class SPECTRAGenerator(nn.Module):
    """
    The Generator takes an input text and returns samples from p(z|x)
    """

    def __init__(
        self,
        embed: nn.Embedding = None,
        hidden_size: int = 200,
        dropout: float = 0.1,
        layer: str = "lstm",
        bidirectional: bool = True,
        budget: int = 0,
        init: bool = False,
        max_iter: int = 100,
        transition: int = 0,
        temperature: float = 0.01,
    ):
        super().__init__()

        emb_size = embed.weight.shape[1]
        enc_size = 2 * hidden_size if bidirectional else hidden_size
        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))
        self.enc_layer = build_sentence_encoder(
            layer, emb_size, hidden_size, bidirectional=bidirectional
        )
        self.layer = nn.Linear(enc_size, 1)
        self.self_scorer = SelfAdditiveScorer(enc_size, enc_size)
        self.init = init
        self.max_iter = max_iter
        self.z = None  # z samples
        self.z_dists = []  # z distribution(s)
        self.transition = transition
        self.budget = budget
        self.temperature = temperature

    def forward(self, x, current_epoch, mask):
        # encode sentence
        batch_size, target_size = x.shape
        lengths = mask.long().sum(1)
        emb = self.embed_layer(x)  # [B, T, E]

        # [B, T, H]
        h, _ = self.enc_layer(emb, mask, lengths)

        # compute attention scores
        # [B, T, H] -> [B, T, 1]
        h1 = self.layer(h)

        t = torch.full((batch_size, target_size + 1), float(self.transition))
        z = []
        num_states = 2

        for k in range(batch_size):
            scores = h1[k].view(-1)
            budget = torch.round(self.budget / 100 * lengths[k])
            length = scores.shape[0]

            # Set unary scores for valid positions
            x = torch.cat(
                (
                    scores.unsqueeze(-1) / self.temperature,
                    torch.zeros((length, 1), device=scores.device),
                ),
                dim=-1,
            )
            x[lengths[k]:, 0] = -1e12
            
            # Set transition scores for valid positions
            transition_scores = torch.tensor(t[k], device=scores.device)
            transition = torch.zeros(
                (length + 1, num_states, num_states), device=scores.device
            )
            transition[: lengths[k] + 1, 0, 0] = (
                transition_scores[: lengths[k] + 1] / self.temperature
            )

            # H:SeqBudget consists of a single factor so, in this particular case, the LP-SparseMAP solution is 
            # indeed the SparseMAP solution and it can be found within a single iteration.
            self.max_iter = 1
            self.step_size = 0.0

            if self.training:
                z_probs = seq_budget_smap(
                    x,
                    transition,
                    budget=budget,
                    temperature=self.temperature,
                    init=self.init,
                    max_iter=self.max_iter,
                    step_size=self.step_size,
                )
            else:
                test_temperature = 1e-3
                z_probs = seq_budget_smap(
                    x / test_temperature,
                    transition / test_temperature,
                    budget=budget,
                    temperature=test_temperature,
                    init=self.init,
                    max_iter=self.max_iter,
                    step_size=self.step_size,
                )

                

            z_probs.cuda()
            z.append(z_probs)

        z = torch.stack(z, dim=0).squeeze(-1)  # [B, T]
        z = z.cuda()
        z = torch.where(mask, z, z.new_zeros([1]))
        self.z = z

        return z


class BernoulliIndependentGenerator(nn.Module):
    """
    The Generator takes an input text and returns samples from p(z|x)
    """

    def __init__(
        self,
        embed: nn.Embedding = None,
        hidden_size: int = 200,
        dropout: float = 0.1,
        layer: str = "lstm",
        budget: int = 10,
        contiguous: bool = False,
        relaxed: bool = False,
        topk: bool = False,
    ):
        super().__init__()

        emb_size = embed.weight.shape[1]
        enc_size = hidden_size * 2

        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))
        self.enc_layer = build_sentence_encoder(layer, emb_size, hidden_size)
        self.self_scorer = SelfAdditiveScorer(enc_size, enc_size)
        self.contiguous = contiguous
        self.budget = budget
        self.relaxed = relaxed
        self.topk = topk

        if self.relaxed:
            self.z_layer = RelaxedBernoulliGate(enc_size)
        else:
            self.z_layer = BernoulliGate(enc_size)

        self.z = None  # z samples
        self.z_dists = []  # z distribution(s)

    def forward(self, x, mask):

        # encode sentence
        lengths = mask.long().sum(1)
        emb = self.embed_layer(x)  # [B, T, E]
        h, _ = self.enc_layer(emb, mask, lengths)

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
                for k in range(x.shape[0]):
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
                for k in range(x.shape[0]):
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
        z = z.view(emb.shape[0], -1)

        self.z = z
        self.z_dists = [z_dist]

        return z

class SparsemaxGenerator(nn.Module):
    """
    The Generator takes an input text and returns samples from p(z|x)
    """

    def __init__(
        self,
        embed: nn.Embedding = None,
        hidden_size: int = 200,
        dropout: float = 0.1,
        layer: str = "lstm",
        bidirectional: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()

        emb_size = embed.weight.shape[1]
        enc_size = 2 * hidden_size if bidirectional else hidden_size
        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))
        self.enc_layer = build_sentence_encoder(
            layer, emb_size, hidden_size, bidirectional=bidirectional
        )
        self.self_scorer = SelfAdditiveScorer(enc_size, enc_size)
        self.z = None  # z samples
        self.temperature = temperature

    def forward(self, x, mask):

        # encode sentence
        lengths = mask.long().sum(1)
        emb = self.embed_layer(x)  # [B, T, E]

        # [B, T, H]
        h, _ = self.enc_layer(emb, mask, lengths)

        # compute sparsemax
        # [B, T, H] -> [B, T]
        h = self.self_scorer(h, h)
        z = sparsemax(h / self.temperature, dim=-1)
        z = torch.where(mask, z, z.new_zeros([1]))
        self.z = z

        return z




class KumaIndependentLatentModel(nn.Module):
    """
    The latent model ("The Generator") takes an input text
    and returns samples from p(z|x)
    This version uses a reparameterizable distribution, e.g. HardKuma.
    """

    def __init__(
        self,
        embed: nn.Embedding = None,
        hidden_size: int = 200,
        dropout: float = 0.1,
        layer: str = "lstm",
        budget: int = 10,
        contiguous: bool = False,
        topk: bool = False,
    ):

        super().__init__()

        self.topk = topk
        self.contiguous = contiguous
        self.budget = budget

        self.layer = layer
        emb_size = embed.weight.shape[1]
        enc_size = hidden_size * 2

        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))
        self.enc_layer = build_sentence_encoder(layer, emb_size, hidden_size)

        self.z_layer = KumaGate(enc_size)
    
        self.z = None  # z samples
        self.z_dists = []  # z distribution(s)

    def forward(self, x, mask, **kwargs):

        # encode sentence
        lengths = mask.sum(1)

        emb = self.embed_layer(x)  # [B, T, E]
        h, _ = self.enc_layer(emb, mask, lengths)

        z_dist = self.z_layer(h/0.01)

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
                for k in range(x.shape[0]):
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
                for k in range(x.shape[0]):
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
                z_sel = torch.where((pc > p0) & (pc > p1), z_dist.mean(), zero_one)  # [B, M]
                z = z_sel.squeeze(-1)
            
        # mask invalid positions
        z = z.squeeze(-1)
        z = torch.where(mask, z, z.new_zeros([1]))

        self.z = z  # [B, T]
        self.z_dists = [z_dist]

        return z


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


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.
        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(
            hypothesis_batch.transpose(2, 1).contiguous()
        )

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(
            similarity_matrix.transpose(1, 2).contiguous(), premise_mask
        )

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(
            premise_batch, hyp_prem_attn, hypothesis_mask
        )

        return attended_premises, attended_hypotheses