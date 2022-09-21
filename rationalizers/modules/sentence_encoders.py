"""Copied from https://github.com/bastings/interpretable_predictions"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from rationalizers.utils import masked_average


class LSTMEncoder(nn.Module):
    """
    This module encodes a sequence into a single vector using an LSTM.
    """

    def __init__(
        self,
        in_features,
        hidden_size: int = 200,
        batch_first: bool = True,
        bidirectional: bool = True,
    ):
        """
        :param in_features: input size of the LSTM
        :param hidden_size: hidden size of the LSTM
        :param batch_first: if True, the input and output tensors are provided as (batch, seq, feature).
        :param bidirectional: if True, the LSTM is bidirectional.
        """
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            in_features,
            hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

    def forward(self, x, mask, lengths=None):
        """
        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :param lengths: the lengths of each input sequence [B]
        :return: the encoded sequence, shape [B, H]
        """
        if lengths is None:
            lengths = mask.long().sum(-1).cpu()
        packed_sequence = pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hx, cx) = self.lstm(packed_sequence)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # classify from concatenation of final states
        if self.lstm.bidirectional:
            final = torch.cat([hx[-2], hx[-1]], dim=-1)
        else:  # classify from final state
            final = hx[-1]
        return outputs, final


class MaskedAverageEncoder(nn.Module):
    """
    This module encodes a sequence into a single vector using a masked average
    """

    def forward(self, x, mask):
        """
        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :return: average of word embeddings, shape [B, E]
        """
        return masked_average(x, mask)
