import torch
from torch import nn
import torch.nn.functional as F
import ipdb

from torch.distributions import RelaxedOneHotCategorical
from rationalizers.modules.matchings_utils import submul, apply_multiple
from rationalizers.builders import build_sentence_encoder
from rationalizers.modules.sparsemap import (
    matching_smap,
    matching_smap_atmostone,
    matching_smap_atmostone_budget,
)

class LPSparseMAPFaithfulMatching(nn.Module):
    """
    ESIM model with SPECTRA strategies for extraction of the sparse alignment.

    For faithful alignments (the only information about the premise that the model 
    has to make a prediction comes from the alignment and its masking of the encoded 
    representation), turn the `faithful` flag on.
    
    """

    def __init__(
        self,
        embed: nn.Embedding = None,
        hidden_size: int = 200,
        dropout: float = 0.1,
        layer: str = "lstm",
        bidirectional: bool = True,
        temperature: float = 1.0,
        budget: float = 1.0,
        nonlinearity: str = "sigmoid",
        output_size: int = 1,
        matching_type: str = "AtMostONE",
        faithful: bool = True,
    ):
        super().__init__()

        self.faithful = faithful 
        self.matching_type = matching_type
        emb_size = embed.weight.shape[1]
        enc_size = 2 * hidden_size if bidirectional else hidden_size
        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))
        self.context_lstm = build_sentence_encoder(
            layer,
            emb_size,
            hidden_size,
            bidirectional=True,
        )
        self.z = None  # z samples
        self.temperature = temperature
        self.budget = budget

        if self.faithful:
            self.projection_x1 = nn.Sequential(nn.Linear(enc_size, hidden_size), nn.ReLU())
            self.projection_x2 = nn.Sequential(
                nn.Linear(enc_size + enc_size, hidden_size), nn.ReLU()
            )
        else:
            self.projection = nn.Sequential(
            nn.Linear(4 * 2 * hidden_size, hidden_size), nn.ReLU()
            )

        self.composition_lstm = build_sentence_encoder(
            layer,
            hidden_size,
            hidden_size,
            bidirectional=True,
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(4 * enc_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1),
        )

    def forward(self, x1, x2, mask):
        """
        :param x1: premise embeddings
        :param x2: hypothesis embeddings
        :param mask: list [mask_x1, mask_x2] -- mask should be true/1 for valid positions, false/0 for invalid ones.
        """
        batch_size, _ = x1.shape

        lengths_x1 = mask[0].long().sum(1)
        lengths_x2 = mask[1].long().sum(1)
        mask_x1 = mask[0]
        mask_x2 = mask[1]

        emb_x1 = self.embed_layer(x1)  # [B, T, E]
        emb_x2 = self.embed_layer(x2)  # [B, D, E]

        # BiLSTM representation of the x1ise and x2thesis
        x1_h, _ = self.context_lstm(emb_x1, mask_x1, lengths_x1)
        x2_h, _ = self.context_lstm(emb_x2, mask_x2, lengths_x2)

        # [B, T, D]
        h_alignments = torch.bmm(x1_h, x2_h.transpose(1, 2))

        z = []
        for k in range(batch_size):
            scores = h_alignments[k] / self.temperature

            if self.matching_type == "AtMostONE":
                if self.training:
                    z_probs = matching_smap_atmostone(scores, max_iter=10)  # [T,D]
                else:
                    z_probs = torch.zeros(scores.shape, device=scores.device)
                    z_probs_sparsemap = matching_smap_atmostone(scores[:lengths_x1[k], :lengths_x2[k]] / 1e-3, max_iter=1000)
                    z_probs[:lengths_x1[k], :lengths_x2[k]] = z_probs_sparsemap

            if self.matching_type == "XOR-AtMostONE":
                if self.training:
                    z_probs = matching_smap(scores, max_iter=10)  # [T,D]
                else:
                    z_probs = torch.zeros(scores.shape, device=scores.device)
                    z_probs_sparsemap = matching_smap(scores[:lengths_x1[k], :lengths_x2[k]] / 1e-3, max_iter=1000)
                    z_probs[:lengths_x1[k], :lengths_x2[k]] = z_probs_sparsemap

            if self.matching_type == "AtMostONE-Budget":
                if self.training:
                    z_probs = matching_smap_atmostone_budget(
                        scores, max_iter=10, budget=self.budget
                    )  # [T,D]
                else:
                    z_probs = torch.zeros(scores.shape, device=scores.device)
                    z_probs_sparsemap = matching_smap_atmostone_budget(
                        scores[:lengths_x1[k], :lengths_x2[k]] / 1e-3, max_iter=1000, budget=self.budget
                    )
                    z_probs[:lengths_x1[k], :lengths_x2[k]] = z_probs_sparsemap

            z_probs = z_probs * mask[1][k].unsqueeze(0)
            z_probs = z_probs * mask[0][k].unsqueeze(-1)
            z.append(z_probs)

        z = torch.stack(z, dim=0).squeeze(-1)  # [B, T, D]
        z = z.to(h_alignments.device)
        self.z = z

        x1_align = torch.matmul(z, x2_h)
        x2_align = torch.matmul(z.transpose(-1, -2), x1_h)

        if self.faithful:
            x1_combined = x1_align
            x2_combined = torch.cat([x2_h, x2_align], -1)

            x1_combined = self.projection_x1(x1_combined)
            x2_combined = self.projection_x2(x2_combined)
        else:
            x1_combined = torch.cat([x1_h, x1_align, submul(x1_h, x1_align)], -1)
            x2_combined = torch.cat([x2_h, x2_align, submul(x2_h, x2_align)], -1)

            x1_combined = self.projection(x1_combined)
            x2_combined = self.projection(x2_combined)

        x1_compose, _ = self.composition_lstm(x1_combined, mask_x1, lengths_x1)
        x2_compose, _ = self.composition_lstm(x2_combined, mask_x2, lengths_x2)

        x1_rep = apply_multiple(x1_compose)
        x2_rep = apply_multiple(x2_compose)

        x = torch.cat([x1_rep, x2_rep], -1)

        y_hat = self.output_layer(x)

        return z, y_hat


class GumbelFaithfulMatching(nn.Module):
    """
    The Matching Generator takes two input texts and returns samples from p(z|x1,x2)
    """

    def __init__(
        self,
        embed: nn.Embedding = None,
        hidden_size: int = 200,
        dropout: float = 0.1,
        layer: str = "lstm",
        bidirectional: bool = True,
        temperature: float = 1.0,
        nonlinearity: str = "sigmoid",
        output_size: int = 1,
        faithful: bool = True,
    ):
        super().__init__()

        self.faithful = faithful
        emb_size = embed.weight.shape[1]
        enc_size = 2 * hidden_size if bidirectional else hidden_size
        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))
        self.context_lstm = build_sentence_encoder(
            layer,
            emb_size,
            hidden_size,
            bidirectional=True,
        )
        self.z = None  # z samples
        self.temperature = temperature

        if self.faithful:
            self.projection_x1 = nn.Sequential(nn.Linear(enc_size, hidden_size), nn.ReLU())
            self.projection_x2 = nn.Sequential(
                nn.Linear(enc_size + enc_size, hidden_size), nn.ReLU()
            )
        else:
            self.projection = nn.Sequential(
            nn.Linear(4 * 2 * hidden_size, hidden_size), nn.ReLU()
            )

        self.composition_lstm = build_sentence_encoder(
            layer,
            hidden_size,
            hidden_size,
            bidirectional=True,
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(4 * enc_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1),
        )

    def forward(self, x1, x2, mask):
        """
        :param x1: premise embeddings
        :param x2: hypothesis embeddings
        :param mask: list [mask_x1, mask_x2] -- mask should be true/1 for valid positions, false/0 for invalid ones.
        """

        batch_size, _ = x1.shape
        lengths_x1 = mask[0].long().sum(1)
        lengths_x2 = mask[1].long().sum(1)
        mask_x1 = mask[0]
        mask_x2 = mask[1]

        emb_x1 = self.embed_layer(x1)  # [B, T, E]
        emb_x2 = self.embed_layer(x2)  # [B, D, E]

        # BiLSTM representation of the x1ise and x2thesis
        x1_h, _ = self.context_lstm(emb_x1, mask_x1, lengths_x1)
        x2_h, _ = self.context_lstm(emb_x2, mask_x2, lengths_x2)

        # [B, T, D]
        h_alignments = torch.bmm(x1_h, x2_h.transpose(1, 2))

        if not self.training:
            row_x1_probs = F.gumbel_softmax(h_alignments / 1e-6, tau = self.temperature, dim=1, hard=True)
            column_x2_probs = F.gumbel_softmax(h_alignments / 1e-6, tau = self.temperature, dim=2, hard=True)
        else:
            row_x1_probs = F.gumbel_softmax(h_alignments, tau = self.temperature, dim=1)
            column_x2_probs = F.gumbel_softmax(h_alignments, tau = self.temperature, dim=2)

        x1_align = torch.matmul(row_x1_probs, x2_h)
        x2_align = torch.matmul(column_x2_probs.transpose(-2, -1), x1_h)

        if self.faithful:
            x1_combined = x1_align
            x2_combined = torch.cat([x2_h, x2_align], -1)

            x1_combined = self.projection_x1(x1_combined)
            x2_combined = self.projection_x2(x2_combined)
        else:
            x1_combined = torch.cat([x1_h, x1_align, submul(x1_h, x1_align)], -1)
            x2_combined = torch.cat([x2_h, x2_align, submul(x2_h, x2_align)], -1)

            x1_combined = self.projection(x1_combined)
            x2_combined = self.projection(x2_combined)

        x1_compose, _ = self.composition_lstm(x1_combined, mask_x1, lengths_x1)
        x2_compose, _ = self.composition_lstm(x2_combined, mask_x2, lengths_x2)

        x1_rep = apply_multiple(x1_compose)
        x2_rep = apply_multiple(x2_compose)

        x = torch.cat([x1_rep, x2_rep], -1)

        y_hat = self.output_layer(x)
        z = [row_x1_probs, column_x2_probs]

        return z, y_hat

class ESIMFaithfulMatching(nn.Module):
    """
    The Matching Generator takes two input texts and returns samples from p(z|x1,x2)
    """

    def __init__(
        self,
        embed: nn.Embedding = None,
        hidden_size: int = 200,
        dropout: float = 0.1,
        layer: str = "lstm",
        bidirectional: bool = True,
        temperature: float = 1.0,
        nonlinearity: str = "sigmoid",
        output_size: int = 1,
        faithful: bool = True,
    ):
        super().__init__()

        self.faithful = faithful
        emb_size = embed.weight.shape[1]
        enc_size = 2 * hidden_size if bidirectional else hidden_size
        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))
        self.context_lstm = build_sentence_encoder(
            layer,
            emb_size,
            hidden_size,
            bidirectional=True,
        )
        self.z = None  # z samples
        self.temperature = temperature

        if self.faithful:
            self.projection_x1 = nn.Sequential(nn.Linear(enc_size, hidden_size), nn.ReLU())
            self.projection_x2 = nn.Sequential(
                nn.Linear(enc_size + enc_size, hidden_size), nn.ReLU()
            )
        else:
            self.projection = nn.Sequential(
            nn.Linear(4 * 2 * hidden_size, hidden_size), nn.ReLU()
            )

        self.composition_lstm = build_sentence_encoder(
            layer,
            hidden_size,
            hidden_size,
            bidirectional=True,
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(4 * enc_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1),
        )

    def forward(self, x1, x2, mask):
        batch_size, _ = x1.shape

        lengths_x1 = mask[0].long().sum(1)
        lengths_x2 = mask[1].long().sum(1)
        mask_x1 = mask[0]
        mask_x2 = mask[1]

        emb_x1 = self.embed_layer(x1)  # [B, T, E]
        emb_x2 = self.embed_layer(x2)  # [B, D, E]

        # BiLSTM representation of the x1ise and x2thesis
        x1_h, _ = self.context_lstm(emb_x1, mask_x1, lengths_x1)
        x2_h, _ = self.context_lstm(emb_x2, mask_x2, lengths_x2)

        # [B, T, D]
        h_alignments = torch.bmm(x1_h, x2_h.transpose(1, 2))

        row_x1_probs = F.softmax(h_alignments, dim=1)
        column_x2_probs = F.softmax(h_alignments, dim=2)

        x1_align = torch.matmul(row_x1_probs, x2_h)
        x2_align = torch.matmul(column_x2_probs.transpose(-2, -1), x1_h)

        if self.faithful:
            x1_combined = x1_align
            x2_combined = torch.cat([x2_h, x2_align], -1)

            x1_combined = self.projection_x1(x1_combined)
            x2_combined = self.projection_x2(x2_combined)
        else:
            x1_combined = torch.cat([x1_h, x1_align, submul(x1_h, x1_align)], -1)
            x2_combined = torch.cat([x2_h, x2_align, submul(x2_h, x2_align)], -1)

            x1_combined = self.projection(x1_combined)
            x2_combined = self.projection(x2_combined)

        x1_compose, _ = self.composition_lstm(x1_combined, mask_x1, lengths_x1)
        x2_compose, _ = self.composition_lstm(x2_combined, mask_x2, lengths_x2)

        x1_rep = apply_multiple(x1_compose)
        x2_rep = apply_multiple(x2_compose)

        x = torch.cat([x1_rep, x2_rep], -1)

        y_hat = self.output_layer(x)
        z = [row_x1_probs, column_x2_probs]

        return z, y_hat

      