import torch
from torch import nn

from rationalizers.builders import build_sentence_encoder
from .hflayers import HopfieldPooling

"""Copied from https://github.com/bastings/interpretable_predictions"""


class SentimentPredictor(nn.Module):
    """
    The Encoder takes an input text (and rationale z) and computes p(y|x,z)

    Supports a sigmoid on the final result (for regression)
    If not sigmoid, will assume cross-entropy loss (for classification)

    """

    def __init__(
        self,
        embed: nn.Embedding = None,
        hidden_size: int = 200,
        output_size: int = 1,
        dropout: float = 0.1,
        layer: str = "rcnn",
        nonlinearity: str = "sigmoid",
    ):

        super().__init__()

        emb_size = embed.weight.shape[1]

        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))

        self.enc_layer = build_sentence_encoder(layer, emb_size, hidden_size)

        if hasattr(self.enc_layer, "cnn"):
            enc_size = self.enc_layer.cnn.out_channels
        else:
            enc_size = hidden_size * 2

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, z, mask=None):
        x = x.cuda()
        z = z.cuda()
        mask = mask.cuda()
        rnn_mask = mask
        emb = self.embed_layer(x)
        # apply z to main inputs
        if z is not None:
            z_mask = (mask.float() * z).unsqueeze(-1)  # [B, T, 1]
            rnn_mask = z_mask.squeeze(-1) > 0.0  # z could be continuous
            emb = emb * z_mask

        # z is also used to control when the encoder layer is active
        lengths = mask.long().sum(1)

        # encode the sentence
        _, final = self.enc_layer(emb, rnn_mask, lengths)
        # predict sentiment from final state(s)
        y = self.output_layer(final)

        return y

class HopfieldSentimentPredictor(nn.Module):
    """
    The Encoder takes an input text (and rationale z) and computes p(y|x,z)

    Supports a sigmoid on the final result (for regression)
    If not sigmoid, will assume cross-entropy loss (for classification)

    """

    def __init__(
        self,
        embed: nn.Embedding = None,
        hidden_size: int = 200,
        output_size: int = 1,
        dropout: float = 0.1,
        layer: str = "lstm",
        nonlinearity: str = "sigmoid",
        alpha: float = 2.0,
        num_heads: int = 4,
        bag_dropout: float = 0.0,
        transition: int = 0,
        temperature: float = 0.01,
        budget: int = 20,

    ):

        super().__init__()

        emb_size = embed.weight.shape[1]
        self.budget = budget
        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))

        self.enc_layer = build_sentence_encoder(
            layer, emb_size, hidden_size, bidirectional=True
        )

        if hasattr(self.enc_layer, "cnn"):
            enc_size = self.enc_layer.cnn.out_channels
        else:
            enc_size = hidden_size * 2

        self.hopfield_pooling = HopfieldPooling(
            input_size=enc_size, output_size=enc_size, num_heads=num_heads, alpha = alpha, sparseMAP=True, scaling = 1/temperature,
        alpha_as_static=True, dropout=bag_dropout)

        self.z = None  # z samples
        self.z_dists = []  # z distribution(s)
        self.transition = transition
        self.temperature = temperature
        self.alpha = alpha
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, z, mask=None):
        x = x.cuda()
        batch_size, target_size = x.shape
        emb = self.embed_layer(x)

        # z is also used to control when the encoder layer is active
        lengths = mask.long().sum(1)

        # encode the sentence
        h, _ = self.enc_layer(emb, mask, lengths)
        hs = []
        zs = []
        if self.training:
            self.hopfield_pooling.hopfield.association_core.scaling = 1/self.temperature
        else:
            test_temperature = 1e-3
            self.hopfield_pooling.hopfield.association_core.scaling = 1/test_temperature
        
        for k in range(batch_size):
            K = torch.round(self.budget / 100 * lengths[k])
            self.hopfield_pooling.hopfield.association_core.k = int(K.item())
            h1, z = self.hopfield_pooling(h[k].unsqueeze(0))
            hs.append(h1)
            zs.append(z)
        # predict sentiment from final state(s)
        h = torch.stack(hs, dim=0).squeeze(1).squeeze(1)
        y = self.output_layer(h)
        z = torch.stack(zs, dim=0).squeeze(1).squeeze(1)  # [B, T]
        z = z.cuda()
        z = torch.where(mask, z, z.new_zeros([1]))
        return y, z

class SBHopfieldSentimentPredictor(nn.Module):
    """
    The Encoder takes an input text (and rationale z) and computes p(y|x,z)

    Supports a sigmoid on the final result (for regression)
    If not sigmoid, will assume cross-entropy loss (for classification)

    """

    def __init__(
        self,
        embed: nn.Embedding = None,
        hidden_size: int = 200,
        output_size: int = 1,
        dropout: float = 0.1,
        layer: str = "lstm",
        nonlinearity: str = "sigmoid",
        alpha: float = 2.0,
        num_heads: int = 4,
        bag_dropout: float = 0.0,
        transition: int = 0,
        temperature: float = 0.01,
        budget: int = 20,
        init: bool = False,

    ):

        super().__init__()

        emb_size = embed.weight.shape[1]
        self.budget = budget
        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))

        self.enc_layer = build_sentence_encoder(
            layer, emb_size, hidden_size, bidirectional=True
        )

        if hasattr(self.enc_layer, "cnn"):
            enc_size = self.enc_layer.cnn.out_channels
        else:
            enc_size = hidden_size * 2

        self.hopfield_pooling = HopfieldPooling(
            input_size=enc_size, output_size=enc_size, num_heads=num_heads, alpha = alpha, sparseMAP=True, Seq_budget=True, scaling = 1/temperature,
        alpha_as_static=True, dropout=bag_dropout)
        self.init = init
        self.z = None  # z samples
        self.z_dists = []  # z distribution(s)
        self.transition = transition
        self.num_heads = num_heads
        self.temperature = temperature
        self.alpha = alpha
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, z, mask=None):
        # encode sentence
        batch_size, target_size = x.shape
        lengths = mask.long().sum(1).cpu()
        emb = self.embed_layer(x)  # [B, T, E]

        # [B, T, H]
        h1, _ = self.enc_layer(emb, mask, lengths)

        t = torch.full((batch_size, self.num_heads, target_size + 1), float(self.transition))
        hs = []
        z = []
        for k in range(batch_size):
            budget = torch.round(self.budget / 100 * lengths[k])
            # H:SeqBudget consists of a single factor so, in this particular case, the LP-SparseMAP solution is
            # indeed the SparseMAP solution and it can be found within a single iteration.
            self.max_iter = 1
            self.step_size = 0.0
            self.hopfield_pooling.hopfield.association_core.init = self.init
            self.hopfield_pooling.hopfield.association_core.max_iter = self.max_iter
            self.hopfield_pooling.hopfield.association_core.step_size = self.step_size
            self.hopfield_pooling.hopfield.association_core.transitions = self.transition
            self.hopfield_pooling.hopfield.association_core.init_step = False
            self.hopfield_pooling.hopfield.association_core.budget = int(budget.item())
            self.hopfield_pooling.hopfield.association_core.t = t[k]
            self.hopfield_pooling.hopfield.association_core.length = lengths[k]
            if self.training:
                self.hopfield_pooling.hopfield.association_core.scaling = 1/self.temperature

                h2, z_probs = self.hopfield_pooling(h1[k].unsqueeze(0))
            else:
                test_temperature = 1e-3
                self.hopfield_pooling.hopfield.association_core.scaling = 1/test_temperature

                h2, z_probs = self.hopfield_pooling(h1[k].unsqueeze(0))
            z.append(z_probs)
            hs.append(h2)

        # predict sentiment from final state(s)
        h = torch.stack(hs, dim=0).squeeze(1).squeeze(1)
        y = self.output_layer(h)
        z = torch.stack(z, dim=0).squeeze(1).squeeze(1)  # [B, T]
        z = z.cuda()
        z = torch.where(mask, z, z.new_zeros([1]))
        return y, z
