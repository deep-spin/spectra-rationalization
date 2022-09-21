import logging

from rationalizers.lightning_models.highlights.base import BaseRationalizer
from rationalizers.utils import (
    masked_average
)

shell_logger = logging.getLogger(__name__)


class TransformerBaselineClassifier(BaseRationalizer):

    def get_factual_flow(self, x, mask=None, z=None):
        """
        Compute the factual flow.

        :param x: input ids tensor. torch.LongTensor of shape [B, T] or input vectors of shape [B, T, |V|]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param z: precomputed latent vector. torch.FloatTensor of shape [B, T] (default None)
        :return: z, y_hat
        """
        ext_mask = mask[:, None, None, :]  # add head and seq dimension
        ext_mask = ext_mask.to(dtype=self.dtype)  # fp16 compatibility
        ext_mask = (1.0 - ext_mask) * -10000.0  # will set softmax to zero

        gen_e = self.gen_emb_layer(x)
        gen_h = self.gen_encoder(gen_e, ext_mask).last_hidden_state
        gen_h = self.explainer_mlp(gen_h) if self.explainer_pre_mlp else gen_h
        z, _ = self.explainer(gen_h, mask)
        summary = masked_average(gen_h, z)
        y_hat = self.output_layer(summary)
        return z, y_hat
