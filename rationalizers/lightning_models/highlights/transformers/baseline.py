import logging

from rationalizers.lightning_models.highlights.transformers.base import TransformerBaseRationalizer
from rationalizers.utils import masked_average, get_ext_mask

shell_logger = logging.getLogger(__name__)


class TransformerBaselineClassifier(TransformerBaseRationalizer):

    def get_factual_flow(self, x, mask=None, z=None):
        """
        Compute the factual flow.

        :param x: input ids tensor. torch.LongTensor of shape [B, T] or input vectors of shape [B, T, |V|]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param z: precomputed latent vector. torch.FloatTensor of shape [B, T] (default None)
        :return: z, y_hat
        """
        gen_e = self.ff_gen_emb_layer(x)

        if 't5' in self.ff_gen_arch:
            # t5 encoder and decoder receives word_emb and a raw mask
            gen_h = self.ff_gen_encoder(inputs_embeds=gen_e, attention_mask=mask)
            if self.ff_gen_use_decoder:
                gen_h = self.ff_gen_decoder(inputs_embeds=gen_e, attention_mask=mask,
                                            encoder_hidden_states=gen_h.last_hidden_state,)
        else:
            # bert encoder receives word_emb + pos_embs and an extended mask
            ext_mask = get_ext_mask(mask)
            gen_h = self.ff_gen_encoder(hidden_states=gen_e, attention_mask=ext_mask)

        # get final hidden states
        # selected_layers = list(map(int, self.selected_layers.split(',')))
        # gen_h = torch.stack(gen_h)[selected_layers].mean(dim=0)
        gen_h = gen_h.last_hidden_state

        gen_h = self.explainer_mlp(gen_h) if self.explainer_pre_mlp else gen_h
        z, _ = self.explainer(gen_h, mask)
        summary = gen_h @ z
        y_hat = self.ff_output_layer(summary)
        return z, y_hat
