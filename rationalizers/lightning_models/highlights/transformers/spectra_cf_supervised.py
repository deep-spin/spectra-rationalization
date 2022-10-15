import logging
from copy import deepcopy

import torch
from transformers import EncoderDecoderModel
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right

from rationalizers import constants
from rationalizers.lightning_models.highlights.transformers.spectra_cf import \
    CounterfactualTransformerSPECTRARationalizer
from rationalizers.lightning_models.utils import make_input_for_t5
from rationalizers.utils import freeze_module

shell_logger = logging.getLogger(__name__)


class SupervisedCfTransformerSPECTRARationalizer(CounterfactualTransformerSPECTRARationalizer):

    def __init__(
        self,
        tokenizer: object,
        nb_classes: int,
        is_multilabel: bool,
        h_params: dict,
    ):
        """
        :param tokenizer (object): torchnlp tokenizer object
        :param nb_classes (int): number of classes used to create the last layer
        :param multilabel (bool): whether the problem is multilabel or not (it depends on the dataset)
        :param h_params (dict): hyperparams dict. See docs for more info.
        """
        super().__init__(tokenizer, nb_classes, is_multilabel, h_params)

        # TODO: add another type of supervision: mlm
        assert 'seq2seq' in self.cf_supervision or 'mlm' in self.cf_supervision

        self.cf_teacher_forcing_during_eval = h_params.get('cf_teacher_forcing_during_eval', True)

        # delete the old cf_gen_hf
        del self.cf_gen_hf

        if 'bert' in self.cf_gen_arch:

            self.cf_gen_hf = EncoderDecoderModel.from_encoder_decoder_pretrained(self.cf_gen_arch, self.cf_gen_arch)

            if 'roberta' in self.cf_gen_arch:
                # detach lm_head from shared weights
                if self.cf_gen_lm_head_requires_grad != self.cf_gen_emb_requires_grad:
                    self.cf_gen_hf.decoder.lm_head = deepcopy(self.cf_gen_hf.decoder.lm_head)
                # set submodules
                self.cf_gen_emb_layer = self.cf_gen_hf.encoder.embeddings
                self.cf_gen_encoder = self.cf_gen_hf.encoder.encoder
                self.cf_gen_decoder = self.cf_gen_hf.decoder.roberta.encoder
                self.cf_gen_lm_head = self.cf_gen_hf.decoder.lm_head
                # tie embeddings for encoder and decoder
                self.cf_gen_hf.decoder.roberta.embeddings = self.cf_gen_hf.encoder.embeddings
            else:
                # detach lm_head from shared weights
                if self.cf_gen_lm_head_requires_grad != self.cf_gen_emb_requires_grad:
                    self.cf_gen_hf.decoder.cls = deepcopy(self.cf_gen_hf.decoder.cls)
                # set submodules
                self.cf_gen_emb_layer = self.cf_gen_hf.encoder.embeddings
                self.cf_gen_encoder = self.cf_gen_hf.encoder.encoder
                self.cf_gen_decoder = self.cf_gen_hf.decoder.bert.encoder
                self.cf_gen_lm_head = self.cf_gen_hf.decoder.cls
                # tie embeddings for encoder and decoder
                self.cf_gen_hf.decoder.bert.embeddings = self.cf_gen_hf.encoder.embeddings

            # update config vars
            self.cf_gen_hidden_size = self.cf_gen_hf.config.encoder.hidden_size
            self.cf_gen_hf.config.decoder_start_token_id = self.tokenizer.cls_token_id
            self.cf_gen_hf.config.pad_token_id = self.tokenizer.pad_token_id
            self.cf_gen_hf.config.vocab_size = self.cf_gen_hf.config.decoder.vocab_size

        # freeze embedding layers
        if not self.cf_gen_emb_requires_grad:
            freeze_module(self.cf_gen_emb_layer)

        # freeze models and set to eval mode to disable dropout
        if not self.cf_gen_encoder_requires_grad:
            freeze_module(self.cf_gen_encoder)
            if self.cf_gen_decoder is not None:
                freeze_module(self.cf_gen_decoder)

        # the lm head is an independent factor, which we can freeze or not
        if not self.cf_gen_lm_head_requires_grad:
            # it should not be shared with the embedding layer
            assert id(self.cf_gen_lm_head.weight) != id(self.cf_gen_emb_layer.weight)
            freeze_module(self.cf_gen_lm_head)

        # shared factual and counterfactual generators (only the LM head remains separate)
        if self.share_generators:
            del self.ff_gen_hf
            del self.ff_gen_emb_layer
            del self.ff_gen_encoder
            del self.ff_gen_hidden_size
            self.ff_gen_hf = self.cf_gen_hf
            self.ff_gen_emb_layer = self.cf_gen_emb_layer
            self.ff_gen_encoder = self.cf_gen_encoder
            self.ff_gen_decoder = self.cf_gen_decoder if self.ff_gen_encoder is not None else None
            self.ff_gen_hidden_size = self.cf_gen_hidden_size

        # by now we should have a decoder for supervising the counterfactual generator
        assert self.cf_gen_decoder is not None
        # and we will not sample manually-crafted counterfactuals from the dataset
        assert self.cf_manual_sample is False

    def forward(
        self,
        x: torch.LongTensor,
        x_cf: torch.LongTensor = None,
        mask: torch.BoolTensor = None,
        mask_cf: torch.BoolTensor = None,
        token_type_ids: torch.BoolTensor = None,
        token_type_ids_cf: torch.BoolTensor = None,
        current_epoch=None,
    ):
        """
        Compute forward-pass.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param x_cf: counterfactual input ids tensor. torch.LongTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param mask_cf: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids: mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids_cf: mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :param current_epoch: int represents the current epoch.
        :return: (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """
        # factual flow
        z, y_hat = self.get_factual_flow(x, mask=mask, token_type_ids=token_type_ids)

        # counterfactual flow
        x_tilde, z_tilde, mask_tilde, y_tilde_hat = self.get_counterfactual_flow(
            x, z, mask=mask, token_type_ids=token_type_ids,
            x_dec=x_cf, mask_dec=mask_cf, token_type_ids_dec=token_type_ids_cf,
        )

        # return everything as output (useful for computing the loss)
        return (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat)

    def get_counterfactual_flow(self, x, z, mask=None, token_type_ids=None,
                                x_dec=None, mask_dec=None, token_type_ids_dec=None):
        """
        Compute the counterfactual flow.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param z: binary variables tensor. torch.FloatTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids: mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :param x_dec: decoder input ids tensor. torch.LongTensor of shape [B, T]
        :param mask_dec: decoder mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids_dec: decoder mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :return: (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """
        # prepare input for the generator LM
        e = self.cf_gen_emb_layer(x) if self.cf_input_space == 'embedding' else x

        # get <mask> vectors
        if 't5' in self.cf_gen_arch:
            # fix inputs for t5 (replace chunked masked positions by a single sentinel token)
            x, e, z, mask = make_input_for_t5(x, e, z, mask, pad_id=constants.PAD_ID)
            # create sentinel tokens
            sentinel_ids = 32100 - (z > 0).long().cumsum(dim=-1)
            # clamp valid input ids (might glue last generations as T5 has only 100 sentinels)
            sentinel_ids = torch.clamp(sentinel_ids, min=32000, max=32099)
            # fix x by replacing selected tokens by sentinel ids
            x = (z > 0).long() * sentinel_ids + (z == 0).long() * x
            # get sentinel embeddings
            e_mask = self.cf_gen_emb_layer(sentinel_ids)
        else:
            e_mask = self.cf_gen_emb_layer(torch.ones_like(x) * self.mask_token_id)

        # create mask for pretrained-LMs
        ext_mask = (1.0 - mask[:, None, None, :].to(self.dtype)) * -10000.0
        ext_mask_dec = (1.0 - mask_dec[:, None, None, :].to(self.dtype)) * -10000.0

        # get the new mask
        z_mask = (z * mask.float()).unsqueeze(-1)

        # set the input for the counterfactual encoder via a differentiable where
        s_bar = e_mask * z_mask + e * (1 - z_mask)

        # pass (1-z)-masked inputs
        if 't5' in self.cf_gen_arch:
            cf_gen_enc_out = self.cf_gen_encoder(inputs_embeds=s_bar, attention_mask=mask)
            h_tilde = cf_gen_enc_out.last_hidden_state
        else:
            cf_gen_enc_out = self.cf_gen_encoder(s_bar, ext_mask)
            h_tilde = cf_gen_enc_out.last_hidden_state

        # pass through the decoder and the LM head
        x_tilde, logits = self._sample_from_lm(
            enc_hidden_states=h_tilde,
            enc_ext_mask=ext_mask,
            dec_ids=x_dec,
            dec_ext_mask=ext_mask_dec,
            encoder_outputs=cf_gen_enc_out
        )
        mask_tilde = x_tilde != self.tokenizer.pad_token_id
        token_type_ids_tilde = token_type_ids_dec

        if 't5' in self.cf_gen_arch:
            # expand z to account for the new generated tokens (important for t5)
            x_tilde, _, mask_tilde, token_type_ids_tilde = self._expand_factual_inputs_from_x_tilde(
                x, z, mask, x_tilde, token_type_ids=token_type_ids
            )

        # use teacher-forcing during training
        if self.training and x_dec is not None:
            x_tilde = x_dec

        # reuse the factual flow to get a prediction for the counterfactual flow
        z_tilde, y_tilde_hat = self.get_factual_flow(
            x_tilde, mask=mask_tilde, token_type_ids=token_type_ids_tilde, from_cf=True
        )

        return x_tilde, z_tilde, mask_tilde, y_tilde_hat

    def _sample_from_lm(self, enc_hidden_states, enc_ext_mask, dec_ids, dec_ext_mask, encoder_outputs=None):  # noqa
        if not self.training and not self.cf_teacher_forcing_during_eval and self.stage == 'test':
            # sample autoregressively
            gen_out = self.cf_gen_hf.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=(enc_ext_mask.squeeze(2).squeeze(1) == 0).long(),
                return_dict_in_generate=True,
                output_scores=True,
                **self.cf_generate_kwargs
            )
            # clear memory because generation is done
            torch.cuda.empty_cache()

            # stack outputs and cut out start token
            logits = torch.stack(gen_out.scores).transpose(0, 1)
            x_tilde = gen_out.sequences[:, 1:] if gen_out.sequences.shape[1] > logits.shape[1] else gen_out.sequences

        else:
            # shift decoder ids to the right
            dec_input_ids = shift_tokens_right(
                dec_ids, self.cf_gen_hf.config.pad_token_id, self.cf_gen_hf.config.decoder_start_token_id
            )

            # no need to sample since we have the ground-truth
            dec_embs = self.cf_gen_emb_layer(dec_input_ids)
            cf_gen_dec_out = self.cf_gen_decoder(
                dec_embs,
                attention_mask=dec_ext_mask,
                encoder_hidden_states=enc_hidden_states,
                encoder_attention_mask=enc_ext_mask,

            )
            logits = self.cf_gen_lm_head(cf_gen_dec_out.last_hidden_state)
            x_tilde = logits.argmax(dim=-1)

        # save variables for computing penalties later
        self.cf_x_tilde = x_tilde.clone()
        self.cf_log_prob_x_tilde = torch.log_softmax(logits, dim=-1)
        return x_tilde, logits

    def get_counterfactual_loss(self, y_hat, y, z_tilde, mask_tilde, prefix, x_tilde=None, x_cf=None):
        """
        Compute loss for the counterfactual flow.

        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param z_tilde: latent selection vector. torch.FloatTensor of shape [B, T]
        :param mask_tilde: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param prefix: prefix for loss statistics (train, val, test)
        :param x_tilde: generated counterfactual input ids tensor.
                        torch.LongTensor of shape [B, T] for reinforce sampling
                        torch.LongTensor of shape [B, T, V] for gumbel-softmax sampling
        :param x_cf: gold counterfactual input ids tensor. torch.LongTensor of shape [B, T]
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        main_loss, stats = super().get_counterfactual_loss(y_hat, y, z_tilde, mask_tilde, prefix, x_tilde, x_cf)

        if not self.training and not self.cf_teacher_forcing_during_eval and self.stage == 'test':
            return main_loss, stats

        # we need to use T5 here, there is no way to do this with an encoder-only model
        # approximate original distribution with the new distribution
        # penalty += - torch.nn.functional.kl_divergence(original_bert_dist, torch.exp(logp_xtilde))
        lm_logits = self.cf_log_prob_x_tilde.view(-1, self.cf_log_prob_x_tilde.size(-1))
        lm_labels = x_cf.view(-1,)
        lm_labels = lm_labels.masked_fill(lm_labels == self.pad_token_id, -100)
        cost = torch.nn.functional.nll_loss(lm_logits, lm_labels)
        penalty = self.penalty_seq2seq * cost.mean()

        # update loss and stats
        main_loss = main_loss + penalty
        stats[prefix + "_cf_penalty"] += penalty.item()

        return main_loss, stats
