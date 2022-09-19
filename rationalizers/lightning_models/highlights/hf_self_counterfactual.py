import logging

import torch
from torch import nn

from rationalizers import cf_constants
from rationalizers.lightning_models.highlights.hf_counterfactual import (
    CounterfactualRationalizer,
    sample_from_logits,
    make_input_for_t5, get_new_frequencies_of_gen_ids_from_t5, repeat_interleave_and_pad, merge_input_and_gen_ids
)

shell_logger = logging.getLogger(__name__)


class SelfCounterfactualRationalizer(CounterfactualRationalizer):

    def __init__(
        self,
        tokenizer: object,
        cf_tokenizer: object,
        nb_classes: int,
        is_multilabel: bool,
        h_params: dict,
    ):
        super().__init__(tokenizer, cf_tokenizer, nb_classes, is_multilabel, h_params)
        self.reuse_factual_flow = h_params.get("reuse_factual_flow", False)

    def get_counterfactual_flow(self, x, z, mask=None):
        """
        Compute the counterfactual flow.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param z: binary variables tensor. torch.FloatTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """
        # prepare input for the generator LM
        e = self.cf_gen_emb_layer(x) if self.cf_input_space == 'embedding' else x

        if 't5' in self.cf_gen_arch:
            # fix inputs for t5 (replace chunked masked positions by a single sentinel token)
            x, e, z, mask = make_input_for_t5(x, e, z, mask, pad_id=cf_constants.PAD_ID)
            # create sentinel tokens
            sentinel_ids = 32100 - (z > 0).long().cumsum(dim=-1)
            # clamp valid input ids (might glue last generations as T5 has only 100 sentinels)
            sentinel_ids = torch.clamp(sentinel_ids, min=32000, max=32099)
            # fix x by replacing selected tokens by sentinel ids
            x = (z > 0).long() * sentinel_ids + (z == 0).long() * x
            # get sentinel embeddings
            e_mask = self.cf_gen_emb_layer(sentinel_ids)
        else:
            e_mask = self.cf_gen_emb_layer(torch.ones_like(x) * self.cf_mask_token_id)

        # create mask for pretrained-LMs
        ext_mask = (1.0 - mask[:, None, None, :].to(self.dtype)) * -10000.0

        # get the new mask
        z_mask = (z * mask.float()).unsqueeze(-1)

        # get mask for the generator LM
        # gen_ext_mask = ext_mask * (z_bar.squeeze(-1)[:, None, None, :] > 0.0).float()
        gen_ext_mask = ext_mask

        # differentiable where
        s_bar = e_mask * z_mask + e * (1 - z_mask)

        # pass (1-z)-masked inputs
        if 't5' in self.cf_gen_arch:
            cf_gen_enc_out = self.cf_gen_encoder(inputs_embeds=s_bar, attention_mask=mask)
            h_tilde = cf_gen_enc_out.last_hidden_state
        else:
            cf_gen_enc_out = self.cf_gen_encoder(s_bar, gen_ext_mask)
            h_tilde = cf_gen_enc_out.last_hidden_state

        # sample from LM
        x_tilde, logits = self._sample_from_lm(x, h_tilde, z_mask, mask, encoder_outputs=cf_gen_enc_out)

        # expand z to account for the new tokens in case of using t5
        if 't5' in self.cf_gen_arch:
            # get the counts needed to expand the input_ids into generated_ids
            gen_counts = get_new_frequencies_of_gen_ids_from_t5(
                x, x_tilde, pad_id=cf_constants.PAD_ID, eos_id=cf_constants.EOS_ID,
            )
            # and vice-versa
            inp_counts = get_new_frequencies_of_gen_ids_from_t5(
                x_tilde, x, pad_id=cf_constants.PAD_ID, eos_id=cf_constants.EOS_ID
            )

            # expand x, z, mask according to gen_counts
            x_rep = repeat_interleave_and_pad(x, gen_counts, pad_id=cf_constants.PAD_ID)
            z_tilde = repeat_interleave_and_pad(z, gen_counts, pad_id=cf_constants.PAD_ID)
            mask_tilde = repeat_interleave_and_pad(mask, gen_counts, pad_id=cf_constants.PAD_ID)

            # expand x_tilde according to inp_counts
            x_tilde_rep = repeat_interleave_and_pad(x_tilde, inp_counts)

            # merge x_rep and x_tilde_rep into a single tensor
            x_tilde = merge_input_and_gen_ids(x_rep, x_tilde_rep, pad_id=cf_constants.PAD_ID)

            # fix the corner case of generating fewer tokens than what was selected
            original_seq_len = z_tilde.shape[-1]
            expanded_seq_len = x_tilde.shape[-1]
            if original_seq_len > expanded_seq_len:
                z_tilde = z_tilde[:, :expanded_seq_len]
                mask_tilde = mask_tilde[:, :expanded_seq_len]

            # remove labels in case they were prepended for mice t5
            # (the prepended prompt has 6 tokens)
            if 'mice' in self.cf_gen_arch and self.cf_prepend_label_for_mice:
                x_tilde = x_tilde[:, 6:]
                z_tilde = z_tilde[:, 6:]
                mask_tilde = mask_tilde[:, 6:]

            # if we generated too much, there isn't much we can do besides truncating
            if x_tilde.shape[-1] > 512 and 'bert' in self.cf_pred_arch:
                x_tilde = x_tilde[:, :512]
                z_tilde = z_tilde[:, :512]
                mask_tilde = mask_tilde[:, :512]

            # fixme: we need to remap t5 generated ids to bert ids or use t5 embeddings
            #        in case of using pretrained transformers as the counterfactual predictor
            #        (for now we are ignoring this combination)
            # if 'bert' in self.cf_pred_arch:
            #     x_tilde, z_tilde, mask_tilde = map_t5_ids_to_bert_ids(x_tilde, z_tilde, mask_tilde)

        else:  # otherwise our dimensions match, so we can reuse the same z and mask
            z_tilde = z
            mask_tilde = mask

        # reuse the factual flow to get a prediction for the counterfactual flow
        if self.reuse_factual_flow:
            z_cloned = self.explainer.z.clone()
            z_tilde, y_tilde_hat = self.get_factual_flow(x_tilde, mask=mask_tilde)
            self.explainer.z = z_cloned

        else:
            # pass inputs or hidden states to the predictor
            if self.cf_selection_faithfulness:
                # get the predictor embeddings corresponding to x_tilde
                if self.cf_use_reinforce:
                    # x_tilde contains indices
                    pred_e = self.cf_pred_emb_layer(x_tilde)
                else:
                    # x_tilde contains one-hot vectors
                    inputs_embeds = x_tilde @ self.cf_pred_emb_layer.word_embeddings.weight
                    pred_e = self.cf_pred_emb_layer(inputs_embeds=inputs_embeds)
            else:
                # get the generator hidden states
                pred_e = h_tilde

            # pass inputs directly
            if self.cf_pred_arch == 'lstm':
                _, summary = self.cf_pred_encoder(pred_e, mask_tilde)
                y_tilde_hat = self.cf_output_layer(summary)
            elif self.cf_pred_arch == 'masked_average':
                summary = self.cf_pred_encoder(pred_e, mask_tilde)
                y_tilde_hat = self.cf_output_layer(summary)
            else:
                # pass the selected inputs through the classifier
                assert self.cf_pred_for_sequence_classfication is True
                if 't5' in self.cf_pred_arch:
                    output = self.cf_pred_encoder.generate(inputs_embeds=pred_e, max_length=2,
                                                           return_dict_in_generate=True, output_scores=True)
                    neg_id = self.tokenizer.vocab['▁negative']
                    pos_id = self.tokenizer.vocab['▁positive']
                    scores = output['scores'][0][:, [neg_id, pos_id]]
                    y_tilde_hat = torch.log_softmax(scores, dim=-1)
                else:
                    output = self.cf_pred_encoder(inputs_embeds=pred_e, attention_mask=mask_tilde)
                    y_tilde_hat = torch.log_softmax(output['logits'], dim=-1)

        self.z_tilde = z_tilde
        return x_tilde, z_tilde, mask_tilde, y_tilde_hat

    def _sample_from_lm(self, x, h_tilde, z_mask, mask, encoder_outputs=None):
        if self.cf_use_reinforce:
            if 't5' in self.cf_gen_arch:
                # recover hidden states from the encoder (.generate() changes the hidden states)
                encoder_hidden_states = h_tilde.clone()

                # deal with min and max length
                gen_kwargs = self.cf_generate_kwargs.copy()
                if 'max_length' not in self.cf_generate_kwargs:
                    gen_kwargs['max_length'] = 512
                if self.cf_generate_kwargs.get('min_length', None) == 'original':
                    # set the minimum length to be at least equal to the number of
                    # sentinel tokens (times 2 since T5 has to generate sentinels too)
                    num_sentinels = (z_mask > 0).long().sum(-1).min().item()
                    gen_kwargs['min_length'] = min(gen_kwargs['max_length'], num_sentinels * 2)

                # sample autoregressively
                gen_out = self.cf_gen_hf.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=mask.long(),
                    return_dict_in_generate=True,
                    **gen_kwargs
                )
                # clear memory because generation is done
                # torch.cuda.empty_cache()

                # idk why but t5 generates a pad symbol as the first token
                # so we cut it out for all samples in the batch
                # (this happens only for .sequences)
                x_tilde = gen_out.sequences[:, 1:]

                # get the logits for x_tilde
                cf_gen_dec_out = self.cf_gen_decoder(
                    input_ids=gen_out.sequences,
                    attention_mask=(gen_out.sequences != cf_constants.PAD_ID).long(),
                    encoder_hidden_states=encoder_hidden_states
                )
                logits = self.cf_gen_lm_head(cf_gen_dec_out.last_hidden_state)[:, :-1]

            else:
                # sample directly from the output layer
                logits = self.cf_gen_lm_head(h_tilde)
                # x_tilde = logits.argmax(dim=-1)
                x_tilde = sample_from_logits(
                    logits=logits,
                    top_k=self.cf_generate_kwargs.get('top_k', 0),
                    top_p=self.cf_generate_kwargs.get('top_p', 1.0),
                    min_tokens_to_keep=self.cf_generate_kwargs.get('min_tokens_to_keep', 1.0),
                    num_samples=self.cf_generate_kwargs.get('num_return_sequences', 1),
                ).squeeze(-1)

                # get gen_ids only for <mask> positions
                z_1 = (z_mask > 0).squeeze(-1).long()
                x_tilde = z_1 * x_tilde + (1 - z_1) * x

            # save variables for computing REINFORCE loss later
            self.cf_x_tilde = x_tilde.clone()
            self.cf_log_prob_x_tilde = torch.log_softmax(logits, dim=-1)

        else:
            # use the ST-gumbel-softmax trick
            logits = self.cf_gen_lm_head(h_tilde)
            x_tilde = nn.functional.gumbel_softmax(logits, hard=True, dim=-1)
            # x_tilde.shape is (bs, seq_len, |V|)

            # get gen_ids only for <mask> positions
            z_1 = (z_mask > 0).long()
            x_one_hot = nn.functional.one_hot(x, num_classes=x_tilde.shape[-1])
            x_tilde = z_1 * x_tilde + (1 - z_1) * x_one_hot

            # save variables for computing penalties later
            self.cf_x_tilde = x_tilde.clone()
            self.cf_log_prob_x_tilde = torch.log_softmax(logits, dim=-1)

        return x_tilde, logits
