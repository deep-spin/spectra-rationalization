import logging
import os
from itertools import chain

import numpy as np
import torch
import torchmetrics
import wandb
from torch import nn
from transformers import AutoModel, AutoTokenizer

from rationalizers import cf_constants, constants
from rationalizers.builders import build_optimizer, build_scheduler
from rationalizers.explainers import available_explainers
from rationalizers.lightning_models.highlights.base import BaseRationalizer
from rationalizers.modules.metrics import evaluate_rationale
from rationalizers.modules.scalar_mix import ScalarMixWithDropout
from rationalizers.modules.sentence_encoders import LSTMEncoder, MaskedAverageEncoder
from rationalizers.utils import get_z_stats, freeze_module, masked_average, get_rationales, unroll, get_html_rationales, \
    save_rationales, save_counterfactuals

shell_logger = logging.getLogger(__name__)


class CounterfactualRationalizer(BaseRationalizer):

    def __init__(
        self,
        tokenizer: object,
        cf_tokenizer: object,
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

        ########################
        # hyperparams
        ########################
        # factual:
        self.gen_arch = h_params.get("gen_arch", "bert-base-multilingual-cased")
        self.pred_arch = h_params.get("pred_arch", "bert-base-multilingual-cased")
        self.gen_emb_requires_grad = h_params.get("gen_emb_requires_grad", False)
        self.pred_emb_requires_grad = h_params.get("pred_emb_requires_grad", False)
        self.gen_encoder_requires_grad = h_params.get("gen_encoder_requires_grad", True)
        self.pred_encoder_requires_grad = h_params.get("pred_encoder_requires_grad", True)
        self.pred_output_requires_grad = h_params.get("pred_output_requires_grad", True)
        self.shared_gen_pred = h_params.get("shared_gen_pred", False)
        self.use_scalar_mix = h_params.get("use_scalar_mix", True)
        self.dropout = h_params.get("dropout", 0.1)
        self.selection_space = h_params.get("selection_space", 'embedding')
        self.selection_vector = h_params.get("selection_vector", 'zero')
        self.selection_mask = h_params.get("selection_mask", True)
        self.selection_faithfulness = h_params.get("selection_faithfulness", True)

        # counterfactual:
        self.cf_gen_arch = h_params.get("cf_gen_arch", "bert-base-multilingual-cased")
        self.cf_pred_arch = h_params.get("cf_pred_arch", "bert-base-multilingual-cased")
        self.cf_gen_emb_requires_grad = h_params.get("cf_gen_emb_requires_grad", False)
        self.cf_pred_emb_requires_grad = h_params.get("cf_pred_emb_requires_grad", False)
        self.cf_gen_encoder_requires_grad = h_params.get("cf_gen_encoder_requires_grad", True)
        self.cf_gen_lm_head_requires_grad = h_params.get("cf_gen_lm_head_requires_grad", True)
        self.cf_pred_encoder_requires_grad = h_params.get("cf_pred_encoder_requires_grad", True)
        self.cf_pred_output_requires_grad = h_params.get("cf_pred_output_requires_grad", True)
        self.cf_shared_gen_pred = h_params.get("cf_shared_gen_pred", False)
        self.cf_use_scalar_mix = h_params.get("cf_use_scalar_mix", True)
        self.cf_dropout = h_params.get("cf_dropout", 0.1)
        self.cf_input_space = h_params.get("cf_input_space", 'embedding')
        self.cf_selection_vector = h_params.get("cf_selection_vector", 'zero')
        self.cf_selection_mask = h_params.get("cf_selection_mask", True)
        self.cf_selection_faithfulness = h_params.get("cf_selection_faithfulness", True)
        self.cf_selection_mode = h_params.get("cf_selection_mode", 'select')
        self.cf_use_reinforce = h_params.get('cf_use_reinforce', True)
        self.cf_use_baseline = h_params.get('cf_use_baseline', True)
        self.cf_lbda = h_params.get('cf_lbda', 1.0)
        self.cf_generate_kwargs = h_params.get('cf_generate_kwargs', dict())
        # both:
        self.explainer_fn = h_params.get("explainer", True)
        self.explainer_pre_mlp = h_params.get("explainer_pre_mlp", True)
        self.explainer_requires_grad = h_params.get("explainer_requires_grad", True)
        self.temperature = h_params.get("temperature", 1.0)
        self.share_predictors = h_params.get("share_predictors", False)
        self.share_generators = h_params.get("share_generators", False)

        if self.shared_gen_pred:
            assert self.gen_arch == self.pred_arch
        if self.cf_shared_gen_pred:
            assert self.cf_gen_arch == self.cf_pred_arch

        ########################
        # factual flow
        ########################
        self.z = None
        self.mask_token_id = tokenizer.mask_token_id
        # generator module
        self.gen_hf = AutoModel.from_pretrained(self.gen_arch)
        self.gen_emb_layer = self.gen_hf.embeddings if 't5' not in self.gen_arch else self.gen_hf.shared
        self.gen_encoder = self.gen_hf.encoder
        self.gen_hidden_size = self.gen_hf.config.hidden_size
        self.gen_scalar_mix = ScalarMixWithDropout(
            mixture_size=self.gen_hf.config.num_hidden_layers+1,
            dropout=self.dropout,
            do_layer_norm=False,
        )

        # explainer
        explainer_cls = available_explainers[self.explainer_fn]
        self.explainer = explainer_cls(h_params, self.gen_hidden_size)
        self.explainer_mlp = nn.Sequential(
            nn.Linear(self.gen_hidden_size, self.gen_hidden_size),
            nn.Tanh(),
        )

        # predictor module
        if self.pred_arch == 'lstm':
            self.pred_encoder = LSTMEncoder(self.gen_hidden_size, self.gen_hidden_size, bidirectional=True)
            self.pred_hidden_size = self.gen_hidden_size * 2
            self.pred_emb_layer = self.gen_emb_layer
            self.pred_hf = None
            self.pred_scalar_mix = None
        elif self.pred_arch == 'masked_average':
            self.pred_encoder = MaskedAverageEncoder()
            self.pred_hidden_size = self.gen_hidden_size
            self.pred_emb_layer = self.gen_emb_layer
            self.pred_hf = None
            self.pred_scalar_mix = None
        else:
            self.pred_hf = self.gen_hf if self.shared_gen_pred else AutoModel.from_pretrained(self.pred_arch, dropout=self.dropout)
            self.pred_hidden_size = self.pred_hf.config.hidden_size
            self.pred_emb_layer = self.pred_hf.embeddings
            self.pred_encoder = self.pred_hf.encoder
            self.pred_scalar_mix = ScalarMixWithDropout(
                mixture_size=self.pred_hf.config.num_hidden_layers+1,
                dropout=self.dropout,
                do_layer_norm=False,
            )
        # predictor output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.pred_hidden_size, self.pred_hidden_size),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.pred_hidden_size, nb_classes),
            nn.Sigmoid() if not self.is_multilabel else nn.LogSoftmax(dim=-1),
        )

        ########################
        # counterfactual flow
        ########################
        self.z_tilde = None
        self.cf_tokenizer = cf_tokenizer
        self.cf_mask_token_id = self.cf_tokenizer.mask_token_id

        # for reinforce
        self.n_points = 0
        self.mean_baseline = 0
        self.log_prob_x_tilde = None

        # counterfactual generator module
        if 't5' in self.cf_gen_arch:
            from transformers import T5ForConditionalGeneration
            self.cf_gen_hf = T5ForConditionalGeneration.from_pretrained(self.cf_gen_arch)
            self.cf_gen_emb_layer = self.cf_gen_hf.shared
            self.cf_gen_lm_head = self.cf_gen_hf.lm_head
            self.cf_gen_encoder = self.cf_gen_hf.encoder
        elif 'bert' in self.cf_gen_arch:
            from transformers import BertForMaskedLM
            bert_with_mlm_head = BertForMaskedLM.from_pretrained(self.cf_gen_arch)
            self.cf_gen_hf = bert_with_mlm_head.bert
            self.cf_gen_lm_head = bert_with_mlm_head.cls
            self.cf_gen_emb_layer = self.cf_gen_hf.embeddings
            self.cf_gen_encoder = self.cf_gen_hf.encoder
        else:
            raise NotImplementedError
        self.cf_gen_hidden_size = self.cf_gen_hf.config.hidden_size
        self.cf_gen_scalar_mix = ScalarMixWithDropout(
            mixture_size=self.cf_gen_hf.config.num_hidden_layers+1,
            dropout=self.cf_dropout,
            do_layer_norm=False,
        )
        # counterfactual predictor module
        if self.cf_pred_arch == 'lstm':
            self.cf_pred_encoder = LSTMEncoder(self.cf_gen_hidden_size, self.cf_gen_hidden_size, bidirectional=True)
            self.cf_pred_hidden_size = self.cf_gen_hidden_size * 2
            self.cf_pred_emb_layer = self.cf_gen_emb_layer
            self.cf_pred_hf = None
            self.cf_pred_scalar_mix = None
        elif self.cf_pred_arch == 'masked_average':
            self.cf_pred_encoder = MaskedAverageEncoder()
            self.cf_pred_hidden_size = self.cf_gen_hidden_size
            self.cf_pred_emb_layer = self.cf_gen_emb_layer
            self.cf_pred_hf = None
            self.cf_pred_scalar_mix = None
        else:
            self.cf_pred_hf = self.cf_gen_hf if self.cf_shared_gen_pred else AutoModel.from_pretrained(self.cf_pred_arch)
            self.cf_pred_hidden_size = self.cf_pred_hf.config.hidden_size
            self.cf_pred_emb_layer = self.cf_pred_hf.embeddings if 't5' not in self.cf_pred_arch else self.cf_pred_hf.shared
            self.cf_pred_encoder = self.cf_pred_hf.encoder
            self.cf_pred_scalar_mix = ScalarMixWithDropout(
                mixture_size=self.cf_pred_hf.config.num_hidden_layers+1,
                dropout=self.cf_dropout,
                do_layer_norm=False,
            )
        # counterfactual predictor output layer
        self.cf_output_layer = nn.Sequential(
            nn.Linear(self.cf_pred_hidden_size, self.cf_pred_hidden_size),
            nn.Tanh(),
            nn.Dropout(self.cf_dropout),
            nn.Linear(self.cf_pred_hidden_size, nb_classes),
            nn.Sigmoid() if not self.is_multilabel else nn.LogSoftmax(dim=-1),
        )

        ########################
        # weights details
        ########################
        # initialize params using xavier initialization for weights and zero for biases
        # (weights of these modules might be loaded later)
        self.init_weights(self.explainer_mlp)
        self.init_weights(self.explainer)
        self.init_weights(self.output_layer)
        self.init_weights(self.cf_output_layer)

        # freeze embedding layers
        if not self.gen_emb_requires_grad:
            freeze_module(self.gen_emb_layer)
        if not self.pred_emb_requires_grad:
            freeze_module(self.pred_emb_layer)
        if not self.cf_gen_emb_requires_grad:
            freeze_module(self.cf_gen_emb_layer)
        if not self.cf_pred_emb_requires_grad:
            freeze_module(self.cf_pred_emb_layer)

        # freeze models (and set to eval mode to disable dropout)
        if not self.gen_encoder_requires_grad:
            freeze_module(self.gen_encoder)
            freeze_module(self.gen_scalar_mix)
        if not self.pred_encoder_requires_grad:
            freeze_module(self.pred_encoder)
            if self.pred_scalar_mix is not None:
                freeze_module(self.pred_scalar_mix)
        if not self.cf_gen_encoder_requires_grad:
            freeze_module(self.cf_gen_encoder)
            freeze_module(self.cf_gen_scalar_mix)
        if not self.cf_gen_lm_head_requires_grad:
            freeze_module(self.cf_gen_lm_head)
        if not self.cf_pred_encoder_requires_grad:
            freeze_module(self.cf_pred_encoder)
            if self.cf_pred_scalar_mix is not None:
                freeze_module(self.cf_pred_scalar_mix)

        # freeze output layers
        if not self.pred_output_requires_grad:
            freeze_module(self.output_layer)
        if not self.cf_pred_output_requires_grad:
            freeze_module(self.cf_output_layer)

        # freeze explainer
        if not self.explainer_requires_grad:
            freeze_module(self.explainer_mlp)
            freeze_module(self.explainer)

        # shared generator and predictor for factual flow
        if self.shared_gen_pred:
            del self.gen_scalar_mix  # unregister generator scalar mix
            self.gen_scalar_mix = self.pred_scalar_mix
        # shared generator and predictor for counterfactual flow
        if self.cf_shared_gen_pred:
            del self.cf_gen_scalar_mix  # unregister generator scalar mix
            self.cf_gen_scalar_mix = self.cf_pred_scalar_mix

        # shared factual and counterfactual generators
        if self.share_generators:
            del self.cf_gen_hf
            del self.cf_gen_emb_layer
            del self.cf_gen_encoder
            del self.cf_gen_hidden_size
            del self.cf_gen_scalar_mix
            self.cf_gen_hf = self.gen_hf
            self.cf_gen_emb_layer = self.gen_emb_layer
            self.cf_gen_encoder = self.gen_encoder
            self.cf_gen_hidden_size = self.gen_hidden_size
            self.cf_gen_scalar_mix = self.gen_scalar_mix

        # shared factual and counterfactual predictors
        if self.share_predictors:
            del self.cf_pred_hf
            del self.cf_pred_emb_layer
            del self.cf_pred_encoder
            del self.cf_pred_hidden_size
            del self.cf_pred_scalar_mix
            del self.cf_output_layer
            self.cf_pred_hf = self.pred_hf
            self.cf_pred_emb_layer = self.pred_emb_layer
            self.cf_pred_encoder = self.pred_encoder
            self.cf_pred_hidden_size = self.pred_hidden_size
            self.cf_pred_scalar_mix = self.pred_scalar_mix
            self.cf_output_layer = self.output_layer

        # counterfactual metrics
        self.cf_train_accuracy = torchmetrics.Accuracy()
        self.cf_val_accuracy = torchmetrics.Accuracy()
        self.cf_test_accuracy = torchmetrics.Accuracy()

    def configure_optimizers(self):
        """Configure optimizers and lr schedulers for Trainer."""
        ff_params = chain(
            self.gen_emb_layer.parameters(),
            self.gen_encoder.parameters(),
            self.gen_scalar_mix.parameters(),
            self.explainer_mlp.parameters(),
            self.explainer.parameters(),
            self.pred_emb_layer.parameters() if not self.shared_gen_pred else [],
            self.pred_encoder.parameters() if not self.shared_gen_pred else [],
            self.pred_scalar_mix.parameters() if not self.shared_gen_pred and self.pred_scalar_mix is not None else [],
            self.output_layer.parameters()
        )
        cf_params = chain(
            self.cf_gen_emb_layer.parameters() if not self.share_generators else [],
            self.cf_gen_encoder.parameters() if not self.share_generators else [],
            self.cf_gen_lm_head.parameters(),
            self.cf_gen_scalar_mix.parameters() if not self.share_generators and self.cf_gen_scalar_mix is not None else [],
            self.cf_pred_emb_layer.parameters() if not self.cf_shared_gen_pred and not self.share_predictors else [],
            self.cf_pred_encoder.parameters() if not self.cf_shared_gen_pred and not self.share_predictors else [],
            self.cf_pred_scalar_mix.parameters() if not self.cf_shared_gen_pred and not self.share_predictors and self.cf_pred_scalar_mix is not None else [],
            self.cf_output_layer.parameters() if not self.share_predictors else []
        )
        grouped_parameters = [
            {"params": ff_params, 'lr': self.hparams['lr']},
            {"params": cf_params, 'lr': self.hparams['cf_lr']}
        ]
        optimizer = build_optimizer(grouped_parameters, self.hparams)
        scheduler = build_scheduler(optimizer, self.hparams)
        output = {"optimizer": optimizer}
        if scheduler is not None:
            output["scheduler"] = scheduler
            output["monitor"] = self.hparams['monitor']  # not sure we need this
        return output

    def forward(
        self,
        x: torch.LongTensor,
        x_cf: torch.LongTensor = None,
        c_cf: list = None,
        mask: torch.BoolTensor = None,
        mask_cf: torch.BoolTensor = None,
        current_epoch=None,
    ):
        """
        Compute forward-pass.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param x_cf: counterfactual input ids tensor. torch.LongTensor of shape [B, T]
        :param c_cf: input counts for counterfactual mapping.
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param mask_cf: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param current_epoch: int represents the current epoch.
        :return: (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """
        # factual flow
        z, y_hat = self.get_factual_flow(x, mask=mask)

        # align factual z with counterfactual ids
        z_cf = repeat_interleave_and_pad(z, c_cf, pad_id=cf_constants.PAD_ID)

        # cut out reminiscent pads (they were repeated at the end of each sequence)
        z_cf = z_cf[:, :x_cf.shape[1]]

        # counterfactual flow
        x_tilde, z_tilde, mask_tilde, y_tilde_hat = self.get_counterfactual_flow(x_cf, z_cf, mask=mask_cf)

        # return everything as output (useful for computing the loss)
        return (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat)

    def get_factual_flow(self, x, mask=None):
        """
        Compute the factual flow.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: z, y_hat
        """
        # create mask for pretrained-LMs
        ext_mask = mask[:, None, None, :]  # add head and seq dimension
        ext_mask = ext_mask.to(dtype=self.dtype)  # fp16 compatibility
        ext_mask = (1.0 - ext_mask) * -10000.0  # will set softmax to zero

        gen_e = self.gen_emb_layer(x)
        if self.use_scalar_mix:
            if 't5' in self.gen_arch:
                gen_h = self.gen_encoder(inputs_embeds=gen_e, attention_mask=mask, output_hidden_states=True)
            else:
                gen_h = self.gen_encoder(gen_e, ext_mask, output_hidden_states=True)
            gen_h = self.gen_scalar_mix(gen_h.hidden_states, mask)
        else:
            # selected_layers = list(map(int, self.selected_layers.split(',')))
            # gen_h = torch.stack(self.gen_encoder(gen_e, ext_mask).hidden_states)
            # gen_h = gen_h[selected_layers].mean(dim=0)
            if 't5' in self.gen_arch:
                gen_h = self.gen_encoder(inputs_embeds=gen_e, attention_mask=mask)
            else:
                gen_h = self.gen_encoder(gen_e, ext_mask)
            gen_h = gen_h.last_hidden_state

        if self.explainer_pre_mlp:
            gen_h = self.explainer_mlp(gen_h)
        z = self.explainer(gen_h, mask)
        z_mask = (z * mask.float()).unsqueeze(-1)

        if self.selection_faithfulness is True:
            pred_e = self.pred_emb_layer(x)
        else:
            pred_e = gen_h

        if self.selection_vector == 'mask':
            # create an input with full mask tokens
            x_mask = torch.ones_like(x) * self.mask_token_id
            pred_e_mask = self.pred_emb_layer(x_mask)
        else:
            pred_e_mask = torch.zeros_like(pred_e)

        if self.selection_space == 'token':
            z_mask_bin = (z_mask > 0).float()
            pred_e = pred_e * z_mask_bin + pred_e_mask * (1 - z_mask_bin)
        elif self.selection_space == 'embedding':
            pred_e = pred_e * z_mask + pred_e_mask * (1 - z_mask)

        if self.selection_mask:
            ext_mask = (1.0 - z_mask.squeeze(-1)[:, None, None, :].to(self.dtype)) * -10000.0

        if self.pred_arch == 'lstm':
            _, summary = self.pred_encoder(pred_e, mask)
        elif self.pred_arch == 'masked_average':
            summary = self.pred_encoder(pred_e, mask)
        else:
            if self.use_scalar_mix:
                if 't5' in self.pred_arch:
                    pred_h = self.pred_encoder(inputs_embeds=pred_e, attention_mask=mask, output_hidden_states=True)
                else:
                    pred_h = self.pred_encoder(pred_e, ext_mask, output_hidden_states=True)
                pred_h = self.pred_scalar_mix(pred_h.hidden_states, mask)
            else:
                if 't5' in self.pred_arch:
                    pred_h = self.pred_encoder(inputs_embeds=pred_e, attention_mask=mask)
                else:
                    pred_h = self.pred_encoder(pred_e, ext_mask)
                pred_h = pred_h.last_hidden_state
            summary = masked_average(pred_h, mask)

        y_hat = self.output_layer(summary)

        return z, y_hat

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
            e, z, mask = make_input_for_t5(e, z, mask)
            # create sentinel tokens
            sentinel_ids = 32100 - (z > 0).long().cumsum(dim=-1)
            # clamp valid input ids (might glue last generations as T5 has only 100 sentinels)
            sentinel_ids = torch.clamp(sentinel_ids, min=32000, max=32099)
            # get sentinel embeddings
            e_mask = self.cf_gen_emb_layer(sentinel_ids)
        else:
            e_mask = self.cf_gen_emb_layer(torch.ones_like(x) * self.cf_mask_token_id)

        # create mask for pretrained-LMs
        ext_mask = (1.0 - mask[:, None, None, :].to(self.dtype)) * -10000.0

        # get the complement of z
        z_mask = (z * mask.float()).unsqueeze(-1)

        # get mask for the generator LM
        # gen_ext_mask = ext_mask * (z_bar.squeeze(-1)[:, None, None, :] > 0.0).float()
        gen_ext_mask = ext_mask

        # differentiable where
        s_bar = e_mask * z_mask + e * (1 - z_mask)

        # pass (1-z)-masked inputs
        if self.cf_use_scalar_mix:
            # it doesnt make a lot of sense to use scalar mix here since
            # the lm head expects the hidden states of the last layer, but
            # we since the lm head can be trained, it can still learn something
            if 't5' in self.cf_gen_arch:
                cf_gen_enc_out = self.cf_gen_encoder(inputs_embeds=s_bar, attention_mask=mask, output_hidden_states=True)
                # no scalar mix for t5
                h_tilde = cf_gen_enc_out.hidden_states
            else:
                cf_gen_enc_out = self.cf_gen_encoder(s_bar, gen_ext_mask, output_hidden_states=True)
                h_tilde = cf_gen_enc_out.hidden_states
                h_tilde = self.cf_gen_scalar_mix(h_tilde, mask)
        else:
            if 't5' in self.cf_gen_arch:
                cf_gen_enc_out = self.cf_gen_encoder(inputs_embeds=s_bar, attention_mask=mask)
            else:
                cf_gen_enc_out = self.cf_gen_encoder(s_bar, gen_ext_mask)
            h_tilde = cf_gen_enc_out.last_hidden_state

        # sample from LM
        if self.cf_use_reinforce:
            if 't5' in self.cf_gen_arch:
                # sample autoregressively
                gen_out = self.cf_gen_hf.generate(
                    encoder_outputs=cf_gen_enc_out,
                    attention_mask=mask.long(),
                    output_scores=True,
                    return_dict_in_generate=True,
                    **self.cf_generate_kwargs
                )
                # idk why but t5 generates a pad symbol as the first token
                # so we cut it out for all samples in the batch
                # (this happens only for .sequences)
                x_tilde = gen_out.sequences[:, 1:]
                logits = torch.stack(gen_out.scores).transpose(0, 1)
            else:
                # sample directly from the output layer
                logits = self.cf_gen_lm_head(h_tilde)
                x_tilde = logits.argmax(dim=-1)

                # get gen_ids only for <mask> positions
                z_1 = (z_mask > 0).squeeze(-1).long()
                x_tilde = z_1 * x_tilde + (1 - z_1) * x

        else:
            # use the ST-gumbel-softmax trick
            logits = self.cf_gen_lm_head(h_tilde)
            x_tilde = nn.functional.gumbel_softmax(logits, hard=True, dim=-1)

            # get gen_ids only for <mask> positions
            z_1 = (z_mask > 0).long()
            x_one_hot = nn.functional.one_hot(x, num_classes=x_tilde.shape[-1])
            x_tilde = z_1 * x_tilde + (1 - z_1) * x_one_hot

        # save log probas for later
        self.log_prob_x_tilde = torch.log_softmax(logits, dim=-1)

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

            # if we generated too much, there isn't much we can do besides truncating
            if expanded_seq_len > 512 and 'bert' in self.cf_pred_arch:
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

        # mask for hf-transformers
        ext_mask_tilde = (1.0 - mask_tilde[:, None, None, :].to(self.dtype)) * -10000.0

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

        # get the replacement vector for non-selected positions
        pred_e_mask = torch.zeros_like(pred_e)
        if self.cf_selection_vector == 'mask':
            if 't5' in self.cf_gen_arch:
                sentinel_ids = torch.zeros_like(x_tilde).long() + 32000
                pred_e_mask = self.cf_pred_emb_layer(sentinel_ids)
            else:
                pred_e_mask = self.cf_pred_emb_layer(torch.ones_like(x_tilde).long() * self.cf_mask_token_id)

        # input selection
        if self.cf_selection_mode == 'select':
            # select inputs by z_tilde
            z_mask_tilde = (z_tilde * mask_tilde.float()).unsqueeze(-1)
            s_tilde = pred_e * z_mask_tilde + pred_e_mask * (1 - z_mask_tilde)
        else:
            # pass inputs directly
            z_mask_tilde = mask_tilde
            s_tilde = pred_e

        # whether we want to mask out non-selected elements during self-attention
        if self.cf_selection_mask:
            ext_mask_tilde = (1.0 - z_mask_tilde.squeeze(-1)[:, None, None, :].to(self.dtype)) * -10000.0

        # get hidden states
        if self.cf_pred_arch == 'lstm':
            _, summary = self.cf_pred_encoder(s_tilde, mask_tilde)
        elif self.cf_pred_arch == 'masked_average':
            summary = self.cf_pred_encoder(s_tilde, mask_tilde)
        else:
            # pass the selected inputs through the encoder
            if self.cf_use_scalar_mix:
                if 't5' in self.cf_pred_arch:
                    pred_h = self.cf_pred_encoder(inputs_embeds=s_tilde, attention_mask=mask_tilde, output_hidden_states=True)
                else:
                    pred_h = self.cf_pred_encoder(s_tilde, ext_mask_tilde, output_hidden_states=True)
                pred_h = self.cf_pred_scalar_mix(pred_h.hidden_states, mask_tilde)
            else:
                if 't5' in self.cf_pred_arch:
                    pred_h = self.cf_pred_encoder(inputs_embeds=s_tilde, attention_mask=mask_tilde)
                else:
                    pred_h = self.cf_pred_encoder(s_tilde, ext_mask_tilde)
                pred_h = pred_h.last_hidden_state
            # get predictions
            summary = masked_average(pred_h, mask_tilde)

        # cf predictor output
        y_tilde_hat = self.cf_output_layer(summary)

        self.z_tilde = z_tilde
        return x_tilde, z_tilde, mask_tilde, y_tilde_hat

    def get_factual_loss(self, y_hat, y, mask, prefix):
        """
        Compute loss for the factual flow.

        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param prefix: prefix for loss statistics (train, val, test)
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        stats = {}
        loss_vec = self.criterion(y_hat, y)  # [B] or [B,C]
        # main MSE loss for p(y | x,z)
        if not self.is_multilabel:
            loss = loss_vec.mean(0)  # [B,C] -> [B]
            stats["mse"] = loss.item()
        else:
            loss = loss_vec.mean()  # [1]
            stats["criterion"] = loss.item()  # [1]

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(self.explainer.z, mask)
        stats[prefix + "_p0"] = num_0 / float(total)
        stats[prefix + "_pc"] = num_c / float(total)
        stats[prefix + "_p1"] = num_1 / float(total)
        stats[prefix + "_ps"] = (num_c + num_1) / float(total)
        stats[prefix + "_main_loss"] = loss.item()

        return loss, stats

    def get_counterfactual_loss(self, y_hat, y, z_tilde, mask_tilde, prefix):
        """
        Compute loss for the conterfactual flow.

        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param z_tilde: latent selection vector. torch.FloatTensor of shape [B, T]
        :param mask_tilde: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param prefix: prefix for loss statistics (train, val, test)
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        stats = {}
        loss_vec = self.criterion(y_hat, y)  # [B] or [B,C]

        # x = h
        # y = y_tilde
        # z = x_tilde

        # main MSE loss for p(y | x, z)
        if not self.is_multilabel:
            loss_vec = loss_vec.mean(1)  # [B,C] -> [B]

        loss = loss_vec.mean()  # [1]
        main_loss = loss

        if not self.is_multilabel:
            stats["cf_mse"] = loss.item()  # [1]
        else:
            stats["cf_criterion"] = loss.item()  # [1]

        if self.cf_use_reinforce:

            # for BERT:
            # the <mask> is on <mask> table
            # the book   is on the    table
            # p1  p2     p3 p4 p5     p6
            # 0   1      0  0  1      0
            # p * z

            # for T5:
            # the <extra_id_0> is on    <extra_id_1>  table
            # the long book    is under the very nice table
            # p1  p2   p3      p4 ...
            # 0   1    0       0 0    1    0
            # 0   1    1       0 0    1 1 1 0
            # p * 1

            # opt 1: independent samples
            # P(gen_id | s)
            # p = softmax(logits_output)

            # opt 2: autoregressive samples
            # P(gen_id_i | gen_id_{0:i-1}, s)
            # p = cumprod(p)

            # log P(x_tilde | s; phi)
            if 't5' in self.cf_gen_arch:
                # compute log prob of all sampled tokens (excluding sentinels)
                logp_xtilde = self.log_prob_x_tilde  # [B, T, |V|]
                gen_ids = logp_xtilde.argmax(dim=-1)
                gen_mask = ~((gen_ids >= 32000) & (gen_ids <= 32099))  # non-sentinel
                gen_mask &= (gen_ids != cf_constants.PAD_ID)  # non-padding
                logp_xtilde = logp_xtilde.max(dim=-1)[0]  # get the probas of the argmax only
                logp_xtilde = logp_xtilde.masked_fill(~gen_mask, 0)
                log_xtilde_scalar = (logp_xtilde * gen_mask.float()).sum(1) / gen_mask.float().sum(1)

            else:
                # compute only the log prob of selected tokens
                logp_xtilde = self.log_prob_x_tilde  # [B, T, |V|]
                logp_xtilde = logp_xtilde.max(dim=-1)[0]  # get the probas of the argmax only
                logp_xtilde = torch.where(z_tilde == 0, 0, logp_xtilde)
                logp_xtilde = logp_xtilde.masked_fill(~mask_tilde, 0)
                log_xtilde_scalar = (logp_xtilde * mask_tilde.float()).sum(1) / mask_tilde.float().sum(1)

            # compute generator loss
            cost_vec = loss_vec.detach()
            # cost_vec is neg reward
            cost_logpz = ((cost_vec - self.mean_baseline) * log_xtilde_scalar).mean(0)

            # MSE with regularizers = neg reward
            obj = cost_vec.mean()
            stats["cf_obj"] = obj.item()

            # add baseline
            if self.cf_use_baseline:
                self.n_points += 1.0
                self.mean_baseline += (cost_vec.detach().mean() - self.mean_baseline) / self.n_points

            # pred diff doesn't do anything if only 1 aspect being trained
            if not self.is_multilabel:
                pred_diff = y_hat.max(dim=1)[0] - y_hat.min(dim=1)[0]
                pred_diff = pred_diff.mean()
                stats["cf_pred_diff"] = pred_diff.item()

            # generator cost
            stats["cf_cost_g"] = cost_logpz.item()

            # predictor cost
            stats["cf_cost_p"] = loss.item()

            main_loss = loss + cost_logpz

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(z_tilde, mask_tilde)
        stats[prefix + "_cf_p0"] = num_0 / float(total)
        stats[prefix + "_cf_pc"] = num_c / float(total)
        stats[prefix + "_cf_p1"] = num_1 / float(total)
        stats[prefix + "_cf_ps"] = (num_c + num_1) / float(total)
        stats[prefix + "_cf_main_loss"] = main_loss.item()

        return main_loss, stats

    def training_step(self, batch: dict, batch_idx: int):
        """
        Compute forward-pass, calculate loss and log metrics.

        :param batch: The dict output from the data module with the following items:
            `input_ids`: torch.LongTensor of shape [B, T],
            `lengths`: torch.LongTensor of shape [B]
            `labels`: torch.LongTensor of shape [B, C]
            `tokens`: list of strings
        :param batch_idx: integer displaying index of this batch
        :return: pytorch_lightning.Result log object
        """
        input_ids = batch["input_ids"]
        cf_input_ids = batch["cf_input_ids"]
        cf_counts = batch["cf_counts"]
        labels = batch["labels"]
        mask = input_ids != constants.PAD_ID
        mask_cf = cf_input_ids != cf_constants.PAD_ID
        prefix = "train"

        (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat) = self(
            input_ids,
            cf_input_ids,
            cf_counts,
            mask=mask,
            mask_cf=mask_cf,
            current_epoch=self.current_epoch
        )

        # compute loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        ff_loss, loss_stats = self.get_factual_loss(y_hat, y, mask, prefix=prefix)

        y_tilde_hat = y_tilde_hat if not self.is_multilabel else y_tilde_hat.view(-1, self.nb_classes)
        y_cf = 1 - y  # todo: write generic function
        cf_loss, cf_loss_stats = self.get_counterfactual_loss(y_tilde_hat, y_cf, z_tilde, mask_tilde, prefix=prefix)

        loss = ff_loss + self.cf_lbda * cf_loss

        # logger=False because they are going to be logged via loss_stats
        self.log("train_ff_ps", loss_stats["train_ps"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_cf_ps", cf_loss_stats["train_cf_ps"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_ff_sum_loss", ff_loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)
        self.log("train_cf_sum_loss", cf_loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)
        self.log("train_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        if self.is_multilabel:
            metrics_to_wandb = {
                "train_ff_p1": loss_stats["train_p1"],
                "train_cf_p1": cf_loss_stats["train_cf_p1"],
                "train_ff_ps": loss_stats["train_ps"],
                "train_cf_ps": cf_loss_stats["train_cf_ps"],
                "train_ff_sum_loss": loss_stats["criterion"],
                "train_cf_sum_loss": cf_loss_stats["cf_criterion"],
            }
        else:
            metrics_to_wandb = {
                "train_ff_p1": loss_stats["train_p1"],
                "train_cf_p1": cf_loss_stats["train_cf_p1"],
                "train_ff_ps": loss_stats["train_ps"],
                "train_cf_ps": cf_loss_stats["train_cf_ps"],
                "train_ff_sum_loss": loss_stats["mse"],
                "train_cf_sum_loss": cf_loss_stats["cf_mse"],
            }
        if "cost_g" in loss_stats:
            metrics_to_wandb["train_cost_g"] = loss_stats["cost_g"]
        if "cost_g" in cf_loss_stats:
            metrics_to_wandb["train_cf_cost_g"] = cf_loss_stats["cf_cost_g"]

        self.logger.log_metrics(metrics_to_wandb, self.global_step)

        # return the loss tensor to PTL
        return {"loss": loss, "ps": loss_stats["train_ps"], "cf_ps": cf_loss_stats["train_cf_ps"]}

    def _shared_eval_step(self, batch: dict, batch_idx: int, prefix: str):
        input_ids = batch["input_ids"]
        cf_input_ids = batch["cf_input_ids"]
        cf_counts = batch["cf_counts"]
        labels = batch["labels"]
        mask = input_ids != constants.PAD_ID
        mask_cf = cf_input_ids != cf_constants.PAD_ID

        # forward-pass
        (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat) = self(
            input_ids,
            cf_input_ids,
            cf_counts,
            mask=mask,
            mask_cf=mask_cf,
            current_epoch=self.current_epoch
        )

        # compute loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        ff_loss, ff_loss_stats = self.get_factual_loss(y_hat, y, prefix=prefix, mask=mask)
        self.logger.agg_and_log_metrics(ff_loss_stats, step=None)

        y_tilde_hat = y_tilde_hat if not self.is_multilabel else y_tilde_hat.view(-1, self.nb_classes)
        y_cf = 1 - y  # todo: write generic function
        cf_loss, cf_loss_stats = self.get_counterfactual_loss(y_tilde_hat, y_cf, z_tilde, mask_tilde, prefix=prefix)
        self.logger.agg_and_log_metrics(cf_loss_stats, step=None)

        loss = ff_loss + self.cf_lbda * cf_loss

        self.log(f"{prefix}_ff_sum_loss", ff_loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True,)
        self.log(f"{prefix}_cf_sum_loss", cf_loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True,)
        self.log(f"{prefix}_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        z_1 = (z > 0).long()  # non-zero probs are considered selections for sparsemap
        ids_rationales, rationales = get_rationales(self.tokenizer, input_ids, z_1, batch["lengths"])
        pieces = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in input_ids.tolist()]

        if 't5' in self.cf_gen_arch:
            gen_ids = x_tilde
        else:
            z_1 = (z_tilde > 0).long()
            gen_ids = x_tilde if self.cf_use_reinforce else x_tilde.argmax(dim=-1)
            gen_ids = z_1 * gen_ids + (1 - z_1) * input_ids
        cfs = [self.cf_tokenizer.convert_ids_to_tokens(idxs) for idxs in gen_ids.tolist()]

        # output to be stacked across iterations
        output = {
            f"{prefix}_sum_loss": loss.item(),
            f"{prefix}_ps": ff_loss_stats[prefix + "_ps"],
            f"{prefix}_ids_rationales": ids_rationales,
            f"{prefix}_rationales": rationales,
            f"{prefix}_pieces": pieces,
            f"{prefix}_tokens": batch["tokens"],
            f"{prefix}_z": z,
            f"{prefix}_predictions": y_hat,
            f"{prefix}_labels": y.tolist(),
            f"{prefix}_lengths": batch["lengths"].tolist(),
            f"{prefix}_cfs": cfs,
            f"{prefix}_cf_labels": y_cf.tolist(),
            f"{prefix}_cf_predictions": y_tilde_hat,
            f"{prefix}_cf_z": z_tilde,
            f"{prefix}_cf_lengths": batch["cf_lengths"].tolist(),
        }

        if "annotations" in batch.keys():
            output[f"{prefix}_annotations"] = batch["annotations"]

        if "mse" in ff_loss_stats.keys():
            output[f"{prefix}_mse"] = ff_loss_stats["mse"]

        return output

    def _shared_eval_epoch_end(self, outputs: list, prefix: str):
        """
        PTL hook. Perform validation at the end of an epoch.

        :param outputs: list of dicts representing the stacked outputs from validation_step
        :param prefix: `val` or `test`
        """
        # assume that `outputs` is a list containing dicts with the same keys
        stacked_outputs = {k: [x[k] for x in outputs] for k in outputs[0].keys()}

        # average across batches
        avg_outputs = {
            f"avg_{prefix}_sum_loss": np.mean(stacked_outputs[f"{prefix}_sum_loss"]),
            f"avg_{prefix}_ps": np.mean(stacked_outputs[f"{prefix}_ps"]),
        }

        shell_logger.info(
            f"Avg {prefix} sum loss: {avg_outputs[f'avg_{prefix}_sum_loss']:.4}"
        )
        shell_logger.info(f"Avg {prefix} ps: {avg_outputs[f'avg_{prefix}_ps']:.4}")

        dict_metrics = {
            f"avg_{prefix}_ps": avg_outputs[f"avg_{prefix}_ps"],
            f"avg_{prefix}_sum_loss": avg_outputs[f"avg_{prefix}_sum_loss"],
        }

        # log rationales
        from random import shuffle
        idxs = list(range(sum(map(len, stacked_outputs[f"{prefix}_pieces"]))))
        if prefix != 'test':
            shuffle(idxs)
            idxs = idxs[:10]
        else:
            idxs = idxs[:100]
        select = lambda v: [v[i] for i in idxs]
        detach = lambda v: [v[i].detach().cpu() for i in range(len(v))]
        pieces = select(unroll(stacked_outputs[f"{prefix}_pieces"]))
        scores = detach(select(unroll(stacked_outputs[f"{prefix}_z"])))
        gold = select(unroll(stacked_outputs[f"{prefix}_labels"]))
        pred = detach(select(unroll(stacked_outputs[f"{prefix}_predictions"])))
        lens = select(unroll(stacked_outputs[f"{prefix}_lengths"]))
        html_string = get_html_rationales(pieces, scores, gold, pred, lens)
        self.logger.experiment.log({f"{prefix}_rationales": wandb.Html(html_string)})

        # save rationales
        if self.hparams.save_rationales:
            scores = detach(unroll(stacked_outputs[f"{prefix}_z"]))
            lens = unroll(stacked_outputs[f"{prefix}_lengths"])
            filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_rationales.txt')
            shell_logger.info(f'Saving rationales in {filename}...')
            save_rationales(filename, scores, lens)

        # log counterfactuals
        cfs = select(unroll(stacked_outputs[f"{prefix}_cfs"]))
        scores = detach(select(unroll(stacked_outputs[f"{prefix}_cf_z"])))
        gold = select(unroll(stacked_outputs[f"{prefix}_cf_labels"]))
        pred = detach(select(unroll(stacked_outputs[f"{prefix}_cf_predictions"])))
        lens = select(unroll(stacked_outputs[f"{prefix}_cf_lengths"]))
        html_string = get_html_rationales(cfs, scores, gold, pred, lens)
        self.logger.experiment.log({f"{prefix}_counterfactuals": wandb.Html(html_string)})

        # save rationales
        if self.hparams.save_counterfactuals:
            pieces = unroll(stacked_outputs[f"{prefix}_cfs"])
            lens = unroll(stacked_outputs[f"{prefix}_cf_lengths"])
            filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_counterfactuals.txt')
            shell_logger.info(f'Saving counterfactuals in {filename}...')
            save_counterfactuals(filename, pieces, lens)

        # only evaluate rationales on the test set and if we have annotation (only for beer dataset)
        if prefix == "test" and "test_annotations" in stacked_outputs.keys():
            rat_metrics = evaluate_rationale(
                stacked_outputs["test_ids_rationales"],
                stacked_outputs["test_annotations"],
                stacked_outputs["test_lengths"],
            )
            shell_logger.info(
                f"Rationales macro precision: {rat_metrics[f'macro_precision']:.4}"
            )
            shell_logger.info(
                f"Rationales macro recall: {rat_metrics[f'macro_recall']:.4}"
            )
            shell_logger.info(f"Rationales macro f1: {rat_metrics[f'f1_score']:.4}")

        # log classification metrics
        if self.is_multilabel:
            preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_predictions"]), dim=-1)
            labels = torch.tensor(unroll(stacked_outputs[f"{prefix}_labels"]), device=preds.device)
            cf_preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_cf_predictions"]), dim=-1)
            cf_labels = torch.tensor(unroll(stacked_outputs[f"{prefix}_cf_labels"]), device=cf_preds.device)
            if prefix == "val":
                accuracy = self.val_accuracy(preds, labels)
                cf_accuracy = self.cf_val_accuracy(cf_preds, cf_labels)
                precision = self.val_precision(preds, labels)
                recall = self.val_recall(preds, labels)
                f1_score = 2 * precision * recall / (precision + recall)
            else:
                accuracy = self.test_accuracy(preds, labels)
                cf_accuracy = self.cf_test_accuracy(cf_preds, cf_labels)
                precision = self.test_precision(preds, labels)
                recall = self.test_recall(preds, labels)
                f1_score = 2 * precision * recall / (precision + recall)

            dict_metrics[f"{prefix}_precision"] = precision
            dict_metrics[f"{prefix}_recall"] = recall
            dict_metrics[f"{prefix}_f1score"] = f1_score
            dict_metrics[f"{prefix}_accuracy"] = accuracy
            dict_metrics[f"{prefix}_cf_accuracy"] = cf_accuracy

            shell_logger.info(f"{prefix} accuracy: {accuracy:.4}")
            shell_logger.info(f"{prefix} cf_accuracy: {cf_accuracy:.4}")
            shell_logger.info(f"{prefix} precision: {precision:.4}")
            shell_logger.info(f"{prefix} recall: {recall:.4}")
            shell_logger.info(f"{prefix} f1: {f1_score:.4}")

            self.log(
                f"{prefix}_f1score",
                dict_metrics[f"{prefix}_f1score"],
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

        else:
            avg_outputs[f"avg_{prefix}_mse"] = np.mean(stacked_outputs[f"{prefix}_mse"])
            shell_logger.info(
                f"Avg {prefix} MSE: {avg_outputs[f'avg_{prefix}_mse']:.4}"
            )
            dict_metrics[f"avg_{prefix}_mse"] = avg_outputs[f"avg_{prefix}_mse"]

            self.log(
                f"{prefix}_MSE",
                dict_metrics[f"avg_{prefix}_mse"],
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

        self.logger.agg_and_log_metrics(dict_metrics, self.current_epoch)

        self.log(
            f"avg_{prefix}_sum_loss",
            dict_metrics[f"avg_{prefix}_sum_loss"],
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        if self.is_multilabel:
            output = {
                f"avg_{prefix}_sum_loss": dict_metrics[f"avg_{prefix}_sum_loss"],
                f"avg_{prefix}_ps": dict_metrics[f"avg_{prefix}_ps"],
                f"{prefix}_precision": precision,
                f"{prefix}_recall": recall,
                f"{prefix}_f1score": f1_score,
                f"{prefix}_accuracy": accuracy,
                f"{prefix}_cf_accuracy": cf_accuracy,
            }
        else:
            output = {
                f"avg_{prefix}_sum_loss": dict_metrics[f"avg_{prefix}_sum_loss"],
                f"avg_{prefix}_ps": dict_metrics[f"avg_{prefix}_ps"],
                f"avg_{prefix}_MSE": dict_metrics[f"avg_{prefix}_mse"],
            }

        return output


def make_input_for_t5(e, z, mask):
    """
    Replace masked positions by sentinel tokens

    :param e: input sequence, torch.FloatTensor [B, T, D]
    :param z: latent selection vector torch.FloarTensor [B, T]
    :param mask: mask for padding positions, torch.BoolTensor [B, T]
    :return: t5_input: new input ids for T5, torch.FloatTensor [B, T', D],
             t5_z: new latent selection vector, torch.FloatTensor [B, T']
             t5_mask: new mask for padding positions, torch.BoolTensor [B, T']
    """
    bs, seq_len, hdim = e.shape

    # for example:
    # z = [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]

    # leave only the first non-zero element in each contiguous chunk
    z_first = torch.relu(z - torch.cat((z.new_zeros(bs, 1), z[:, :-1]), dim=-1))
    # z_first = [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    # create a matrix of indices of shape (bs, seq_len)
    ar = torch.arange(seq_len, device=z.device).unsqueeze(0).expand(bs, -1).long()

    # select all tokens but the 2nd-n in each chunk
    z_all_but_succ = (1 - (z > 0).long() * (1 - (z_first > 0).long()))

    # get the ids of these tokens (and set 2nd-n tokens in each chunk to 999999)
    idxs_all_but_succ = z_all_but_succ * (ar + 1) + (1 - z_all_but_succ) * 999999

    # sort ids so that 999999 are at the end
    # (there are better ways to do this but sort is not very slow after all)
    idxs_all_but_succ = torch.sort(idxs_all_but_succ, stable=True, dim=-1).values

    # get the mask pointing to valid tokens
    z_mask = idxs_all_but_succ < 999999

    # discount 1 to get 0-index ids
    idxs_all_but_succ = idxs_all_but_succ - 1

    # for using gather, we need to replace these tokens by some valid index
    # so we use seq_len - 1 here, which will lead to pad tokens anyway
    idxs_all_but_succ = idxs_all_but_succ.masked_fill(~z_mask, seq_len - 1)

    # expand as `e` so we can gather from it
    idxs_all_but_succ = idxs_all_but_succ.unsqueeze(-1).expand(-1, -1, hdim)

    # gather vectors from `e`
    t5_input = e.gather(1, idxs_all_but_succ)

    # the new mask is points to non padding tokens + eliminated tokens
    t5_mask = idxs_all_but_succ < mask.sum(-1).unsqueeze(-1)

    # the new mask is simply z_first (the places where sentinels were inserted)
    t5_z = z_first * t5_mask.float()

    return t5_input, t5_z, t5_mask


def get_new_frequencies_of_gen_ids_from_t5(x_ids, g_ids, pad_id=0, eos_id=1, idx_a=32000, idx_b=32099):
    """
    Get the number of tokens generated by T5 for each position of the input.

    :param x_ids: original input ids, torch.LongTensor of shape [B, T]
    :param g_ids: generated ids, torch.LongTensor of shape [B, T]
    :param pad_id: id of padding token
    :param eos_id: id of end-of-sequence token
    :param idx_a: id of first token of the new frequency range
    :param idx_b: id of last token of the new frequency range
    :return: new_freqs, torch.LongTensor of shape [B, T]
    """
    new_freqs = []
    for i in range(x_ids.shape[0]):
        # for example:
        # x = ['▁UN', '▁Chief', '▁says', '▁there', '▁is', '▁no', '▁way', '▁to', '<extra_id_0>', '▁in', '▁Syria', '</s>']
        # g = ['<extra_id_0>', '▁do', '▁so', '<extra_id_1>', '▁do', '▁so', '.', '</s>', '<pad>', '<pad>']

        # recover z from x_ids
        z_x = (x_ids[i] >= idx_a) & (x_ids[i] <= idx_b)          # select sentinel tokens
        z_x = z_x & (x_ids[i] != pad_id) & (x_ids[i] != eos_id)  # but ignore <pad> and </s>
        # for example:
        # z_x = tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

        # recover the complement of z from g_ids
        z_y = ~((g_ids[i] >= idx_a) & (g_ids[i] <= idx_b))       # select non sentinel tokens
        z_y = z_y & (g_ids[i] != pad_id) & (g_ids[i] != eos_id)  # but ignore <pad> and </s>
        # for example:
        # z_y = tensor([0, 1, 1, 0, 1, 1, 1, 0, 0, 0])

        # count the number of consecutive non-sentinel tokens
        outputs, counts = torch.unique_consecutive(z_y, return_counts=True)
        # for example:
        # outputs = tensor([0, 1, 0, 1, 0])
        # counts = tensor([1, 2, 1, 3, 3])

        # mask out invalid outputs due to end-of-sequence generation
        m = torch.arange(len(outputs), device=z_x.device) < (z_x.sum() * 2)
        # for example:
        # m = tensor([1, 1, 0, 0, 0])

        # count only the valid outputs
        outputs = outputs.masked_fill(~m, 0)
        counts = counts.masked_fill(~m, 1)
        # for example:
        # outputs = tensor([0, 1, 0, 0, 0])
        # counts = tensor([1, 2, 1, 1, 1])

        # convert counts to repeat_interleave frequencies
        n_x = 1 - z_x.clone().long()
        # for example:
        # n_x = tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1])

        n_x[n_x == 0] = counts[outputs]
        # n_x = tensor([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1])

        # now we can repeat according to this new count, so we save them
        new_freqs.append(n_x)
    return new_freqs


def repeat_interleave_and_pad(x, counts, pad_id=0):
    """
    batch-wise repeat_interleave x according to counts,
    and then pad reminiscent positions with pad_id

    :param x: tensor with shape (batch_size, seq_len)
    :param counts: list of tensors with shape (seq_len,)
    :param pad_id: padding value
    """
    if counts[0] is None:
        return x
    seq_len = x.shape[-1]
    # if any(len(c) != seq_len for c in counts):
    #     print(list(map(len, counts)))  # why?
    x_new = [x[i].repeat_interleave(counts[i][:seq_len], dim=-1) for i in range(len(counts))]
    x_new = torch.nn.utils.rnn.pad_sequence(x_new, batch_first=True, padding_value=pad_id)
    return x_new.to(x.device)


def merge_input_and_gen_ids(input_ids_rep, generated_ids_rep, pad_id=0, idx_a=32000, idx_b=32099):
    """
    Merge the input ids and generated ids into one tensor.

    :param input_ids_rep: input ids after repeated_interleave, tensor of shape [B, T1]
    :param generated_ids_rep: generated ids after repeated_interleave, tensor of shape [B, T2]
    :param pad_id: id of padding token
    :param idx_a: id of first token of the new frequency range
    :param idx_b: id of last token of the new frequency range
    """
    # recover z from input ids and generated ids
    mask_inp = ~((input_ids_rep >= idx_a) & (input_ids_rep <= idx_b))
    mask_gen = ~((generated_ids_rep >= idx_a) & (generated_ids_rep <= idx_b))

    # get the length of the repeated tensors
    len_inp = input_ids_rep.shape[1]
    len_gen = generated_ids_rep.shape[1]

    # if we have less tokens in the original input, we truncate the generated ones
    if len_inp <= len_gen:
        m = mask_inp[:, :len_inp].long()
        merged_ids_rep = m * input_ids_rep[:, :len_inp] + (1 - m) * generated_ids_rep[:, :len_inp]

    # otherwise, we truncate the input ones
    else:
        m = mask_gen[:, :len_gen].long()
        merged_ids_rep = m * generated_ids_rep[:, :len_gen] + (1 - m) * input_ids_rep[:, :len_gen]

    # cap the new tensor to the new length since the merged_ids
    # can have more pad tokens than the necessary
    new_max_len = (merged_ids_rep != pad_id).sum(1).max().item()
    merged_ids_rep = merged_ids_rep[:, :new_max_len]

    return merged_ids_rep
