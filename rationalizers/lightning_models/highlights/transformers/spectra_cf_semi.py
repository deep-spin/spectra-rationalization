import logging
import os
from copy import deepcopy
from itertools import chain
from random import shuffle

import numpy as np
import torch
import torchmetrics
import wandb
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, RobertaForMaskedLM, BertForMaskedLM
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right

from rationalizers import constants
from rationalizers.builders import build_optimizer, build_scheduler
from rationalizers.explainers import available_explainers
from rationalizers.lightning_models.highlights.transformers.base import TransformerBaseRationalizer
from rationalizers.lightning_models.utils import (
    prepend_label_for_mice_t5, make_input_for_t5,
    get_new_frequencies_of_gen_ids_from_t5, repeat_interleave_and_pad,
    merge_input_and_gen_ids, sample_from_logits, get_contrast_label, prepend_label_for_t5, make_input_for_ct5,
    fix_t5_generated_inputs, merge_inputs_for_t5, is_sentinel
)
from rationalizers.modules.metrics import evaluate_rationale
from rationalizers.modules.sentence_encoders import LSTMEncoder, MaskedAverageEncoder
from rationalizers.utils import (
    get_z_stats, freeze_module, masked_average, get_rationales, unroll, get_html_rationales,
    save_rationales, save_counterfactuals, load_torch_object, is_trainable, get_ext_mask
)

shell_logger = logging.getLogger(__name__)


class SemiSupervisedCfTransformerSPECTRARationalizer(TransformerBaseRationalizer):

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
        super(TransformerBaseRationalizer, self).__init__(tokenizer, nb_classes, is_multilabel, h_params)

        # manual update constants
        constants.update_constants(self.tokenizer)
        self.has_countertfactual_flow = True

        ########################
        # hyperparams
        ########################
        # factual:
        self.ff_gen_arch = h_params.get("gen_arch", "bert-base-multilingual-cased")
        self.ff_gen_emb_requires_grad = h_params.get("gen_emb_requires_grad", False)
        self.ff_gen_encoder_requires_grad = h_params.get("gen_encoder_requires_grad", True)
        self.ff_gen_use_decoder = h_params.get("gen_use_decoder", False)
        self.ff_pred_arch = h_params.get("pred_arch", "bert-base-multilingual-cased")
        self.ff_pred_emb_requires_grad = h_params.get("pred_emb_requires_grad", False)
        self.ff_pred_encoder_requires_grad = h_params.get("pred_encoder_requires_grad", True)
        self.ff_pred_output_requires_grad = h_params.get("pred_output_requires_grad", True)
        self.ff_shared_gen_pred = h_params.get("shared_gen_pred", False)
        self.ff_dropout = h_params.get("dropout", 0.1)
        self.ff_selection_vector = h_params.get("selection_vector", 'zero')
        self.ff_selection_mask = h_params.get("selection_mask", True)
        self.ff_selection_faithfulness = h_params.get("selection_faithfulness", True)
        self.ff_lbda = h_params.get('ff_lbda', 1.0)

        # counterfactual:
        self.cf_gen_arch = h_params.get("cf_gen_arch", "bert-base-multilingual-cased")
        self.cf_gen_emb_requires_grad = h_params.get("cf_gen_emb_requires_grad", False)
        self.cf_gen_encoder_requires_grad = h_params.get("cf_gen_encoder_requires_grad", True)
        self.cf_gen_lm_head_requires_grad = h_params.get("cf_gen_lm_head_requires_grad", True)
        self.cf_gen_use_decoder = h_params.get("cf_gen_use_decoder", True)
        self.cf_dropout = h_params.get("cf_dropout", 0.1)
        self.cf_input_space = h_params.get("cf_input_space", 'embedding')
        self.cf_selection_vector = h_params.get("cf_selection_vector", 'zero')
        self.cf_selection_mask = h_params.get("cf_selection_mask", True)
        self.cf_use_reinforce = h_params.get('cf_use_reinforce', True)
        self.cf_use_reinforce_baseline = h_params.get('cf_use_reinforce_baseline', True)
        self.cf_lbda = h_params.get('cf_lbda', 1.0)
        self.cf_generate_kwargs = h_params.get('cf_generate_kwargs', dict())

        # self.cf_manual_sample = h_params.get("cf_manual_sample", True)
        self.cf_prepend_label_type = h_params.get("cf_prepend_label_type", "gold")
        self.cf_z_type = h_params.get("cf_z_type", "gold")
        self.cf_task_name = h_params.get("cf_task_name", "binary_classification")
        self.cf_explainer_mask_token_type_id = h_params.get("cf_explainer_mask_token_type_id", None)
        self.cf_gen_ckpt = h_params.get("cf_gen_ckpt", None)
        self.cf_lbda_gen = h_params.get('cf_lbda_gen', 0.01)

        self.cf_x_tilde = None
        self.cf_log_prob_x_tilde = None
        self.cf_manual_sample = False
        self.generation_mode = False

        # explainer:
        self.explainer_pre_mlp = h_params.get("explainer_pre_mlp", True)
        self.explainer_requires_grad = h_params.get("explainer_requires_grad", True)
        self.explainer_mask_token_type_id = h_params.get("explainer_mask_token_type_id", None)
        self.temperature = h_params.get("temperature", 1.0)

        # both
        self.share_generators = h_params.get("share_generators", False)

        ########################
        # useful vars
        ########################
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id
        if 'mt5' in self.cf_gen_arch:
            self.sentinel_a, self.sentinel_b = 250000, 250099
        else:
            self.sentinel_a, self.sentinel_b = 32000, 32099

        ########################
        # build factual flow
        ########################
        # generator module
        self.ff_gen_hf = AutoModel.from_pretrained(self.ff_gen_arch)
        self.ff_gen_emb_layer = self.ff_gen_hf.shared if 't5' in self.ff_gen_arch else self.ff_gen_hf.embeddings
        self.ff_gen_encoder = self.ff_gen_hf.encoder
        self.ff_gen_decoder = self.ff_gen_hf.decoder if hasattr(self.ff_gen_hf, 'decoder') else None
        self.ff_gen_hidden_size = self.ff_gen_hf.config.hidden_size

        # explainer
        explainer_cls = available_explainers['sparsemap']
        self.explainer = explainer_cls(h_params, self.ff_gen_hidden_size)
        self.explainer_mlp = nn.Sequential(
            nn.Linear(self.ff_gen_hidden_size, self.ff_gen_hidden_size),
            nn.Tanh(),
        )

        # predictor module
        if self.ff_pred_arch == 'lstm':
            self.ff_pred_encoder = LSTMEncoder(self.ff_gen_hidden_size, self.ff_gen_hidden_size, bidirectional=True)
            self.ff_pred_hidden_size = self.ff_gen_hidden_size * 2
            self.ff_pred_emb_layer = self.ff_gen_emb_layer
            self.ff_pred_hf = None
            self.ff_pred_decoder = None
        elif self.ff_pred_arch == 'masked_average':
            self.ff_pred_encoder = MaskedAverageEncoder()
            self.ff_pred_hidden_size = self.ff_gen_hidden_size
            self.ff_pred_emb_layer = self.ff_gen_emb_layer
            self.ff_pred_hf = None
            self.ff_pred_decoder = None
        else:
            self.ff_pred_hf = AutoModel.from_pretrained(self.ff_pred_arch)
            self.ff_pred_hidden_size = self.ff_pred_hf.config.hidden_size
            self.ff_pred_emb_layer = self.ff_pred_hf.shared if 't5' in self.ff_pred_arch else self.ff_pred_hf.embeddings
            self.ff_pred_encoder = self.ff_pred_hf.encoder
            self.ff_pred_decoder = self.ff_pred_hf.decoder if hasattr(self.ff_pred_hf, 'decoder') else None

        # predictor output layer
        self.ff_output_layer = nn.Sequential(
            nn.Linear(self.ff_pred_hidden_size, self.ff_pred_hidden_size),
            nn.Tanh(),
            nn.Dropout(self.ff_dropout),
            nn.Linear(self.ff_pred_hidden_size, self.nb_classes),
            nn.Sigmoid() if not self.is_multilabel else nn.LogSoftmax(dim=-1),
        )

        ########################
        # counterfactual flow
        ########################
        # for reinforce
        self.rf_n_points = 0
        self.rf_mean_baseline = 0

        # counterfactual generator module
        if 't5' in self.cf_gen_arch:
            from transformers import T5Config, T5ForConditionalGeneration
            if 'mice' in self.cf_gen_arch:
                t5_config = T5Config.from_pretrained("t5-base", n_positions=512)
                self.cf_gen_hf = T5ForConditionalGeneration.from_pretrained("t5-base", config=t5_config)
                self.cf_gen_hf.load_state_dict(load_torch_object(self.cf_gen_arch), strict=False)
            else:
                if 'mt5' in self.cf_gen_arch:
                    from transformers import MT5ForConditionalGeneration
                    self.cf_gen_hf = MT5ForConditionalGeneration.from_pretrained(self.cf_gen_arch)
                else:
                    self.cf_gen_hf = T5ForConditionalGeneration.from_pretrained(self.cf_gen_arch)
                if self.cf_gen_ckpt is not None:
                    shell_logger.info('Loading cf generator from checkpoint: {}'.format(self.cf_gen_ckpt))
                    cf_gen_state_dict = load_torch_object(self.cf_gen_ckpt)['state_dict']
                    self.cf_gen_hf.load_state_dict(cf_gen_state_dict, strict=False)
            # self.cf_gen_emb_layer = deepcopy(self.cf_gen_hf.shared)
            self.cf_gen_emb_layer = self.cf_gen_hf.shared
            self.cf_gen_encoder = self.cf_gen_hf.encoder
            self.cf_gen_decoder = self.cf_gen_hf.decoder
            if self.cf_gen_lm_head_requires_grad != self.cf_gen_emb_requires_grad:
                self.cf_gen_hf.lm_head = deepcopy(self.cf_gen_hf.lm_head)  # detach lm_head from shared weights
            self.cf_gen_lm_head = self.cf_gen_hf.lm_head
            self.cf_gen_hidden_size = self.cf_gen_hf.config.hidden_size
        elif 'roberta' in self.cf_gen_arch:
            self.cf_gen_hf = RobertaForMaskedLM.from_pretrained(self.cf_gen_arch)
            if self.cf_gen_lm_head_requires_grad != self.cf_gen_emb_requires_grad:
                self.cf_gen_hf.lm_head = deepcopy(self.cf_gen_hf.lm_head)  # detach lm_head from shared weights
            self.cf_gen_emb_layer = self.cf_gen_hf.roberta.embeddings
            self.cf_gen_encoder = self.cf_gen_hf.roberta.encoder
            self.cf_gen_lm_head = self.cf_gen_hf.lm_head
            self.cf_gen_decoder = None
            self.cf_gen_hidden_size = self.cf_gen_hf.roberta.config.hidden_size
        elif 'bert' in self.cf_gen_arch:
            self.cf_gen_hf = BertForMaskedLM.from_pretrained(self.cf_gen_arch)
            if self.cf_gen_lm_head_requires_grad != self.cf_gen_emb_requires_grad:
                self.cf_gen_hf.cls = deepcopy(self.cf_gen_hf.cls)  # detach lm_head from shared weights
            self.cf_gen_emb_layer = self.cf_gen_hf.bert.embeddings
            self.cf_gen_encoder = self.cf_gen_hf.bert.encoder
            self.cf_gen_lm_head = self.cf_gen_hf.cls
            self.cf_gen_decoder = None
            self.cf_gen_hidden_size = self.cf_gen_hf.bert.config.hidden_size
        else:
            raise NotImplementedError

        ########################
        # loss functions
        ########################
        criterion_cls = nn.MSELoss if not self.is_multilabel else nn.NLLLoss
        self.ff_criterion = criterion_cls(reduction="none")
        self.cf_criterion = criterion_cls(reduction="none")

        ########################
        # weights details
        ########################
        self.init_all_weights()
        self.freeze_all_weights()

        ########################
        # logging
        ########################
        self.report_all_weights()

    def init_all_weights(self):
        # initialize params using xavier initialization for weights and zero for biases
        # (weights of these modules might be loaded later)
        self.init_weights(self.explainer_mlp)
        self.init_weights(self.explainer)
        self.init_weights(self.ff_output_layer)

    def freeze_all_weights(self):
        # freeze embedding layers
        if not self.ff_gen_emb_requires_grad:
            freeze_module(self.ff_gen_emb_layer)
        if not self.ff_pred_emb_requires_grad:
            freeze_module(self.ff_pred_emb_layer)
        if not self.cf_gen_emb_requires_grad:
            freeze_module(self.cf_gen_emb_layer)

        # freeze models and set to eval mode to disable dropout
        if not self.ff_gen_encoder_requires_grad:
            freeze_module(self.ff_gen_encoder)
            if self.ff_gen_decoder is not None:
                freeze_module(self.ff_gen_decoder)

        # freeze models and set to eval mode to disable dropout
        if not self.ff_pred_encoder_requires_grad:
            freeze_module(self.ff_pred_encoder)
            if self.ff_pred_decoder is not None:
                freeze_module(self.ff_pred_decoder)

        # freeze models and set to eval mode to disable dropout
        if not self.cf_gen_encoder_requires_grad:
            freeze_module(self.cf_gen_encoder)
            if self.cf_gen_decoder is not None:
                freeze_module(self.cf_gen_decoder)

        # the lm head is an independent factor, which we can freeze or not
        if not self.cf_gen_lm_head_requires_grad:
            # it should not be shared with the embedding layer
            # assert id(self.cf_gen_lm_head.weight) != id(self.cf_gen_emb_layer.weight)
            freeze_module(self.cf_gen_lm_head)

        # freeze output layers
        if not self.ff_pred_output_requires_grad:
            freeze_module(self.ff_output_layer)

        # freeze explainer
        if not self.explainer_requires_grad:
            freeze_module(self.explainer_mlp)
            freeze_module(self.explainer)

        # share generator and predictor for the factual flow
        if self.ff_shared_gen_pred:
            assert self.ff_gen_arch == self.ff_pred_arch
            self.ff_pred_hf = self.ff_gen_hf
            self.ff_pred_hidden_size = self.ff_gen_hidden_size
            self.ff_pred_emb_layer = self.ff_gen_emb_layer
            self.ff_pred_encoder = self.ff_gen_encoder
            self.ff_pred_decoder = self.ff_gen_decoder

        # shared factual and counterfactual generators (only the LM head remains separate)
        if self.share_generators:
            del self.cf_gen_hf
            del self.cf_gen_emb_layer
            del self.cf_gen_encoder
            del self.cf_gen_hidden_size
            self.cf_gen_hf = self.ff_gen_hf
            self.cf_gen_emb_layer = self.ff_gen_emb_layer
            self.cf_gen_encoder = self.ff_gen_encoder
            self.cf_gen_decoder = self.ff_gen_decoder
            self.cf_gen_hidden_size = self.ff_gen_hidden_size

    def report_all_weights(self):
        # manual check requires_grad for all modules
        for name, module in self.named_children():
            shell_logger.info('is_trainable({}): {}'.format(name, is_trainable(module)))

    def configure_optimizers(self):
        """Configure optimizers and lr schedulers for Trainer."""
        ff_params = chain(
            self.ff_gen_emb_layer.parameters(),
            self.ff_gen_encoder.parameters(),
            self.ff_gen_decoder.parameters() if self.ff_gen_decoder is not None else [],
            self.ff_pred_emb_layer.parameters() if not self.ff_shared_gen_pred else [],
            self.ff_pred_encoder.parameters() if not self.ff_shared_gen_pred else [],
            self.ff_output_layer.parameters(),
            self.explainer_mlp.parameters(),
            self.explainer.parameters(),
        )
        cf_params = chain(
            self.cf_gen_emb_layer.parameters() if not self.share_generators else [],
            self.cf_gen_encoder.parameters() if not self.share_generators else [],
            self.cf_gen_decoder.parameters() if not self.share_generators and self.cf_gen_decoder is not None else [],
            self.cf_gen_lm_head.parameters(),
        )
        grouped_parameters = []
        if self.ff_lbda > 0:
            grouped_parameters += [{"params": ff_params, 'lr': self.hparams['lr']}]
        if self.cf_lbda > 0:
            grouped_parameters += [{"params": cf_params, 'lr': self.hparams['cf_lr']}]

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
        mask: torch.BoolTensor = None,
        mask_cf: torch.BoolTensor = None,
        token_type_ids: torch.BoolTensor = None,
        token_type_ids_cf: torch.BoolTensor = None,
        y: torch.LongTensor = None,
        y_cf: torch.LongTensor = None,
        z: torch.LongTensor = None,
        z_cf: torch.LongTensor = None,
        has_cf: bool = True,
        current_epoch: int = None,
    ):
        """
        Compute forward-pass.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param x_cf: counterfactual input ids tensor. torch.LongTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param mask_cf: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids: mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids_cf: mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :param y: factual labels. torch.LongTensor of shape [B, L]
        :param y_cf: counterfactual labels. torch.LongTensor of shape [B, L]
        :param z: rationales tensor. torch.LongTensor of shape [B, T]
        :param z_cf: rationales tensor. torch.LongTensor of shape [B, T]
        :param has_cf: whether the batch contains counterfactuals
        :param current_epoch: int represents the current epoch.
        :return: (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """
        # factual flow
        z_hat, y_hat = self.get_factual_flow(x, mask=mask, token_type_ids=token_type_ids)

        if has_cf:
            with torch.no_grad():
                z_cf_hat, y_cf_hat = self.get_factual_flow(x_cf, mask=mask_cf, token_type_ids=token_type_ids_cf)
            x_prepend = x_cf.clone()
            y_prepend = y_cf.clone() if y_cf is not None and self.cf_prepend_label_type == "gold" else y_cf_hat.argmax(-1)
            z_prepend = z_cf.clone() if z_cf is not None and self.cf_z_type == "gold" else z_cf_hat.detach()
            mask_prepend = mask_cf.clone()
            token_type_ids_prepend = token_type_ids_cf.clone() if token_type_ids_cf is not None else None
        else:
            x_prepend = x.clone()
            y_factual = y.clone() if y is not None and self.cf_prepend_label_type == "gold" else y_hat.argmax(-1)
            y_prepend = get_contrast_label(y_factual, self.nb_classes, self.cf_task_name)
            z_prepend = z.clone() if z is not None and self.cf_z_type == "gold" else z_hat
            mask_prepend = mask.clone()
            token_type_ids_prepend = token_type_ids.clone() if token_type_ids is not None else None

        # edit only a single input in case we have concatenated inputs
        if self.cf_explainer_mask_token_type_id is not None and token_type_ids_prepend is not None:
            e_mask = mask_prepend & (token_type_ids_prepend == self.cf_explainer_mask_token_type_id)
            z_prepend = z_prepend.masked_fill(~e_mask, 0)

        # prepend label to input and rationale
        x_prepend, z_prepend, mask_prepend = prepend_label_for_t5(
            x=x_prepend,
            y=y_prepend,
            z=z_prepend,
            mask=mask_prepend,
            tokenizer=self.tokenizer,
            max_length=512,
            task_name=self.cf_task_name
        )

        # counterfactual flow
        cf_output = self.get_counterfactual_flow(
            x=x_prepend,
            z=z_prepend,
            mask=mask_prepend,
            token_type_ids=token_type_ids_prepend,
            has_cf=has_cf
        )

        # return everything as output (useful for computing the loss)
        return (z_hat, y_hat), cf_output

    def get_counterfactual_flow(self, x, z, mask=None, token_type_ids=None, has_cf=True):
        """
        Compute the counterfactual flow.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param z: binary variables tensor. torch.FloatTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids: mask tensor for explanation positions. torch.BoolTensor of shape [B, T]
        :param has_cf: whether the batch contains counterfactuals
        :return: (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """
        # prepare input for the generator LM
        e = self.cf_gen_emb_layer(x)
        z_pos = z.masked_fill(~mask, 0)
        z_comp = (1 - z_pos).masked_fill(~mask, 0)

        if 't5' in self.cf_gen_arch:
            # fix inputs for t5
            x_enc, e_enc, z_enc, mask_enc = make_input_for_t5(
                x, e, z_pos, mask, pad_id=self.pad_token_id, idx_a=self.sentinel_a, idx_b=self.sentinel_b
            )
            x_dec, e_dec, z_dec, mask_dec = make_input_for_t5(
                x, e, z_comp, mask, pad_id=self.pad_token_id, idx_a=self.sentinel_a, idx_b=self.sentinel_b
            )
            sentinel_ids = torch.clamp(
                self.sentinel_b + 1 - (z_enc > 0).long().cumsum(dim=-1),
                min=self.sentinel_a,
                max=self.sentinel_b
            )
            e_mask = self.cf_gen_emb_layer(sentinel_ids)
        else:
            # fix inputs for bert
            mask_ids = torch.ones_like(x) * self.mask_token_id
            x_enc = mask_ids * z_pos + x * z_comp
            x_dec = x_enc
            e_enc = e_dec = e
            z_enc = z_dec = z_pos
            mask_enc = mask_dec = mask
            e_mask = self.cf_gen_emb_layer(torch.ones_like(x_enc) * self.mask_token_id)

        if has_cf:
            #
            # return the gold counterfactual & supervise the generator
            #

            gen_kwargs = {}
            if 't5' in self.cf_gen_arch:
                # shift token ids one to the right -> <s>, x0, x1 ...
                x_dec_shifted = shift_tokens_right(
                    x_dec, self.pad_token_id, self.cf_gen_hf.config.decoder_start_token_id
                )
                mask_dec_shifted = ~torch.eq(x_dec_shifted, self.pad_token_id)
                gen_kwargs = dict(
                    decoder_input_ids=x_dec_shifted,
                    decoder_attention_mask=mask_dec_shifted.long()
                )

            # pass through the model
            outputs = self.cf_gen_hf(
                input_ids=x_enc,
                attention_mask=mask_enc.long(),
                **gen_kwargs
            )
            logits = outputs.logits
            # logits[:, :, bad_tokens_ids] = -999999.0
            x_tilde = logits.argmax(dim=-1)

            if 't5' in self.cf_gen_arch:
                x_tilde = x_tilde.masked_fill(~mask_dec, self.pad_token_id)

            # use the gold counterfactual to compute the loss
            x_edit, mask_edit, token_type_ids_edit = x, mask, token_type_ids

        else:
            # return a sampled counterfactual from the generator & do reinforce

            # set the input for the counterfactual encoder via a differentiable where
            e_bar = e_mask * z_enc.unsqueeze(-1) + e_enc * (1 - z_enc).unsqueeze(-1)

            # pass masked inputs
            if 't5' in self.cf_gen_arch:
                cf_gen_enc_out = self.cf_gen_encoder(inputs_embeds=e_bar, attention_mask=mask_enc)
            else:
                ext_mask = (1.0 - mask_enc[:, None, None, :].to(self.dtype)) * -10000.0
                cf_gen_enc_out = self.cf_gen_encoder(e_bar, ext_mask)

            # recover last hidden state
            h_tilde = cf_gen_enc_out.last_hidden_state

            # sample from the LM head
            x_tilde, logits = self._sample_from_lm(h_tilde, mask_enc, encoder_outputs=cf_gen_enc_out, x_enc=x_enc)

            # get the edits from x_tilde
            x_edit, mask_edit, token_type_ids_edit = self._get_edits_from_x_tilde(x_tilde, x_enc, z_enc)

        # reuse the factual flow to get a prediction for the counterfactual flow
        z_edit, y_edit_hat = self.get_factual_flow(x_edit, mask=mask_edit, token_type_ids=token_type_ids_edit)

        return x_tilde, logits, x_enc, x_dec, z_enc, z_dec, x_edit, z_edit, mask_edit, y_edit_hat

    def _get_edits_from_x_tilde(self, x_tilde, x_enc, z_enc):
        if self.cf_use_reinforce and 't5' in self.cf_gen_arch:
            with torch.no_grad():
                x_edit = x_tilde if x_tilde.dim() == 2 else x_tilde.argmax(dim=-1)
                x_edit = fix_t5_generated_inputs(
                    x_edit,
                    pad_id=self.pad_token_id,
                    idx_a=self.sentinel_a,
                    idx_b=self.sentinel_b
                )
                x_edit, _ = merge_inputs_for_t5(
                    x_enc, x_edit, z_enc,
                    pad_id=self.pad_token_id,
                    eos_id=self.eos_token_id,
                    idx_a=self.sentinel_a,
                    idx_b=self.sentinel_b
                )
                x_edit = x_edit[:, 2:]  # remove the prepended label
        else:
            if x_tilde.dim() == 2:
                # get gen_ids only for <mask> positions
                z_1 = (z_enc > 0).long()
                x_edit = z_1 * x_tilde + (1 - z_1) * x_enc

            else:
                # get gen_ids only for <mask> positions
                vocab_size = x_tilde.shape[-1]
                z_1 = (z_enc > 0).long().unsqueeze(-1)
                x_one_hot = nn.functional.one_hot(x_enc, num_classes=vocab_size)
                x_edit = z_1 * x_tilde + (1 - z_1) * x_one_hot

        # get edit ids
        x_edit_ids = x_edit.argmax(dim=-1) if x_edit.dim() == 3 else x_edit

        # recover mask
        mask_edit = x_edit_ids != self.pad_token_id

        # recreate token_type_ids (works only for two concatenated inputs)
        token_type_ids_edit = 1 - torch.cumprod(x_edit_ids != self.eos_token_id, dim=-1)
        # mask padding positions with `2`
        token_type_ids_edit = token_type_ids_edit.masked_fill(~mask_edit, 2)

        return x_edit, mask_edit, token_type_ids_edit

    def _sample_from_lm(self, h_tilde, mask, encoder_outputs=None, x_enc=None):
        if self.cf_use_reinforce:
            if 't5' in self.cf_gen_arch:
                # recover hidden states from the encoder (.generate() changes the hidden states)
                # encoder_hidden_states = h_tilde.clone()

                # sample autoregressively
                gen_out = self.cf_gen_hf.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=mask.long(),
                    return_dict_in_generate=True,
                    output_scores=True,
                    **self.cf_generate_kwargs
                )
                # clear memory because generation is done
                torch.cuda.empty_cache()

                # idk why but t5 generates a pad symbol as the first token
                # so we cut it out for all samples in the batch
                # (this happens only for .sequences)
                x_tilde = gen_out.sequences[:, 1:]

                # stack and reshape logits
                logits = torch.stack(gen_out.scores).transpose(0, 1)
                nb = self.cf_generate_kwargs.get('num_beams', 1)
                bs = mask.shape[0]
                ts_new = logits.shape[1]
                logits = logits.reshape(bs, nb, ts_new, -1)

                # get the logits for x_tilde
                # cf_gen_dec_out = self.cf_gen_decoder(
                #     input_ids=gen_out.sequences,
                #     attention_mask=(gen_out.sequences != self.pad_token_id).long(),
                #     encoder_hidden_states=encoder_hidden_states
                # )
                # logits = self.cf_gen_lm_head(cf_gen_dec_out.last_hidden_state)[:, :-1]

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

        else:
            # use the ST-gumbel-softmax trick
            logits = self.cf_gen_lm_head(h_tilde)
            # x_tilde.shape is (bs, seq_len, |V|)
            x_tilde = nn.functional.gumbel_softmax(logits, hard=True, dim=-1)

        # save variables for computing REINFORCE loss later
        self.cf_x_tilde = x_tilde
        self.cf_logits = logits

        return x_tilde, logits

    def get_factual_loss(self, y_hat, y):
        # main loss for p(y | x, z)
        loss_vec = self.ff_criterion(y_hat, y)  # [B] or [B,C]
        loss = loss_vec.mean()
        return loss

    def get_counterfactual_loss(
        self,
        y_hat,
        y,
        x_dec=None,
        x_tilde=None,
        x_tilde_logits=None,
        z_enc=None,
        has_cf=False,
        prefix='train'
    ):
        # main loss for p(y | x, z)
        loss_vec = self.cf_criterion(y_hat, y)  # [B] or [B,C]
        main_loss = loss_vec.mean()

        # ideas for later (penalties):
        # use an "adaptor" layer which is a pretrained LM, such that new logits ~= adaptor logits
        # logits = alpha * self.adaptor_logits(x) + (1 - alpha) * self.cf_flow_logits(x)

        stats = {}
        if self.stage in ['test', 'predict']:
            gen_loss = torch.zeros(1, device=self.device)
        else:
            if has_cf:
                # supervise the generator
                lm_logits = x_tilde_logits.view(-1, x_tilde_logits.size(-1))
                lm_labels = x_dec.masked_fill(x_dec == self.pad_token_id, -100).view(-1, )
                gen_loss = torch.nn.functional.cross_entropy(lm_logits, lm_labels, ignore_index=-100)

            else:
                # do reinforce

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

                # recover the probabilities of x_tilde with opt 1.
                # log P(x_tilde | s; phi)
                # x:        the           <input_id_0> is on the <input_id_1>
                # x_tilde: <input_id_0>  long book    <input_id_1> table </s>
                # recover log prob of all sampled tokens
                logits = x_tilde_logits.squeeze(1) if x_tilde_logits.dim() == 4 else x_tilde_logits
                logp_xtilde = torch.log_softmax(logits, dim=-1)  # [B, T, |V|]

                # get probas of generated tokens [B, T]
                logp_xtilde = logp_xtilde.gather(-1, x_tilde.unsqueeze(-1)).squeeze(-1)

                # compute logprob only for the new sampled tokens
                if 't5' in self.cf_gen_arch:
                    # non-sentinel tokens
                    gen_mask = ~is_sentinel(x_tilde, self.sentinel_a, self.sentinel_b)
                    # non-pad tokens
                    gen_mask = gen_mask & ~torch.eq(x_tilde, self.pad_token_id)
                else:
                    gen_mask = (z_enc > 0).float()  # [B, T]

                # mask averaged log prob of generated tokens
                logp_xtilde_vec = (logp_xtilde * gen_mask).sum(1) / gen_mask.sum(1)

                # compute generator loss
                cost_vec = loss_vec.detach()
                # cost_vec is neg reward
                cost_logpz = ((cost_vec - self.rf_mean_baseline) * logp_xtilde_vec).mean(0)
                # add baseline
                if self.cf_use_reinforce_baseline:
                    self.rf_n_points += 1.0
                    self.rf_mean_baseline += (cost_vec.mean() - self.rf_mean_baseline) / self.rf_n_points

                # neg reward
                stats["obj"] = cost_vec.mean().item()
                # generator cost
                stats["gen_loss"] = cost_logpz.item()
                # predictor cost
                stats["pred_loss"] = main_loss.item()
                gen_loss = cost_logpz

        main_loss = main_loss + self.cf_lbda_gen * gen_loss
        return main_loss, stats

    def _get_z_stats(self, z, mask, prefix: str):
        # latent selection stats
        stats = {}
        num_0, num_c, num_1, total = get_z_stats(z, mask)
        stats[prefix + "_p0"] = num_0 / float(total)
        stats[prefix + "_pc"] = num_c / float(total)
        stats[prefix + "_p1"] = num_1 / float(total)
        stats[prefix + "_ps"] = (num_c + num_1) / float(total)
        return stats

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
        prefix = "train"
        input_ids = batch["input_ids"]
        mask = input_ids != self.pad_token_id
        labels = batch["labels"]
        cf_input_ids = batch.get("cf_input_ids", None)
        cf_mask = cf_input_ids != self.pad_token_id if cf_input_ids is not None else None
        cf_labels = batch.get("cf_labels", None)
        token_type_ids = batch.get("token_type_ids", None)
        cf_token_type_ids = batch.get("cf_token_type_ids", None)
        z_gold = batch.get("z", None)
        z_cf_gold = batch.get("z_cf", None)
        has_cf = cf_input_ids is not None

        # forward pass
        (z, y_hat), (x_tilde, x_tilde_logits, x_enc, x_dec, z_enc, z_dec, x_edit, z_edit, mask_edit, y_edit_hat) = self(
            x=input_ids,
            x_cf=cf_input_ids,
            mask=mask,
            mask_cf=cf_mask,
            token_type_ids=token_type_ids,
            token_type_ids_cf=cf_token_type_ids,
            y=labels,
            y_cf=cf_labels,
            z=z_gold,
            z_cf=z_cf_gold,
            has_cf=has_cf,
            current_epoch=self.current_epoch
        )

        # compute factual loss
        y_ff_pred = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y_ff_gold = labels if not self.is_multilabel else labels.view(-1)
        ff_loss = self.get_factual_loss(y_ff_pred, y_ff_gold)
        ff_z_stats = self._get_z_stats(z, mask, prefix=prefix)

        # compute counterfactual loss
        y_cf_pred = y_edit_hat if not self.is_multilabel else y_edit_hat.view(-1, self.nb_classes)
        y_cf_gold = cf_labels if not self.is_multilabel else cf_labels.view(-1)
        cf_loss, cf_loss_stats = self.get_counterfactual_loss(
            y_cf_pred,
            y_cf_gold,
            x_dec=x_dec,
            x_tilde=x_tilde,
            x_tilde_logits=x_tilde_logits,
            z_enc=z_enc,
            has_cf=has_cf,
            prefix=prefix,
        )
        cf_z_stats = self._get_z_stats(z_edit, mask_edit, prefix=prefix)

        # combine losses
        loss = self.ff_lbda * ff_loss + self.cf_lbda * cf_loss

        # logger=False because they are going to be logged via loss_stats
        self.log("train_ff_ps", ff_z_stats["train_ps"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_cf_ps", cf_z_stats["train_ps"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_ff_sum_loss", ff_loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)
        self.log("train_cf_sum_loss", cf_loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)
        self.log("train_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        metrics_to_wandb = {
            "train_ff_p1": ff_z_stats["train_p1"],
            "train_cf_p1": cf_z_stats["train_p1"],
            "train_ff_ps": ff_z_stats["train_ps"],
            "train_cf_ps": cf_z_stats["train_ps"],
            "train_ff_sum_loss": ff_loss.item(),
            "train_cf_sum_loss": cf_loss.item(),
        }
        if "gen_loss" in cf_loss_stats:
            metrics_to_wandb["train_cf_gen_loss"] = cf_loss_stats["gen_loss"]

        self.logger.log_metrics(metrics_to_wandb, self.global_step)

        # return the loss tensor to PTL
        return {
            "loss": loss,
            "ff_ps": ff_z_stats["train_ps"],
            "cf_ps": cf_z_stats["train_ps"]
        }

    def _shared_eval_step(self, batch: dict, batch_idx: int, prefix: str):
        input_ids = batch["input_ids"]
        mask = input_ids != self.pad_token_id
        labels = batch["labels"]
        cf_input_ids = batch.get("cf_input_ids", None)
        cf_mask = cf_input_ids != self.pad_token_id if cf_input_ids is not None else None
        cf_labels = batch.get("cf_labels", None)
        token_type_ids = batch.get("token_type_ids", None)
        cf_token_type_ids = batch.get("cf_token_type_ids", None)
        z_gold = batch.get("z", None)
        z_cf_gold = batch.get("z_cf", None)
        has_cf = cf_input_ids is not None
        # has_cf = self.stage not in ['test', 'predict'] and cf_input_ids is not None

        # forward pass
        (z, y_hat), (x_tilde, x_tilde_logits, x_enc, x_dec, z_enc, z_dec, x_edit, z_edit, mask_edit, y_edit_hat) = self(
            x=input_ids,
            x_cf=cf_input_ids,
            mask=mask,
            mask_cf=cf_mask,
            token_type_ids=token_type_ids,
            token_type_ids_cf=cf_token_type_ids,
            y=labels,
            y_cf=cf_labels,
            z=z_gold,
            z_cf=z_cf_gold,
            has_cf=has_cf,
            current_epoch=self.current_epoch
        )

        # compute factual loss
        y_ff_pred = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y_ff_gold = labels if not self.is_multilabel else labels.view(-1)
        ff_loss = self.get_factual_loss(y_ff_pred, y_ff_gold)
        ff_z_stats = self._get_z_stats(z, mask, prefix=prefix)
        self.logger.agg_and_log_metrics(ff_z_stats, step=None)

        # compute counterfactual loss
        y_cf_pred = y_edit_hat if not self.is_multilabel else y_edit_hat.view(-1, self.nb_classes)
        y_cf_gold = cf_labels if not self.is_multilabel else cf_labels.view(-1)
        cf_loss, cf_loss_stats = self.get_counterfactual_loss(
            y_cf_pred,
            y_cf_gold,
            x_dec=x_dec,
            x_tilde=x_tilde,
            x_tilde_logits=x_tilde_logits,
            z_enc=z_enc,
            has_cf=has_cf,
            prefix=prefix,
        )
        cf_z_stats = self._get_z_stats(z_edit, mask_edit, prefix=prefix)
        self.logger.agg_and_log_metrics(cf_loss_stats, step=None)
        self.logger.agg_and_log_metrics(cf_z_stats, step=None)

        # combine losses
        loss = self.ff_lbda * ff_loss + self.cf_lbda * cf_loss

        # log metrics
        self.log(f"{prefix}_ff_sum_loss", ff_loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True,)
        self.log(f"{prefix}_cf_sum_loss", cf_loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True,)
        self.log(f"{prefix}_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        # get factual tokens
        z_1 = (z > 0).long()  # non-zero probs are considered selections
        ff_rat_ids, ff_rat_tokens = get_rationales(self.tokenizer, input_ids, z_1, batch["lengths"])
        ff_tokens = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in input_ids.tolist()]
        ff_lengths = mask.long().sum(-1).tolist()

        # get counterfactual tokens
        gen_ids = x_edit if x_edit.dim() == 2 else x_edit.argmax(dim=-1)
        cf_tokens = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in gen_ids.tolist()]
        cf_lengths = mask_edit.long().sum(-1).tolist()

        # output to be stacked across iterations
        output = {
            f"{prefix}_ff_sum_loss": ff_loss.item(),
            f"{prefix}_cf_sum_loss": cf_loss.item(),
            f"{prefix}_ff_ps": ff_z_stats[prefix + "_ps"],
            f"{prefix}_cf_ps": cf_z_stats[prefix + "_ps"],
            f"{prefix}_ids_rationales": ff_rat_ids,
            f"{prefix}_rationales": ff_rat_tokens,
            f"{prefix}_pieces": ff_tokens,
            f"{prefix}_tokens": batch["tokens"],
            f"{prefix}_z": z,
            f"{prefix}_predictions": y_ff_pred,
            f"{prefix}_labels": y_ff_gold,
            f"{prefix}_lengths": ff_lengths,
            f"{prefix}_cfs": cf_tokens,
            f"{prefix}_cf_labels": y_cf_gold,
            f"{prefix}_cf_predictions": y_cf_pred,
            f"{prefix}_cf_z": z_edit,
            f"{prefix}_cf_lengths": cf_lengths,
        }
        if "annotations" in batch.keys():
            output[f"{prefix}_annotations"] = batch["annotations"]
        return output

    def _shared_eval_epoch_end(self, outputs: list, prefix: str):
        """
        PTL hook. Perform validation at the end of an epoch.

        :param outputs: list of dicts representing the stacked outputs from validation_step
        :param prefix: `val` or `test`
        """
        # assume that `outputs` is a list containing dicts with the same keys
        stacked_outputs = {k: [x[k] for x in outputs] for k in outputs[0].keys()}

        # sample a few examples to be logged in wandb
        idxs = list(range(sum(map(len, stacked_outputs[f"{prefix}_pieces"]))))
        shuffle(idxs)
        idxs = idxs[:10] if prefix != 'test' else idxs[:100]

        # useful functions
        select = lambda v: [v[i] for i in idxs]
        detach = lambda v: [v[i].detach().cpu() for i in range(len(v))]

        if self.log_rationales_in_wandb:
            # log rationales
            pieces = select(unroll(stacked_outputs[f"{prefix}_pieces"]))
            scores = detach(select(unroll(stacked_outputs[f"{prefix}_z"])))
            gold = select(unroll(stacked_outputs[f"{prefix}_labels"]))
            pred = detach(select(unroll(stacked_outputs[f"{prefix}_predictions"])))
            lens = select(unroll(stacked_outputs[f"{prefix}_lengths"]))
            html_string = get_html_rationales(pieces, scores, gold, pred, lens)
            self.logger.experiment.log({f"{prefix}_rationales": wandb.Html(html_string)})

            # log counterfactuals
            cfs = select(unroll(stacked_outputs[f"{prefix}_cfs"]))
            scores = detach(select(unroll(stacked_outputs[f"{prefix}_cf_z"])))
            gold = select(unroll(stacked_outputs[f"{prefix}_cf_labels"]))
            pred = detach(select(unroll(stacked_outputs[f"{prefix}_cf_predictions"])))
            lens = select(unroll(stacked_outputs[f"{prefix}_cf_lengths"]))
            html_string = get_html_rationales(cfs, scores, gold, pred, lens)
            self.logger.experiment.log({f"{prefix}_counterfactuals": wandb.Html(html_string)})

        # save rationales
        if self.hparams.save_rationales:
            # factual rationales
            scores = detach(unroll(stacked_outputs[f"{prefix}_z"]))
            lens = unroll(stacked_outputs[f"{prefix}_lengths"])
            filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_ff_rationales.txt')
            shell_logger.info(f'Saving rationales in {filename}...')
            save_rationales(filename, scores, lens)

            # counterfactual rationales
            scores = detach(unroll(stacked_outputs[f"{prefix}_cf_z"]))
            lens = unroll(stacked_outputs[f"{prefix}_cf_lengths"])
            filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_cf_rationales.txt')
            shell_logger.info(f'Saving rationales in {filename}...')
            save_rationales(filename, scores, lens)

        # save counterfactuals
        if self.hparams.save_counterfactuals:
            pieces = unroll(stacked_outputs[f"{prefix}_cfs"])
            lens = unroll(stacked_outputs[f"{prefix}_cf_lengths"])
            filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_counterfactuals.txt')
            shell_logger.info(f'Saving counterfactuals in {filename}...')
            save_counterfactuals(filename, pieces, lens)

        # log metrics
        dict_metrics = {
            f"{prefix}_ff_ps": np.mean(stacked_outputs[f"{prefix}_ff_ps"]),
            f"{prefix}_ff_sum_loss": np.mean(stacked_outputs[f"{prefix}_ff_sum_loss"]),
            f"{prefix}_cf_ps": np.mean(stacked_outputs[f"{prefix}_cf_ps"]),
            f"{prefix}_cf_sum_loss": np.mean(stacked_outputs[f"{prefix}_cf_sum_loss"]),
        }

        # only evaluate rationales on the test set and if we have annotation (only for beer dataset)
        if prefix == "test" and "test_annotations" in stacked_outputs.keys():
            rat_metrics = evaluate_rationale(
                stacked_outputs["test_ids_rationales"],
                stacked_outputs["test_annotations"],
                stacked_outputs["test_lengths"],
            )
            dict_metrics[f"{prefix}_ff_rat_precision"] = rat_metrics["macro_precision"]
            dict_metrics[f"{prefix}_ff_rat_recall"] = rat_metrics["macro_recall"]
            dict_metrics[f"{prefix}_ff_rat_f1"] = rat_metrics["f1_score"]

        # log classification metrics
        if self.is_multilabel:
            preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_predictions"]), dim=-1)
            labels = torch.tensor(unroll(stacked_outputs[f"{prefix}_labels"]), device=preds.device)
            cf_preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_cf_predictions"]), dim=-1)
            cf_labels = torch.tensor(unroll(stacked_outputs[f"{prefix}_cf_labels"]), device=cf_preds.device)
            ff_accuracy = torchmetrics.functional.accuracy(
                preds, labels, num_classes=self.nb_classes, average="macro"
            )
            ff_precision = torchmetrics.functional.precision(
                preds, labels, num_classes=self.nb_classes, average="macro"
            )
            ff_recall = torchmetrics.functional.recall(
                preds, labels, num_classes=self.nb_classes, average="macro"
            )
            ff_f1_score = 2 * ff_precision * ff_recall / (ff_precision + ff_recall)
            cf_accuracy = torchmetrics.functional.accuracy(
                cf_preds, cf_labels, num_classes=self.nb_classes, average="macro"
            )
            cf_precision = torchmetrics.functional.precision(
                cf_preds, cf_labels, num_classes=self.nb_classes, average="macro"
            )
            cf_recall = torchmetrics.functional.recall(
                cf_preds, cf_labels, num_classes=self.nb_classes, average="macro"
            )
            cf_f1_score = 2 * cf_precision * cf_recall / (cf_precision + cf_recall)
            dict_metrics[f"{prefix}_ff_accuracy"] = ff_accuracy
            dict_metrics[f"{prefix}_ff_precision"] = ff_precision
            dict_metrics[f"{prefix}_ff_recall"] = ff_recall
            dict_metrics[f"{prefix}_ff_f1score"] = ff_f1_score
            dict_metrics[f"{prefix}_cf_accuracy"] = cf_accuracy
            dict_metrics[f"{prefix}_cf_precision"] = cf_precision
            dict_metrics[f"{prefix}_cf_recall"] = cf_recall
            dict_metrics[f"{prefix}_cf_f1score"] = cf_f1_score
        else:
            dict_metrics[f"{prefix}_ff_mse"] = np.mean(stacked_outputs[f"{prefix}_ff_mse"])
            dict_metrics[f"{prefix}_cf_mse"] = np.mean(stacked_outputs[f"{prefix}_cf_mse"])

        # log all saved metrics
        for metric_name, metric_value in dict_metrics.items():
            shell_logger.info("{}: {:.4f}".format(metric_name, metric_value))
            self.log(
                metric_name,
                metric_value,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

        # aggregate across epochs
        self.logger.agg_and_log_metrics(dict_metrics, self.current_epoch)

        return dict_metrics
