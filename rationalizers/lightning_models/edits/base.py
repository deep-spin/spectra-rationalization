import logging
import os
from random import shuffle

import numpy as np
import torch
import torchmetrics
import wandb
from torch.nn import CrossEntropyLoss
from transformers import RobertaForMaskedLM, BertForMaskedLM, T5Config, T5ForConditionalGeneration
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right

from rationalizers import constants
from rationalizers.builders import build_optimizer, build_scheduler
from rationalizers.explainers import available_explainers
from rationalizers.lightning_models.highlights.transformers.base import TransformerBaseRationalizer
from rationalizers.lightning_models.utils import make_input_for_t5, prepend_label_for_t5, merge_inputs_for_t5, \
    get_t5_sentinel_ids, make_input_for_ct5, fix_t5_generated_inputs, get_contrast_label
from rationalizers.modules.metrics import evaluate_rationale
from rationalizers.utils import (
    get_z_stats, freeze_module, get_rationales, unroll, get_html_rationales,
    save_rationales, load_torch_object, is_trainable, save_counterfactuals
)

shell_logger = logging.getLogger(__name__)


class BaseEditor(TransformerBaseRationalizer):

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
        if h_params['explainer'] == 'sparsemap':
            explainer_cls = available_explainers['sparsemap']
            self.explainer = explainer_cls(h_params, self.ff_gen_hidden_size)

        # freeze factual flow
        freeze_module(self.ff_gen_hf)
        freeze_module(self.ff_pred_hf)
        freeze_module(self.explainer_mlp)
        freeze_module(self.explainer)
        freeze_module(self.ff_output_layer)
        self.ff_gen_hf.eval()
        self.explainer_mlp.eval()
        self.explainer.eval()
        self.ff_pred_hf.eval()
        self.ff_output_layer.eval()

        # counterfactual generator
        self.cf_x_tilde = None
        self.cf_log_prob_x_tilde = None
        self.cf_gen_arch = h_params.get("cf_gen_arch", "t5-small")
        self.cf_lbda = h_params.get('cf_lbda', 1.0)
        self.cf_generate_kwargs = h_params.get('cf_generate_kwargs', dict())
        self.cf_prepend_label_type = h_params.get("cf_prepend_label_type", "gold")
        self.cf_z_type = h_params.get("cf_z_type", "gold")
        self.cf_task_name = h_params.get("cf_task_name", "binary_classification")
        self.cf_explainer_mask_token_type_id = h_params.get("cf_explainer_mask_token_type_id", None)
        self.cf_classify_edits = h_params.get("cf_classify_edits", False)
        self.generation_mode = False

        # define sentinel range
        # self.sentinel_a, self.sentinel_b = len(self.tokenizer) - 100, len(self.tokenizer) - 1
        if 'mt5' in self.cf_gen_arch:
            self.sentinel_a, self.sentinel_b = 250000, 250099
        else:
            self.sentinel_a, self.sentinel_b = 32000, 32099

        # counterfactual generator module
        if 't5' in self.cf_gen_arch:
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
        elif 'roberta' in self.cf_gen_arch:
            self.cf_gen_hf = RobertaForMaskedLM.from_pretrained(self.cf_gen_arch)
        elif 'bert' in self.cf_gen_arch:
            self.cf_gen_hf = BertForMaskedLM.from_pretrained(self.cf_gen_arch)
        else:
            raise NotImplementedError

        # manual check requires_grad for all modules
        for name, module in self.named_children():
            shell_logger.info('is_trainable({}): {}'.format(name, is_trainable(module)))

    def configure_optimizers(self):
        """Configure optimizers and lr schedulers for Trainer."""
        parameters = [{"params": self.cf_gen_hf.parameters(), 'lr': self.hparams['lr']}]
        optimizer = build_optimizer(parameters, self.hparams)
        scheduler = build_scheduler(optimizer, self.hparams)
        output = {"optimizer": optimizer}
        if scheduler is not None:
            output["scheduler"] = scheduler
            output["monitor"] = self.hparams['monitor']  # not sure we need this
        return output

    def forward(
        self,
        x: torch.LongTensor,
        mask: torch.BoolTensor = None,
        token_type_ids: torch.LongTensor = None,
        y: torch.LongTensor = None,
        z: torch.LongTensor = None,
        contrast_label: bool = False,
        current_epoch=None,
    ):
        """
        Compute forward-pass.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param token_type_ids: mask tensor for explanation positions. torch.LongTensor of shape [B, T]
        :param y: labels tensor. torch.LongTensor of shape [B]
        :param z: rationales tensor. torch.LongTensor of shape [B, T]
        :param contrast_label: whether to prepend a contrastive label or not
        :param current_epoch: int represents the current epoch.
        :return: (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """

        if self.cf_z_type == "pred" or self.cf_prepend_label_type == "pred":
            with torch.no_grad():
                # factual flow
                z_hat, y_hat = self.get_factual_flow(x, mask=mask, token_type_ids=token_type_ids)

        # select which y and z to use
        y_prepend = y.clone() if y is not None and self.cf_prepend_label_type == "gold" else y_hat.argmax(-1)
        z_prepend = z.clone() if z is not None and self.cf_z_type == "gold" else z_hat

        # get a contrast label
        if contrast_label:
            y_prepend = get_contrast_label(y_prepend, self.nb_classes, self.cf_task_name)

        # edit only a single input in case we have concatenated inputs
        if self.cf_explainer_mask_token_type_id is not None and token_type_ids is not None:
            e_mask = mask & (token_type_ids == self.cf_explainer_mask_token_type_id)
            z_prepend = z_prepend.masked_fill(~e_mask, 0)

        # prepend label to input and rationale
        x_prepend, z_prepend, mask_prepend = prepend_label_for_t5(
            x=x,
            y=y_prepend,
            z=z_prepend,
            mask=mask,
            tokenizer=self.tokenizer,
            max_length=512,
            task_name=self.cf_task_name
        )

        # counterfactual flow
        x_tilde, logits, x_enc, x_dec, z_enc, z_dec = self.get_counterfactual_flow(
            x=x_prepend,
            z=z_prepend,
            mask=mask_prepend
        )

        # return everything as output (useful for computing the loss)
        return (z_hat, y_hat), (x_tilde, logits, x_enc, x_dec, z_enc, z_dec)

    def classify_edits(self, x_tilde, x_enc, z_enc):
        """
        Classify edits.

        :param x_tilde: decoder output. torch.LongTensor of shape [B, T]
        :param x_enc: encoder input. torch.LongTensor of shape [B, T]
        :param z_enc: rationale tensor. torch.FloatTensor of shape [B, T]
        """
        with torch.no_grad():
            x_edit = x_tilde if x_tilde.dim() == 2 else x_tilde.argmax(dim=-1)
            x_edit = fix_t5_generated_inputs(
                x_edit, pad_id=self.pad_token_id, idx_a=self.sentinel_a, idx_b=self.sentinel_b
            )
            x_edit, _ = merge_inputs_for_t5(
                x_enc, x_edit, z_enc,
                pad_id=self.pad_token_id, eos_id=self.eos_token_id,
                idx_a=self.sentinel_a, idx_b=self.sentinel_b
            )
            x_edit = x_edit[:, 2:]  # remove the prepend label
            mask_edit = x_edit != self.tokenizer.pad_token_id
            token_type_ids_edit = 1 - torch.cumprod(x_edit != self.eos_token_id, dim=-1)
            token_type_ids_edit = token_type_ids_edit.masked_fill(~mask_edit, 2)
            z_cf_hat, y_cf_hat = self.get_factual_flow(x_edit, mask=mask_edit, token_type_ids=token_type_ids_edit)
        return z_cf_hat, y_cf_hat

    def get_counterfactual_flow(self, x, z, mask=None):
        """
        Compute the counterfactual flow.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param z: binary variables tensor. torch.FloatTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """
        e = x.unsqueeze(-1)  # dummy embeddings
        z_pos = (z > 0).long()
        z_comp = (1 - z_pos) * mask.long()

        if 't5' in self.cf_gen_arch:
            # fix inputs for t5
            # replace chunked masked positions by a single sentinel token
            if 'ct5' in self.cf_gen_arch:
                x_enc, _, z_enc, mask_enc = make_input_for_ct5(
                    x, e, z_pos, mask, pad_id=self.pad_token_id, idx_a=self.sentinel_a, idx_b=self.sentinel_b
                )
            else:
                x_enc, _, z_enc, mask_enc = make_input_for_t5(
                    x, e, z_pos, mask, pad_id=self.pad_token_id, idx_a=self.sentinel_a, idx_b=self.sentinel_b
                )
            # do the same to get decoder inputs with [1 - z] masks
            if 'ct5' in self.cf_gen_arch:
                x_dec, _, z_dec, mask_dec = make_input_for_ct5(
                    x, e, z_comp, mask, pad_id=self.pad_token_id, idx_a=self.sentinel_a, idx_b=self.sentinel_b
                )
            else:
                x_dec, _, z_dec, mask_dec = make_input_for_t5(
                    x, e, z_comp, mask, pad_id=self.pad_token_id, idx_a=self.sentinel_a, idx_b=self.sentinel_b
                )
        else:
            # fix inputs for bert
            mask_ids = torch.ones_like(x) * self.mask_token_id
            x_enc = z_pos * mask_ids + (1 - z_pos) * x
            z_enc = z_pos
            mask_enc = mask
            x_dec = x_enc.clone()
            z_dec = z_comp
            mask_dec = mask_enc.clone()

        # pass through the model
        x_tilde, logits = self._sample_from_lm(x_enc, x_dec, mask=mask_enc, mask_dec=mask_dec)

        return x_tilde, logits, x_enc, x_dec, z_enc, z_dec

    def _sample_from_lm(self, x_enc, x_dec, mask=None, mask_dec=None):  # noqa
        if 't5' in self.cf_gen_arch:
            if self.generation_mode:
                # set ids that should not be generated
                # bad_tokens_ids = get_t5_sentinel_ids(idx_a=self.self.sentinel_a, idx_b=self.self.sentinel_b)
                # bad_tokens_ids += [self.eos_token_id]
                do_output_logits = False
                logits = None

                # sample autoregressively
                gen_out = self.cf_gen_hf.generate(
                    input_ids=x_enc,
                    attention_mask=mask.long(),
                    return_dict_in_generate=True,
                    output_scores=do_output_logits,
                    # bad_tokens_ids=bad_tokens_ids,
                    **self.cf_generate_kwargs
                )
                # clear memory because generation is done
                torch.cuda.empty_cache()

                # stack outputs and
                if do_output_logits:
                    logits = torch.stack(gen_out.scores).transpose(0, 1)
                    # reshape
                    bs = x_enc.shape[0]
                    nb = self.cf_generate_kwargs.get('num_beams', 1)
                    ts = logits.shape[1]
                    logits = logits.reshape(bs, nb, ts, -1)

                # cut out start token
                x_tilde = gen_out.sequences[:, 1:]
            else:
                # set ids that should not be generated
                # bad_tokens_ids = get_t5_sentinel_ids(idx_a=self.self.sentinel_a, idx_b=self.self.sentinel_b)
                # bad_tokens_ids += [self.eos_token_id]
                # shift token ids one to the right -> <s>, x0, x1 ...
                x_dec_shifted = shift_tokens_right(
                    x_dec, self.pad_token_id, self.cf_gen_hf.config.decoder_start_token_id
                )
                mask_dec_shifted = ~torch.eq(x_dec_shifted, self.pad_token_id)
                # pass through the model
                outputs = self.cf_gen_hf(
                    input_ids=x_enc,
                    attention_mask=mask.long(),
                    decoder_input_ids=x_dec_shifted,
                    decoder_attention_mask=mask_dec_shifted.long()
                )
                logits = outputs.logits
                # logits[:, :, bad_tokens_ids] = -999999.0
                x_tilde = logits.argmax(dim=-1)
                x_tilde = x_tilde.masked_fill(~mask_dec, self.pad_token_id)

        else:
            outputs = self.cf_gen_hf(
                input_ids=x_enc,
                attention_mask=mask.long(),
            )
            logits = outputs.logits
            x_tilde = logits.argmax(dim=-1)
            x_tilde = x_tilde.masked_fill(~mask_dec, self.pad_token_id)

        return x_tilde, logits

    def get_counterfactual_loss(self, y_hat, y, z, mask, prefix):
        """
        Compute loss for the counterfactual flow.
        """
        if self.stage in ['test', 'predict']:
            main_loss = torch.zeros(1, device=self.device)
        else:
            # lm_logits = y_hat.log_softmax(dim=-1).view(-1, y_hat.size(-1))
            # lm_labels = y.masked_fill(y == self.pad_token_id, -100).view(-1,)
            # main_loss = torch.nn.functional.nll_loss(lm_logits, lm_labels, ignore_index=-100)
            lm_logits = y_hat.view(-1, y_hat.size(-1))
            lm_labels = y.masked_fill(y == self.pad_token_id, -100).view(-1, )
            main_loss = torch.nn.functional.cross_entropy(lm_logits, lm_labels, ignore_index=-100)

        stats = {}
        num_0, num_c, num_1, total = get_z_stats(z, mask)
        stats[prefix + "_p0"] = num_0 / float(total)
        stats[prefix + "_pc"] = num_c / float(total)
        stats[prefix + "_p1"] = num_1 / float(total)
        stats[prefix + "_ps"] = (num_c + num_1) / float(total)
        stats[prefix + "_main_loss"] = main_loss.item()
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
        token_type_ids = batch.get("token_type_ids", None)
        mask = input_ids != constants.PAD_ID
        labels = batch["labels"]
        z_gold = batch.get("z", None)
        prefix = "train"

        (z, preds), (x_tilde, logits, x_enc, x_dec, z_enc, z_dec) = self(
            x=input_ids,
            mask=mask,
            token_type_ids=token_type_ids,
            y=labels,
            z=z_gold,
            current_epoch=self.current_epoch
        )
        # compute factual loss
        loss, loss_stats = self.get_counterfactual_loss(logits, x_dec, z, mask, prefix=prefix)

        # logger=False because they are going to be logged via loss_stats
        self.log("train_ps", loss_stats["train_ps"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        metrics_to_wandb = {
            "train_ps": loss_stats["train_ps"],
            "train_sum_loss": loss.item(),
        }
        self.logger.log_metrics(metrics_to_wandb, self.global_step)

        # return the loss tensor to PTL
        return {"loss": loss, "ps": loss_stats["train_ps"]}

    def _shared_eval_step(self, batch: dict, batch_idx: int, prefix: str):
        input_ids = batch["input_ids"]
        token_type_ids = batch.get("token_type_ids", None)
        mask = input_ids != constants.PAD_ID
        labels = batch["labels"]
        z_gold = batch.get("z", None)

        # forward-pass
        (z, y_hat), (x_tilde, logits, x_enc, x_dec, z_enc, z_dec) = self(
            x=input_ids,
            mask=mask,
            token_type_ids=token_type_ids,
            y=labels,
            z=z_gold,
            contrast_label=False,
            current_epoch=self.current_epoch
        )
        # classify the generated edits
        z_edit, y_edit_hat = z, y_hat
        if self.cf_classify_edits:
            z_edit, y_edit_hat = self.classify_edits(x_tilde, x_enc, z_enc)

        # forward pass with contrastive labels
        _, (ct_x_tilde, _, ct_x_enc, _, ct_z_enc, _) = self(
            x=input_ids,
            mask=mask,
            token_type_ids=token_type_ids,
            y=labels,
            z=z_gold,
            contrast_label=True,
            current_epoch=self.current_epoch
        )
        # classify the generated contrastive edits
        ct_z_edit, ct_y_edit_hat = z, y_hat
        if self.cf_classify_edits:
            ct_z_edit, ct_y_edit_hat = self.classify_edits(ct_x_tilde, ct_x_enc, ct_z_enc)

        # compute factual loss
        loss, loss_stats = self.get_counterfactual_loss(logits, x_dec, z, mask, prefix=prefix)
        self.logger.agg_and_log_metrics(loss_stats, step=None)

        # log metrics
        self.log(f"{prefix}_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        # recover tokens from ids
        def get_edit_tokens(x_tilde_, x_enc_, z_enc_):
            # merge rationales
            gen_ids = x_tilde_.clone() if x_tilde_.dim() == 2 else x_tilde_.argmax(dim=-1)
            # fix repeated inputs for t5 (up to repetitions of size 3 -> k=2)
            gen_ids = fix_t5_generated_inputs(
                gen_ids, pad_id=self.pad_token_id, idx_a=self.sentinel_a, idx_b=self.sentinel_b
            )
            # get edits
            if 't5' not in self.cf_gen_arch:
                z_tilde = (z_enc_ > 0).long()
                gen_ids = z_tilde * gen_ids + (1 - z_tilde) * input_ids
            else:
                gen_ids, z_tilde = merge_inputs_for_t5(
                    x_enc_, gen_ids, z_enc_,
                    pad_id=self.pad_token_id, eos_id=self.eos_token_id,
                    idx_a=self.sentinel_a, idx_b=self.sentinel_b
                )
            tokens = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in gen_ids.tolist()]
            lengths = (gen_ids != self.pad_token_id).long().sum(-1).tolist()
            return tokens, z_tilde, lengths

        pieces = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in input_ids.tolist()]
        edit_tokens, edit_z_tilde, edit_lengths = get_edit_tokens(x_tilde, x_enc, z_enc)
        ct_edit_tokens, ct_edit_z_tilde, ct_edit_lengths = get_edit_tokens(ct_x_tilde, ct_x_enc, ct_z_enc)

        # output to be stacked across iterations
        output = {
            f"{prefix}_sum_loss": loss.item(),
            f"{prefix}_ps": loss_stats[prefix + "_ps"],

            f"{prefix}_pieces": pieces,
            f"{prefix}_z": z,
            f"{prefix}_predictions": y_hat,
            f"{prefix}_labels": labels,
            f"{prefix}_lengths": batch["lengths"],

            f"{prefix}_edits": edit_tokens,
            f"{prefix}_edits_z": z_edit,
            f"{prefix}_edits_z_tilde": edit_z_tilde,
            f"{prefix}_edits_predictions": y_edit_hat,
            f"{prefix}_edits_lengths": edit_lengths,

            f"{prefix}_contrast_edits": ct_edit_tokens,
            f"{prefix}_contrast_edits_z": ct_z_edit,
            f"{prefix}_contrast_edits_predictions": ct_y_edit_hat,
            f"{prefix}_contrast_edits_z_tilde": ct_edit_z_tilde,
            f"{prefix}_contrast_edits_lengths": ct_edit_lengths,
        }
        if "annotations" in batch.keys():
            output[f"{prefix}_annotations"] = batch["annotations"]
        if "mse" in loss_stats.keys():
            output[f"{prefix}_mse"] = loss_stats["mse"]
        if "is_original" in batch.keys():
            output[f"{prefix}_is_original"] = batch["is_original"]

        return output

    def _shared_eval_epoch_end(self, outputs: list, prefix: str):
        """
        PTL hook. Perform validation at the end of an epoch.

        :param outputs: list of dicts representing the stacked outputs from validation_step
        :param prefix: `val` or `test`
        """
        stacked_outputs = {k: [x[k] for x in outputs] for k in outputs[0].keys()}

        # sample a few examples to be logged in wandb
        idxs = list(range(sum(map(len, stacked_outputs[f"{prefix}_pieces"]))))
        shuffle(idxs)
        idxs = idxs[:10] if prefix != 'test' else idxs[:100]

        # useful functions
        select = lambda v: [v[i] for i in idxs]
        detach = lambda v: [v[i].detach().cpu() for i in range(len(v))]

        if self.log_rationales_in_wandb:
            gold = select(unroll(stacked_outputs[f"{prefix}_labels"]))
            gold_ct = get_contrast_label(torch.tensor(gold), self.nb_classes, self.cf_task_name).tolist()

            # log rationales
            pieces = select(unroll(stacked_outputs[f"{prefix}_pieces"]))
            scores = detach(select(unroll(stacked_outputs[f"{prefix}_z"])))
            pred = detach(select(unroll(stacked_outputs[f"{prefix}_predictions"])))
            lens = select(unroll(stacked_outputs[f"{prefix}_lengths"]))
            html_string = get_html_rationales(pieces, scores, gold, pred, lens)
            self.logger.experiment.log({f"{prefix}_rationales": wandb.Html(html_string)})

            if self.cf_classify_edits:
                # log rationales of the edited input
                pieces = select(unroll(stacked_outputs[f"{prefix}_edits"]))
                scores = detach(select(unroll(stacked_outputs[f"{prefix}_edits_z"])))
                pred = detach(select(unroll(stacked_outputs[f"{prefix}_edits_predictions"])))
                lens = select(unroll(stacked_outputs[f"{prefix}_edits_lengths"]))
                # remove prepended labels
                pieces = [p[2:] for p in pieces]
                lens = [l - 2 for l in lens]
                html_string = get_html_rationales(pieces, scores, gold, pred, lens)
                self.logger.experiment.log({f"{prefix}_edits_rationales": wandb.Html(html_string)})

                # log rationales of the contrastive edited input
                pieces = select(unroll(stacked_outputs[f"{prefix}_contrast_edits"]))
                scores = detach(select(unroll(stacked_outputs[f"{prefix}_contrast_edits_z"])))
                pred = detach(select(unroll(stacked_outputs[f"{prefix}_contrast_edits_predictions"])))
                lens = select(unroll(stacked_outputs[f"{prefix}_contrast_edits_lengths"]))
                pieces = [p[2:] for p in pieces]
                lens = [l - 2 for l in lens]
                html_string = get_html_rationales(pieces, scores, gold_ct, pred, lens)
                self.logger.experiment.log({f"{prefix}_contrast_edits_rationales": wandb.Html(html_string)})

            # log edits
            cfs = select(unroll(stacked_outputs[f"{prefix}_edits"]))
            scores = detach(select(unroll(stacked_outputs[f"{prefix}_edits_z_tilde"])))
            pred = detach(select(unroll(stacked_outputs[f"{prefix}_edits_predictions"])))
            lens = select(unroll(stacked_outputs[f"{prefix}_edits_lengths"]))
            html_string = get_html_rationales(cfs, scores, gold, pred, lens)
            self.logger.experiment.log({f"{prefix}_edits": wandb.Html(html_string)})

            # log contrastive edits
            cfs = select(unroll(stacked_outputs[f"{prefix}_contrast_edits"]))
            scores = detach(select(unroll(stacked_outputs[f"{prefix}_contrast_edits_z_tilde"])))
            pred = detach(select(unroll(stacked_outputs[f"{prefix}_contrast_edits_predictions"])))
            lens = select(unroll(stacked_outputs[f"{prefix}_contrast_edits_lengths"]))
            html_string = get_html_rationales(cfs, scores, gold_ct, pred, lens)
            self.logger.experiment.log({f"{prefix}_contrast_edits": wandb.Html(html_string)})

        # save rationales
        if self.hparams.save_rationales:
            # factual rationales
            scores = detach(unroll(stacked_outputs[f"{prefix}_z"]))
            lens = unroll(stacked_outputs[f"{prefix}_lengths"])
            filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_rationales.txt')
            shell_logger.info(f'Saving rationales to: {filename}')
            save_rationales(filename, scores, lens)

            if self.cf_classify_edits:
                # edit rationales
                scores = detach(unroll(stacked_outputs[f"{prefix}_edits_z"]))
                lens = unroll(stacked_outputs[f"{prefix}_edits_lengths"])
                filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_edit_rationales.txt')
                shell_logger.info(f'Saving edit rationales to: {filename}')
                save_rationales(filename, scores, lens)

                # contrast edit rationales
                scores = detach(unroll(stacked_outputs[f"{prefix}_contrast_edits_z"]))
                lens = unroll(stacked_outputs[f"{prefix}_contrast_edits_lengths"])
                filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_contrast_edit_rationales.txt')
                shell_logger.info(f'Saving contrast edit rationales to: {filename}')
                save_rationales(filename, scores, lens)

        # save edits
        if self.hparams.save_edits:
            # factual edits
            pieces = unroll(stacked_outputs[f"{prefix}_edits"])
            lens = unroll(stacked_outputs[f"{prefix}_edits_lengths"])
            filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_edits.txt')
            shell_logger.info(f'Saving edits to: {filename}')
            save_counterfactuals(filename, pieces, lens)

            # contrast edits
            pieces = unroll(stacked_outputs[f"{prefix}_contrast_edits"])
            lens = unroll(stacked_outputs[f"{prefix}_contrast_edits_lengths"])
            filename = os.path.join(self.hparams.default_root_dir, f'{prefix}_contrast_edits.txt')
            shell_logger.info(f'Saving contrast edits to: {filename}')
            save_counterfactuals(filename, pieces, lens)

        # log metrics
        dict_metrics = {
            f"{prefix}_ps": np.mean(stacked_outputs[f"{prefix}_ps"]),
            f"{prefix}_sum_loss": np.mean(stacked_outputs[f"{prefix}_sum_loss"]),
        }

        # log classification metrics
        preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_predictions"]), dim=-1)
        labels = torch.tensor(unroll(stacked_outputs[f"{prefix}_labels"]), device=preds.device)
        acc = torchmetrics.functional.accuracy(preds, labels, num_classes=self.nb_classes, average="macro")
        dict_metrics[f"{prefix}_accuracy"] = acc

        if self.cf_classify_edits:
            preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_edits_predictions"]), dim=-1)
            acc = torchmetrics.functional.accuracy(preds, labels, num_classes=self.nb_classes, average="macro")
            dict_metrics[f"{prefix}_edit_accuracy"] = acc

            if self.cf_task_name == 'nli':
                preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_contrast_edits_predictions"]), dim=-1)
                acc = 1 - torchmetrics.functional.accuracy(preds, labels, num_classes=self.nb_classes, average="macro")
            else:
                labels = get_contrast_label(labels, self.nb_classes, self.cf_task_name)
                preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_contrast_edits_predictions"]), dim=-1)
                acc = torchmetrics.functional.accuracy(preds, labels, num_classes=self.nb_classes, average="macro")
            dict_metrics[f"{prefix}_contrast_edit_accuracy"] = acc

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

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = None):
        input_ids = batch["input_ids"]
        token_type_ids = batch.get("token_type_ids", None)
        labels = batch.get("labels", None)
        z_gold = batch.get("z", None)
        mask = input_ids != constants.PAD_ID

        # forward-pass
        (z, y_hat), (x_tilde, logits, x_enc, x_dec, z_enc, z_dec) = self(
            x=input_ids,
            mask=mask,
            token_type_ids=token_type_ids,
            y=labels,
            z=z_gold,
            contrast_label=True,
            current_epoch=self.current_epoch
        )

        # recover tokens from ids
        def get_edit_tokens(x_tilde_, x_enc_, z_enc_):
            # merge rationales
            gen_ids = x_tilde_.clone() if x_tilde_.dim() == 2 else x_tilde_.argmax(dim=-1)
            # fix repeated inputs for t5 (up to repetitions of size 3 -> k=2)
            gen_ids = fix_t5_generated_inputs(
                gen_ids, pad_id=self.pad_token_id, idx_a=self.sentinel_a, idx_b=self.sentinel_b
            )
            # get edits
            if 't5' not in self.cf_gen_arch:
                z_tilde = (z_enc_ > 0).long()
                gen_ids = z_tilde * gen_ids + (1 - z_tilde) * input_ids
            else:
                gen_ids, z_tilde = merge_inputs_for_t5(
                    x_enc_, gen_ids, z_enc_,
                    pad_id=self.pad_token_id, eos_id=self.eos_token_id,
                    idx_a=self.sentinel_a, idx_b=self.sentinel_b
                )
            tokens = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in gen_ids.tolist()]
            lengths = (gen_ids != self.pad_token_id).long().sum(-1).tolist()
            return tokens, z_tilde, lengths

        edit_tokens, edit_z_tilde, edit_lengths = get_edit_tokens(x_tilde, x_enc, z_enc)
        edit_tokens = [e[:l] for e, l in zip(edit_tokens, edit_lengths)]
        edit_z_tilde = [z[:l] for z, l in zip(edit_z_tilde, edit_lengths)]

        output = {
            "predictions": y_hat,
            "z": z,
            "edits": edit_tokens,
            "edits_z": edit_z_tilde,
        }
        return output
