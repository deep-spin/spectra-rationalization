import logging
import os
from random import shuffle

import numpy as np
import torch
import torchmetrics
import wandb

from rationalizers.lightning_models.highlights.transformers.spectra import TransformerSPECTRARationalizer
from rationalizers.modules.metrics import evaluate_rationale
from rationalizers.utils import (
    get_z_stats, get_rationales, unroll, get_html_rationales,
    save_rationales, save_counterfactuals
)

shell_logger = logging.getLogger(__name__)


class ExplSupervisedTransformerSPECTRARationalizer(TransformerSPECTRARationalizer):

    def __init__(
        self,
        tokenizer: object,
        nb_classes: int,
        is_multilabel: bool,
        h_params: dict,
    ):
        super().__init__(tokenizer, nb_classes, is_multilabel, h_params)
        self.cf_lbda = h_params.get('cf_lbda', 1.0)
        self.expl_lbda = h_params.get('expl_lbda', 1.0)
        self.sparsemap_budget_strategy = h_params.get('sparsemap_budget_strategy', 'fixed')  # fixed, adaptive
        self.sparsemap_orig_budget = h_params.get('sparsemap_budget', 0)
        # self.cf_gold_strategy = h_params.get('cf_gold_strategy', 'fixed')  # fixed or dynamic

    def get_factual_loss(self, y_hat, y, **kwargs):
        # main loss for p(y | x, z)
        loss_vec = self.ff_criterion(y_hat, y)  # [B] or [B,C]
        loss = loss_vec.mean()
        return loss

    def get_explainer_loss(self, z_ff, z_ff_star, z_cf, z_cf_star, mask_ff, mask_cf):
        """
        Compute loss for the explainer flow.

        :param z_ff: latent selection vector from factual flow. torch.FloatTensor of shape [B, T]
        :param z_ff_star: latent selection vector from factual flow (star). torch.FloatTensor of shape [B, T]
        :param z_cf: latent selection vector from counterfactual flow. torch.FloatTensor of shape [B, T]
        :param z_cf_star: latent selection vector from counterfactual flow (star). torch.FloatTensor of shape [B, T]
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
        """
        ff_z_loss = torch.norm(z_ff - z_ff_star, p=2, dim=-1) ** 2 / mask_ff.float().sum(dim=-1)
        cf_z_loss = torch.norm(z_cf - z_cf_star, p=2, dim=-1) ** 2 / mask_cf.float().sum(dim=-1)
        # ff_z_loss = (torch.relu(z_ff - z_ff_star) * mask_ff.float()).sum(-1) / mask_ff.float().sum(-1)
        # cf_z_loss = (torch.relu(z_cf - z_cf_star) * mask_cf.float()).sum(-1) / mask_cf.float().sum(-1)
        loss = ff_z_loss.mean() + cf_z_loss.mean()
        return loss

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

        # factual inputs
        input_ids = batch["input_ids"]
        mask = input_ids != self.pad_token_id
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)
        z_gold = batch.get("z", None)

        # counterfactual inputs
        cf_input_ids = batch.get("cf_input_ids", None)
        cf_mask = cf_input_ids != self.pad_token_id if cf_input_ids is not None else None
        cf_labels = batch.get("cf_labels", None)
        cf_token_type_ids = batch.get("cf_token_type_ids", None)
        cf_z_pre_gold = batch.get("cf_z_pre", None)
        cf_z_pos_gold = batch.get("cf_z_pos", None)

        # define sparsemap budget
        if self.sparsemap_budget_strategy == 'adaptive_dynamic':
            orig_budget = 100 * z_gold.sum(-1).float() / mask.sum(-1).float()
        else:
            orig_budget = 1.0 * self.sparsemap_orig_budget

        # compute factual loss
        self.explainer.budget = orig_budget
        z, y_hat = self(input_ids, mask, token_type_ids=token_type_ids, current_epoch=self.current_epoch)
        y_ff_pred = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y_ff_gold = labels if not self.is_multilabel else labels.view(-1)
        ff_loss = self.get_factual_loss(y_ff_pred, y_ff_gold)
        ff_z_stats = self._get_z_stats(z, mask, prefix=prefix)
        self.logger.agg_and_log_metrics(ff_z_stats, step=None)

        # compute counterfactual loss
        if self.sparsemap_budget_strategy in ['adaptive', 'adaptive_dynamic']:
            budget_increase_rate = torch.clamp(cf_z_pre_gold.sum(-1) / z_gold.sum(-1), min=1.0)
            self.explainer.budget = orig_budget * budget_increase_rate
        cf_z, cf_y_hat = self(cf_input_ids, cf_mask, token_type_ids=cf_token_type_ids, current_epoch=self.current_epoch)
        y_cf_pred = cf_y_hat if not self.is_multilabel else cf_y_hat.view(-1, self.nb_classes)
        y_cf_gold = cf_labels if not self.is_multilabel else cf_labels.view(-1)
        cf_loss = self.get_factual_loss(y_cf_pred, y_cf_gold)
        cf_z_stats = self._get_z_stats(cf_z, cf_mask, prefix=prefix)
        self.logger.agg_and_log_metrics(cf_z_stats, step=None)

        # compute explainer loss
        expl_loss = self.get_explainer_loss(z, z_gold, cf_z, cf_z_pre_gold, mask, cf_mask)

        # combine losses
        loss = self.ff_lbda * ff_loss + self.cf_lbda * cf_loss + self.expl_lbda * expl_loss

        # logger=False because they are going to be logged via loss_stats
        self.log("train_ff_ps", ff_z_stats["train_ps"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_cf_ps", cf_z_stats["train_ps"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_ff_sum_loss", ff_loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)
        self.log("train_cf_sum_loss", cf_loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)
        self.log("train_expl_sum_loss", expl_loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False, )
        self.log("train_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        metrics_to_wandb = {
            "train_ff_p1": ff_z_stats["train_p1"],
            "train_cf_p1": cf_z_stats["train_p1"],
            "train_ff_ps": ff_z_stats["train_ps"],
            "train_cf_ps": cf_z_stats["train_ps"],
            "train_ff_sum_loss": ff_loss.item(),
            "train_cf_sum_loss": cf_loss.item(),
            "train_expl_sum_loss": expl_loss.item(),
        }
        self.logger.log_metrics(metrics_to_wandb, self.global_step)

        # return the loss tensor to PTL
        return {
            "loss": loss,
            "ff_ps": ff_z_stats["train_ps"],
            "cf_ps": cf_z_stats["train_ps"]
        }

    def _shared_eval_step(self, batch: dict, batch_idx: int, prefix: str):
        # factual inputs
        input_ids = batch["input_ids"]
        mask = input_ids != self.pad_token_id
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)
        z_gold = batch.get("z", None)

        # counterfactual inputs
        cf_input_ids = batch.get("cf_input_ids", None)
        cf_mask = cf_input_ids != self.pad_token_id if cf_input_ids is not None else None
        cf_labels = batch.get("cf_labels", None)
        cf_token_type_ids = batch.get("cf_token_type_ids", None)
        cf_z_pre_gold = batch.get("cf_z_pre", None)
        cf_z_pos_gold = batch.get("cf_z_pos", None)

        # if not isinstance(self.explainer.budget, (int, float)):
        #     self.explainer.budget = self.explainer.budget.mean().item()

        if self.sparsemap_budget_strategy == 'adaptive_dynamic' and z_gold is not None:
            orig_budget = 100 * z_gold.sum(-1).float() / mask.sum(-1).float()
        else:
            orig_budget = 1.0 * self.sparsemap_orig_budget

        # compute factual loss
        self.explainer.budget = orig_budget
        z, y_hat = self(input_ids, mask, token_type_ids=token_type_ids, current_epoch=self.current_epoch)
        y_ff_pred = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y_ff_gold = labels if not self.is_multilabel else labels.view(-1)
        ff_loss = self.get_factual_loss(y_ff_pred, y_ff_gold)
        ff_z_stats = self._get_z_stats(z, mask, prefix=prefix)
        self.logger.agg_and_log_metrics(ff_z_stats, step=None)

        if cf_input_ids is not None:
            # compute counterfactual loss
            if self.sparsemap_budget_strategy in ['adaptive', 'adaptive_dynamic'] and z_gold is not None:
                budget_increase_rate = torch.clamp(cf_z_pre_gold.sum(-1) / z_gold.sum(-1), min=1.0)
                self.explainer.budget = orig_budget * budget_increase_rate
            cf_z, cf_y_hat = self(cf_input_ids, cf_mask, token_type_ids=cf_token_type_ids, current_epoch=self.current_epoch)
            y_cf_pred = cf_y_hat if not self.is_multilabel else cf_y_hat.view(-1, self.nb_classes)
            y_cf_gold = cf_labels if not self.is_multilabel else cf_labels.view(-1)
            cf_loss = self.get_factual_loss(y_cf_pred, y_cf_gold)
            cf_z_stats = self._get_z_stats(cf_z, cf_mask, prefix=prefix)
            self.logger.agg_and_log_metrics(cf_z_stats, step=None)

            # compute explainer loss
            expl_loss = self.get_explainer_loss(z, z_gold, cf_z, cf_z_pre_gold, mask, cf_mask)
        else:
            cf_z = z
            y_cf_pred = y_ff_pred
            y_cf_gold = y_ff_gold
            cf_loss = torch.tensor(0.0, device=self.device)
            expl_loss = torch.tensor(0.0, device=self.device)
            cf_z_stats = ff_z_stats

        # combine losses
        loss = self.ff_lbda * ff_loss + self.cf_lbda * cf_loss + self.expl_lbda * expl_loss

        # log metrics
        self.log(f"{prefix}_ff_sum_loss", ff_loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True,)
        self.log(f"{prefix}_cf_sum_loss", cf_loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True,)
        self.log(f"{prefix}_expl_sum_loss", expl_loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log(f"{prefix}_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        # get factual tokens
        z_1 = (z > 0).long()  # non-zero probs are considered selections
        ff_rat_ids, ff_rat_tokens = get_rationales(self.tokenizer, input_ids, z_1, batch["lengths"])
        ff_tokens = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in input_ids.tolist()]
        ff_lengths = mask.long().sum(-1).tolist()

        # get counterfactual tokens
        if cf_input_ids is not None:
            cf_tokens = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in cf_input_ids.tolist()]
            cf_lengths = cf_mask.long().sum(-1).tolist()
        else:
            cf_tokens = ff_tokens
            cf_lengths = ff_lengths

        # output to be stacked across iterations
        output = {
            f"{prefix}_ff_sum_loss": ff_loss.item(),
            f"{prefix}_cf_sum_loss": cf_loss.item(),
            f"{prefix}_expl_sum_loss": expl_loss.item(),
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
            f"{prefix}_cf_z": cf_z,
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
