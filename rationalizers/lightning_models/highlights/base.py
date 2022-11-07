import logging
import math
import os
from pprint import pformat

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch import nn
import torchmetrics

from rationalizers import constants
from rationalizers.builders import build_optimizer, build_scheduler
from rationalizers.modules.metrics import evaluate_rationale
from rationalizers.utils import get_rationales, get_html_rationales, unroll, save_rationales

shell_logger = logging.getLogger(__name__)


class BaseRationalizer(pl.LightningModule):
    """Base module for rationalizers."""

    def __init__(
        self,
        tokenizer: object,
        nb_classes: int,
        is_multilabel: bool,
        h_params: dict,
    ):
        """
        :param tokenizer (object): torchnlp tokenizer object
        :param h_params (dict): hyperparams dict. See docs for more info.
        :param nb_classes (int): number of classes used to create the last layer
        :param multilabel (bool): whether the problem is multilabel or not (it depends on the dataset)
        """
        super().__init__()
        # the tokenizer will be used to convert indices to strings
        self.tokenizer = tokenizer
        # to be used at the output layer
        self.nb_classes = nb_classes
        self.is_multilabel = is_multilabel
        # whether we should log rationales in wandb
        self.log_rationales_in_wandb = True
        # prefix is used to know whether we are training, testing or validating
        self.stage = None
        # define vocab size
        self.vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
        # save hyperparams to be accessed via self.hparams
        self.save_hyperparameters(h_params)
        shell_logger.info(pformat(h_params))

    def forward(
        self, x: torch.LongTensor, current_epoch=None, mask: torch.BoolTensor = None
    ):
        """
        Compute forward-pass.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: the output from SentimentPredictor. Torch.Tensor of shape [B, C]
        """
        z = self.generator(x, mask=mask)
        y_hat = self.predictor(x, z, mask=mask)
        return z, y_hat

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
        labels = batch["labels"]
        mask = input_ids != constants.PAD_ID
        prefix = "train"

        # forward-pass
        z, y_hat = self(input_ids, mask=mask, current_epoch=self.current_epoch)

        # compute loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        loss, loss_stats = self.get_loss(y_hat, y, prefix=prefix, mask=mask)

        # logger=False because they are going to be logged via loss_stats
        self.log(
            "ps",
            loss_stats[prefix + "_ps"],
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train_sum_loss",
            loss.item(),
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
        )

        if self.is_multilabel:
            metrics_to_wandb = {
                "train_ps": loss_stats["train_ps"],
                "train_loss": loss_stats["criterion"],
            }
        else:
            metrics_to_wandb = {
                "train_ps": loss_stats["train_ps"],
                "train_loss": loss_stats["mse"],
            }

        self.logger.log_metrics(metrics_to_wandb, self.global_step)

        # return the loss tensor to PTL
        return {"loss": loss, "ps": loss_stats[prefix + "_ps"]}

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = None):
        self.stage = "predict"
        output = self._shared_eval_step(batch, batch_idx, prefix="test")
        return output

    def validation_step(self, batch: dict, batch_idx: int):
        self.stage = "val"
        output = self._shared_eval_step(batch, batch_idx, prefix="val")
        return output

    def test_step(self, batch: dict, batch_idx: int):
        self.stage = "test"
        output = self._shared_eval_step(batch, batch_idx, prefix="test")
        return output

    def _shared_eval_step(self, batch: dict, batch_idx: int, prefix: str):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        mask = input_ids != constants.PAD_ID
        # forward-pass
        z, y_hat = self(input_ids, mask=mask)

        # compute loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        loss, loss_stats = self.get_loss(y_hat, y, prefix=prefix, mask=mask)

        self.logger.agg_and_log_metrics(loss_stats, step=None)

        z_1 = (z > 0).long()  # non-zero probs are considered selections for sparsemap
        ids_rationales, rationales = get_rationales(
            self.tokenizer, input_ids, z_1, batch["lengths"]
        )
        if hasattr(self.tokenizer, 'convert_ids_to_tokens'):
            pieces = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in input_ids.tolist()]
        else:
            pieces = [[self.tokenizer.index_to_token[i] for i in idxs] for idxs in input_ids.tolist()]

        self.log(
            f"{prefix}_sum_loss",
            loss.item(),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        # output to be stacked across iterations
        output = {
            f"{prefix}_sum_loss": loss.item(),
            f"{prefix}_ps": loss_stats[prefix + "_ps"],
            f"{prefix}_ids_rationales": ids_rationales,
            f"{prefix}_rationales": rationales,
            f"{prefix}_pieces": pieces,
            f"{prefix}_tokens": batch["tokens"],
            f"{prefix}_z": z,
            f"{prefix}_predictions": y_hat,
            f"{prefix}_labels": batch["labels"].tolist(),
            f"{prefix}_lengths": batch["lengths"].tolist(),
        }

        if "annotations" in batch.keys():
            output[f"{prefix}_annotations"] = batch["annotations"]

        if "mse" in loss_stats.keys():
            output[f"{prefix}_mse"] = loss_stats["mse"]

        return output

    def training_epoch_end(self, outputs: list):
        """
        PTL hook.

        :param outputs: list of dicts representing the stacked outputs from training_step
        """
        print("\nEpoch ended.\n")

    def validation_epoch_end(self, outputs: list):
        self._shared_eval_epoch_end(outputs, prefix="val")

    def test_epoch_end(self, outputs: list):
        self._shared_eval_epoch_end(outputs, prefix="test")

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
            preds = torch.argmax(
                torch.cat(stacked_outputs[f"{prefix}_predictions"]), dim=-1
            )
            labels = torch.tensor(
                [
                    item
                    for sublist in stacked_outputs[f"{prefix}_labels"]
                    for item in sublist
                ],
                device=preds.device,
            )
            if prefix == "val":
                accuracy = self.val_accuracy(preds, labels)
                precision = self.val_precision(preds, labels)
                recall = self.val_recall(preds, labels)
                f1_score = 2 * precision * recall / (precision + recall)
            else:
                accuracy = self.test_accuracy(preds, labels)
                precision = self.test_precision(preds, labels)
                recall = self.test_recall(preds, labels)
                f1_score = 2 * precision * recall / (precision + recall)

            dict_metrics[f"{prefix}_precision"] = precision
            dict_metrics[f"{prefix}_recall"] = recall
            dict_metrics[f"{prefix}_f1score"] = f1_score
            dict_metrics[f"{prefix}_accuracy"] = accuracy

            shell_logger.info(f"{prefix} accuracy: {accuracy:.4}")
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
            }
        else:
            output = {
                f"avg_{prefix}_sum_loss": dict_metrics[f"avg_{prefix}_sum_loss"],
                f"avg_{prefix}_ps": dict_metrics[f"avg_{prefix}_ps"],
                f"avg_{prefix}_MSE": dict_metrics[f"avg_{prefix}_mse"],
            }

        return output

    def configure_optimizers(self):
        """Configure optimizers and lr schedulers for Trainer."""
        optimizer = build_optimizer(self.parameters(), self.hparams)
        scheduler = build_scheduler(optimizer, self.hparams)
        output = {"optimizer": optimizer}
        if scheduler is not None:
            output["scheduler"] = scheduler
            # output["monitor"] = self.criterion  # not sure we need this
        return output

    def init_weights(self, module=None):
        """
        Model initialization.
        """

        def xavier_uniform_n_(w, gain=1.0, n=4):
            """
            Xavier initializer for parameters that combine multiple matrices in one
            parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
            where e.g. all gates are computed at the same time by 1 big matrix.
            :param w:
            :param gain:
            :param n:
            :return:
            """
            with torch.no_grad():
                fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
                assert fan_out % n == 0, "fan_out should be divisible by n"
                fan_out = fan_out // n
                std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                a = math.sqrt(3.0) * std
                torch.nn.init.uniform_(w, -a, a)
        named_params = module.named_parameters() if module is not None else self.named_parameters()
        for name, p in named_params:
            if name.startswith("emb") or "lagrange" in name:
                shell_logger.info("{:10s} {:20s} {}".format("unchanged", name, p.shape))
            elif "lstm" in name and len(p.shape) > 1:
                shell_logger.info("{:10s} {:20s} {}".format("xavier_n", name, p.shape))
                xavier_uniform_n_(p)
            elif len(p.shape) > 1:
                shell_logger.info("{:10s} {:20s} {}".format("xavier", name, p.shape))
                torch.nn.init.xavier_uniform_(p)
            elif "bias" in name:
                shell_logger.info("{:10s} {:20s} {}".format("zeros", name, p.shape))
                torch.nn.init.constant_(p, 0.0)
            else:
                shell_logger.info("{:10s} {:20s} {}".format("unchanged", name, p.shape))
