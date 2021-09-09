import logging
import math

from datetime import datetime
import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out
from torchnlp.encoders.text import StaticTokenizerEncoder
from torch import nn
from rationalizers import constants
from rationalizers.builders import build_optimizer, build_scheduler

shell_logger = logging.getLogger(__name__)


class BaseMatching(pl.LightningModule):
    """Base module for matchings."""

    def __init__(
        self,
        tokenizer: StaticTokenizerEncoder,
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
        self.dataset = h_params.get("dm", None)

        # define metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
        self.train_precision = pl.metrics.Precision(
            num_classes=nb_classes, average="macro"
        )
        self.val_precision = pl.metrics.Precision(
            num_classes=nb_classes, average="macro"
        )
        self.test_precision = pl.metrics.Precision(
            num_classes=nb_classes, average="macro"
        )
        self.train_recall = pl.metrics.Recall(num_classes=nb_classes, average="macro")
        self.val_recall = pl.metrics.Recall(num_classes=nb_classes, average="macro")
        self.test_recall = pl.metrics.Recall(num_classes=nb_classes, average="macro")

        # define loss function
        criterion_cls = nn.MSELoss if not self.is_multilabel else nn.NLLLoss
        self.criterion = criterion_cls(reduction="none")

        # model arch:
        self.vocab_size = tokenizer.vocab_size
        self.emb_type = h_params.get("emb_type", "random")
        self.emb_path = h_params.get("emb_path", None)
        self.emb_size = h_params.get("emb_size", 300)
        self.emb_requires_grad = not h_params.get("embed_fixed", True)
        self.hidden_size = h_params.get("hidden_size", 150)
        self.dropout = h_params.get("dropout", 0.5)
        self.sentence_encoder_layer_type = h_params.get(
            "sentence_encoder_layer_type", "lstm"
        )

        self.predicting = h_params.get("predicting", False)

        # save hyperparams to be accessed via self.hparams
        self.save_hyperparameters(h_params)
        shell_logger.info(h_params)

    def forward(
        self,
        x1: torch.LongTensor,
        x2: torch.LongTensor,
        mask_x1: torch.BoolTensor = None,
        mask_x2: torch.BoolTensor = None,
    ):
        """
        Compute forward-pass.

        :param x1: input x1 ids tensor. torch.LongTensor of shape [B, T]
        :param x2: input x2 ids tensor. torch.LongTensor of shape [B, D]
        :param mask_x1: mask tensor for padding positions for the x1ise. torch.BoolTensor of shape [B, T]
        :param mask_x2: mask tensor for padding positions for the x2thesis. torch.BoolTensor of shape [B, D]

        :return: the output from SentimentPredictor. Torch.Tensor of shape [B, C]
        """
        z, y_hat = self.matching_model(x1, x2, mask=[mask_x1, mask_x2])
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
        prefix = "train"
        x1 = batch["x1_ids"]
        x2 = batch["x2_ids"]
        labels = batch["labels"]
        mask_x1 = x1 != constants.PAD_ID
        mask_x2 = x2 != constants.PAD_ID

        # forward-pass
        z, y_hat = self(x1, x2, mask_x1=mask_x1, mask_x2=mask_x2)

        # compute loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        loss, loss_stats = self.get_loss(
            y_hat, y, prefix=prefix, mask_x1=mask_x1, mask_x2=mask_x2
        )

        # logger=False because they are going to be logged via loss_stats
        self.log(
            "train_sum_loss",
            loss.item(),
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
        )

        self.logger.agg_and_log_metrics(loss_stats, self.global_step)

        # return the loss tensor to PTL
        return {"loss": loss}  # "ps": loss_stats[prefix + "_ps"]}

    def validation_step(self, batch: dict, batch_idx: int):
        output = self._shared_eval_step(batch, batch_idx, prefix="val")
        return output

    def test_step(self, batch: dict, batch_idx: int):
        output = self._shared_eval_step(batch, batch_idx, prefix="test")
        return output

    def _shared_eval_step(self, batch: dict, batch_idx: int, prefix: str):
        x1 = batch["x1_ids"]
        x2 = batch["x2_ids"]
        labels = batch["labels"]
        mask_x1 = x1 != constants.PAD_ID
        mask_x2 = x2 != constants.PAD_ID

        # forward-pass
        z, y_hat = self(x1, x2, mask_x1=mask_x1, mask_x2=mask_x2)

        # compute loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        loss, loss_stats = self.get_loss(
            y_hat, y, prefix=prefix, mask_x1=mask_x1, mask_x2=mask_x2
        )

        metrics_to_wandb = {f"{prefix}_loss": loss_stats[f"{prefix}_criterion"]}
        self.logger.log_metrics(metrics_to_wandb, self.global_step)

        # output to be stacked across iterations
        output = {
            f"{prefix}_sum_loss": loss.item(),
            f"{prefix}_predictions": y_hat,
            f"{prefix}_labels": batch["labels"].tolist(),
            f"{prefix}_tokens_x1": batch["x1"],
            f"{prefix}_tokens_x2": batch["x2"],
        }
        if prefix == "test":
            output["test_probs"] = z

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
            # f"avg_{prefix}_ps": np.mean(stacked_outputs[f"{prefix}_ps"]),
        }
        shell_logger.info(
            f"\nAvg {prefix} sum loss: {avg_outputs[f'avg_{prefix}_sum_loss']:.4}"
        )
        # shell_logger.info(f"Avg {prefix} ps: {avg_outputs[f'avg_{prefix}_ps']:.4}")

        dict_metrics = {f"avg_{prefix}_sum_loss": avg_outputs[f"avg_{prefix}_sum_loss"]}
        self.logger.agg_and_log_metrics(dict_metrics, self.current_epoch)

        # log classification metrics
        if self.is_multilabel:
            if prefix == "val":
                val_preds = torch.argmax(
                    torch.cat(stacked_outputs["val_predictions"]), dim=-1
                )
                val_labels = torch.tensor(
                    [
                        item
                        for sublist in stacked_outputs["val_labels"]
                        for item in sublist
                    ],
                    device=val_preds.device,
                )
                accuracy = self.val_accuracy(val_preds, val_labels)
                precision = self.val_precision(val_preds, val_labels)
                recall = self.val_recall(val_preds, val_labels)
                f1_score = 2 * precision * recall / (precision + recall)
                class_metrics = {
                    f"{prefix}_precision": precision,
                    f"{prefix}_recall": recall,
                    f"{prefix}_f1score": f1_score,
                    f"{prefix}_accuracy": accuracy,
                }

            else:
                test_preds = torch.argmax(
                    torch.cat(stacked_outputs["test_predictions"]), dim=-1
                )
                test_labels = torch.tensor(
                    [
                        item
                        for sublist in stacked_outputs["test_labels"]
                        for item in sublist
                    ],
                    device=test_preds.device,
                )
                accuracy = self.test_accuracy(test_preds, test_labels)
                precision = self.test_precision(test_preds, test_labels)
                recall = self.test_recall(test_preds, test_labels)
                f1_score = 2 * precision * recall / (precision + recall)
                class_metrics = {
                    f"{prefix}_precision": precision,
                    f"{prefix}_recall": recall,
                    f"{prefix}_f1score": f1_score,
                    f"{prefix}_accuracy": accuracy,
                }

            self.logger.log_metrics(class_metrics, step=None)
            shell_logger.info(f"{prefix} accuracy: {accuracy:.4}")
            shell_logger.info(f"{prefix} precision: {precision:.4}")
            shell_logger.info(f"{prefix} recall: {recall:.4}")
            shell_logger.info(f"{prefix} f1: {f1_score:.4}")

            self.log(
                f"{prefix}_f1score",
                f1_score,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

        self.log(
            f"avg_{prefix}_sum_loss",
            dict_metrics[f"avg_{prefix}_sum_loss"],
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        output = {
            f"avg_{prefix}_sum_loss": dict_metrics[f"avg_{prefix}_sum_loss"],
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_f1score": f1_score,
            f"{prefix}_accuracy": accuracy,
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

    def init_weights(self):
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

        for name, p in self.named_parameters():
            if name.startswith("emb") or "lagrange" in name:
                print("{:10s} {:20s} {}".format("unchanged", name, p.shape))
            elif "lstm" in name and len(p.shape) > 1:
                print("{:10s} {:20s} {}".format("xavier_n", name, p.shape))
                xavier_uniform_n_(p)
            elif len(p.shape) > 1:
                print("{:10s} {:20s} {}".format("xavier", name, p.shape))
                torch.nn.init.xavier_uniform_(p)
            elif "bias" in name:
                print("{:10s} {:20s} {}".format("zeros", name, p.shape))
                torch.nn.init.constant_(p, 0.0)
            else:
                print("{:10s} {:20s} {}".format("unchanged", name, p.shape))
