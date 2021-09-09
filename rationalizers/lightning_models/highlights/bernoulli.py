import logging
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchnlp.encoders.text import StaticTokenizerEncoder

from rationalizers import constants
from rationalizers.builders import build_embedding_weights
from rationalizers.lightning_models.highlights.base import BaseRationalizer
from rationalizers.modules.generators import (
    BernoulliIndependentGenerator,
)
from rationalizers.modules.metrics import evaluate_rationale
from rationalizers.modules.predictors import SentimentPredictor
from rationalizers.utils import get_z_stats, get_rationales

shell_logger = logging.getLogger(__name__)


class BernoulliRationalizer(BaseRationalizer):
    """Rationalizer that uses latent Bernoulli variables to get sparse and contiguous selections."""

    def __init__(
        self,
        tokenizer: StaticTokenizerEncoder,
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

        # model arch:
        self.vocab_size = tokenizer.vocab_size
        self.emb_type = h_params.get("emb_type", "random")
        self.emb_path = h_params.get("emb_path", None)
        self.emb_size = h_params.get("emb_size", 300)
        self.emb_requires_grad = not h_params.get("embed_fixed", True)
        self.hidden_size = h_params.get("hidden_size", 150)
        self.dropout = h_params.get("dropout", 0.5)
        self.sentence_encoder_layer_type = h_params.get(
            "sentence_encoder_layer_type", "rcnn"
        )
        self.contiguous = h_params.get("contiguous", False)
        self.topk = h_params.get("topk", False)
        self.relaxed = h_params.get("relaxed", False)
        self.budget = h_params.get("budget", 10)

        # loss fn:
        self.lambda_0 = h_params.get("lambda_0", 0.0)
        self.lambda_1 = h_params.get("lambda_1", 0.0)
        self.baseline = h_params.get("baseline", False)

        if self.baseline:
            self.mean_baseline = 0
            self.n_points = 0

        # global steps for board loggers
        self.eval_global_step = {"val": 0, "test": 0}

        # save hyperparams to checkpoint
        self.save_hyperparameters(h_params)

        # define metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
        self.train_precision = pl.metrics.Precision(
            num_classes=nb_classes,
            average="macro",
        )
        self.val_precision = pl.metrics.Precision(
            num_classes=nb_classes,
            average="macro",
        )
        self.test_precision = pl.metrics.Precision(
            num_classes=nb_classes,
            average="macro",
        )
        self.train_recall = pl.metrics.Recall(
            num_classes=nb_classes,
            average="macro",
        )
        self.val_recall = pl.metrics.Recall(
            num_classes=nb_classes,
            average="macro",
        )
        self.test_recall = pl.metrics.Recall(
            num_classes=nb_classes,
            average="macro",
        )

        # load word embedding weights based on `emb_type` and define the embedding layer
        embedding_weights = build_embedding_weights(
            self.tokenizer.vocab, self.emb_type, self.emb_path, self.emb_size
        )
        self.emb_layer = nn.Embedding(
            self.vocab_size,
            self.emb_size,
            padding_idx=constants.PAD_ID,
            _weight=embedding_weights,
        )
        self.emb_layer.weight.requires_grad = self.emb_requires_grad

        # create generator        
        self.generator = BernoulliIndependentGenerator(
            embed=self.emb_layer,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            layer=self.sentence_encoder_layer_type,
            budget=self.budget,
            contiguous=self.contiguous,
            topk=self.topk,
            relaxed=self.relaxed,
        )

        # create predictor
        nonlinearity_str = "sigmoid" if not self.is_multilabel else "log_softmax"
        self.predictor = SentimentPredictor(
            embed=self.emb_layer,
            hidden_size=self.hidden_size,
            output_size=self.nb_classes,
            dropout=self.dropout,
            layer=self.sentence_encoder_layer_type,
            nonlinearity=nonlinearity_str,
        )

        # initialize params using xavier initialization for weights and zero for biases
        self.init_weights()

    def get_loss(self, y_hat, y, mask=None, baseline=False):
        """
        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        stats = {}

        loss_vec = self.criterion(y_hat, y)  # [B] or [B,C]

        # main MSE loss for p(y | x,z)
        if not self.is_multilabel:
            loss_vec = loss_vec.mean(1)  # [B,C] -> [B]
        loss = loss_vec.mean()  # [1]
        if not self.is_multilabel:
            stats["mse"] = loss.item()  # [1]

        coherent_factor = self.lambda_1
        sparsity_factor = self.lambda_0

        # compute generator loss
        z = self.generator.z.squeeze()  # [B, T]

        # get P(z = 0 | x) and P(z = 1 | x)
        m = self.generator.z_dists[0]
        logp_z0 = m.log_prob(0.0).squeeze(2)  # [B,T], log P(z = 0 | x)
        logp_z1 = m.log_prob(1.0).squeeze(2)  # [B,T], log P(z = 1 | x)
       

        # compute log p(z|x) for each case (z==0 and z==1) and mask
        logpz = torch.where(z == 0, logp_z0, logp_z1)
        logpz = torch.where(mask, logpz, logpz.new_zeros([1]))

        z = z.view(loss_vec.shape[0], -1)

        # sparsity regularization
        zsum = z.sum(1)
        zdiff = z[:, 1:] - z[:, :-1]
        zdiff = zdiff.abs().sum(1)  # [B]

        zsum_cost = sparsity_factor * zsum.mean(0)
        stats["zsum_cost"] = zsum_cost.item()

        zdiff_cost = coherent_factor * zdiff.mean(0)
        stats["zdiff_cost"] = zdiff_cost.mean().item()

        sparsity_cost = zsum_cost + zdiff_cost
        stats["sparsity_cost"] = sparsity_cost.item()

        cost_vec = loss_vec.detach() + zsum * sparsity_factor + zdiff * coherent_factor

        if self.baseline:
            cost_logpz = ((cost_vec - self.mean_baseline) * logpz.sum(1)).mean(0)  # cost_vec is neg reward
        else:
            cost_logpz = (cost_vec * logpz.sum(1)).mean(0)  # cost_vec is neg reward
        obj = cost_vec.mean()  # MSE with regularizers = neg reward
        stats["obj"] = obj.item()

        if self.baseline:
            self.n_points += 1.0
            self.mean_baseline += (cost_vec.detach().mean() - self.mean_baseline) / self.n_points

        # pred diff doesn't do anything if only 1 aspect being trained
        if not self.is_multilabel:
            pred_diff = y_hat.max(dim=1)[0] - y_hat.min(dim=1)[0]
            pred_diff = pred_diff.mean()
            stats["pred_diff"] = pred_diff.item()

        # generator cost
        stats["cost_g"] = cost_logpz.item()

        # predictor cost
        stats["cost_p"] = loss.item()

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(self.generator.z, mask)
        stats["p0"] = num_0 / float(total)
        stats["pc"] = num_c / float(total)
        stats["p1"] = num_1 / float(total)
        stats["selected"] = num_1
        stats["total"] = float(total)

        main_loss = loss + cost_logpz
        stats["main_loss"] = main_loss.item()
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
        labels = batch["labels"]
        mask = input_ids != constants.PAD_ID

        # forward-pass
        z, y_hat = self(input_ids, mask=mask)
        # compute loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        loss, loss_stats = self.get_loss(y_hat, y, mask=mask)

        # logger=False because they are going to be logged via loss_stats
        self.log(
            "p1",
            loss_stats["p1"],
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=False,
        )
        self.log(
            "train_sum_loss",
            loss.item(),
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=False,
        )
        self.log(
            "train_obj_loss",
            loss_stats["obj"],
            prog_bar=True,
            logger=False,
            on_step=False,
            on_epoch=False,
        )

        # log directly to tensorboard using self.logger.experiment
        # self.logger.experiment.add_scalar(
        # "train_batch_number", batch_idx, self.global_step
        # )
        # for metric, val in loss_stats.items():
        # self.logger.experiment.add_scalar(f"train_{metric}", val, self.global_step)

        # compute metrics for this step
        if not self.is_multilabel:
            y = (y >= 0.5).long()

        # return the loss tensor to PTL
        return {"loss": loss, "obj": loss_stats["obj"], "p1": loss_stats["p1"]}

    def validation_step(self, batch: dict, batch_idx: int):
        output = self._shared_eval_step(batch, batch_idx, prefix="val")
        return output

    def test_step(self, batch: dict, batch_idx: int):
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
        loss, loss_stats = self.get_loss(y_hat, y, mask=mask)

        # log stats
        self.log(
            f"{prefix}_sum_loss",
            loss.item(),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{prefix}_obj_loss",
            loss_stats["obj"],
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        # log rationales
        ids_rationales, rationales = get_rationales(
            self.tokenizer, input_ids, z, batch["lengths"]
        )

        # compute metrics for this step
        if not self.is_multilabel:
            y = (y >= 0.5).long()

        # output to be stacked across iterations
        output = {
            f"{prefix}_sum_loss": loss.item(),
            f"{prefix}_obj_loss": loss_stats["obj"],
            f"{prefix}_p1": loss_stats["p1"],
            f"{prefix}_ids_rationales": ids_rationales,
            f"{prefix}_rationales": rationales,
            f"{prefix}_predictions": y_hat,
            f"{prefix}_tokens": batch["tokens"],
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
        print("\n Epoch Ended. \n")

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
            f"avg_{prefix}_obj_loss": np.mean(stacked_outputs[f"{prefix}_obj_loss"]),
            f"avg_{prefix}_p1": np.mean(stacked_outputs[f"{prefix}_p1"]),
        }


        shell_logger.info(
            f"Avg {prefix} sum loss: {avg_outputs[f'avg_{prefix}_sum_loss']:.4}"
        )
        shell_logger.info(
            f"Avg {prefix} obj loss: {avg_outputs[f'avg_{prefix}_obj_loss']:.4}"
        )
        shell_logger.info(f"Avg {prefix} p1: {avg_outputs[f'avg_{prefix}_p1']:.4}")

        dict_metrics = {
            f"avg_{prefix}_p1": avg_outputs[f"avg_{prefix}_p1"],
            f"avg_{prefix}_sum_loss": avg_outputs[f"avg_{prefix}_sum_loss"],
        }

        if not self.is_multilabel:
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

        # only evaluate rationales on the test set and if we have annotation (only for beer dataset)
        if prefix == "test" and "test_annotations" in stacked_outputs.keys():
            metrics = evaluate_rationale(
                stacked_outputs["test_ids_rationales"],
                stacked_outputs["test_annotations"],
                stacked_outputs["test_lengths"],
            )

            shell_logger.info(
                f"Rationales macro precision: {metrics[f'macro_precision']:.4}"
            )
            shell_logger.info(f"Rationales macro recall: {metrics[f'macro_recall']:.4}")
            shell_logger.info(f"Rationales macro f1: {metrics[f'f1_score']:.4}")

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

        if self.is_multilabel:
            output = {
                f"avg_{prefix}_sum_loss": dict_metrics[f"avg_{prefix}_sum_loss"],
                f"avg_{prefix}_p1": dict_metrics[f"avg_{prefix}_p1"],
                f"{prefix}_precision": precision,
                f"{prefix}_recall": recall,
                f"{prefix}_f1score": f1_score,
                f"{prefix}_accuracy": accuracy,
            }
        else:
            output = {
                f"avg_{prefix}_sum_loss": dict_metrics[f"avg_{prefix}_sum_loss"],
                f"avg_{prefix}_p1": dict_metrics[f"avg_{prefix}_p1"],
                f"avg_{prefix}_MSE": dict_metrics[f"avg_{prefix}_mse"],
            }
