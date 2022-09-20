import logging
import os
from itertools import chain
from random import shuffle

import numpy as np
import torch
import torchmetrics
import wandb
from torch import nn
from transformers import AutoModel

from rationalizers import constants
from rationalizers.builders import build_optimizer, build_scheduler
from rationalizers.explainers import available_explainers
from rationalizers.lightning_models.highlights.base import BaseRationalizer
from rationalizers.modules.metrics import evaluate_rationale
from rationalizers.modules.sentence_encoders import LSTMEncoder, MaskedAverageEncoder
from rationalizers.utils import (
    get_z_stats, freeze_module, masked_average, get_rationales, unroll, get_html_rationales,
    save_rationales, is_trainable, get_ext_mask
)

shell_logger = logging.getLogger(__name__)


class TransformerRationalizer(BaseRationalizer):

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

        # manual update constants
        constants.update_constants(self.tokenizer)

        ########################
        # hyperparams
        ########################
        # factual:
        self.ff_gen_arch = h_params.get("gen_arch", "bert-base-multilingual-cased")
        self.ff_gen_emb_requires_grad = h_params.get("gen_emb_requires_grad", False)
        self.ff_gen_encoder_requires_grad = h_params.get("gen_encoder_requires_grad", True)
        self.ff_gen_use_decoder = h_params.get("gen_use_encoder", False)
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

        # explainer:
        self.explainer_fn = h_params.get("explainer", True)
        self.explainer_pre_mlp = h_params.get("explainer_pre_mlp", True)
        self.explainer_requires_grad = h_params.get("explainer_requires_grad", True)
        self.temperature = h_params.get("temperature", 1.0)

        # both
        self.share_generators = h_params.get("share_generators", False)

        ########################
        # useful vars
        ########################
        self.ff_z = None
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

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
        explainer_cls = available_explainers[self.explainer_fn]
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
            self.ff_pred_hf = AutoModel.from_pretrained(self.ff_pred_arch, dropout=self.dropout)
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
        # weights details
        ########################
        # initialize params using xavier initialization for weights and zero for biases
        # (weights of these modules might be loaded later)
        self.init_weights(self.explainer_mlp)
        self.init_weights(self.explainer)
        self.init_weights(self.ff_output_layer)

        # freeze embedding layers
        if not self.ff_gen_emb_requires_grad:
            freeze_module(self.ff_gen_emb_layer)
        if not self.ff_pred_emb_requires_grad:
            freeze_module(self.ff_pred_emb_layer)

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

        ########################
        # loss functions
        ########################
        criterion_cls = nn.MSELoss if not self.is_multilabel else nn.NLLLoss
        self.ff_criterion = criterion_cls(reduction="none")

        ########################
        # metrics
        ########################
        self.ff_train_accuracy = torchmetrics.Accuracy()
        self.ff_val_accuracy = torchmetrics.Accuracy()
        self.ff_test_accuracy = torchmetrics.Accuracy()
        self.ff_train_precision = torchmetrics.Precision(num_classes=nb_classes, average="macro")
        self.ff_val_precision = torchmetrics.Precision(num_classes=nb_classes, average="macro")
        self.ff_test_precision = torchmetrics.Precision(num_classes=nb_classes, average="macro")
        self.ff_train_recall = torchmetrics.Recall(num_classes=nb_classes, average="macro")
        self.ff_val_recall = torchmetrics.Recall(num_classes=nb_classes, average="macro")
        self.ff_test_recall = torchmetrics.Recall(num_classes=nb_classes, average="macro")

        # remove leftover from the base class
        del self.train_accuracy, self.val_accuracy, self.test_accuracy
        del self.train_precision, self.val_precision, self.test_precision
        del self.train_recall, self.val_recall, self.test_recall

        ########################
        # logging
        ########################
        # manual check requires_grad for all modules
        for name, module in self.named_children():
            shell_logger.info('is_trainable({}): {}'.format(name, is_trainable(module)))

    def configure_optimizers(self):
        """Configure optimizers and lr schedulers for Trainer."""
        ff_params = chain(
            self.ff_gen_emb_layer.parameters(),
            self.ff_gen_encoder.parameters(),
            self.ff_pred_emb_layer.parameters() if not self.ff_shared_gen_pred else [],
            self.ff_pred_encoder.parameters() if not self.ff_shared_gen_pred else [],
            self.ff_output_layer.parameters(),
            self.explainer_mlp.parameters(),
            self.explainer.parameters(),
        )
        parameters = [{"params": ff_params, 'lr': self.hparams['lr']}]
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
        current_epoch=None,
    ):
        """
        Compute forward-pass.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param current_epoch: int represents the current epoch.
        :return: (z, y_hat), (x_tilde, z_tilde, mask_tilde, y_tilde_hat)
        """
        # factual flow
        z, y_hat = self.get_factual_flow(x, mask=mask)
        # return everything as output (useful for computing the loss)
        return z, y_hat

    def get_factual_flow(self, x, mask=None, z=None):
        """
        Compute the factual flow.

        :param x: input ids tensor. torch.LongTensor of shape [B, T] or input vectors of shape [B, T, |V|]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param z: precomputed latent vector. torch.FloatTensor of shape [B, T] (default None)
        :return: z, y_hat
        """
        # receive input ids
        gen_e = self.ff_gen_emb_layer(x)

        # pass though the generator encoder
        if 't5' in self.ff_gen_arch:
            # t5 encoder and decoder receives word_emb and a raw mask
            gen_h = self.ff_gen_encoder(
                inputs_embeds=gen_e,
                attention_mask=mask
            )
            if self.ff_gen_use_decoder:
                gen_h = self.ff_gen_decoder(
                    inputs_embeds=gen_e,
                    attention_mask=mask,
                    encoder_hidden_states=gen_h.last_hidden_state,
                )
        else:
            # bert encoder receives word_emb + pos_embs and an extended mask
            ext_mask = get_ext_mask(mask)
            gen_h = self.ff_gen_encoder(
                hidden_states=gen_e,
                attention_mask=ext_mask
            )

        # get final hidden states
        # selected_layers = list(map(int, self.selected_layers.split(',')))
        # gen_h = torch.stack(gen_h)[selected_layers].mean(dim=0)
        gen_h = gen_h.last_hidden_state

        # pass through the explainer
        gen_h = self.explainer_mlp(gen_h) if self.explainer_pre_mlp else gen_h
        z = self.explainer(gen_h, mask) if z is None else z
        z_mask = (z * mask.float()).unsqueeze(-1)

        # decide if we pass embeddings or hidden states to the predictor
        if self.ff_selection_faithfulness is True:
            # receive input ids
            pred_e = self.ff_pred_emb_layer(x)
        else:
            pred_e = gen_h

        # decide if we will use a <mask>/<pad> vector or a <zero> vector
        if self.ff_selection_vector == 'mask':
            if 't5' in self.gen_arch:
                # create an input with sentinel tokens for T5
                sentinel_ids = 32100 - (z > 0).long().cumsum(dim=-1)
                x_mask = torch.clamp(sentinel_ids, min=32000, max=32099)
            else:
                # create an input with full mask tokens for other archs
                x_mask = torch.ones(x.shape[:2], device=x.device) * self.mask_token_id
            pred_e_mask = self.ff_pred_emb_layer(x_mask)
        elif self.ff_selection_vector == 'pad':
            x_mask = torch.ones(x.shape[:2], device=x.device) * self.pad_token_id
            pred_e_mask = self.ff_pred_emb_layer(x_mask)
        else:
            pred_e_mask = torch.zeros_like(pred_e)

        # create the input that will be passed to the predictor
        pred_e = pred_e * z_mask + pred_e_mask * (1 - z_mask)

        # pass through the predictor
        if self.ff_pred_arch == 'lstm':
            _, summary = self.ff_pred_encoder(pred_e, mask)
        elif self.ff_pred_arch == 'masked_average':
            summary = self.ff_pred_encoder(pred_e, mask)
        else:
            if 't5' in self.ff_pred_arch:
                # decide if we will mask the self-attention of the predictor
                attn_mask = (1 - z_mask.squeeze(-1)).long() if self.ff_selection_mask else mask
                pred_h = self.ff_pred_encoder(
                    inputs_embeds=pred_e,
                    attention_mask=attn_mask
                )
                if self.ff_gen_use_decoder:
                    pred_h = self.ff_pred_decoder(
                        inputs_embeds=pred_e,
                        attention_mask=attn_mask,
                        encoder_hidden_states=pred_h.last_hidden_state,
                    )
            else:
                # decide if we will mask the self-attention of the predictor
                ext_mask = get_ext_mask(1 - z_mask.squeeze(-1)) if self.ff_selection_mask else get_ext_mask(mask)
                pred_h = self.ff_pred_encoder(
                    hidden_states=pred_e,
                    attention_mask=ext_mask
                )

            # get final hidden states
            # selected_layers = list(map(int, self.selected_layers.split(',')))
            # pred_h = torch.stack(pred_h)[selected_layers].mean(dim=0)
            pred_h = pred_h.last_hidden_state

            # compute a sentence vector
            summary = masked_average(pred_h, mask)

        # pass through the final mlp
        y_hat = self.ff_output_layer(summary)

        return z, y_hat

    def get_factual_loss(self, y_hat, y, z, mask, prefix):
        """
        Compute loss for the factual flow.

        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param z: latent selection vector. torch.FloatTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :param prefix: prefix for loss statistics (train, val, test)
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        stats = {}
        loss_vec = self.ff_criterion(y_hat, y)  # [B] or [B,C]
        # main MSE loss for p(y | x,z)
        if not self.is_multilabel:
            loss = loss_vec.mean(0)  # [B,C] -> [B]
            stats["mse"] = loss.item()
        else:
            loss = loss_vec.mean()  # [1]
            stats["criterion"] = loss.item()  # [1]

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(z, mask)
        stats[prefix + "_p0"] = num_0 / float(total)
        stats[prefix + "_pc"] = num_c / float(total)
        stats[prefix + "_p1"] = num_1 / float(total)
        stats[prefix + "_ps"] = (num_c + num_1) / float(total)
        stats[prefix + "_main_loss"] = loss.item()

        return loss, stats

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
        mask = input_ids != constants.PAD_ID
        labels = batch["labels"]
        prefix = "train"

        z, y_hat = self(input_ids, mask, current_epoch=self.current_epoch)

        # compute factual loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        loss, loss_stats = self.get_factual_loss(y_hat, y, z, mask, prefix=prefix)

        # logger=False because they are going to be logged via loss_stats
        self.log("train_ff_ps", loss_stats["train_ps"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("train_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        if self.is_multilabel:
            metrics_to_wandb = {
                "train_ff_p1": loss_stats["train_p1"],
                "train_ff_ps": loss_stats["train_ps"],
                "train_ff_sum_loss": loss_stats["criterion"],
            }
        else:
            metrics_to_wandb = {
                "train_ff_p1": loss_stats["train_p1"],
                "train_ff_ps": loss_stats["train_ps"],
                "train_ff_sum_loss": loss_stats["mse"],
            }
        if "cost_g" in loss_stats:
            metrics_to_wandb["train_cost_g"] = loss_stats["cost_g"]

        self.logger.log_metrics(metrics_to_wandb, self.global_step)

        # return the loss tensor to PTL
        return {"loss": loss, "ps": loss_stats["train_ps"]}

    def _shared_eval_step(self, batch: dict, batch_idx: int, prefix: str):
        input_ids = batch["input_ids"]
        mask = input_ids != constants.PAD_ID
        labels = batch["labels"]

        # forward-pass
        z, y_hat = self(input_ids, mask, current_epoch=self.current_epoch)

        # compute factual loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        loss, ff_loss_stats = self.get_factual_loss(y_hat, y, z, mask, prefix=prefix)
        self.logger.agg_and_log_metrics(ff_loss_stats, step=None)

        # log metrics
        self.log(f"{prefix}_sum_loss", loss.item(), prog_bar=True, logger=False, on_step=True, on_epoch=False,)

        # get factual rationales
        z_1 = (z > 0).long()  # non-zero probs are considered selections
        ids_rationales, rationales = get_rationales(self.tokenizer, input_ids, z_1, batch["lengths"])
        pieces = [self.tokenizer.convert_ids_to_tokens(idxs) for idxs in input_ids.tolist()]

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
        idxs = list(range(sum(map(len, stacked_outputs[f"{prefix}_pieces"]))))
        if prefix != 'test':
            shuffle(idxs)
            idxs = idxs[:10]
        else:
            shuffle(idxs)
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
            preds = torch.argmax(torch.cat(stacked_outputs[f"{prefix}_predictions"]), dim=-1)
            labels = torch.tensor(unroll(stacked_outputs[f"{prefix}_labels"]), device=preds.device)
            if prefix == "val":
                ff_accuracy = self.ff_val_accuracy(preds, labels)
                ff_precision = self.ff_val_precision(preds, labels)
                ff_recall = self.ff_val_recall(preds, labels)
                ff_f1_score = 2 * ff_precision * ff_recall / (ff_precision + ff_recall)
            else:
                ff_accuracy = self.ff_test_accuracy(preds, labels)
                ff_precision = self.ff_test_precision(preds, labels)
                ff_recall = self.ff_test_recall(preds, labels)
                ff_f1_score = 2 * ff_precision * ff_recall / (ff_precision + ff_recall)

            dict_metrics[f"{prefix}_ff_accuracy"] = ff_accuracy
            dict_metrics[f"{prefix}_ff_precision"] = ff_precision
            dict_metrics[f"{prefix}_ff_recall"] = ff_recall
            dict_metrics[f"{prefix}_ff_f1score"] = ff_f1_score

            shell_logger.info(f"{prefix} ff_acc: {ff_accuracy:.4}")
            shell_logger.info(f"{prefix} ff_f1: {ff_f1_score:.4}")

            self.log(
                f"{prefix}_ff_accuracy",
                dict_metrics[f"{prefix}_ff_accuracy"],
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{prefix}_ff_f1score",
                dict_metrics[f"{prefix}_ff_f1score"],
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
                f"{prefix}_ff_accuracy": ff_accuracy,
                f"{prefix}_ff_precision": ff_precision,
                f"{prefix}_ff_recall": ff_recall,
                f"{prefix}_ff_f1score": ff_f1_score,
            }
        else:
            output = {
                f"avg_{prefix}_sum_loss": dict_metrics[f"avg_{prefix}_sum_loss"],
                f"avg_{prefix}_ps": dict_metrics[f"avg_{prefix}_ps"],
                f"avg_{prefix}_MSE": dict_metrics[f"avg_{prefix}_mse"],
            }

        return output
