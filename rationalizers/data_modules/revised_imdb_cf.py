from functools import partial
from itertools import chain

import datasets as hf_datasets
import nltk
import torch
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
from torchnlp.utils import collate_tensors
from transformers import PreTrainedTokenizerBase

from rationalizers import constants
from rationalizers.data_modules.base import BaseDataModule


class CounterfactualRevisedIMDBDataModule(BaseDataModule):
    """DataModule for the Revised IMDB Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params)
        # hard-coded stuff
        self.path = "./rationalizers/custom_hf_datasets/revised_imdb_cf.py"
        self.is_multilabel = True
        self.nb_classes = 2

        # hyperparams
        self.batch_size = d_params.get("batch_size", 32)
        self.num_workers = d_params.get("num_workers", 0)
        self.vocab_min_occurrences = d_params.get("vocab_min_occurrences", 1)
        self.max_seq_len = d_params.get("max_seq_len", 99999999)
        self.is_original = d_params.get("is_original", None)

        # objects
        self.dataset = None
        self.label_encoder = None
        self.tokenizer = tokenizer
        self.tokenizer_cls = partial(
            StaticTokenizerEncoder,
            tokenize=nltk.wordpunct_tokenize,
            min_occurrences=self.vocab_min_occurrences,
            reserved_tokens=[
                constants.PAD,
                constants.UNK,
                constants.EOS,
                constants.SOS,
                "<copy>",
            ],
            padding_index=constants.PAD_ID,
            unknown_index=constants.UNK_ID,
            eos_index=constants.EOS_ID,
            sos_index=constants.SOS_ID,
            append_sos=False,
            append_eos=False,
        )

    def _collate_fn(self, samples: list, are_samples_batched: bool = False):
        """
        :param samples: a list of dicts
        :param are_samples_batched: in case a batch/bucket sampler are being used
        :return: dict of features, label (Tensor)
        """
        if are_samples_batched:
            # dataloader batch size is 1 -> the sampler is responsible for batching
            samples = samples[0]

        # convert list of dicts to dict of lists
        collated_samples = collate_tensors(samples, stack_tensors=list)

        # pad and stack input ids
        input_ids, lengths = stack_and_pad_tensors(
            collated_samples["input_ids"], padding_index=constants.PAD_ID
        )
        cf_input_ids, cf_lengths = stack_and_pad_tensors(
            collated_samples["cf_input_ids"], padding_index=constants.PAD_ID
        )
        # stack labels
        labels = collated_samples["label"]
        if isinstance(labels, list):
            labels = torch.stack(labels, dim=0)

        cf_labels = collated_samples["cf_label"]
        if isinstance(cf_labels, list):
            cf_labels = torch.stack(cf_labels, dim=0)

        # keep tokens in raw format
        tokens = collated_samples["tokens"]
        cf_tokens = collated_samples["cf_tokens"]

        # metadata
        batch_id = collated_samples["batch_id"]
        is_original = collated_samples["is_original"]

        # return batch to the data loader
        batch = {
            "input_ids": input_ids,
            "lengths": lengths,
            "labels": labels,
            "tokens": tokens,
            "cf_input_ids": cf_input_ids,
            "cf_lengths": cf_lengths,
            "cf_labels": cf_labels,
            "cf_tokens": cf_tokens,
            "batch_id": batch_id,
            "is_original": is_original,
        }
        return batch

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )

        # build tokenizer
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["tokens"],
                self.dataset["train"]["cf_tokens"],
                self.dataset["validation"]["tokens"],
                self.dataset["validation"]["cf_tokens"]
            )
            self.tokenizer = self.tokenizer_cls(tok_samples)

        # map strings to ids
        def _encode(example: dict):
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                example["input_ids"] = self.tokenizer(
                    example["tokens"].strip(),
                    padding=False,  # do not pad, padding will be done later
                    truncation=True,  # truncate to max length accepted by the model
                )["input_ids"]
                example["cf_input_ids"] = self.tokenizer(
                    example["cf_tokens"].strip(),
                    padding=False,  # do not pad, padding will be done later
                    truncation=True,  # truncate to max length accepted by the model
                )["input_ids"]
            else:
                example["input_ids"] = self.tokenizer.encode(example["tokens"].strip())
                example["cf_input_ids"] = self.tokenizer.encode(example["cf_tokens"].strip())
            return example

        self.dataset = self.dataset.map(_encode)
        self.dataset = self.dataset.filter(lambda example: len(example["input_ids"]) <= self.max_seq_len)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "label", "cf_input_ids", "cf_label", "batch_id", "is_original"],
            output_all_columns=True
        )
