from functools import partial
from itertools import chain

import datasets as hf_datasets
import torch
from torchnlp.encoders.text import WhitespaceEncoder, stack_and_pad_tensors, pad_tensor
from torchnlp.utils import collate_tensors
from transformers import PreTrainedTokenizerBase

from rationalizers import constants
from rationalizers.data_modules.base import BaseDataModule


class BeerDataModule(BaseDataModule):
    """DataModule for BeerAdvocate Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params)
        # hard-coded stuff
        self.path = "./rationalizers/custom_hf_datasets/beer.py"

        # hyperparams
        self.aspect_subset = d_params.get("aspect_subset", "aspect0")
        self.batch_size = d_params.get("batch_size", 32)
        self.num_workers = d_params.get("num_workers", 0)
        self.vocab_min_occurrences = d_params.get("vocab_min_occurrences", 1)
        self.max_seq_len = d_params.get("max_seq_len", 99999999)
        self.transform_to_multiclass = d_params.get("transform_to_multiclass", False)
        self.is_multilabel = self.transform_to_multiclass

        # deal with single aspect experiments
        self.nb_classes = 5 if self.aspect_subset == "260k" else 1
        self.aspect_id = (
            -1 if self.aspect_subset == "260k" else int(self.aspect_subset[-1])
        )

        # objects
        self.dataset = None
        self.label_encoder = None  # no label encoder for this dataset
        self.tokenizer = tokenizer
        self.tokenizer_cls = partial(
            WhitespaceEncoder,  # Beer dataset was already tokenized by Lei et al.
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

        # allowed_samples = []
        # for sample in samples:
        #     if len(sample["input_ids"]) < 257:
        #         allowed_samples.append(sample)

        # samples = allowed_samples

        # convert list of dicts to dict of lists
        collated_samples = collate_tensors(samples, stack_tensors=list)

        # pad and stack input ids
        def pad_and_stack_ids(x):
            x_ids, x_lengths = stack_and_pad_tensors(x, padding_index=constants.PAD_ID)
            return x_ids, x_lengths

        def stack_labels(y):
            if isinstance(y, list):
                return torch.stack(y, dim=0)
            return y

        input_ids, lengths = pad_and_stack_ids(collated_samples["input_ids"])
        labels = stack_labels(collated_samples["scores"])

        # transform vector to label (multilabel to multiclass)
        if self.transform_to_multiclass:
            labels = labels.argmax(dim=-1)

        # if aspect-only training, get that aspect as a single score (nb_classes will be 1)
        if self.aspect_id > -1:
            labels = labels[:, self.aspect_id].unsqueeze(1)

        # keep annotations and tokens in raw format
        annotations = collated_samples["annotations"]
        tokens = collated_samples["tokens"]

        # return batch to the data loader
        batch = {
            "input_ids": input_ids,
            "lengths": lengths,
            "annotations": annotations,
            "tokens": tokens,
            "labels": labels,
        }
        return batch

    def prepare_data(self):
        # download data, prepare and store it (do not assign to self vars)
        # _ = hf_datasets.load_dataset(
        #     path=self.path,
        #     save_infos=True,
        # )
        pass

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            aspect_subset=self.aspect_subset,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )

        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["tokens"],
                self.dataset["validation"]["tokens"]
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
            else:
                example["input_ids"] = self.tokenizer.encode(example["tokens"].strip())
            return example

        # function to filter out examples longer than max_seq_len
        def _filter(example: dict):
            return len(example["input_ids"]) <= self.max_seq_len

        # apply encode and filter
        self.dataset = self.dataset.map(_encode)
        self.dataset = self.dataset.filter(_filter)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "scores"],
            output_all_columns=True,
        )
