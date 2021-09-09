from functools import partial
from itertools import chain

import datasets as hf_datasets
import torch
from torchnlp.encoders.label_encoder import LabelEncoder
from torchnlp.encoders.text import WhitespaceEncoder, stack_and_pad_tensors
from torchnlp.utils import collate_tensors

from rationalizers import constants
from rationalizers.data_modules.base import BaseDataModule


class SSTDataModule(BaseDataModule):
    """DataModule for Stanford Sentiment Treebank Dataset."""

    def __init__(self, d_params: dict):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params)
        # hard-coded stuff
        self.path = "./rationalizers/custom_hf_datasets/sst.py"
        self.is_multilabel = True

        # to be set later (it depends on the desired granularity)
        self.nb_classes = None

        # hyperparams
        self.batch_size = d_params.get("batch_size", 32)
        self.num_workers = d_params.get("num_workers", 0)
        self.vocab_min_occurrences = d_params.get("vocab_min_occurrences", 1)
        self.granularity = d_params.get(
            "granularity", "2"
        )  # default: binary classification / no neutrals
        self.subtrees = d_params.get("subtrees", False)  # default: do not use subtrees

        # objects
        self.dataset = None
        self.label_encoder = None
        self.label_encoder_cls = partial(
            LabelEncoder, reserved_labels=[]
        )  # no unknown & pad symbols for labels
        self.tokenizer = None
        self.tokenizer_cls = partial(
            WhitespaceEncoder,  # SST dataset is already tokenized by default
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
        self.label_encoder = None
        self.label_encoder_cls = partial(
            LabelEncoder, reserved_labels=[]
        )  # no unknown & pad symbols for labels

        # to be set later (it depends on the desired granularity)
        self.nb_classes = None

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
            collated_samples["input_ids"], padding_index=self.tokenizer.padding_index
        )

        # stack labels
        labels = collated_samples["label"]
        if isinstance(labels, list):
            labels = torch.stack(labels, dim=0)

        # keep tokens in raw format
        tokens = collated_samples["tokens"]

        # return batch to the data loader
        batch = {
            "input_ids": input_ids,
            "lengths": lengths,
            "tokens": tokens,
            "labels": labels,
        }
        return batch

    def prepare_data(self):
        # download data, prepare and store it (do not assign to self vars)
        _ = hf_datasets.load_dataset(
            path=self.path,
            granularity=self.granularity,
            subtrees=self.subtrees,
            download_mode=hf_datasets.GenerateMode.REUSE_CACHE_IF_EXISTS,
            save_infos=True,
        )

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            granularity=self.granularity,
            subtrees=self.subtrees,
            download_mode=hf_datasets.GenerateMode.REUSE_CACHE_IF_EXISTS,
        )

        # build tokenize rand label encoder
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["tokens"], self.dataset["validation"]["tokens"]
            )
            self.tokenizer = self.tokenizer_cls(tok_samples)

        if self.label_encoder is None:
            label_samples = chain(self.dataset["train"]["label"])
            self.label_encoder = self.label_encoder_cls(label_samples)
        self.nb_classes = len(self.label_encoder.vocab)
        # self.nb_classes = len(self.dataset['train'].features['label'].names)

        # map strings to ids
        def _encode(example: dict):
            example["input_ids"] = self.tokenizer.encode(example["tokens"].strip())
            example["label"] = self.label_encoder.encode(example["label"])
            return example

        self.dataset = self.dataset.map(_encode)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch", columns=["input_ids", "label"], output_all_columns=True
        )
    