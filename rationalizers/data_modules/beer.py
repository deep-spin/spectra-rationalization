from functools import partial
from itertools import chain

import datasets as hf_datasets
import torch
from torchnlp.encoders.text import WhitespaceEncoder, stack_and_pad_tensors
from torchnlp.utils import collate_tensors

from rationalizers import constants
from rationalizers.data_modules.base import BaseDataModule


class BeerDataModule(BaseDataModule):
    """DataModule for BeerAdvocate Dataset."""

    def __init__(self, d_params: dict):
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
        self.transform_to_multiclass = d_params.get("transform_to_multiclass", False)
        self.is_multilabel = self.transform_to_multiclass

        # deal with single aspect experiments
        self.nb_classes = 5 if self.aspect_subset == "260k" else 1
        self.aspect_id = (
            -1 if self.aspect_subset == "260k" else int(self.aspect_subset[-1])
        )

        # objects
        self.dataset = None
        self.tokenizer = None
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
        self.label_encoder = (
            None  # no need of label encoder -> predefined labels from self.dataset
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
        input_ids, lengths = stack_and_pad_tensors(
            collated_samples["input_ids"], padding_index=self.tokenizer.padding_index
        )

        # stack scores
        scores = collated_samples["scores"]
        if isinstance(scores, list):
            scores = torch.stack(scores, dim=0)

        # transform vector to label (multilabel to multiclass)
        if self.transform_to_multiclass:
            scores = scores.argmax(dim=-1)

        # if aspect-only training, get that aspect as a single score (nb_classes will be 1)
        if self.aspect_id > -1:
            scores = scores[:, self.aspect_id].unsqueeze(1)

        # keep annotations and tokens in raw format
        annotations = collated_samples["annotations"]
        tokens = collated_samples["tokens"]

        # return batch to the data loaders
        batch = {
            "input_ids": input_ids,
            "lengths": lengths,
            "annotations": annotations,
            "tokens": tokens,
            "labels": scores,
        }
        return batch

    def prepare_data(self):
        # download data, prepare and store it (do not assign to self vars)
        _ = hf_datasets.load_dataset(
            path=self.path,
            aspect_subset=self.aspect_subset,
            download_mode=hf_datasets.GenerateMode.REUSE_DATASET_IF_EXISTS,
            save_infos=True,
        )

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            aspect_subset=self.aspect_subset,
            download_mode=hf_datasets.GenerateMode.REUSE_DATASET_IF_EXISTS,
        )

        # build tokenizer info (vocab + special tokens) based on train and validation set
        tok_samples = chain(
            self.dataset["train"]["tokens"], self.dataset["validation"]["tokens"]
        )
        self.tokenizer = self.tokenizer_cls(tok_samples)

        # map strings to ids
        def _encode(example: dict):
            example["input_ids"] = self.tokenizer.encode(example["tokens"].strip())
            return example

        self.dataset = self.dataset.map(_encode)
        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch", columns=["input_ids", "scores"], output_all_columns=True
        )
