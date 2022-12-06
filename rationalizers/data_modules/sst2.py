from itertools import chain

import datasets as hf_datasets
from transformers import PreTrainedTokenizerBase

from rationalizers.data_modules.imdb import ImdbDataModule


class SST2DataModule(ImdbDataModule):
    """DataModule for Stanford Sentiment Treebank Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params, tokenizer=tokenizer)
        # hard-coded stuff
        self.path = "gpt3mix/sst2"
        self.test_only = True

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )
        for split in ["train", "validation"]:
            if split in self.dataset and self.test_only:
                del self.dataset[split]

        # build tokenize rand label encoder
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["text"],
                self.dataset["validation"]["text"]
            )
            self.tokenizer = self.tokenizer_cls(tok_samples)

        # function to map strings to ids
        def _encode(example: dict):
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                example["input_ids"] = self.tokenizer(
                    example["text"].strip(),
                    padding=False,  # do not pad, padding will be done later
                    truncation=True,  # truncate to max length accepted by the model
                )["input_ids"]
            else:
                example["input_ids"] = self.tokenizer.encode(example["text"].strip())
            return example

        # function to filter out examples longer than max_seq_len
        def _filter(example: dict):
            return len(example["input_ids"]) <= self.max_seq_len

        # apply encode and filter
        self.dataset = self.dataset.map(_encode)
        self.dataset = self.dataset.filter(_filter)

        # fix labels
        def _fix_label(example: dict):
            # originally, 0 is positive and 1 is negative
            # we want 0 to be negative and 1 to be positive
            example["label"] = 1 - example["label"]
            return example
        self.dataset = self.dataset.map(_fix_label)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "label"],
            output_all_columns=True,
        )
