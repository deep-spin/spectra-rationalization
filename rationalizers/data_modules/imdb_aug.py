from itertools import chain
import datasets as hf_datasets
from transformers import PreTrainedTokenizerBase

from rationalizers.data_modules.imdb import ImdbDataModule


class AugmentedImdbDataModule(ImdbDataModule):
    """DataModule for IMDB Dataset augmented with contrastive examples from the RevisedIMDB dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        # True: select only factuals for the augmented examples
        # False: select only counterfactuals for the augmented examples
        # None: select both
        self.is_original = d_params.get("is_original", False)

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
            ignore_verifications=True  # weird checksum mismatch from hf_datasets???
        )
        modified_dataset = self.dataset["train"].train_test_split(test_size=0.1)
        self.dataset["train"] = modified_dataset["train"]
        self.dataset["validation"] = modified_dataset["test"]

        # remove unnecessary data
        del self.dataset['unsupervised']

        # cap dataset size - useful for quick testing
        if self.max_dataset_size is not None:
            self.dataset["train"] = self.dataset["train"].select(range(self.max_dataset_size))
            self.dataset["validation"] = self.dataset["validation"].select(range(self.max_dataset_size))
            self.dataset["test"] = self.dataset["test"].select(range(self.max_dataset_size))

        # augment dataset
        d_aug = hf_datasets.load_dataset(
            path="./rationalizers/custom_hf_datasets/revised_imdb_for_aug.py",
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )
        # select only the contrastive samples
        if self.is_original is not None:
            d_aug = d_aug.filter(lambda ex: ex['is_original'] == self.is_original)
        d_aug = d_aug.rename_column("tokens", "text")
        d_aug = d_aug.remove_columns(["batch_id", "is_original"])
        # augment train and validation set
        self.dataset["train"] = hf_datasets.concatenate_datasets([self.dataset["train"], d_aug["train"]])
        self.dataset["validation"] = hf_datasets.concatenate_datasets([self.dataset["validation"], d_aug["validation"]])

        # build tokenize rand label encoder
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["text"],
                self.dataset["validation"]["text"],
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

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "label"],
            output_all_columns=True,
        )
