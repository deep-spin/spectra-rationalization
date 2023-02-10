from itertools import chain
import datasets as hf_datasets
import torch
from transformers import PreTrainedTokenizerBase

from rationalizers.data_modules.snli import SNLIDataModule
from rationalizers.data_modules.utils import token_type_ids_from_input_ids


class AugmentedSNLIDataModule(SNLIDataModule):
    """DataModule for SNLI Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        # True: select only factuals for the augmented exs
        # False: select only counterfactuals for the augmented exs
        # None: select both
        self.is_original = d_params.get("is_original", False)

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )

        # filter out invalid samples
        self.dataset = self.dataset.filter(lambda ex: ex["label"] != -1)

        # cap dataset size - useful for quick testing
        if self.max_dataset_size is not None:
            self.dataset["train"] = self.dataset["train"].select(range(self.max_dataset_size))
            self.dataset["validation"] = self.dataset["validation"].select(range(self.max_dataset_size))
            self.dataset["test"] = self.dataset["test"].select(range(self.max_dataset_size))

        # augment dataset
        d_aug = hf_datasets.load_dataset(
            path="./rationalizers/custom_hf_datasets/revised_snli.py",
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
            side='both'
        )
        # select only the contrastive samples
        if self.is_original is not None:
            d_aug = d_aug.filter(lambda ex: ex['is_original'] == self.is_original)
        d_aug = d_aug.rename_column("prem", "premise")
        d_aug = d_aug.rename_column("hyp", "hypothesis")
        d_aug = d_aug.remove_columns(["batch_id", "is_original"])
        # augment train and validation set
        self.dataset["train"] = hf_datasets.concatenate_datasets([self.dataset["train"], d_aug["train"]])
        self.dataset["validation"] = hf_datasets.concatenate_datasets([self.dataset["validation"], d_aug["validation"]])

        # build tokenize rand label encoder
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["premise"],
                self.dataset["train"]["hypothesis"],
                self.dataset["validation"]["premise"],
                self.dataset["validation"]["hypothesis"],
            )
            self.tokenizer = self.tokenizer_cls(tok_samples)

        # map strings to ids
        def _encode(ex: dict):
            if self.concat_inputs:
                if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                    if not self.swap_pair:
                        input_enc = self.tokenizer(
                            ex["premise"].strip(),
                            ex["hypothesis"].strip(),
                            padding=False,  # do not pad, padding will be done later
                            truncation=True,  # truncate to max length accepted by the model
                        )
                    else:
                        input_enc = self.tokenizer(
                            ex["hypothesis"].strip(),
                            ex["premise"].strip(),
                            padding=False,  # do not pad, padding will be done later
                            truncation=True,  # truncate to max length accepted by the model
                        )
                    ex["input_ids"] = input_enc["input_ids"]
                    ex["token_type_ids"] = token_type_ids_from_input_ids(ex["input_ids"], self.sep_token_id)
                else:
                    if not self.swap_pair:
                        ex["input_ids"] = self.tokenizer.encode(
                            ex["premise"].strip() + ' ' + self.sep_token + ' ' + ex["hypothesis"].strip()
                        )
                    else:
                        ex["input_ids"] = self.tokenizer.encode(
                            ex["hypothesis"].strip() + ' ' + self.sep_token + ' ' + ex["premise"].strip()
                        )
                    ex["token_type_ids"] = token_type_ids_from_input_ids(ex["input_ids"], self.sep_token_id)
            else:
                if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                    ex["prem_ids"] = self.tokenizer(
                        ex["premise"].strip(),
                        padding=False,  # do not pad, padding will be done later
                        truncation=True,  # truncate to max length accepted by the model
                    )["input_ids"]
                    ex["hyp_ids"] = self.tokenizer(
                        ex["hypothesis"].strip(),
                        padding=False,  # do not pad, padding will be done later
                        truncation=True,  # truncate to max length accepted by the model
                    )["input_ids"]
                else:
                    ex["prem_ids"] = self.tokenizer.encode(ex["premise"].strip())
                    ex["hyp_ids"] = self.tokenizer.encode(ex["hypothesis"].strip())
            return ex

        self.dataset = self.dataset.map(_encode)

        if self.concat_inputs:
            self.dataset = self.dataset.filter(lambda ex: len(ex["input_ids"]) <= self.max_seq_len)
        else:
            self.dataset = self.dataset.filter(lambda ex: len(ex["prem_ids"]) <= self.max_seq_len)
            self.dataset = self.dataset.filter(lambda ex: len(ex["hyp_ids"]) <= self.max_seq_len)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        if self.concat_inputs:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "label",],
                output_all_columns=True,
            )
        else:
            self.dataset.set_format(
                type="torch",
                columns=["prem_ids", "hyp_ids", "label",],
                output_all_columns=True,
            )
