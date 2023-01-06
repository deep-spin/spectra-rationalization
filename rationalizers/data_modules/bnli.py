from itertools import chain
import datasets as hf_datasets
import torch
from transformers import PreTrainedTokenizerBase

from rationalizers.data_modules.snli import SNLIDataModule


class BreakNLIDataModule(SNLIDataModule):

    def __init__(self, d_params: dict, tokenizer: object = None):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params, tokenizer=tokenizer)
        # hard-coded stuff
        self.path = 'pietrolesci/breaking_nli'
        self.is_multilabel = True
        self.nb_classes = 3  # entailment, neutral, contradiction

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )
        self.dataset = self.dataset.rename_column("sentence1", "premise")
        self.dataset = self.dataset.rename_column("sentence2", "hypothesis")
        self.dataset = self.dataset.remove_columns(
            ['category', 'annotator_labels', 'pairID', 'label']
        )
        self.dataset = self.dataset.rename_column("gold_label", "label")
        self.dataset = self.dataset.cast_column(
            'label', hf_datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]))
        self.dataset['train'] = self.dataset['full']
        self.dataset['validation'] = self.dataset['full']
        self.dataset['test'] = self.dataset['full']

        # cap dataset size - useful for quick testing
        if self.max_dataset_size is not None:
            self.dataset["train"] = self.dataset["train"].select(range(self.max_dataset_size))
            self.dataset["validation"] = self.dataset["validation"].select(range(self.max_dataset_size))
            self.dataset["test"] = self.dataset["test"].select(range(self.max_dataset_size))

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
        def _encode(example: dict):
            if self.concat_inputs:
                if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                    if not self.swap_pair:
                        input_enc = self.tokenizer(
                            example["premise"].strip(),
                            example["hypothesis"].strip(),
                            padding=False,  # do not pad, padding will be done later
                            truncation=True,  # truncate to max length accepted by the model
                        )
                    else:
                        input_enc = self.tokenizer(
                            example["hypothesis"].strip(),
                            example["premise"].strip(),
                            padding=False,  # do not pad, padding will be done later
                            truncation=True,  # truncate to max length accepted by the model
                        )
                    example["input_ids"] = input_enc["input_ids"]
                    if 'token_type_ids' in input_enc:
                        example["token_type_ids"] = torch.tensor(input_enc["token_type_ids"])
                    else:
                        example["token_type_ids"] = 1 - torch.cumprod(
                            torch.tensor(example["input_ids"]) != self.sep_token_id, dim=0)
                else:
                    if not self.swap_pair:
                        example["input_ids"] = self.tokenizer.encode(
                            example["premise"].strip() + ' ' + self.sep_token + ' ' + example["hypothesis"].strip()
                        )
                    else:
                        example["input_ids"] = self.tokenizer.encode(
                            example["hypothesis"].strip() + ' ' + self.sep_token + ' ' + example["premise"].strip()
                        )
                    example["token_type_ids"] = 1 - torch.cumprod(
                        torch.tensor(example["input_ids"]) != self.sep_token_id, dim=0)
            else:
                if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                    example["prem_ids"] = self.tokenizer(
                        example["premise"].strip(),
                        padding=False,  # do not pad, padding will be done later
                        truncation=True,  # truncate to max length accepted by the model
                    )["input_ids"]
                    example["hyp_ids"] = self.tokenizer(
                        example["hypothesis"].strip(),
                        padding=False,  # do not pad, padding will be done later
                        truncation=True,  # truncate to max length accepted by the model
                    )["input_ids"]
                else:
                    example["prem_ids"] = self.tokenizer.encode(example["premise"].strip())
                    example["hyp_ids"] = self.tokenizer.encode(example["hypothesis"].strip())
            return example

        self.dataset = self.dataset.map(_encode)

        if self.concat_inputs:
            self.dataset = self.dataset.filter(lambda example: len(example["input_ids"]) <= self.max_seq_len)
        else:
            self.dataset = self.dataset.filter(lambda example: len(example["prem_ids"]) <= self.max_seq_len)
            self.dataset = self.dataset.filter(lambda example: len(example["hyp_ids"]) <= self.max_seq_len)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        if self.concat_inputs:
            self.dataset.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "label", ],
                output_all_columns=True,
            )
        else:
            self.dataset.set_format(
                type="torch",
                columns=["prem_ids", "hyp_ids", "label", ],
                output_all_columns=True,
            )
