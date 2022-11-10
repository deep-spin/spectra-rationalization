from itertools import chain

import datasets as hf_datasets
from transformers import PreTrainedTokenizerBase

from rationalizers.data_modules.revised_imdb import RevisedIMDBDataModule


class SyntheticRevisedIMDBDataModule(RevisedIMDBDataModule):

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        self.synthetic_edits_path = d_params["synthetic_edits_path"]

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )

        # remove manual edits
        self.dataset['train'] = self.dataset['train'].filter(lambda example: example["is_original"] is True)

        # read synthetic edits
        ds_syn = hf_datasets.load_dataset(
            "csv",
            data_files=self.synthetic_edits_path,
            sep='\t',
            features=hf_datasets.Features({
               "text": hf_datasets.Value("string"),
               "gold_label": hf_datasets.ClassLabel(names=["Negative", "Positive"]),
            }),
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )

        # fix columns
        def replace_content_fn(ex):
            ex['text'] = ex['text'].replace('br />', '<br />')  # recover <br /> tags
            ex['is_original'] = False
            return ex
        ds_syn = ds_syn.map(replace_content_fn)

        # match column names
        ds_syn = ds_syn.rename_column("text", "tokens")
        ds_syn = ds_syn.rename_column("gold_label", "label")

        # add synthetic edits to the original dataset
        self.dataset["train"] = hf_datasets.concatenate_datasets([self.dataset["train"], ds_syn["train"]])

        # cap dataset size - useful for quick testing
        if self.max_dataset_size is not None:
            self.dataset["train"] = self.dataset["train"].select(range(self.max_dataset_size))
            self.dataset["validation"] = self.dataset["validation"].select(range(self.max_dataset_size))
            self.dataset["test"] = self.dataset["test"].select(range(self.max_dataset_size))

        # build tokenizer
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["tokens"],
                self.dataset["validation"]["tokens"],
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

        self.dataset = self.dataset.map(_encode)
        self.dataset = self.dataset.filter(lambda example: len(example["input_ids"]) <= self.max_seq_len)

        if self.is_original is not None:
            self.dataset = self.dataset.filter(lambda example: example["is_original"] == self.is_original)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=[
                "input_ids", "label",
                "batch_id", "is_original"
            ],
            output_all_columns=True
        )
