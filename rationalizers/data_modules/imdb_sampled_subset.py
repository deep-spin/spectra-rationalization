import os
import datasets as hf_datasets
from transformers import PreTrainedTokenizerBase

from rationalizers.data_modules import ImdbDataModule


class SampledSubsetImdbDataModule(ImdbDataModule):
    """DataModule for IMDB Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        self.path = d_params['path']

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        data = {
            "text": read_texts(os.path.join(self.path, 'inputs.txt')),
            "label": read_labels(os.path.join(self.path, 'labels.txt')),
        }
        dataset = hf_datasets.Dataset.from_dict(
            data,
            features=hf_datasets.Features({
                "text": hf_datasets.Value("string"),
                "label": hf_datasets.Value("int32"),
            }),
        )

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
        dataset = dataset.map(_encode)
        dataset = dataset.filter(_filter)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        dataset.set_format(
            type="torch",
            columns=["input_ids", "label"],
            output_all_columns=True,
        )
        self.dataset = {
            'train': dataset,
            'validation': dataset,
            'test': dataset
        }


def read_texts(path):
    texts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            texts.append(line.strip())
    return texts


def read_labels(path):
    labels = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            labels.append(int(line.strip()))
    return labels
