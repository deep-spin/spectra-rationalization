import re
from itertools import chain

import datasets as hf_datasets
import torch
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
from torchnlp.utils import collate_tensors
from transformers import PreTrainedTokenizerBase

from rationalizers import constants
from rationalizers.data_modules.imdb import ImdbDataModule


class SyntheticImdbDataModule(ImdbDataModule):
    """DataModule for IMDB Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        self.synthetic_edits_path = d_params["synthetic_edits_path"]
        self.filter_invalid_edits = d_params.get("filter_invalid_edits", False)
        self.max_synthetic_edits = d_params.get("max_synthetic_edits", 99999999)
        self.pct_synthetic_dataset_size = d_params.get("pct_synthetic_dataset_size", 1.0)

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
        def pad_and_stack_ids(x):
            x_ids, x_lengths = stack_and_pad_tensors(x, padding_index=constants.PAD_ID)
            return x_ids, x_lengths

        def stack_labels(y):
            if isinstance(y, list):
                return torch.stack(y, dim=0)
            return y

        input_ids, lengths = pad_and_stack_ids(collated_samples["input_ids"])
        labels = stack_labels(collated_samples["label"])

        # keep tokens in raw format
        tokens = collated_samples["text"]

        is_original = torch.tensor(collated_samples["is_original"])

        # return batch to the data loader
        batch = {
            "input_ids": input_ids,
            "lengths": lengths,
            "tokens": tokens,
            "labels": labels,
            "is_original": is_original,
        }
        return batch

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )

        # remove unnecessary data
        del self.dataset['unsupervised']

        # create a validation set
        modified_dataset = self.dataset["train"].train_test_split(test_size=0.1)
        self.dataset["train"] = modified_dataset["train"]
        self.dataset["validation"] = modified_dataset["test"]

        # cap dataset size - useful for quick testing
        if self.max_dataset_size is not None:
            self.dataset["train"] = self.dataset["train"].select(range(self.max_dataset_size))
            self.dataset["validation"] = self.dataset["validation"].select(range(self.max_dataset_size))
            self.dataset["test"] = self.dataset["test"].select(range(self.max_dataset_size))

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
            example["is_original"] = 1
            return example

        # apply encode and filter
        self.dataset = self.dataset.map(_encode)

        # read synthetic edits
        ds_syn = hf_datasets.load_dataset(
            "csv",
            data_files=self.synthetic_edits_path,
            sep='\t',
            usecols=['orig_predictions', 'orig_z',
                     'edits_texts', 'edits_labels', 'edits_predictions', 'edits_z_pre', 'edits_z_pos'],
            features=hf_datasets.Features({
                # "orig_texts": hf_datasets.Value("string"),
                # "orig_labels": hf_datasets.ClassLabel(names=["Negative", "Positive"]),
                "orig_predictions": hf_datasets.Value("string"),
                "orig_z": hf_datasets.Value("string"),
                "edits_texts": hf_datasets.Value("string"),
                "edits_labels": hf_datasets.Value("string"),
                "edits_predictions": hf_datasets.Value("string"),
                "edits_z_pre": hf_datasets.Value("string"),
                "edits_z_pos": hf_datasets.Value("string"),
            }),
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )

        # transform z from string to list of floats and fix labels
        def fix_data(ex):
            label_map = {"Negative": 'neg', "Positive": 'pos'}
            ids = self.tokenizer.convert_tokens_to_ids(ex['edits_texts'].split())
            ex['edits_texts'] = fix_edits(self.tokenizer.decode(ids))
            ex['input_ids'] = self.tokenizer(ex['edits_texts'], padding=False, truncation=True)['input_ids']
            ex['edits_labels'] = label_map[ex['edits_labels']]
            ex['edits_predictions'] = label_map[ex['edits_predictions']]
            ex['is_original'] = 0
            return ex
        ds_syn = ds_syn.map(fix_data)

        if self.filter_invalid_edits:
            # filter out examples with wrong predictions (i.e. edits that do not change the prediction)
            ds_syn = ds_syn.filter(lambda ex: ex["edits_labels"] == ex["edits_predictions"])

        # match column names and features
        ds_syn = ds_syn.rename_column("edits_texts", "text")
        ds_syn = ds_syn.rename_column("edits_labels", "label")
        ds_syn = ds_syn.remove_columns(
            ["orig_predictions", "orig_z", "edits_predictions", "edits_z_pre", "edits_z_pos"]
        )
        ds_syn = ds_syn.cast_column('label', hf_datasets.ClassLabel(names=['neg', 'pos']))

        # limit synthetic dataset size
        if self.pct_synthetic_dataset_size < 1:
            train_size = len(self.dataset["train"])
            syn_size = int(train_size * self.pct_synthetic_dataset_size)
            df_syn_train = ds_syn["train"].to_pandas()
            df_neg = df_syn_train[df_syn_train['label'] == 0].sample(frac=1).iloc[:syn_size // 2]
            df_pos = df_syn_train[df_syn_train['label'] == 1].sample(frac=1).iloc[:syn_size // 2]
            sel_idxs = df_neg.index.tolist() + df_pos.index.tolist()
            ds_syn["train"] = ds_syn["train"].select(sel_idxs)
            print(ds_syn)

        # add synthetic edits as new examples to the original dataset
        self.dataset["train"] = hf_datasets.concatenate_datasets([self.dataset["train"], ds_syn["train"]], axis=0)
        print(self.dataset)

        # function to filter out examples longer than max_seq_len
        def _filter(example: dict):
            return len(example["input_ids"]) <= self.max_seq_len
        self.dataset = self.dataset.filter(_filter)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "label"],
            output_all_columns=True,
        )


def fix_edits(text):
    text = text.replace('<', ' <').replace('>', '> ')
    text = text.replace('""', ' "').replace("''", "'")
    text = text.replace('<unk> br', '<br')
    text = re.sub(r'( </s>)+', ' </s>', text)
    text = re.sub(r'</s>[\S\ ]+', '</s>', text)
    text = text.replace("</s>", "")
    text = re.sub(r'\ +', ' ', text).strip()
    return text
