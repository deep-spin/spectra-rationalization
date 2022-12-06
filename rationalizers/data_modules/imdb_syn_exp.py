from itertools import chain

import datasets as hf_datasets
import torch
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors
from transformers import PreTrainedTokenizerBase

from rationalizers import constants
from rationalizers.data_modules.imdb import ImdbDataModule


class SyntheticExplainImdbDataModule(ImdbDataModule):
    """DataModule for IMDB Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        self.synthetic_edits_path = d_params["synthetic_edits_path"]
        self.filter_invalid_edits = d_params.get("filter_invalid_edits", False)

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
            collated_samples["input_ids"], padding_index=constants.PAD_ID
        )
        # stack labels
        labels = collated_samples["label"]
        if isinstance(labels, list):
            labels = torch.stack(labels, dim=0)
        # keep tokens in raw format
        tokens = collated_samples["text"]

        batch_extra = {}
        if "cf_input_ids" in collated_samples:
            cf_input_ids, cf_lengths = stack_and_pad_tensors(
                collated_samples["cf_input_ids"], padding_index=constants.PAD_ID
            )
            cf_labels = collated_samples["cf_label"]
            if isinstance(cf_labels, list):
                cf_labels = torch.stack(cf_labels, dim=0)
            cf_tokens = collated_samples["cf_text"]
            z, _ = stack_and_pad_tensors(collated_samples["z"], padding_index=0)
            cf_z_pre, _ = stack_and_pad_tensors(collated_samples["cf_z_pre"], padding_index=0)
            cf_z_pos, _ = stack_and_pad_tensors(collated_samples["cf_z_pos"], padding_index=0)
            batch_extra = {
                "z": z,
                "cf_input_ids": cf_input_ids,
                "cf_lengths": cf_lengths,
                "cf_labels": cf_labels,
                "cf_tokens": cf_tokens,
                "cf_z_pre": cf_z_pre,
                "cf_z_pos": cf_z_pos,
            }

        # return batch to the data loader
        batch = {
            "input_ids": input_ids,
            "lengths": lengths,
            "labels": labels,
            "tokens": tokens,
            **batch_extra
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
            ex['orig_predictions'] = label_map[ex['orig_predictions']]
            ex['edits_labels'] = label_map[ex['edits_labels']]
            ex['edits_predictions'] = label_map[ex['edits_predictions']]
            ex['orig_z'] = eval(ex['orig_z'])
            ex['edits_z_pre'] = eval(ex['edits_z_pre'])
            ex['edits_z_pos'] = eval(ex['edits_z_pos'])
            return ex
        ds_syn = ds_syn.map(fix_data)

        # match column names and features
        ds_syn = ds_syn.rename_column("orig_z", "z")
        ds_syn = ds_syn.rename_column("edits_z_pre", "cf_z_pre")
        ds_syn = ds_syn.rename_column("edits_z_pos", "cf_z_pos")
        ds_syn = ds_syn.rename_column("edits_labels", "cf_label")
        ds_syn = ds_syn.rename_column("edits_predictions", "cf_pred")
        ds_syn = ds_syn.rename_column("edits_texts", "cf_text")
        # ds_syn = ds_syn.remove_columns(["orig_predictions", "edits_predictions"])

        # add synthetic edits as columns to the original dataset
        self.dataset["train"] = hf_datasets.concatenate_datasets([self.dataset["train"], ds_syn["train"]], axis=1)

        if self.filter_invalid_edits:
            # filter out examples with wrong predictions (i.e. edits that do not change the prediction)
            self.dataset["train"] = self.dataset["train"].filter(lambda ex: ex["cf_label"] == ex["cf_pred"])

        # cast cf_label to correct type
        self.dataset["train"] = self.dataset["train"].cast_column('cf_label', hf_datasets.ClassLabel(names=['neg', 'pos']))

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
            if 'cf_text' in example:
                example["z"] = torch.as_tensor((example["z"]), dtype=torch.float32)
                example["cf_z_pre"] = torch.as_tensor((example["cf_z_pre"][2:]), dtype=torch.float32)
                example["cf_z_pos"] = torch.as_tensor((example["cf_z_pos"]), dtype=torch.float32)
                if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                    example["cf_input_ids"] = self.tokenizer.convert_tokens_to_ids(example['cf_text'].strip().split())
                else:
                    example["cf_input_ids"] = self.tokenizer.encode(example["cf_text"].strip())
                max_len = self.tokenizer.model_max_length or self.max_seq_len
                example["cf_input_ids"] = example["cf_input_ids"][:max_len]
                example["cf_z_pre"] = example["cf_z_pre"][:max_len]
                example["cf_z_pos"] = example["cf_z_pos"][:max_len]
            return example

        # function to filter out examples longer than max_seq_len
        def _filter(example: dict):
            return len(example["input_ids"]) <= self.max_seq_len

        # apply encode and filter
        self.dataset = self.dataset.map(_encode)
        self.dataset = self.dataset.filter(_filter)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset['train'].set_format(
            type="torch",
            columns=["input_ids", "label", "z", "cf_input_ids", "cf_label", "cf_z_pre", "cf_z_pos"],
            output_all_columns=True,
        )
        self.dataset['validation'].set_format(
            type="torch",
            columns=["input_ids", "label", "z", "cf_input_ids", "cf_label", "cf_z_pre", "cf_z_pos"],
            output_all_columns=True,
        )
        self.dataset['test'].set_format(
            type="torch",
            columns=["input_ids", "label"],
            output_all_columns=True,
        )
