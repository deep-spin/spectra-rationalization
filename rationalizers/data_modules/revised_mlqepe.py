from functools import partial
from itertools import chain

import datasets as hf_datasets
import nltk
import torch
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
from torchnlp.utils import collate_tensors

from rationalizers import constants
from rationalizers.data_modules.base import BaseDataModule
from rationalizers.data_modules.utils import concat_sequences


class RevisedMLQEPEDataModule(BaseDataModule):
    """DataModule for the Revised MLQEPE Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params)
        # hard-coded stuff
        self.path = "./rationalizers/custom_hf_datasets/revised_mlqepe.py"
        self.is_multilabel = True
        self.nb_classes = 2

        # hyperparams
        self.batch_size = d_params.get("batch_size", 32)
        self.num_workers = d_params.get("num_workers", 0)
        self.vocab_min_occurrences = d_params.get("vocab_min_occurrences", 1)
        self.max_seq_len = d_params.get("max_seq_len", 99999999)

        # objects
        self.dataset = None
        self.label_encoder = None
        self.tokenizer = tokenizer
        self.tokenizer_cls = partial(
            StaticTokenizerEncoder,
            tokenize=nltk.wordpunct_tokenize,
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

        def stack_and_pad_ids(x):
            input_ids, lengths = stack_and_pad_tensors(x, padding_index=constants.PAD_ID)
            if self.max_seq_len != 99999999:
                input_ids = pad_tensor(x.t(), self.max_seq_len, padding_index=constants.PAD_ID).t()
            return input_ids, lengths

        def stack_labels(y):
            if isinstance(y, list):
                return torch.stack(y, dim=0)
            return y

        # pad and stack input ids
        input_ids, lengths = stack_and_pad_ids(collated_samples["input_ids"])
        cf_input_ids, cf_lengths = stack_and_pad_ids(collated_samples["cf_input_ids"])
        token_type_ids, _ = stack_and_pad_ids(collated_samples["token_type_ids"])
        cf_token_type_ids, _ = stack_and_pad_ids(collated_samples["cf_token_type_ids"])

        # stack labels
        labels = stack_labels(collated_samples["label"])
        cf_labels = stack_labels(collated_samples["cf_label"])

        # keep tokens in raw format
        src_tokens = collated_samples["src"]
        mt_tokens = collated_samples["mt"]
        cf_src_tokens = collated_samples["cf_src"]
        cf_mt_tokens = collated_samples["cf_mt"]

        # metadata
        batch_id = collated_samples["batch_id"]
        is_original = collated_samples["is_original"]

        # return batch to the data loader
        batch = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "lengths": lengths,
            "labels": labels,
            "src_tokens": src_tokens,
            "mt_tokens": mt_tokens,
            "cf_input_ids": cf_input_ids,
            "cf_token_type_ids": cf_token_type_ids,
            "cf_lengths": cf_lengths,
            "cf_labels": cf_labels,
            "cf_src_tokens": cf_src_tokens,
            "cf_mt_tokens": cf_mt_tokens,
            "batch_id": batch_id,
            "is_original": is_original,
        }
        return batch

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.GenerateMode.REUSE_DATASET_IF_EXISTS,
        )

        # build tokenizer
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["src"],
                self.dataset["train"]["mt"],
                self.dataset["train"]["cf_src"],
                self.dataset["train"]["cf_mt"],
                self.dataset["validation"]["src"],
                self.dataset["validation"]["mt"],
                self.dataset["validation"]["cf_src"],
                self.dataset["validation"]["cf_mt"],
            )
            self.tokenizer = self.tokenizer_cls(tok_samples)

        # map strings to ids
        def _encode(example: dict):
            src_ids = self.tokenizer.encode(example["src"].strip())
            mt_ids = self.tokenizer.encode(example["mt"].strip())
            cf_src_ids = self.tokenizer.encode(example["cf_src"].strip())
            cf_mt_ids = self.tokenizer.encode(example["cf_mt"].strip())
            input_ids, token_type_ids = concat_sequences(src_ids, mt_ids)
            cf_input_ids, cf_token_type_ids = concat_sequences(cf_src_ids, cf_mt_ids)
            example["input_ids"] = input_ids
            example["token_type_ids"] = token_type_ids
            example["cf_input_ids"] = cf_input_ids
            example["cf_token_type_ids"] = cf_token_type_ids
            return example

        self.dataset = self.dataset.map(_encode)
        self.dataset = self.dataset.filter(lambda example: len(example["input_ids"]) <= self.max_seq_len)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=[
                "input_ids", "token_type_ids", "label",
                "cf_input_ids", "cf_token_type_ids", "cf_label",
                "batch_id", "is_original"
            ],
            output_all_columns=True
        )
