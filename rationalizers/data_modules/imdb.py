from functools import partial
from itertools import chain
import datasets as hf_datasets
import nltk
import torch
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
from torchnlp.utils import collate_tensors

from rationalizers import constants, cf_constants
from rationalizers.data_modules.base import BaseDataModule
from rationalizers.data_modules.utils import remap_input_to_cf_vocab


class ImdbDataModule(BaseDataModule):
    """DataModule for IMDB Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None, cf_tokenizer: object = None, set_cf_inputs: bool = False):
        super().__init__(d_params)
        # hard-coded stuff
        self.path = "imdb"  # hf_datasets will handle everything
        self.is_multilabel = True
        self.nb_classes = 2  # neg, pos

        # hyperparams
        self.batch_size = d_params.get("batch_size", 64)
        self.num_workers = d_params.get("num_workers", 0)
        self.vocab_min_occurrences = d_params.get("vocab_min_occurrences", 1)
        self.max_seq_len = d_params.get("max_seq_len", 99999999)
        self.max_dataset_size = d_params.get("max_dataset_size", None)

        # objects
        self.dataset = None
        self.label_encoder = None  # no label encoder for this dataset
        self.tokenizer = tokenizer
        self.tokenizer_cls = partial(
            # WhitespaceEncoder,
            # TreebankEncoder,
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
        self.cf_tokenizer = cf_tokenizer
        self.cf_tokenizer_cls = self.tokenizer_cls
        self.set_cf_inputs = set_cf_inputs

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
        input_ids, lengths = stack_and_pad_tensors(collated_samples["input_ids"], padding_index=constants.PAD_ID)
        if self.max_seq_len != 99999999:
            input_ids = pad_tensor(input_ids.t(), self.max_seq_len, padding_index=constants.PAD_ID).t()

        # stack labels
        labels = collated_samples["label"]
        if isinstance(labels, list):
            labels = torch.stack(labels, dim=0)

        # keep tokens in raw format
        tokens = collated_samples["text"]

        # return batch to the data loader
        batch = {
            "input_ids": input_ids,
            "lengths": lengths,
            "tokens": tokens,
            "labels": labels,
        }
        # check if we have counterfactual inputs and do the same for them
        has_cf_inputs = "cf_input_ids" in collated_samples.keys()
        if has_cf_inputs:
            cf_input_ids, cf_lengths = stack_and_pad_tensors(collated_samples["cf_input_ids"],
                                                             padding_index=cf_constants.PAD_ID)
            if self.max_seq_len != 99999999:
                cf_input_ids = pad_tensor(cf_input_ids.t(), self.max_seq_len, padding_index=cf_constants.PAD_ID).t()

            batch["cf_input_ids"] = cf_input_ids
            batch["cf_lengths"] = cf_lengths
            batch["cf_counts"] = [None] * len(cf_input_ids)
            if self.tokenizer != self.cf_tokenizer:
                batch["cf_counts"] = remap_input_to_cf_vocab(input_ids, self.tokenizer, self.cf_tokenizer)

        return batch

    def prepare_data(self):
        # download data, prepare and store it (do not assign to self vars)
        # _ = hf_datasets.load_dataset(
        #     path=self.path,
        #     save_infos=True,
        # )
        pass

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(path=self.path,)
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

        # build tokenize rand label encoder
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(self.dataset["train"]["text"],)
            self.tokenizer = self.tokenizer_cls(tok_samples)

        # do the same for the counterfactual tokenizer
        if self.cf_tokenizer is None and self.set_cf_inputs:
            tok_samples = chain(self.dataset["train"]["text"],)
            self.cf_tokenizer = self.cf_tokenizer_cls(tok_samples)

        # function to map strings to ids
        def _encode(example: dict):
            if self.set_cf_inputs:
                example["cf_input_ids"] = self.cf_tokenizer.encode(example["text"].strip())
            example["input_ids"] = self.tokenizer.encode(example["text"].strip())
            return example

        # function to filter out examples longer than max_seq_len
        def _filter(example: dict):
            if self.set_cf_inputs:
                return max(len(example["input_ids"]), len(example["cf_input_ids"])) <= self.max_seq_len
            return len(example["input_ids"]) <= self.max_seq_len

        # apply encode and filter
        self.dataset = self.dataset.map(_encode)
        self.dataset = self.dataset.filter(_filter)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "cf_input_ids", "label"] if self.set_cf_inputs else ["input_ids", "label"],
            output_all_columns=True,
        )
