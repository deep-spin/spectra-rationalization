from functools import partial
from itertools import chain
import datasets as hf_datasets
import nltk
import torch
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors
from torchnlp.utils import collate_tensors

from rationalizers import constants
from rationalizers.data_modules.base import BaseDataModule


class HANSDataModule(BaseDataModule):
    """DataModule for HANS Dataset."""

    def __init__(self, d_params: dict):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params)
        # hard-coded stuff
        self.path = "hans"  # hf_datasets will handle everything
        self.is_multilabel = True
        self.nb_classes = 2  # entailment, neutral, contradiction

        # hyperparams
        self.batch_size = d_params.get("batch_size", 64)
        self.num_workers = d_params.get("num_workers", 0)
        self.vocab_min_occurrences = d_params.get("vocab_min_occurrences", 1)

        # objects
        self.dataset = None
        self.label_encoder = None  # no label encoder for this dataset
        self.tokenizer = None
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
        self.version = d_params.get("version", "vanilla")

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
        x1_ids, x1_lengths = stack_and_pad_tensors(
            collated_samples["x1_ids"], padding_index=self.tokenizer.padding_index
        )
        x2_ids, x2_lengths = stack_and_pad_tensors(
            collated_samples["x2_ids"], padding_index=self.tokenizer.padding_index
        )

        # stack labels
        labels = collated_samples["label"]
        if isinstance(labels, list):
            labels = torch.stack(labels, dim=0)

        # keep tokens in raw format
        x2 = collated_samples["hypothesis"]
        x1 = collated_samples["premise"]

        # return batch to the data loader   
        batch = {
            "x1_ids": x1_ids,
            "x2_ids": x2_ids,
            "x1_lengths": x1_lengths,
            "x2_lengths": x2_lengths,
            "x1": x1,
            "x2": x2,
            "labels": labels,
        }
        return batch

    def prepare_data(self):
        # download data, prepare and store it (do not assign to self vars)
        _ = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.GenerateMode.REUSE_DATASET_IF_EXISTS,
            save_infos=True,
        )
        _ = hf_datasets.load_dataset(
            path="esnli",
            download_mode=hf_datasets.GenerateMode.REUSE_DATASET_IF_EXISTS,
            save_infos=True,
        )
        _ = hf_datasets.load_dataset(
            path="multi_nli",
            download_mode=hf_datasets.GenerateMode.REUSE_DATASET_IF_EXISTS,
            save_infos=True,
        )

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.hans_dataset = hf_datasets.load_dataset(
            path=self.path,
        )

        self.hans_dataset["test"] = self.hans_dataset["validation"]

        self.multinli_dataset = hf_datasets.load_dataset(
            path="multi_nli",
        )

        # TRAIN MLNI; VALIDATION: MNLI; TEST: HANS
        if self.version == "vanilla":
            self.hans_dataset["train"] = self.multinli_dataset["train"]
            self.hans_dataset["validation"] = self.multinli_dataset["validation_matched"]
        else:
        # TRAIN MLNI + 30 000 SAMPLES HANS; VALIDATION: MNLI; TEST: HANS
            concat_data_train = hf_datasets.concatenate_datasets([self.hans_dataset["train"].flatten_indices(), self.multinli_dataset["train"].flatten_indices()])
            self.hans_dataset["train"] = concat_data_train
            self.hans_dataset["validation"] = self.multinli_dataset["validation_matched"]

        # change multinli labels to match hans labels
        def _collapselabels(example: dict):
            if example["label"] == 2:
                example["label"] = 1
            return example
        

        self.dataset = self.hans_dataset.map(_collapselabels)
        
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
            example["x2_ids"] = self.tokenizer.encode(example["hypothesis"].strip())
            example["x1_ids"] = self.tokenizer.encode(example["premise"].strip())
            return example

        self.dataset = self.dataset.map(_encode)
        
        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=["x1_ids", "x2_ids", "label"],
            output_all_columns=True,
        )
