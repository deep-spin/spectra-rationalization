from functools import partial
from itertools import chain

import datasets as hf_datasets
import nltk
import torch
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
from torchnlp.utils import collate_tensors
from transformers import PreTrainedTokenizerBase

from rationalizers import constants
from rationalizers.data_modules.base import BaseDataModule
from rationalizers.data_modules.utils import concat_sequences


class MLQEPEDataModule(BaseDataModule):
    """DataModule for the MLQEPE Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        """
        :param d_params: hyperparams dict. See docs for more info.
        """
        super().__init__(d_params)
        # hard-coded stuff
        self.path = "./rationalizers/custom_hf_datasets/mlqepe.py"

        # hyperparams
        self.lp = d_params.get("lp", "all-all")
        self.use_hter_as_label = d_params.get("use_hter_as_label", False)
        self.transform_scores_to_labels = d_params.get("transform_scores_to_labels", False)
        self.batch_size = d_params.get("batch_size", 32)
        self.num_workers = d_params.get("num_workers", 0)
        self.vocab_min_occurrences = d_params.get("vocab_min_occurrences", 1)
        self.max_seq_len = d_params.get("max_seq_len", 99999999)

        # fix nb classes
        if self.transform_scores_to_labels:
            self.is_multilabel = True
            self.nb_classes = 2
        else:
            self.is_multilabel = False
            self.nb_classes = 1

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
        self.sep_token = self.tokenizer.sep_token or self.tokenizer.eos_token
        self.bos_token = self.tokenizer.bos_token or self.tokenizer.sep_token

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

        def pad_and_stack_ids(x, pad_id=constants.PAD_ID):
            x_ids, x_lengths = stack_and_pad_tensors(x, padding_index=pad_id)
            return x_ids, x_lengths

        def stack_labels(y):
            if isinstance(y, list):
                return torch.stack(y, dim=0)
            return y

        # pad and stack input ids
        input_ids, lengths = pad_and_stack_ids(collated_samples["input_ids"])
        token_type_ids, _ = pad_and_stack_ids(collated_samples["token_type_ids"], pad_id=2)

        # stack labels
        if self.use_hter_as_label:
            labels = stack_labels(collated_samples["hter"])
            if self.transform_scores_to_labels:
                labels = ((labels >= 0.30) & (labels <= 1.0)).long()
        else:
            labels = stack_labels(collated_samples["da"])
            if self.transform_scores_to_labels:
                labels = (labels <= 70).long()

        # keep tokens in raw format
        src_tokens = collated_samples["src"]
        mt_tokens = collated_samples["mt"]
        pe_tokens = collated_samples["pe"]
        tokens = [y.strip() + f' {self.sep_token} {self.bos_token} ' + x.strip() for y, x in zip(mt_tokens, src_tokens)]

        # metadata useful for evaluation
        src_tags = collated_samples["src_tags"]
        mt_tags = collated_samples["mt_tags"]
        src_mt_aligns = collated_samples["src_mt_aligns"]
        lps = collated_samples["lp"]

        # return batch to the data loader
        batch = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "lengths": lengths,
            "labels": labels,
            "tokens": tokens,
            "src_tokens": src_tokens,
            "mt_tokens": mt_tokens,
            "pe_tokens": pe_tokens,
            "src_tags": src_tags,
            "mt_tags": mt_tags,
            "src_mt_aligns": src_mt_aligns,
            "lps": lps,
        }
        return batch

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
            lp=self.lp,
        )

        # build tokenizer
        if self.tokenizer is None:
            # build tokenizer info (vocab + special tokens) based on train and validation set
            tok_samples = chain(
                self.dataset["train"]["src"],
                self.dataset["train"]["mt"],
                self.dataset["validation"]["src"],
                self.dataset["validation"]["mt"],
            )
            self.tokenizer = self.tokenizer_cls(tok_samples)

        # map strings to ids
        def _encode(example: dict):
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                src_ids = self.tokenizer(
                    example["src"].strip(),
                    padding=False,  # do not pad, padding will be done later
                    truncation=True,  # truncate to max length accepted by the model
                    max_length=512,
                )["input_ids"]
                mt_ids = self.tokenizer(
                    example["mt"].strip(),
                    padding=False,  # do not pad, padding will be done later
                    truncation=True,  # truncate to max length accepted by the model
                    max_length=512,
                )["input_ids"]
            else:
                src_ids = self.tokenizer.encode(example["src"].strip())
                mt_ids = self.tokenizer.encode(example["mt"].strip())
            input_ids, token_type_ids = concat_sequences(mt_ids, src_ids)
            example["input_ids"] = input_ids
            example["token_type_ids"] = token_type_ids
            return example

        self.dataset = self.dataset.map(_encode)
        self.dataset = self.dataset.filter(lambda example: len(example["input_ids"]) <= self.max_seq_len)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "da", "hter"],
            output_all_columns=True
        )
