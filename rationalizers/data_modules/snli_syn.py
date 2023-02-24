from functools import partial
from itertools import chain
import datasets as hf_datasets
import nltk
import numpy as np
import torch
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor
from torchnlp.utils import collate_tensors
from transformers import PreTrainedTokenizerBase

from rationalizers import constants
from rationalizers.data_modules.snli import SNLIDataModule
from rationalizers.data_modules.utils import token_type_ids_from_input_ids, fix_saved_inputs_for_t5


class SyntheticSNLIDataModule(SNLIDataModule):
    """DataModule for SNLI Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        self.synthetic_edits_path = d_params["synthetic_edits_path"]
        self.filter_invalid_edits = d_params.get("filter_invalid_edits", False)
        self.pct_synthetic_dataset_size = d_params.get("pct_synthetic_dataset_size", 1.0)
        assert self.concat_inputs is True

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
        def pad_and_stack_ids(x, pad_id=constants.PAD_ID):
            x_ids, x_lengths = stack_and_pad_tensors(x, padding_index=pad_id)
            return x_ids, x_lengths

        def stack_labels(y):
            if isinstance(y, list):
                return torch.stack(y, dim=0)
            return y

        input_ids, lengths = pad_and_stack_ids(collated_samples["input_ids"])
        token_type_ids, _ = pad_and_stack_ids(collated_samples["token_type_ids"], pad_id=2)

        # stack labels
        labels = stack_labels(collated_samples["label"])

        # keep tokens in raw format
        prem_tokens = collated_samples["premise"]
        hyp_tokens = collated_samples["hypothesis"]

        if not self.swap_pair:
            tokens = [p.strip() + ' ' + self.sep_token + ' ' + h.strip() for p, h in zip(prem_tokens, hyp_tokens)]
        else:
            tokens = [p.strip() + ' ' + self.sep_token + ' ' + h.strip() for p, h in zip(hyp_tokens, prem_tokens)]

        is_original = torch.tensor(collated_samples["is_original"])

        batch = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "lengths": lengths,
            "labels": labels,
            "tokens": tokens,
            "is_original": is_original
        }

        return batch

    def prepare_data(self):
        pass

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
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                input_enc = self.tokenizer(
                    ex["premise"].strip(),
                    ex["hypothesis"].strip(),
                    padding=False,  # do not pad, padding will be done later
                    truncation=True,  # truncate to max length accepted by the model
                )
                ex["input_ids"] = input_enc["input_ids"]
                ex["token_type_ids"] = token_type_ids_from_input_ids(ex["input_ids"], self.sep_token_id)
            else:
                ex["input_ids"] = self.tokenizer.encode(
                    ex["premise"].strip() + ' ' + self.sep_token + ' ' + ex["hypothesis"].strip()
                )
                ex["token_type_ids"] = token_type_ids_from_input_ids(ex["input_ids"], self.sep_token_id)
            ex['is_original'] = 1
            return ex

        self.dataset = self.dataset.map(_encode)

        # read synthetic edits
        ds_syn = hf_datasets.load_dataset(
            "csv",
            data_files=self.synthetic_edits_path,
            sep='\t',
            usecols=['orig_labels', 'orig_predictions', 'orig_z',
                     'edits_texts', 'edits_labels', 'edits_predictions', 'edits_z_pre', 'edits_z_pos'],
            features=hf_datasets.Features({
                # "orig_texts": hf_datasets.Value("string"),
                "orig_labels": hf_datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                "orig_predictions": hf_datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                "orig_z": hf_datasets.Value("string"),
                "edits_texts": hf_datasets.Value("string"),
                "edits_labels": hf_datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                "edits_predictions": hf_datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                "edits_z_pre": hf_datasets.Value("string"),
                "edits_z_pos": hf_datasets.Value("string"),
            }),
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )

        def fix_data(ex):
            ex['input_ids'] = self.tokenizer.convert_tokens_to_ids(ex['edits_texts'].split())
            ex['input_ids'] = fix_saved_inputs_for_t5(ex['input_ids'], sep_id=self.sep_token_id)
            ex['premise'] = ex['edits_texts'].split('</s>')[0]
            ex['hypothesis'] = ex['edits_texts'].split('</s>')[1]
            ex["token_type_ids"] = token_type_ids_from_input_ids(ex["input_ids"], self.sep_token_id)
            ex['is_original'] = 0
            return ex
        ds_syn = ds_syn.map(fix_data)

        if self.filter_invalid_edits:
            # filter out exs with wrong predictions (i.e. edits that do not change the prediction)
            ds_syn = ds_syn.filter(lambda ex: ex["edits_labels"] == ex["edits_predictions"])

        # match column names and features
        ds_syn = ds_syn.rename_column("edits_labels", "label")

        # limit synthetic dataset size
        if self.pct_synthetic_dataset_size < 1:
            train_size = len(self.dataset["train"])
            syn_size = int(train_size * self.pct_synthetic_dataset_size)
            df_syn_train = ds_syn["train"].to_pandas()
            df_ent = df_syn_train[df_syn_train['label'] == 0].sample(frac=1).iloc[:syn_size // 3]
            df_neu = df_syn_train[df_syn_train['label'] == 1].sample(frac=1).iloc[:syn_size // 3]
            df_con = df_syn_train[df_syn_train['label'] == 2].sample(frac=1).iloc[:syn_size // 3]
            sel_idxs = df_ent.index.tolist() + df_neu.index.tolist() + df_con.index.tolist()
            ds_syn["train"] = ds_syn["train"].select(sel_idxs)
            print(ds_syn)
            print(np.unique(self.dataset["train"]["label"], return_counts=True))
            print(np.unique(ds_syn["train"]["label"], return_counts=True))

        # remove columns that are not needed
        ds_syn = ds_syn.remove_columns(
            ["orig_labels", "orig_predictions", "orig_z", "edits_predictions", "edits_z_pre", "edits_z_pos"]
        )

        # add synthetic edits as new exs to the original dataset
        self.dataset["train"] = hf_datasets.concatenate_datasets([self.dataset["train"], ds_syn["train"]], axis=0)
        print(self.dataset)

        # filter length
        self.dataset = self.dataset.filter(lambda ex: len(ex["input_ids"]) <= self.max_seq_len)

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "label"],
            output_all_columns=True,
        )
