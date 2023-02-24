from itertools import chain
import datasets as hf_datasets
import numpy as np
import torch
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors
from transformers import PreTrainedTokenizerBase

from rationalizers import constants
from rationalizers.data_modules.snli import SNLIDataModule
from rationalizers.data_modules.utils import token_type_ids_from_input_ids, fix_saved_inputs_for_t5


class SyntheticExplainSNLIDataModule(SNLIDataModule):
    """DataModule for SNLI Dataset."""

    def __init__(self, d_params: dict, tokenizer: object = None):
        super().__init__(d_params, tokenizer)
        self.path = "esnli"
        self.synthetic_edits_path = d_params["synthetic_edits_path"]
        self.filter_invalid_edits = d_params.get("filter_invalid_edits", False)
        self.ignore_neutral_edits = d_params.get("ignore_neutral_edits", False)
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
        a_tokens = collated_samples["premise"] if not self.swap_pair else collated_samples["hypothesis"]
        b_tokens = collated_samples["hypothesis"] if not self.swap_pair else collated_samples["premise"]
        tokens = [p.strip() + ' ' + self.sep_token + ' ' + h.strip() for p, h in zip(a_tokens, b_tokens)]

        batch_extra = {}
        if "cf_input_ids" in collated_samples:
            cf_input_ids, cf_lengths = pad_and_stack_ids(collated_samples["cf_input_ids"])
            cf_token_type_ids, _ = pad_and_stack_ids(collated_samples["cf_token_type_ids"], pad_id=2)
            cf_labels = stack_labels(collated_samples["cf_label"])

            a_tokens = collated_samples["cf_premise"] if not self.swap_pair else collated_samples["cf_hypothesis"]
            b_tokens = collated_samples["cf_hypothesis"] if not self.swap_pair else collated_samples["cf_premise"]
            cf_tokens = [p.strip() + ' ' + self.sep_token + ' ' + h.strip() for p, h in zip(a_tokens, b_tokens)]

            z, _ = stack_and_pad_tensors(collated_samples["z"], padding_index=0)
            cf_z_pre, _ = stack_and_pad_tensors(collated_samples["cf_z_pre"], padding_index=0)
            cf_z_pos, _ = stack_and_pad_tensors(collated_samples["cf_z_pos"], padding_index=0)
            cf_is_valid = torch.stack(collated_samples["cf_is_valid"], dim=0)
            batch_extra = {
                "z": z,
                "cf_input_ids": cf_input_ids,
                "cf_token_type_ids": cf_token_type_ids,
                "cf_lengths": cf_lengths,
                "cf_labels": cf_labels,
                "cf_tokens": cf_tokens,
                "cf_z_pre": cf_z_pre,
                "cf_z_pos": cf_z_pos,
                "cf_is_valid": cf_is_valid,
            }

        batch = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "lengths": lengths,
            "labels": labels,
            "tokens": tokens,
            **batch_extra
        }

        return batch

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        # read synthetic edits
        ds_syn = hf_datasets.load_dataset(
            "csv",
            data_files=self.synthetic_edits_path,
            sep='\t',
            usecols=['orig_texts', 'orig_labels', 'orig_predictions', 'orig_z',
                     'edits_texts', 'edits_labels', 'edits_predictions', 'edits_z_pre', 'edits_z_pos'],
            features=hf_datasets.Features({
                "orig_texts": hf_datasets.Value("string"),
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
            ex['orig_input_ids'] = self.tokenizer.convert_tokens_to_ids(ex['orig_texts'].split())
            ex['orig_input_ids'] = fix_saved_inputs_for_t5(ex['orig_input_ids'], sep_id=self.sep_token_id)
            ex['orig_z'] = eval(ex['orig_z'])[:len(ex['orig_input_ids'])]
            ex["orig_token_type_ids"] = token_type_ids_from_input_ids(ex["orig_input_ids"], self.sep_token_id)
            ex['orig_premise'] = ex['orig_texts'].split('</s>')[0]
            ex['orig_hypothesis'] = ex['orig_texts'].split('</s>')[1]

            ex['edits_input_ids'] = self.tokenizer.convert_tokens_to_ids(ex['edits_texts'].split())
            ex['edits_input_ids'] = fix_saved_inputs_for_t5(ex['edits_input_ids'], sep_id=self.sep_token_id)
            ex['edits_z_pre'] = eval(ex['edits_z_pre'])[:len(ex['edits_input_ids'])]
            ex['edits_z_pos'] = eval(ex['edits_z_pos'])[:len(ex['edits_input_ids'])]
            ex["edits_token_type_ids"] = token_type_ids_from_input_ids(ex["edits_input_ids"], self.sep_token_id)
            ex['edits_premise'] = ex['edits_texts'].split('</s>')[0]
            ex['edits_hypothesis'] = ex['edits_texts'].split('</s>')[1]

            ex['edits_is_valid'] = 1
            if self.ignore_neutral_edits:
                if isinstance(ex['orig_labels'], str):
                    ex['edits_is_valid'] = int(ex['orig_labels'] != 'neutral')
                else:
                    ex['edits_is_valid'] = int(ex['orig_labels'] != 1)
            if self.filter_invalid_edits:
                if ex['edits_is_valid'] == 1:
                    ex['edits_is_valid'] = int(ex['edits_labels'] == ex['edits_predictions'])
            return ex
        ds_syn = ds_syn.map(fix_data)

        # match column names and features
        ds_syn = ds_syn.rename_column("orig_input_ids", "input_ids")
        # ds_syn = ds_syn.rename_column("orig_texts", "text")
        ds_syn = ds_syn.rename_column("orig_labels", "label")
        ds_syn = ds_syn.rename_column("orig_predictions", "pred")
        ds_syn = ds_syn.rename_column("orig_z", "z")
        ds_syn = ds_syn.rename_column("orig_token_type_ids", "token_type_ids")
        ds_syn = ds_syn.rename_column("orig_premise", "premise")
        ds_syn = ds_syn.rename_column("orig_hypothesis", "hypothesis")
        ds_syn = ds_syn.rename_column("edits_input_ids", "cf_input_ids")
        # ds_syn = ds_syn.rename_column("edits_texts", "cf_text")
        ds_syn = ds_syn.rename_column("edits_labels", "cf_label")
        ds_syn = ds_syn.rename_column("edits_predictions", "cf_pred")
        ds_syn = ds_syn.rename_column("edits_z_pre", "cf_z_pre")
        ds_syn = ds_syn.rename_column("edits_z_pos", "cf_z_pos")
        ds_syn = ds_syn.rename_column("edits_token_type_ids", "cf_token_type_ids")
        ds_syn = ds_syn.rename_column("edits_premise", "cf_premise")
        ds_syn = ds_syn.rename_column("edits_hypothesis", "cf_hypothesis")
        ds_syn = ds_syn.rename_column("edits_is_valid", "cf_is_valid")
        # ds_syn = ds_syn.remove_columns(["orig_predictions", "edits_predictions"])

        # Assign train/val/test datasets for use in dataloaders
        self.dataset = hf_datasets.load_dataset(
            path=self.path,
            download_mode=hf_datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )
        del self.dataset['train']

        # filter out invalid samples
        self.dataset = self.dataset.filter(lambda ex: ex["label"] != -1)

        # cap dataset size - useful for quick testing
        if self.max_dataset_size is not None:
            ds_syn["train"] = ds_syn["train"].select(range(self.max_dataset_size))
            self.dataset["validation"] = self.dataset["validation"].select(range(self.max_dataset_size))
            self.dataset["test"] = self.dataset["test"].select(range(self.max_dataset_size))

        # map strings to ids
        def _encode(ex: dict):
            input_enc = self.tokenizer(
                ex["premise"].strip(),
                ex["hypothesis"].strip(),
                padding=False,  # do not pad, padding will be done later
                truncation=True,  # truncate to max length accepted by the model
            )
            ex["input_ids"] = input_enc["input_ids"]
            ex["token_type_ids"] = token_type_ids_from_input_ids(ex["input_ids"], self.sep_token_id)
            return ex
        self.dataset = self.dataset.map(_encode)

        # add synthetic edits as new exs to the original dataset
        self.dataset["train"] = ds_syn["train"]

        # if self.filter_invalid_edits:
        #     # filter out exs with wrong predictions (i.e. edits that do not change the prediction)
        #     self.dataset["train"] = self.dataset["train"].filter(lambda ex: ex["cf_label"] == ex["cf_pred"])

        # convert `columns` to pytorch tensors and keep un-formatted columns
        self.dataset['train'].set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "label", "z",
                     "cf_input_ids", "cf_token_type_ids", "cf_label", "cf_z_pre", "cf_z_pos", "cf_is_valid"],
            output_all_columns=True,
        )
        self.dataset['validation'].set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "label"],
            output_all_columns=True,
        )
        self.dataset['test'].set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "label"],
            output_all_columns=True,
        )
