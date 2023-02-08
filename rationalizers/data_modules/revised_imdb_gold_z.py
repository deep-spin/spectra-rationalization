import torch

from rationalizers.data_modules.revised_imdb_cf import CounterfactualRevisedIMDBDataModule


class GoldRationaleRevisedIMDBDataModule(CounterfactualRevisedIMDBDataModule):
    """DataModule for the Revised IMDB Dataset."""

    def _collate_fn(self, samples: list, are_samples_batched: bool = False):
        from Levenshtein import editops

        batch = super()._collate_fn(samples, are_samples_batched)
        input_ids = batch['input_ids']
        cf_input_ids = batch['cf_input_ids']

        z = torch.zeros_like(input_ids)
        for i, (x_f, x_c) in enumerate(zip(input_ids, cf_input_ids)):
            x_f_str = [str(e.item()) for e in x_f if e.item() != self.tokenizer.pad_token_id]
            x_c_str = [str(e.item()) for e in x_c if e.item() != self.tokenizer.pad_token_id]
            sel_idxs = [min(t[1], z.shape[-1] - 1) for t in editops(x_f_str, x_c_str)]
            z[i, sel_idxs] = 1
        batch['z'] = z

        del batch['cf_input_ids']
        del batch['cf_lengths']
        del batch['cf_labels']
        del batch['cf_tokens']

        return batch
