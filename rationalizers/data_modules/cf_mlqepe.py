from rationalizers.data_modules import MLQEPEDataModule


class CounterfactualMLQEPEDataModule(MLQEPEDataModule):
    """DataModule for MLQEPE Dataset with counterfactuals."""

    def _collate_fn(self, samples: list, are_samples_batched: bool = False):
        batch = super()._collate_fn(samples, are_samples_batched)
        batch.update({
            'cf_input_ids': batch['input_ids'].clone(),
            'cf_lengths': batch['lengths'].clone(),
            'cf_labels': 1 - batch['lengths'].clone(),
        })
        return batch
