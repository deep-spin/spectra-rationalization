import torch

from rationalizers.data_modules import SNLIDataModule


class CounterfactualSNLIDataModule(SNLIDataModule):
    """DataModule for SNLI Dataset with counterfactuals."""

    def _collate_fn(self, samples: list, are_samples_batched: bool = False):
        batch = super()._collate_fn(samples, are_samples_batched)
        # sample a different label for the counterfactual
        labels = batch['labels']
        batch_size = len(labels)
        all_labels = torch.arange(self.nb_classes).unsqueeze(0).expand(batch_size, -1)
        cf_labels = all_labels[all_labels != labels.unsqueeze(-1)].view(batch_size, -1)
        random_idxs = torch.randint(0, self.nb_classes - 1, size=(batch_size, ))
        cf_labels = cf_labels.gather(1, random_idxs.unsqueeze(-1)).squeeze(-1)
        batch.update({
            'cf_input_ids': batch['input_ids'].clone(),
            'cf_lengths': batch['lengths'].clone(),
            'cf_labels': cf_labels,
        })
        return batch
