import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from rationalizers.utils import load_object, save_object


class BaseDataModule(pl.LightningDataModule):
    """Base DataModule for all data modules."""

    def __init__(self, d_params: dict):
        """
        :param d_params: hyperparams dict.
        """
        super().__init__()
        # base hyperparams
        self.batch_size = 2
        self.num_workers = 0
        # base objects
        self.dataset = None
        self.label_encoder = None
        self.tokenizer = None

    def load_encoders(self, root_dir, load_tokenizer, load_label_encoder):
        if load_tokenizer:
            self.tokenizer = load_object(os.path.join(root_dir, "tokenizer.pickle"))
        if load_label_encoder:
            self.label_encoder = load_object(
                os.path.join(root_dir, "label_encoder.pickle")
            )

    def save_encoders(self, root_dir, save_tokenizer, save_label_encoder):
        if save_tokenizer:
            save_object(self.tokenizer, os.path.join(root_dir, "tokenizer.pickle"))
        if save_label_encoder:
            save_object(
                self.label_encoder, os.path.join(root_dir, "label_encoder.pickle")
            )

    def _collate_fn(self, samples: list, are_samples_batched: bool = False):
        raise NotImplementedError

    def train_dataloader(self):
        # use a standard random sampler:
        sampler = RandomSampler(self.dataset["train"])
        return DataLoader(
            self.dataset["train"],
            sampler=sampler,
            collate_fn=self._collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        # bucket examples of similar size together:
        # sampler = SequentialSampler(self.dataset['train'])
        # sampler = BucketBatchSampler(
        #     sampler,
        #     drop_last=False,
        #     batch_size=self.batch_size,
        #     sort_key=lambda i: len(self.dataset['train'][i]['input_ids'])
        # )
        # return DataLoader(
        #     self.dataset['train'],
        #     sampler=sampler,
        #     collate_fn=self._collate_fn,
        #     collate_fn=partial(self._collate_fn, are_samples_batched=True),
        #     num_workers=self.num_workers,
        # )

    def val_dataloader(self):
        sampler = SequentialSampler(self.dataset["validation"])
        return DataLoader(
            self.dataset["validation"],
            sampler=sampler,
            collate_fn=self._collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        sampler = SequentialSampler(self.dataset["test"])
        return DataLoader(
            self.dataset["test"],
            sampler=sampler,
            collate_fn=self._collate_fn,
            batch_size=1,
            num_workers=self.num_workers,
        )
