import argparse

import datasets
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split


def collate_fn(data):
    x, y, z, l = list(zip(*data))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    z = torch.nn.utils.rnn.pad_sequence(z, batch_first=True, padding_value=0.0)
    l = torch.stack(l)
    return x, y, z, l


def fix_t5_saved_inputs(gen_ids, pad_id=0, eos_id=1):
    is_eos = gen_ids == eos_id
    is_fused = gen_ids == gen_ids.roll(-1, dims=-1)
    return gen_ids[~(is_eos & is_fused)]


class RationaleDataset(data.Dataset):
    def __init__(self, fname, tokenizer, split_size=None):
        super().__init__()
        self.df = pd.read_csv(
            fname,
            sep='\t',
            usecols=['orig_texts', 'orig_labels', 'orig_predictions', 'orig_z']
        )
        self.tokenizer = tokenizer
        self.label_map = {
            'Negative': 0, 'Neg': 0, 'negative': 0, 'neg': 0, '0': 0, 0: 0,
            'Positive': 1, 'Pos': 1, 'pegative': 1, 'pos': 1, '1': 1, 1: 1,
            'entailment': 0, 'Entailment': 0,
            'neutral': 1, 'Neutral': 1,
            'contradiction': 2, 'Contradiction': 2, '2': 2, 2: 2
        }
        self.nb_classes = len(self.df['orig_labels'].unique())
        if split_size is not None:
            self.df, _ = train_test_split(self.df, train_size=split_size, shuffle=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        input_ids = self.tokenizer.convert_tokens_to_ids(row['orig_texts'].strip().split())
        clf_label = self.label_map[row['orig_predictions']]
        rationale = eval(row['orig_z'])

        x = torch.tensor(input_ids)
        y = torch.tensor(clf_label)
        z = torch.tensor(rationale)

        x = x[(z > 0) & (x != self.tokenizer.eos_token_id)]
        l = torch.tensor(len(x))
        return x, y, z, l


class BowRationaleDataset(RationaleDataset):
    def make_bow_vector(self, message):
        vocab_size = len(self.tokenizer.vocab)
        bow = torch.zeros(vocab_size)
        return torch.scatter_add(bow, 0, torch.tensor(message), torch.ones_like(bow))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        input_ids = self.tokenizer.convert_tokens_to_ids(row['orig_texts'].strip().split())
        clf_label = self.label_map[row['orig_predictions']]
        rationale = eval(row['orig_z'])
        message = [
            idx for idx, z in zip(input_ids, rationale)
            if z > 0 and idx not in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]
        ]

        x = self.make_bow_vector(message)
        y = torch.tensor(clf_label)
        z = torch.tensor(rationale)
        l = torch.tensor(len(rationale))
        return x, y, z, l


class MoviesDataset(data.Dataset):
    def __init__(self, tokenizer, split='train'):
        super(MoviesDataset, self).__init__()
        ds = datasets.load_dataset(
            'movie_rationales',
            download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
        )
        assert split in ['train', 'test', 'validation']
        self.tokenizer = tokenizer
        self.df = self._flatten_dataset_with_evidences(ds[split].to_pandas())
        self.nb_classes = len(self.df['label'].unique())

    def _flatten_dataset_with_evidences(self, df):
        new_data = {
            'review': [],
            'label': [],
            'evidence': [],
        }
        for i, row in df.iterrows():
            for e in row['evidences']:
                new_data['review'].append(row['review'])
                new_data['label'].append(row['label'])
                new_data['evidence'].append(e)
        return pd.DataFrame(new_data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        input_ids = self.tokenizer.encode(row['evidence'], add_special_tokens=False)
        clf_label = int(row['label'])
        rationale = input_ids

        x = torch.tensor(input_ids)
        y = torch.tensor(clf_label)
        z = torch.tensor(rationale)
        l = torch.tensor(len(x))
        return x, y, z, l


class Student(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x, z=None, l=None):
        pass

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        print(optim)
        return optim

    def training_step(self, batch, batch_idx):
        x, y, z, l = batch
        y_hat = self(x, z, l)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, z, l = batch
        y_hat = self(x, z, l)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return {
            'loss': loss.item(),
            'gold': y,
            'pred': y_hat.argmax(-1),
        }

    def test_step(self, batch, batch_idx):
        x, y, z, l = batch
        y_hat = self(x, z, l)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        return {
            'loss': loss.item(),
            'gold': y,
            'pred': y_hat.argmax(-1),
        }

    def predict_step(self, batch, batch_idx):
        x, y, z, l = batch
        y_hat = self(x, z, l)
        return {
            'gold': y,
            'pred': y_hat.argmax(-1),
        }

    def validation_epoch_end(self, outputs: list):
        return self._shared_eval_epoch_end(outputs, prefix="val")

    def test_epoch_end(self, outputs: list):
        return self._shared_eval_epoch_end(outputs, prefix="test")

    def _shared_eval_epoch_end(self, outputs: list, prefix: str):
        stacked_outputs = {k: [x[k] for x in outputs] for k in outputs[0].keys()}

        loss = torch.tensor(stacked_outputs['loss'])
        gold = torch.cat(stacked_outputs['gold'])
        pred = torch.cat(stacked_outputs['pred'])

        loss = torch.mean(loss).item()
        acc = torch.mean((gold == pred).float()).item()
        print('loss: {:.4f}'.format(loss))
        print('accuracy: {:.4f}'.format(acc))
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_acc", acc)

        return loss, acc


class BowStudent(Student):
    def __init__(self, message_size, nb_classes, lr=0.001, weight_decay=0.0):
        super().__init__()
        self.linear = nn.Linear(message_size, nb_classes)
        self.save_hyperparameters()

    def forward(self, x, z=None, l=None):
        return self.linear(x)


class LSTMStudent(Student):
    def __init__(self, emb_layer, message_size, nb_classes, lr=0.001, weight_decay=0.0, hidden_size=50,
                 bidirectional=True):
        super().__init__()
        self.emb_layer = emb_layer
        for param in self.emb_layer.parameters():
            param.requires_grad = False
        self.lstm = nn.LSTM(
            message_size,
            hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        n = 2 if bidirectional else 1
        self.linear = nn.Linear(n * hidden_size, nb_classes)
        self.save_hyperparameters(ignore=['emb_layer'])

    def forward(self, x, z=None, l=None):
        h = self.emb_layer(x)
        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(
            h, l.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hx, cx) = self.lstm(packed_sequence)
        # outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        if self.lstm.bidirectional:
            final = torch.cat([hx[-2], hx[-1]], dim=-1)
        else:
            final = hx[-1]
        return self.linear(final)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-s", "--student-type",
        type=str,
        choices=["bow", "lstm"],
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')
    t5_model = AutoModel.from_pretrained('t5-small')
    t5_emb_layer = t5_model.shared

    if args.student_type == "bow":
        train_dataset = BowRationaleDataset(args.train_data, t5_tokenizer, split_size=0.9)
        val_dataset = BowRationaleDataset(args.train_data, t5_tokenizer, split_size=0.1)
        test_dataset = BowRationaleDataset(args.test_data, t5_tokenizer, split_size=None)
    elif args.student_type == "esnli":
        train_dataset = MoviesDataset(t5_tokenizer, split='train')
        val_dataset = MoviesDataset(t5_tokenizer, split='validation')
        test_dataset = MoviesDataset(t5_tokenizer, split='test')
    else:
        train_dataset = MoviesDataset(t5_tokenizer, split='train')
        val_dataset = MoviesDataset(t5_tokenizer, split='validation')
        test_dataset = MoviesDataset(t5_tokenizer, split='test')

    train_dataloader = data.DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn, shuffle=False)
    test_dataloader = data.DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn, shuffle=False)

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    if args.student_type == "bow":
        model = BowStudent(
            message_size=len(t5_tokenizer.vocab),
            nb_classes=train_dataset.nb_classes,
            lr=0.001,
            weight_decay=0.0000001
        )
    else:
        model = LSTMStudent(
            emb_layer=t5_emb_layer,
            message_size=t5_emb_layer.embedding_dim,
            nb_classes=train_dataset.nb_classes,
            lr=0.001,
            weight_decay=0.0000001,
            hidden_size=50,
            bidirectional=True
        )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        gradient_clip_val=5.0,
        min_epochs=3,
        max_epochs=10,
        callbacks=[early_stop_callback]
    )
    print(trainer.checkpoint_callback.best_model_path)

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    trainer.test(dataloaders=test_dataloader)
