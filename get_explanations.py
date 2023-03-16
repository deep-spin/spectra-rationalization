import os
import sys
import re
import warnings
from argparse import Namespace
from pprint import pprint
from tqdm import tqdm

import torch
import random
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from pytorch_lightning import Trainer
from transformers import AutoTokenizer
import datasets as hf_datasets

from rationalizers import constants
from rationalizers.data_modules import available_data_modules
from rationalizers.lightning_models import available_models
from rationalizers.utils import load_ckpt_config, unroll

pd.options.display.float_format = '{:,.4f}'.format


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_tokenizer(args):
    # 1. Load a huggingface tokenizer
    print('Loading tokenizer: {}...'.format(args.tokenizer))
    if args.tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = None
    return tokenizer


def load_data_module(args, tokenizer):
    # 2. Load data module
    print('Loading data module: {}...'.format(args.dm))
    dm_cls = available_data_modules[args.dm]
    dm = dm_cls(d_params=vars(args), tokenizer=tokenizer)
    dm.load_encoders(
        root_dir=os.path.dirname(args.ckpt),
        load_tokenizer=args.load_tokenizer and tokenizer is None,
        load_label_encoder=args.load_label_encoder,
    )
    constants.update_constants(dm.tokenizer)
    return dm


def load_model(args, dm):
    # 3. Load model
    print('Loading model from {}...'.format(args.ckpt))
    model_cls = available_models[args.model]
    model = model_cls.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        map_location=lambda storage, loc: storage,
        strict=False,
        tokenizer=dm.tokenizer,
        nb_classes=dm.nb_classes,
        is_multilabel=dm.is_multilabel,
        h_params=vars(args),
    )
    # set the model to eval mode
    model.log_rationales_in_wandb = False
    model.generation_mode = True
    model.eval()
    return model


def load(args):
    tokenizer = load_tokenizer(args)
    dm = load_data_module(args, tokenizer)
    model = load_model(args, dm)
    return tokenizer, dm, model


def get_args_from_ckpt(ckpt_path, new_args):
    old_args = load_ckpt_config(ckpt_path)
    args = Namespace(**{**old_args, **new_args})
    args.ckpt = ckpt_path
    return args


def predict(ckpt_path, dm_name, dm_args, dataloader='train', verbose=True, disable_progress_bar=False, return_tokenizer=False):
    # disable hf_dataset progress bar
    if disable_progress_bar:
        hf_datasets.logging.disable_progress_bar()
    else:
        hf_datasets.logging.enable_progress_bar()
    if verbose:
        hf_datasets.logging.set_verbosity(20)
    else:
        hf_datasets.logging.set_verbosity(50)
    
    # load args
    base_args = dict(
        seed=0,
        load_tokenizer=False,
        load_label_encoder=False,
        save_rationales=False,
        save_edits=False,
        cf_classify_edits=False,
        sparsemap_budget=30,
    )
    new_args = {**base_args, **dm_args}
    args = get_args_from_ckpt(ckpt_path, new_args)
    
    # fix cf_explainer_mask_token_type_id
    if hasattr(args, 'explainer_mask_token_type_id') and args.explainer_mask_token_type_id == '':
        args.explainer_mask_token_type_id = None
    if hasattr(args, 'cf_explainer_mask_token_type_id') and args.cf_explainer_mask_token_type_id == '':
        args.cf_explainer_mask_token_type_id = None
        
    pprint(vars(args))
    pprint(dm_args)
    
    # set global seed
    configure_seed(args.seed)
    
    # load tokenizer and model
    tokenizer, _, model = load(args)
    
    # load data module
    dm_cls = available_data_modules[dm_name]
    dm = dm_cls(d_params=dm_args, tokenizer=tokenizer)
    dm.prepare_data()
    dm.setup()
    
    # predict
    model.generation_mode = True
    trainer = Trainer(accelerator='gpu', devices=1)
    if dataloader == 'train':
        outputs = trainer.predict(model, dm.train_dataloader(shuffle=False))
    elif dataloader == 'val':
        outputs = trainer.predict(model, dm.val_dataloader())
    else:
        outputs = trainer.predict(model, dm.test_dataloader())
    
    # empty cache (beam search uses a lot of caching)
    torch.cuda.empty_cache()
    
    # stack outputs
    outputs = {k: [x[k] for x in outputs] for k in outputs[0].keys()}
    
    if return_tokenizer:
        return outputs, tokenizer
    return outputs


def clean_tokens(t5_tokenizer, raw_tokens):
    texts = []
    labels = []
    for tks in raw_tokens:
        tks = ['<unk>' if t is None else t for t in tks]
        text = ' '.join(tks).strip()
        texts.append(text)
    return texts


def save_explanations(fname, tokens, labels, predictions, z):
    label_map = {0: 0, 1: 1, 2: 2}
    if 'imdb' in fname in 'movies' in fname:
        label_map = {0: 'Negative', 1: 'Positive'}
    elif 'nli' in fname:
        label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    df = pd.DataFrame({
        'tokens': tokens,
        'labels': [label_map[y] for y in labels] if labels is not None else None, 
        'predictions': [label_map[y] for y in predictions], 
        'z': [z_.detach().cpu().tolist() for z_ in z],
    })
    df.to_csv(fname, sep='\t', index=False)
    print('Saved to:', fname)

    
if __name__ == '__main__':
    ckpt_name = sys.argv[1]
    ckpt_path = sys.argv[2]
    dm_name = sys.argv[3]
    dm_dataloader = "train" if len(sys.argv) <= 4 else sys.argv[4]
    print(ckpt_name, ckpt_path)
    print(dm_name, dm_dataloader)

    dm_args = dict(
        batch_size=64 if 'nli' in dm_name else 8, 
        max_seq_len=512, 
        num_workers=1, 
        vocab_min_occurrences=1, 
        is_original=None,
        max_dataset_size=None,
    )
    
    # load tokenizer, data, and the model, and then get predictions for a specified dataloader ('train', 'val', 'test')
    outputs, tokenizer = predict(
        ckpt_path, 
        dm_name, 
        dm_args, 
        dataloader=dm_dataloader, 
        verbose=True, 
        disable_progress_bar=False, 
        return_tokenizer=True
    )

    # get originals
    orig_tokens = clean_tokens(tokenizer, unroll(outputs['tokens']))
    orig_labels = unroll(outputs['labels'])  # predictions for original inputs
    orig_predictions = torch.cat(outputs['predictions']).argmax(dim=-1).tolist()
    orig_z = unroll(outputs['z'])
    
    # compute accuracy
    y_pred = torch.tensor(orig_predictions)
    y_gold = torch.tensor(orig_labels)
    print('Orig acc:', (y_pred == y_gold).float().mean().item())

    # save everything
    save_explanations(
        f'data/rationales/{dm_name}_{dm_dataloader}_{ckpt_name}.tsv',
        orig_tokens,
        orig_labels,
        orig_predictions, 
        orig_z, 
    )
