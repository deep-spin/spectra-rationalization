import argparse
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
from rationalizers.utils import load_ckpt_config, load_torch_object
from rationalizers.utils import unroll


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


def trim(text):
    text = re.sub(r'\ +', ' ', text)
    text = re.sub(r'(</s> )+', '</s> ', text)
    text = text.replace('</s> </s>', '</s>')
    return text.strip()


def tokens_to_text(raw_tokens):
    texts = []
    for tks in raw_tokens:
        texts.append(' '.join(['<unk>' if t is None else t for t in tks]))
    return texts


def predict(
    ckpt_path,
    factual_path,
    cf_generate_kwargs,
    dm_name,
    dm_args,
    dataloader='train',
    verbose=True,
    disable_progress_bar=False,
    return_tokenizer=False,
    backwards_compat=False,
    sparsemap_budget=None
):
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
        save_rationales=True,
        save_edits=True,
        cf_classify_edits=False,
        cf_generate_kwargs=cf_generate_kwargs
    )
    new_args = {**base_args, **dm_args}
    if sparsemap_budget is not None:
        new_args['sparsemap_budget'] = sparsemap_budget

    args = get_args_from_ckpt(ckpt_path, new_args)

    # fix cf_explainer_mask_token_type_id
    if hasattr(args, 'explainer_mask_token_type_id'):
        args.explainer_mask_token_type_id = None if args.explainer_mask_token_type_id == '' else args.explainer_mask_token_type_id
    if hasattr(args, 'cf_explainer_mask_token_type_id'):
        args.cf_explainer_mask_token_type_id = None if args.cf_explainer_mask_token_type_id == '' else args.cf_explainer_mask_token_type_id
    pprint(vars(args))
    pprint(dm_args)

    # set global seed
    configure_seed(args.seed)

    # load tokenizer and model
    tokenizer, _, model = load(args)

    # factual model
    if factual_path is not None:
        print("Loading factual rationalizer from {}...".format(factual_path))
        factual_state_dict = load_torch_object(factual_path)['state_dict']
        model.load_state_dict(factual_state_dict, strict=False)

    # load data module
    dm_cls = available_data_modules[dm_name]
    dm = dm_cls(d_params=dm_args, tokenizer=tokenizer)
    dm.prepare_data()
    dm.setup()
    
    # predict
    model.generation_mode = True
    model.log_rationales_in_wandb = False
    model.backwards_compat = backwards_compat
    trainer = Trainer(accelerator='gpu', devices=1)
    if dataloader == 'train':
        outputs = trainer.predict(model, dm.train_dataloader(shuffle=False))
    elif dataloader == 'val':
        outputs = trainer.predict(model, dm.val_dataloader())
    else:
        outputs = trainer.predict(model, dm.test_dataloader())

    # stack outputs
    outputs = {k: [x[k] for x in outputs] for k in outputs[0].keys()}

    if return_tokenizer:
        return outputs, tokenizer
    return outputs


def save_edits(
    fname,
    orig_texts,
    orig_labels,
    orig_predictions,
    orig_z,
    edits_texts,
    edits_labels,
    edits_predictions,
    edits_z_pre,
    edits_z_pos,
):
    df = pd.DataFrame({
        'orig_texts': orig_texts,
        'orig_labels': orig_labels,
        'orig_predictions': orig_predictions,
        'orig_z': [z_.detach().cpu().tolist() for z_ in orig_z],
        'edits_texts': edits_texts,
        'edits_labels': edits_labels,
        'edits_predictions': edits_predictions,
        'edits_z_pre': [z_.detach().cpu().tolist() for z_ in edits_z_pre],
        'edits_z_pos': [z_.detach().cpu().tolist() for z_ in edits_z_pos],
    })
    df.to_csv(fname, sep='\t', index=False)
    print('Saved to:', fname)


def get_edits(
    ckpt_path,
    factual_path,
    dm_name,
    dm_dataloader,
    dm_args,
    cf_generate_kwargs,
    backwards_compat=False,
    sparsemap_budget=None
):
    # set seed
    configure_seed(0)

    # empty cache (beam search uses a lot of caching)
    torch.cuda.empty_cache()

    # load tokenizer, data, and the model, and then get predictions for a specified dataloader ('train', 'val', 'test')
    outputs, tokenizer = predict(
        ckpt_path,
        factual_path,
        cf_generate_kwargs,
        dm_name,
        dm_args,
        dataloader=dm_dataloader,
        verbose=True,
        disable_progress_bar=False,
        return_tokenizer=True,
        backwards_compat=backwards_compat,
        sparsemap_budget=sparsemap_budget
    )

    # get originals
    orig_texts = tokens_to_text(unroll(outputs['texts']))
    orig_labels = unroll(outputs['labels'])
    orig_predictions = torch.cat(outputs['predictions']).argmax(dim=-1).tolist()  # predictions for original inputs
    orig_z = unroll(outputs['z'])  # the z given to the original input by the rationalizer

    # get edits
    edits_texts = tokens_to_text(unroll(outputs['edits']))
    edits_labels = unroll(outputs['edits_labels'])
    edits_predictions = torch.cat(outputs['edits_predictions']).argmax(dim=-1).tolist()  # predictions for edits
    edits_z_pre = unroll(outputs['edits_z'])  # before passing through the rationalizer to mask tokens-to-be-edited
    edits_z_pos = unroll(outputs['edits_z_pos'])  # the z given to the edit by the rationalizer

    return {
        'orig_texts': orig_texts,
        'orig_labels': orig_labels,
        'orig_predictions': orig_predictions,
        'orig_z': orig_z,
        'edits_texts': edits_texts,
        'edits_labels': edits_labels,
        'edits_predictions': edits_predictions,
        'edits_z_pre': edits_z_pre,
        'edits_z_pos': edits_z_pos
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-name", type=str, help="Name used to save edits.", required=True)
    parser.add_argument("--ckpt-path", type=str, help="Path to the editor checkpoint.", required=True)
    parser.add_argument("--ckpt-path-factual", type=str, help="Path to the factual rationalizer ckpt.", default=None)
    parser.add_argument("--dm-name", type=str, help="Name of the data module.", required=True)
    parser.add_argument("--dm-dataloader", type=str, help="Name of the dataloader to use.", default='test')
    parser.add_argument("--num-beams", type=int, help="Number of beams to use for beam search.", default=15)
    parser.add_argument("--do-sample", action='store_true', help="Whether to use sampling instead of beam search.")
    parser.add_argument("--sparsemap-budget", type=int, help="Budget for sparsemap.", default=None)
    parser.add_argument("--backwards-compat", action='store_true', help="Whether to use backwards compatibility mode.")
    parser.add_argument("--ignore-neutrals", action='store_true', help="Whether to ignore neutral examples.")
    parser.add_argument("--random-subset-dirpath", type=str, help="Path to the dir of a subset of a dataset.", default=None)
    args = parser.parse_args()

    ckpt_name = args.ckpt_name
    ckpt_path = args.ckpt_path
    factual_path = args.ckpt_path_factual
    dm_name = args.dm_name
    dm_dataloader = args.dm_dataloader
    num_beams = args.num_beams
    do_sample = args.do_sample
    sparsemap_budget = args.sparsemap_budget
    backwards_compat = args.backwards_compat
    ignore_neutrals = args.ignore_neutrals
    random_subset_dirpath = args.random_subset_dirpath
    
    dm_args = dict(
        batch_size=16,
        max_seq_len=512,
        num_workers=1,
        vocab_min_occurrences=1,
        is_original=True,
        max_dataset_size=None,
        ignore_neutrals=ignore_neutrals,
        path=random_subset_dirpath,
    )
    cf_generate_kwargs = dict(
        do_sample=do_sample,
        num_beams=num_beams,
        num_beam_groups=1,
        early_stopping=True,
        length_penalty=1.0,
        top_k=50,
        top_p=0.9,
        typical_p=None,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        min_length=None,
        max_length=512,
    )

    out_dict = get_edits(
        ckpt_path,
        factual_path,
        dm_name,
        dm_dataloader,
        dm_args,
        cf_generate_kwargs,
        backwards_compat,
        sparsemap_budget=sparsemap_budget
    )

    sample_mode = 'sample' if cf_generate_kwargs['do_sample'] else 'beam'
    num_beams = cf_generate_kwargs['num_beams']
    save_edits(
        f'data/edits/{dm_name}_{dm_dataloader}_{sample_mode}_{num_beams}_{ckpt_name}_raw.tsv',
        out_dict['orig_texts'],
        out_dict['orig_labels'],
        out_dict['orig_predictions'],
        out_dict['orig_z'],
        out_dict['edits_texts'],
        out_dict['edits_labels'],
        out_dict['edits_predictions'],
        out_dict['edits_z_pre'],
        out_dict['edits_z_pos']
    )

    # compute accuracy
    y_pred = np.array(out_dict['orig_predictions'])
    y_gold = np.array(out_dict['orig_labels'])
    y_edit_pred = np.array(out_dict['edits_predictions'])
    y_edit_gold = np.array(out_dict['edits_labels'])
    print('Orig acc:', np.mean(y_pred == y_gold))
    print('Edit acc:', np.mean(y_edit_pred == y_edit_gold))
    print('Cont acc:', np.mean(y_edit_pred != y_gold))
