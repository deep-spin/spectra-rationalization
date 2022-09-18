import csv
import logging
import os
import pickle
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.core.saving import load_hparams_from_tags_csv
from pytorch_lightning.loggers import WandbLogger
from torchnlp.encoders.text import StaticTokenizerEncoder
from torchnlp.word_to_vector import GloVe
from transformers import PreTrainedTokenizer

from rationalizers import constants


def configure_output_dir(output_dir: str):
    """
    Create a directory (recursively) and ignore errors if they already exist.

    :param output_dir: path to the output directory
    :return: output_path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def configure_seed(seed: int):
    """
    Seed everything: python, random, numpy, torch, torch.cuda.

    :param seed: seed integer (if None, a random seed will be created)
    :return: seed integer
    """
    seed = seed_everything(seed)
    return seed


def configure_shell_logger(output_dir: str):
    """Configure logger with a proper log format and save log to a file."""
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    if output_dir is not None:
        fh = logging.FileHandler(os.path.join(output_dir, "out.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)


def save_object(obj: object, path: str):
    """
    Dump an object (e.g. tokenizer or label encoder) via pickle.

    :param obj: any object (e.g. pytorch-nlp's tokenizer instance)
    :param path: path to save the object
    """
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_config_to_csv(dict_args: dict, path: str):
    """
    Save the meta data config to csv in the run folder as "meta_tags.csv"

    :param obj: dict with the data
    :param path: path to save the object
    """
    meta_tags_path = os.path.join(path, "meta_tags.csv")
    if not os.path.exists(path):
        os.mkdir(path)
    with open(meta_tags_path, "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_args.items():
            writer.writerow([key, value])


def load_object(path: str, not_exist_ok: bool = True):
    """
    Unpickle a saved object.

    :param path: path to a pickled object
    :param not_exist_ok: whether it is ok if the obj does not exist
    :return: the object
    """
    if not os.path.exists(path) and not_exist_ok:
        return None
    with open(path, "rb") as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def load_torch_object(path: str):
    return torch.load(path, map_location=lambda storage, loc: storage)


def load_yaml_config(path: str):
    """
    From: https://github.com/joeynmt/joeynmt/

    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dict
    """
    with open(path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def setup_wandb_logger(default_root_dir: str):
    """
    Function that sets the WanbLogger to be used.

    :param default_root_dir: logs save dir.
    """
    id = uuid.uuid4()
    return WandbLogger(
        project="SPECTRA",
        entity="deepspin-cf-rationalizers",
        save_dir=default_root_dir,
        # version=str(id.fields[1]),
    )


def find_last_checkpoint_version(path_to_logs: str):
    """Sort the log directory to pick the last timestamped checkpoint filename."""

    def get_time_from_version_name(name: str):
        # name format example `version_16-10-2020_08-12-48`
        timestamp = name[6:]
        return timestamp

    ckpt_versions = os.listdir(path_to_logs)
    if len(ckpt_versions) == 0:
        return None
    ckpt_versions.sort(key=get_time_from_version_name)

    ckpt_dir = os.path.join(path_to_logs, ckpt_versions[-1], "checkpoints/")
    ckpt_epochs = os.listdir(ckpt_dir)
    if len(ckpt_epochs) == 0:
        return None
    ckpt_epochs.sort(key=lambda x: int(x[6:].split(".")[0]))  # e.g. epoch=2.ckpt

    return os.path.join(ckpt_dir, ckpt_epochs[-1])


def load_ckpt_config(ckpt_path: str):
    """
    Load the .csv config file stored with the checkpoint and transform it to a dict object.
    :param ckpt_path: path to a saved checkpoint.
    :return: config dict
    """
    csv_config_dir = os.path.dirname(os.path.dirname(ckpt_path))
    csv_config_path = os.path.join(csv_config_dir, "meta_tags.csv")
    config_dict = load_hparams_from_tags_csv(csv_config_path)
    return config_dict


def get_rationales(
    tokenizer: object,
    input_ids: torch.LongTensor,
    z: torch.Tensor,
    lengths: torch.LongTensor,
):
    """
    Get rationales from a list of tokens masked by the generator's selection z.

    :param tokenizer:
    :param input_ids: ids LongTensor with shape [B, T]
    :param z: binary FloatTensor with shape [B, T]
    :param lengths: original length LongTensor of each sample with shape [B, ]

    :return: list of lists containing the selected rationales
    """
    z = z.cuda()
    selected_ids = (z * input_ids).long()
    if isinstance(tokenizer, StaticTokenizerEncoder):
        selected_rationales = tokenizer.batch_decode(selected_ids, lengths)
    else:
        selected_rationales = tokenizer.batch_decode(selected_ids)
        selected_rationales = [s[:l] for s, l in zip(selected_rationales, lengths.tolist())]

    return selected_ids, selected_rationales


def get_z_stats(z=None, mask=None):
    """
    From: https://github.com/bastings/interpretable_predictions

    Computes statistics about how many zs are
    exactly 0, continuous (between 0 and 1), or exactly 1.

    :param z:
    :param mask: mask in [B, T]
    :return:
    """
    z = z.cuda()
    z = torch.where(mask, z, z.new_full([1], 1e2))

    num_0 = (z == 0.0).sum().item()
    num_c = ((z > 0.0) & (z < 1.0)).sum().item()
    num_1 = (z == 1.0).sum().item()

    num_0 + num_c + num_1
    mask_total = mask.sum().item()
    # assert total == mask_total, "total mismatch"
    return num_0, num_c, num_1, mask_total


def load_glove_embeddings(vocab: list, name: str, emb_size: int):
    """
    Load pre-trained Glove embeddings using PyTorch-NLP interface:
    https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.word_to_vector.html

    :param vocab: list of tokens
    :param name: Glove name version (e.g. ‘840B’, ‘twitter.27B’, ‘6B’, ‘42B’)
    :param emb_size: word embedding size
    :return: Torch.FloatTensor with shape (vocab_size, emb_dim)
    """
    vocab_set = set(vocab)
    unk_vector = torch.FloatTensor(emb_size).uniform_(-0.05, 0.05)
    unk_init = lambda v: unk_vector
    pretrained_embedding = GloVe(
        name=name,
        dim=emb_size,
        unk_init=unk_init,
        is_include=lambda w: w in vocab_set,
    )
    embedding_weights = torch.FloatTensor(len(vocab), emb_size)
    for idx, token in enumerate(vocab):
        if token in [constants.PAD, constants.SOS, constants.EOS]:
            if token in pretrained_embedding.token_to_index:
                embedding_weights[idx] = pretrained_embedding[token]
            else:
                if token == constants.PAD:  # zero vector for padding
                    embedding_weights[idx] = torch.zeros(emb_size)
                else:  # random token for everything else
                    embedding_weights[idx] = torch.FloatTensor(emb_size).uniform_(
                        -0.05, 0.05
                    )
        else:
            embedding_weights[idx] = pretrained_embedding[token]
    return embedding_weights


def unroll(list_of_lists, rec=False):
    """
    Unroll a list of lists
    Args:
        list_of_lists (list): a list that contains lists
        rec (bool): unroll recursively
    Returns:
        a single list
    """
    if not isinstance(list_of_lists[0], (np.ndarray, list, torch.Tensor)):
        return list_of_lists
    new_list = [item for ell in list_of_lists for item in ell]
    if rec and isinstance(new_list[0], (np.ndarray, list, torch.Tensor)):
        return unroll(new_list, rec=rec)
    return new_list


def freeze_module(module: torch.nn.Module, ignored_weights: list = None):
    if ignored_weights is None:
        ignored_weights = []
    for name, param in module.named_parameters():
        if len(ignored_weights) > 0 and any(w in name for w in ignored_weights):
            continue
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = True


def is_trainable(module: torch.nn.Module):
    if module is None:
        return None
    return any(p.requires_grad for p in module.parameters())


def masked_average(tensor, mask):
    """ Performs masked average of a given tensor at time dim. """
    tensor_sum = (tensor * mask.float().unsqueeze(-1)).sum(1)
    tensor_mean = tensor_sum / mask.sum(-1).float().unsqueeze(-1)
    return tensor_mean


def get_html_rationales(all_tokens, all_scores, all_gold_labels, all_pred_labels, all_lengths):
    def colorize_two_way(tokens, scores, gold_l, pred_l, leng):
        template_pos = '<span style="color: black; background-color: rgba(0, 255, 0, {}); ' \
                       'display:inline-block; font-size:12px;">&nbsp {} &nbsp</span>'
        template_neg = '<span style="color: black; background-color: rgba(255, 0, 0, {}); ' \
                       'display:inline-block; font-size:12px;">&nbsp {} &nbsp</span>'
        text = ''
        f = lambda w: w.replace('<', 'ᐸ').replace('>', 'ᐳ')
        for word, color in zip(tokens[:leng], scores[:leng]):
            if color >= 0:
                text += template_pos.format(color, f(word))
            else:
                text += template_neg.format(-color, f(word))
        html_text = '<div style="width:100%">g: {} | p: {}:&nbsp;&nbsp; {}</div>'.format(
            gold_l, pred_l.argmax(), text
        )
        return html_text
    html_texts = []
    for t, s, gl, pl, l in zip(all_tokens, all_scores, all_gold_labels, all_pred_labels, all_lengths):
        html_texts.append(colorize_two_way(t, s, gl, pl, l))
    return '<br>'.join(html_texts)


def save_rationales(filename, all_scores, all_lengths):
    f = open(filename, 'w', encoding='utf8')
    for scores, leng in zip(all_scores, all_lengths):
        text = ' '.join(['{:.4f}'.format(z) for z in scores[:leng].tolist()])
        f.write(text + '\n')
    f.close()


def save_counterfactuals(filename, all_pieces, all_lengths):
    f = open(filename, 'w', encoding='utf8')
    for pieces, leng in zip(all_pieces, all_lengths):
        text = ' '.join(pieces[:leng])
        f.write(text + '\n')
    f.close()
