import logging
import os

from pytorch_lightning import Trainer
from transformers import AutoTokenizer
import datasets as hf_datasets

from rationalizers import constants
from rationalizers.data_modules import available_data_modules
from rationalizers.lightning_models import available_models

shell_logger = logging.getLogger(__name__)


def run(args):
    dict_args = vars(args)

    tokenizer = None
    if args.tokenizer is not None:
        shell_logger.info("Loading tokenizer: {}...".format(args.tokenizer))
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # /a/b/c.ext to a/b/
    checkpoint_dir = os.path.dirname(args.ckpt)

    # load data and tokenizer
    shell_logger.info("Building data: {}...".format(args.dm))
    dm_cls = available_data_modules[args.dm]
    dm = dm_cls(d_params=dict_args, tokenizer=tokenizer)
    shell_logger.info("Loading encoders from {}".format(checkpoint_dir))
    shell_logger.info("Loading tokenizer: {}...".format(args.load_tokenizer))
    shell_logger.info("Loading label encoder: {}...".format(args.load_label_encoder))
    dm.load_encoders(
        root_dir=checkpoint_dir,
        load_tokenizer=args.load_tokenizer and tokenizer is None,
        load_label_encoder=args.load_label_encoder,
    )

    # update constants
    constants.update_constants(dm.tokenizer)

    # rebuild model and load weights from last checkpoint
    shell_logger.info("Building model and loading checkpoint...")
    model_cls = available_models[args.model]
    model = model_cls.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        map_location=lambda storage, loc: storage,
        tokenizer=dm.tokenizer,
        nb_classes=dm.nb_classes,
        is_multilabel=dm.is_multilabel,
        h_params=dict_args,  # note that dict_args should match training's
    )

    # set the model to eval mode
    model.log_rationales_in_wandb = False
    model.generation_mode = True
    model.eval()

    # test
    shell_logger.info("Testing...")
    dm.is_original = dict_args.get("is_original", None)
    trainer = Trainer.from_argparse_args(args)
    trainer.test(model, datamodule=dm, verbose=True)
