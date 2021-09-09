import logging
import os

from pytorch_lightning import Trainer

from rationalizers.data_modules import available_data_modules
from rationalizers.lightning_models import available_models

shell_logger = logging.getLogger(__name__)


def run(args):
    dict_args = vars(args)

    checkpoint_dir = os.path.dirname(args.ckpt)

    # load data and tokenizer
    shell_logger.info("Building data: {}...".format(args.dm))
    dm_cls = available_data_modules[args.dm]
    dm = dm_cls(d_params=dict_args)
    shell_logger.info("Loading encoders from {}".format(checkpoint_dir))
    shell_logger.info("Loading tokenizer: {}...".format(args.load_tokenizer))
    shell_logger.info("Loading label encoder: {}...".format(args.load_label_encoder))
    dm.load_encoders(
        checkpoint_dir,
        load_tokenizer=args.load_tokenizer,
        load_label_encoder=args.load_label_encoder,
    )
    dm.prepare_data()
    dm.setup()

    # rebuild model and load weights from last checkpoint
    shell_logger.info("Building model and loading checkpoint...")
    model_cls = available_models[args.model]
    model = model_cls.load_from_checkpoint(
        args.ckpt,
        tokenizer=dm.tokenizer,
        nb_classes=dm.nb_classes,
        is_multilabel=dm.is_multilabel,
        h_params=dict_args,  # note that dict_args should match training's
    )

    # resume training
    shell_logger.info("Resuming training...")
    trainer = Trainer.from_argparse_args(args, resume_from_checkpoint=args.ckpt)
    trainer.fit(model, datamodule=dm)

    # perform test
    shell_logger.info("Testing...")
    # load the best checkpoint automatically
    trainer.test(datamodule=dm, verbose=True)

