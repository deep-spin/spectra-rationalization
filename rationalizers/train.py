import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from rationalizers.data_modules import available_data_modules
from rationalizers.lightning_models import available_models
from rationalizers.utils import (
    setup_wandb_logger,
    save_config_to_csv,
)

shell_logger = logging.getLogger(__name__)


def run(args):
    dict_args = vars(args)
    shell_logger.info("Building data: {}...".format(args.dm))
    dm_cls = available_data_modules[args.dm]
    dm = dm_cls(d_params=dict_args)
    dm.prepare_data()
    dm.setup()

    shell_logger.info("Building board loggers...")
    logger = setup_wandb_logger(args.default_root_dir)

    if "ckpt" in dict_args.keys():
        shell_logger.info("Building model: {}...".format(args.model))
        model_cls = available_models[args.model]
        model = model_cls(
            dm.tokenizer, dm.nb_classes, dm.is_multilabel, h_params=dict_args
        )
        trainer = Trainer(resume_from_checkpoint=args.ckpt)
    else:
        shell_logger.info("Building callbacks...")
        callbacks = []
        early_stop_callback = EarlyStopping(
            monitor=args.monitor,
            mode=args.monitor_mode,
            patience=args.monitor_patience,
            verbose=True,
        )
        callbacks.append(early_stop_callback)

        early_stopping = (
            dict_args["early_stopping"]
            if "early_stopping" in dict_args.keys()
            else True
        )

        # Disregard Early Stopping and save the model from last epoch
        if not early_stopping:
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(
                    args.default_root_dir,
                    f"version{logger.version}",
                    "checkpoints",
                ),
                filename="{}-{}-{epoch}-{val_sum_loss:.2f}".format(args.seed, args.transition),
                verbose=True,
                save_last=True,
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(
                    args.default_root_dir,
                    f"version{logger.version}",
                    "checkpoints",
                ),
                filename="{epoch}-{val_sum_loss:.2f}",
                monitor=args.monitor,
                verbose=True,
                mode=args.monitor_mode,
                save_top_k=1,
            )
        callbacks.append(checkpoint_callback)
        version_path = os.path.split(checkpoint_callback.dirpath)[0]
        shell_logger.info("Building model: {}...".format(args.model))
        model_cls = available_models[args.model]
        model = model_cls(
            dm.tokenizer, dm.nb_classes, dm.is_multilabel, h_params=dict_args
        )

        shell_logger.info("Building trainer...")
        trainer = Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
            checkpoint_callback=checkpoint_callback,
            weights_summary="full",
        )

        # log stuff
        shell_logger.info("Vocab size: {}".format(dm.tokenizer.vocab_size))
        shell_logger.info("Nb labels: {}".format(dm.nb_classes))
        shell_logger.info(
            "Total params: {}".format(sum(p.numel() for p in model.parameters()))
        )
        shell_logger.info(
            "Learnable params: {}".format(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            )
        )
        save_config_to_csv(dict_args, version_path)

    # start training
    shell_logger.info("Starting fit...")
    trainer.fit(model, dm)

    # save encoders in the best model dir
    shell_logger.info("Saving encoders in {}".format(checkpoint_callback.dirpath))
    shell_logger.info("Saving tokenizer: {}...".format(args.save_tokenizer))
    shell_logger.info("Saving label encoder: {}...".format(args.save_label_encoder))
    dm.save_encoders(
        checkpoint_callback.dirpath, args.save_tokenizer, args.save_label_encoder
    )

    # perform test
    shell_logger.info("Testing...")
    # load the best checkpoint automatically
    trainer.test(datamodule=dm, verbose=True)
