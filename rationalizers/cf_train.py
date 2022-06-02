import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer
import constants
import cf_constants

from rationalizers.data_modules import available_data_modules
from rationalizers.lightning_models import available_models
from rationalizers.utils import (
    setup_wandb_logger,
    save_config_to_csv,
    load_torch_object
)

shell_logger = logging.getLogger(__name__)


def run(args):
    dict_args = vars(args)

    tokenizer = None
    if args.tokenizer is not None:
        shell_logger.info("Loading tokenizer: {}...".format(args.tokenizer))
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        dict_args['max_length'] = tokenizer.model_max_length
        constants.update_constants(tokenizer)

    cf_tokenizer = tokenizer if args.share_tokenizers else None
    if cf_tokenizer is None and args.cf_tokenizer is not None:
        shell_logger.info("Loading tokenizer: {}...".format(args.cf_tokenizer))
        cf_tokenizer = AutoTokenizer.from_pretrained(args.cf_tokenizer)
        dict_args['max_length'] = cf_tokenizer.model_max_length
        cf_constants.update_constants(cf_tokenizer)

    shell_logger.info("Building data: {}...".format(args.dm))
    dm_cls = available_data_modules[args.dm]
    dm = dm_cls(d_params=dict_args, tokenizer=tokenizer, cf_tokenizer=cf_tokenizer, set_cf_inputs=True)
    if args.factual_ckpt is not None:
        # load factual rationalizer tokenizer and label encoder
        dm.load_encoders(
            root_dir=os.path.dirname(args.factual_ckpt),
            load_tokenizer=args.load_tokenizer and tokenizer is None,
            load_cf_tokenizer=False,
            load_label_encoder=args.load_label_encoder,
        )
    dm.prepare_data()
    dm.setup()

    shell_logger.info("Building board loggers...")
    logger = setup_wandb_logger(args.default_root_dir)

    if "ckpt" in dict_args.keys():
        shell_logger.info("Building model: {}...".format(args.model))
        model_cls = available_models[args.model]
        model = model_cls(dm.tokenizer, dm.cf_tokenizer, dm.nb_classes, dm.is_multilabel, h_params=dict_args)
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
                filename="{epoch}",
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
                filename="{epoch}",
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
            dm.tokenizer, dm.cf_tokenizer, dm.nb_classes, dm.is_multilabel, h_params=dict_args
        )

        if args.factual_ckpt is not None:
            shell_logger.info("Loading factual rationalizer from {}...".format(args.factual_ckpt))
            factual_state_dict = load_torch_object(args.factual_ckpt)['state_dict']
            model.load_state_dict(factual_state_dict, strict=False)

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
        shell_logger.info("Total params: {}".format(sum(p.numel() for p in model.parameters())))
        shell_logger.info("Learnable params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        save_config_to_csv(dict_args, version_path)

    # start training
    shell_logger.info("Starting fit...")
    trainer.fit(model, dm)

    # save encoders in the best model dir
    shell_logger.info("Saving encoders in {}".format(checkpoint_callback.dirpath))
    shell_logger.info("Saving tokenizer: {}...".format(args.save_tokenizer))
    shell_logger.info("Saving cf tokenizer: {}...".format(args.save_cf_tokenizer))
    shell_logger.info("Saving label encoder: {}...".format(args.save_label_encoder))
    dm.save_encoders(
        root_dir=checkpoint_callback.dirpath,
        save_tokenizer=args.save_tokenizer,
        save_cf_tokenizer=args.save_cf_tokenizer,
        save_label_encoder=args.save_label_encoder
    )

    # perform test
    shell_logger.info("Testing...")
    # load the best checkpoint automatically
    trainer.test(datamodule=dm, verbose=True)
