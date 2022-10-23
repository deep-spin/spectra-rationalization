import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer
import datasets as hf_datasets

from rationalizers import constants
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

    shell_logger.info("Building board loggers...")
    logger = setup_wandb_logger(args.default_root_dir, project=args.wandb_project, entity=args.wandb_entity)

    tokenizer = None
    if args.tokenizer is not None:
        shell_logger.info("Loading tokenizer: {}...".format(args.tokenizer))
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    shell_logger.info("Building data: {}...".format(args.dm))
    dm_cls = available_data_modules[args.dm]
    dm = dm_cls(d_params=dict_args, tokenizer=tokenizer)
    if "factual_ckpt" in dict_args.keys() and dict_args['factual_ckpt'] is not None:
        # load rationalizer tokenizer and label encoder
        dm.load_encoders(
            root_dir=os.path.dirname(args.factual_ckpt),
            load_tokenizer=args.load_tokenizer and tokenizer is None,
            load_label_encoder=args.load_label_encoder,
        )

    # update constants
    constants.update_constants(dm.tokenizer)

    # if the tokenizer is not loaded, we need to setup the data module
    if dm.tokenizer is None:
        dm.prepare_data()
        dm.setup()

    shell_logger.info("Building callbacks...")
    callbacks = []
    early_stop_callback = EarlyStopping(
        monitor=args.monitor,
        mode=args.monitor_mode,
        patience=args.monitor_patience,
        verbose=True,
    )
    callbacks.append(early_stop_callback)

    # Disregard Early Stopping and save the model from last epoch
    do_early_stopping = dict_args.get("early_stopping", True)
    if not do_early_stopping:
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

    # load or define model
    if "ckpt" in dict_args.keys() and dict_args["ckpt"] is not None:
        shell_logger.info("Building model: {}...".format(args.model))
        model_cls = available_models[args.model]
        model = model_cls(dm.tokenizer, dm.nb_classes, dm.is_multilabel, h_params=dict_args)
        trainer = Trainer(resume_from_checkpoint=args.ckpt, logger=logger, callbacks=callbacks)
    else:

        shell_logger.info("Building model: {}...".format(args.model))
        model_cls = available_models[args.model]
        model = model_cls(dm.tokenizer, dm.nb_classes, dm.is_multilabel, h_params=dict_args)

        if "factual_ckpt" in dict_args.keys() and dict_args['factual_ckpt'] is not None:
            shell_logger.info("Loading rationalizer from {}...".format(args.factual_ckpt))
            factual_state_dict = load_torch_object(args.factual_ckpt)['state_dict']
            model.load_state_dict(factual_state_dict, strict=False)

        shell_logger.info("Building trainer...")
        trainer = Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
        )

        # log stuff
        shell_logger.info("Vocab size: {}".format(dm.tokenizer.vocab_size))
        shell_logger.info("Nb labels: {}".format(dm.nb_classes))
        shell_logger.info("Total params: {}".format(sum(p.numel() for p in model.parameters())))
        shell_logger.info("Learnable params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        version_path = os.path.split(checkpoint_callback.dirpath)[0]
        save_config_to_csv(dict_args, version_path)

    # start training
    if not args.no_fit:
        shell_logger.info("Starting fit...")
        trainer.fit(model, dm)

    # save encoders in the best model dir
    shell_logger.info("Saving encoders in {}".format(checkpoint_callback.dirpath))
    shell_logger.info("Saving tokenizer: {}...".format(args.save_tokenizer))
    shell_logger.info("Saving label encoder: {}...".format(args.save_label_encoder))
    dm.save_encoders(
        root_dir=checkpoint_callback.dirpath,
        save_tokenizer=args.save_tokenizer,
        save_label_encoder=args.save_label_encoder
    )

    # perform test
    shell_logger.info("Starting test...")
    hf_datasets.logging.disable_progress_bar()

    def log_final_outputs(final_outputs, flow_name=None, prefix_name=None):
        for stat_dict in final_outputs:
            new_stat_dict = {}
            for stat_name, stat_value in stat_dict.items():
                if flow_name is not None:
                    stat_name = stat_name.replace('ff_', flow_name + '_')
                if prefix_name is not None:
                    stat_name = stat_name.replace('test_', 'test_' + prefix_name + '_')
                new_stat_dict[stat_name] = stat_value
            logger.log_metrics(new_stat_dict)

    # perform test
    shell_logger.info("Testing on all samples...")
    dm.is_original = None
    outputs = trainer.test(datamodule=dm, verbose=True)
    log_final_outputs(outputs, flow_name=None, prefix_name=None)

    # log the best model path
    logger.log_hyperparams({'best_model_path': checkpoint_callback.best_model_path})

    # bye bye
    shell_logger.info("Bye bye!")
