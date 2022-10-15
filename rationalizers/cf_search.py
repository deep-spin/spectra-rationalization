import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from transformers import AutoTokenizer
import datasets as hf_datasets
import optuna

from rationalizers import constants
from rationalizers.data_modules import available_data_modules
from rationalizers.lightning_models import available_models
from rationalizers.utils import (
    setup_wandb_logger,
    load_torch_object
)

shell_logger = logging.getLogger(__name__)


def run(args):
    dict_args = vars(args)

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

    def objective(trial):
        # set hyperparam search space
        for h_name, h_values in dict_args["search_space"]:
            if isinstance(h_values[0], int):
                x = trial.suggest_int(h_name, *h_values)  # h_values = [low, high]
            elif isinstance(h_values[0], float):
                x = trial.suggest_float(h_name, *h_values)  # h_values = [low, high]
            elif isinstance(h_values[0], str):
                x = trial.suggest_categorical(h_name, h_values)  # h_values = [list of values]
            else:
                raise ValueError("Unknown type for hyperparam: {}".format(h_values[0]))
            dict_args[h_name] = x
            args.setattr(h_name, x)

        # reset dm.is_original
        dm.is_original = dict_args['is_original']

        # build trainer
        if "ckpt" in dict_args.keys() and dict_args["ckpt"] is not None:
            shell_logger.info("Building model: {}...".format(args.model))
            model_cls = available_models[args.model]
            model = model_cls(dm.tokenizer, dm.nb_classes, dm.is_multilabel, h_params=dict_args)
            shell_logger.info("Building trainer...")
            trainer = Trainer(resume_from_checkpoint=args.ckpt)
        else:
            callbacks = [
                EarlyStopping(
                    monitor=args.monitor,
                    mode=args.monitor_mode,
                    patience=args.monitor_patience,
                    verbose=True,
                )
            ]
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
                enable_checkpointing=False,
                callbacks=callbacks,
            )

        # start training
        shell_logger.info("Starting fit...")
        trainer.fit(model, dm)

        # perform test
        shell_logger.info("Starting test...")
        hf_datasets.logging.disable_progress_bar()

        if not hasattr(model, 'has_countertfactual_flow') or model.has_countertfactual_flow is False:
            shell_logger.info("Testing on counterfactuals...")
            dm.is_original = False
            outputs = trainer.test(datamodule=dm, verbose=True)
        else:
            shell_logger.info("Testing on all samples...")
            dm.is_original = None
            outputs = trainer.test(datamodule=dm, verbose=True)

        output_c = {k.replace('ff_', 'cf_'): v for k, v in outputs[0].items()}
        return output_c["test_cf_accuracy"]

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=args.n_trials)
    shell_logger.info("Number of finished trials: {}".format(len(study.trials)))

    best_trial = study.best_trial
    shell_logger.info("Best trial:")
    shell_logger.info("  Value: {}".format(best_trial.value))
    shell_logger.info("  Params: ")
    for key, value in best_trial.params.items():
        shell_logger.info("    {}: {}".format(key, value))

    # bye bye
    shell_logger.info("Bye bye!")
