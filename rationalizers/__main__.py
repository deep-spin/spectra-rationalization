import argparse
import os

from rationalizers import predict, train, resume, cf_train, cf_predict, cf_search
from rationalizers.utils import (
    configure_output_dir,
    configure_seed,
    configure_shell_logger,
    find_last_checkpoint_version,
    load_ckpt_config,
    load_yaml_config,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=["train", "cf_train", "predict", "cf_predict", "resume"])
    parser.add_argument("--config", type=str, help="Path to YAML config file.")
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to a saved model checkpoint. Used for `predict` only. Will overwrite config file's option.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If not specified, will be read from config file.",
    )
    tmp_args = parser.parse_args()
    tmp_dict_args = vars(tmp_args)

    config_dict = {}
    yaml_config_dict = load_yaml_config(tmp_args.config)
    general_dict = {
        "seed": yaml_config_dict["seed"] if tmp_args.seed is None else tmp_args.seed,
        "default_root_dir": yaml_config_dict["default_root_dir"],
    }

    # define args for each task
    if tmp_args.task in ["train", "cf_train"]:
        config_dict = {**general_dict, **yaml_config_dict["train"]}

    elif tmp_args.task in ["predict", "cf_predict"]:
        ckpt_path = None
        if tmp_args.ckpt is not None:
            ckpt_path = tmp_args.ckpt
        elif (
            "ckpt" in yaml_config_dict["predict"].keys()
            and yaml_config_dict["predict"]["ckpt"] is not None
        ):
            ckpt_path = yaml_config_dict["predict"]["ckpt"]
        else:
            ckpt_path = find_last_checkpoint_version(
                os.path.join(general_dict["default_root_dir"], "wandb/")
            )

        # in case a checkpoint was not defined or it doesnt exist
        # if ckpt_path is None:
        #     raise FileNotFoundError(
        #         "Checkpoint path not defined and there is no checkpoint saved in {}".format(
        #             os.path.join(general_dict["default_root_dir"], "wandb/")
        #         )
        #     )

        # load the config dict saved with the checkpoint
        ckpt_config_dict = load_ckpt_config(ckpt_path)

        # overwrite ckpt_config_dict with -> args from the yaml file -> general dict
        config_dict = {
            **ckpt_config_dict,
            **yaml_config_dict["predict"],
            **general_dict,
            "ckpt": ckpt_path,
        }

    elif tmp_args.task == "resume":
        # define path to checkpoint by following this priority:
        # 1. argparse
        # 2. yaml file
        # 3. find last checkpoint version
        ckpt_path = None
        if tmp_args.ckpt is not None:
            ckpt_path = tmp_args.ckpt

        # load the config dict saved with the checkpoint
        ckpt_config_dict = load_ckpt_config(ckpt_path)

        # overwrite ckpt_config_dict with -> args from the yaml file -> general dict
        config_dict = {
            **ckpt_config_dict,
            **yaml_config_dict["resume"],
            **general_dict,
            "ckpt": ckpt_path,
        }

    # define args
    args = argparse.Namespace(**config_dict)
    # configure general stuff: seed and output dir
    configure_seed(args.seed)

    # set a general default root dir in case it was not set by the user and create nested directories
    if args.default_root_dir is None:
        args.default_root_dir = os.path.join("experiments/", args.dm, args.model)
    args.default_root_dir = configure_output_dir(args.default_root_dir)

    # configure shell logger
    configure_shell_logger(args.default_root_dir)

    # train or predict!
    if tmp_args.task == "train":
        train.run(args)
    elif tmp_args.task == "cf_train":
        cf_train.run(args)
    elif tmp_args.task == "resume":
        resume.run(args)
    elif tmp_args.task == "predict":
        predict.run(args)
    elif tmp_args.task == "cf_predict":
        cf_predict.run(args)
    elif tmp_args.task == "search":
        pass
    elif tmp_args.task == "cf_search":
        cf_search.run(args)
