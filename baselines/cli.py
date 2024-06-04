import argparse

import baselines.cfg as cfg

"""TO DO"""


def parse_arguments() -> argparse.Namespace:
    """
    Parse sys.argv.

    :returns: Namespace object with values for the defined flags.
    """

    parser = argparse.ArgumentParser(
        prog="spiff - baselines", description="Baseline experiments for SpiFF"
    )

    config_group = parser.add_argument_group(
        "Configuration", "Configuration overloading."
    )
    config_group.add_argument(
        "-C",
        "--config",
        help="path to a json file with configuration overload",
        metavar="CONFIG_PATH",
    )

    wandb_group = parser.add_argument_group(
        "Wandb",
        "Flags relating to the usage of Weights & Biases. This flags are mandatory, "
        "unless --no-wandb is used or their values are overloaded in the configuration "
        "file.",
    )
    wandb_group.add_argument(
        "-E",
        "--entity",
        default=cfg.BaselineSystemConfig.wandb_entity,
        help="your wandb entity",
        metavar="WANDB_ENTITY",
    )
    wandb_group.add_argument(
        "-P",
        "--project",
        default=cfg.BaselineSystemConfig.wandb_project,
        help="your wandb project name",
        metavar="WANDB_PROJECT",
    )
    wandb_group.add_argument("--no-wandb", action="store_true", help="do not use wandb")

    system_group = parser.add_argument_group("System", "Paths and system settings.")
    system_group.add_argument(
        "-R",
        "--results",
        default=cfg.BaselineSystemConfig.results_dir,
        help="path to directory where results are saved (default: %(default)s)",
        metavar="RESULTS_DIR",
    )
    system_group.add_argument(
        "-d",
        "--device",
        choices=["cuda", "cpu", "mps", "auto"],
        default="cuda",
        help="device to use (default: %(default)s)",
    )
    system_group.add_argument(
        "--checkpoint",
        help="path to SpiFF checkpoint. "
        "Necessary for frozen and tuned types of experiment.",
    )

    params_group = parser.add_argument_group(
        "Experiment Parameters", "Hyperparameters of the experiment."
    )
    params_group.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=cfg.BaselineConfig.learning_rate,
        help="learning rate of the gradient descent (default: %(default)s)",
    )
    params_group.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=cfg.BaselineConfig.batch_size,
        help="batch size; should be divisible by 3 (default: %(default)s)",
    )
    params_group.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=cfg.BaselineConfig.epochs,
        help="number of training epochs (default: %(default)s)",
    )
    params_group.add_argument(
        "-t",
        "--type",
        choices=cfg.TYPES,
        default=cfg.BaselineConfig.type,
        help="type of the baseline experiment",
    )
    params_group.add_argument(
        "-s",
        "--set",
        choices=cfg.DATASETS,
        default=cfg.BaselineConfig.dataset,
        help="dataset used during the experiment",
    )

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "-n",
        "--name",
        help="run name; "
        "by default a random coolname is generated; "
        "always appended with a timestamp",
        metavar="RUN_NAME",
    )

    return parser.parse_args()


def override_with_flags(config: cfg.BaselineConfig, args: argparse.Namespace) -> None:
    config.system_config.wandb_entity = args.entity
    config.system_config.wandb_project = args.project
    config.system_config.results_dir = args.results
    config.system_config.checkpoint_path = args.checkpoint

    config.dataset = args.set
    config.type = args.type

    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.epochs = args.epochs
