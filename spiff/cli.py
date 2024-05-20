import argparse

import spiff.cfg as cfg


def parse_arguments() -> argparse.Namespace:
    """
    Parse sys.argv.

    :returns: Namespace object with values for the defined flags.
    """

    parser = argparse.ArgumentParser(
        prog="spiff", description="SpiFF - Spatial Features Fingerprint"
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
        default=cfg.SystemConfig.wandb_entity,
        help="your wandb entity",
        metavar="WANDB_ENTITY",
    )
    wandb_group.add_argument(
        "-P",
        "--project",
        default=cfg.SystemConfig.wandb_project,
        help="your wandb project name",
        metavar="WANDB_PROJECT",
    )
    wandb_group.add_argument("--no-wandb", action="store_true", help="do not use wandb")

    system_group = parser.add_argument_group("System", "Paths and system settings.")
    system_group.add_argument(
        "-D",
        "--data",
        default=cfg.SystemConfig.dataset_path,
        help="path to the dataset (default: %(default)s)",
        metavar="DATASET_PATH",
    )
    system_group.add_argument(
        "-R",
        "--results",
        default=cfg.SystemConfig.results_dir,
        help="path to directory where results are saved (default: %(default)s)",
        metavar="RESULTS_DIR",
    )
    system_group.add_argument(
        "-d",
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="device to use (default: %(default)s)",
    )

    params_group = parser.add_argument_group(
        "Experiment Parameters", "Hyperparameters of the experiment."
    )
    params_group.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=cfg.ExperimentConfig.learning_rate,
        help="learning rate of the gradient descent (default: %(default)s)",
    )
    params_group.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=cfg.ExperimentConfig.batch_size,
        help="batch size; should be divisible by 3 (default: %(default)s)",
    )
    params_group.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=cfg.ExperimentConfig.epochs,
        help="number of training epochs (default: %(default)s)",
    )
    params_group.add_argument(
        "-m",
        "--margin",
        type=float,
        default=cfg.ExperimentConfig.margin,
        help="margin of the Triplet Margin Loss (default: %(default)s)",
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


def override_with_flags(config: cfg.ExperimentConfig, args: argparse.Namespace) -> None:
    """
    Override configuration values with values from the commandline flags.

    :param config: ExperimentConfig object whose values are overridden.
    :param args: Namespace with values of the flags.
    """
    config.system_config.wandb_entity = args.entity
    config.system_config.wandb_project = args.project
    config.system_config.results_dir = args.results
    config.system_config.dataset_path = args.data

    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.epochs = args.epochs
    config.margin = args.margin
