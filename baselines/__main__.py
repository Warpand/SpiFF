import json
import logging
import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from baselines.cfg import BaselineConfig
from baselines.cli import override_with_flags, parse_arguments
from baselines.factory import get_baseline_factory
from common.utils import get_run_name
from spiff.cfg import Config

logger = logging.getLogger(__name__)


def set_up_experiment_logger(
    config: BaselineConfig, name: str, use_wandb: bool
) -> pl_loggers.Logger:
    hyperparams = {
        key: val for key, val in config.__dict__.items() if not isinstance(val, Config)
    }
    hyperparams.update({"spiff": config.spiff_config.__dict__})
    if use_wandb:
        os.makedirs(cfg.system_config.results_dir, exist_ok=True)
        wandb_logger = pl_loggers.WandbLogger(
            project=config.system_config.wandb_project,
            entity=config.system_config.wandb_entity,
            name=name,
            save_dir=cfg.system_config.results_dir,
            tags=[cfg.type, cfg.dataset],
        )
        wandb_logger.experiment.config.update(hyperparams)
        return wandb_logger
    else:
        results_dir = os.path.join(config.system_config.results_dir, "csv")
        csv_logger = pl_loggers.CSVLogger(save_dir=results_dir, name=name)
        csv_logger.log_hyperparams(hyperparams)
        return csv_logger


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    args = parse_arguments()
    cfg = BaselineConfig()
    if args.config:
        logger.info(f"Overriding configuration with contents of {args.config}.")
        with open(args.config, "r") as cfg_file:
            overrides = json.load(cfg_file)
        try:
            cfg.override(overrides)
        except ValueError as e:
            logger.error(f"Error while overriding configuration: {e}.")
            sys.exit(1)
    override_with_flags(cfg, args)
    using_wandb = not args.no_wandb
    try:
        cfg.validate(using_wandb)
    except ValueError as e:
        logger.error(f"Error while validating configuration: {e}.")
        sys.exit(1)
    run_name = get_run_name(args.name)
    logger.info(f"Using run name: {run_name}.")
    logger.info("Setting up the PyTorchLightning logger.")
    pl_logger = set_up_experiment_logger(cfg, run_name, using_wandb)

    baseline_factory = get_baseline_factory(cfg)
    trainer = Trainer(
        accelerator=args.device,
        logger=pl_logger,
        max_epochs=cfg.epochs,
        default_root_dir=cfg.system_config.results_dir,
        enable_checkpointing=False,
        log_every_n_steps=50 if cfg.dataset.lower() != "bace" else 1,
    )

    trainer.fit(
        baseline_factory.get_experiment(), datamodule=baseline_factory.get_datamodule()
    )
