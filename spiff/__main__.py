import json
import logging
import os
import sys
from datetime import datetime

import coolname
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

import spiff.models as models
from chem.sim import SCSimilarity
from data.datamodules import ZincDatamodule
from data.featurizer import GraphFeaturizerFactory
from spiff.cfg import Config, ExperimentConfig
from spiff.cli import override_with_flags, parse_arguments
from spiff.experiments import SPiFFModule
from spiff.mining import TripletMiner

logger = logging.getLogger(__name__)


def get_run_name(base: str) -> str:
    if base:
        name = base
    else:
        name = coolname.generate_slug(2)
    return f"{name}-{datetime.now().strftime('%d.%m.%Y-%H:%M')}"


def set_up_experiment_logger(
    config: ExperimentConfig, name: str, use_wandb: bool
) -> pl_loggers.Logger:
    hyperparams = {
        key: val for key, val in config.__dict__.items() if not isinstance(val, Config)
    }
    hyperparams.update(config.model_config.__dict__)
    if use_wandb:
        os.makedirs(cfg.system_config.results_dir, exist_ok=True)
        wandb_logger = pl_loggers.WandbLogger(
            project=config.system_config.wandb_project,
            entity=config.system_config.wandb_entity,
            name=name,
            save_dir=cfg.system_config.results_dir,
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
    cfg = ExperimentConfig()
    if args.config:
        logger.info(f"Overriding configuration with contents of {args.config}.")
        with open(args.config, "r") as cfg_file:
            overrides = json.load(cfg_file)
        try:
            cfg.override(overrides)
        except ValueError as e:
            logger.error(f"Error while overriding configuration: {e}")
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

    # Constructing the experiment
    featurizer = GraphFeaturizerFactory(cfg.chem_features)()
    datamodule = ZincDatamodule(
        cfg.system_config.dataset_path, featurizer, cfg.batch_size
    )

    gnn_factory = models.GNNFactory(
        cfg.model_config.gnn,
        cfg.model_config.hidden_size,
        cfg.model_config.gnn_layers,
        models.ActivationFuncFactory(
            cfg.model_config.gnn_activation, *cfg.model_config.gnn_activation_args
        ),
    )
    mlp_factory = models.LinearModelFactory(
        cfg.model_config.linear_layer_sizes,
        models.ActivationFuncFactory(
            cfg.model_config.linear_activation, *cfg.model_config.linear_activation_args
        ),
    )
    spiff = models.SPiFF(
        featurizer.num_features(),
        cfg.model_config.intermediate_size,
        cfg.model_config.latent_size,
        gnn_factory,
        mlp_factory,
        models.ReadoutFuncFactory(
            cfg.model_config.readout, *cfg.model_config.readout_args
        ),
        cfg.model_config.projection_head_size,
    )

    experiment = SPiFFModule(
        spiff,
        torch.nn.TripletMarginLoss(cfg.margin),
        TripletMiner(SCSimilarity()),
        cfg.learning_rate,
        using_wandb,
    )

    trainer = Trainer(
        accelerator=args.device,
        logger=pl_logger,
        max_epochs=cfg.epochs,
        default_root_dir=cfg.system_config.results_dir,
        enable_checkpointing=True,
    )

    trainer.fit(experiment, datamodule=datamodule)
