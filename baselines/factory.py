import logging
import sys
from abc import ABC, abstractmethod

import torch

import baselines.experiments as experiments
import data.datasets as datasets
from baselines.cfg import BaselineConfig
from common.utils import freeze
from data.datamodules import BaselineDatamodule
from data.featurizer import GraphFeaturizerFactory
from spiff import models

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger = logging.getLogger(__name__)


class SpiFFWrapper(models.SPiFF):
    """
    Ugly hack to avoid being automatically set to train mode by PyTorch Lightning,
    when SpiFF is supposed to be frozen.
    """

    def train(self, mode: bool = True) -> Self:
        if not mode:
            return super().train(False)
        return self


class BaselineExperimentFactory(ABC):
    def __init__(self, cfg: BaselineConfig) -> None:
        self.cfg = cfg
        self.featurizer = GraphFeaturizerFactory(cfg.chem_features)()

    def is_spiff_frozen(self):
        return self.cfg.type.lower() in ["frozen", "random"]

    def is_checkpoint_needed(self):
        return self.cfg.type.lower() in ["tuned", "frozen"]

    def _get_spiff(self) -> models.SPiFF:
        gnn_factory = models.GNNFactory(
            self.cfg.spiff_config.gnn,
            self.cfg.spiff_config.hidden_size,
            self.cfg.spiff_config.gnn_layers,
            models.ActivationFuncFactory(
                self.cfg.spiff_config.gnn_activation,
                *self.cfg.spiff_config.gnn_activation_args,
            ),
        )
        mlp_factory = models.LinearModelFactory(
            self.cfg.spiff_config.linear_layer_sizes,
            models.ActivationFuncFactory(
                self.cfg.spiff_config.linear_activation,
                *self.cfg.spiff_config.linear_activation_args,
            ),
        )
        spiff_class = models.SPiFF if not self.is_spiff_frozen() else SpiFFWrapper
        spiff = spiff_class(
            self.featurizer.num_features(),
            self.cfg.spiff_config.intermediate_size,
            self.cfg.spiff_config.latent_size,
            gnn_factory,
            mlp_factory,
            models.ReadoutFuncFactory(
                self.cfg.spiff_config.readout, *self.cfg.spiff_config.readout_args
            ),
            None,
        )
        if self.is_checkpoint_needed():
            checkpoint_path = self.cfg.system_config.checkpoint_path
            logger.info(f"Loading weights from checkpoint at {checkpoint_path}")
            state_dict = torch.load(checkpoint_path)["state_dict"]
            spiff.load_state_dict(
                {
                    key.removeprefix("model."): val
                    for key, val in state_dict.items()
                    if "projection_head" not in key
                }
            )
        if self.is_spiff_frozen():
            logger.info("Freezing the SpiFF model.")
            freeze(spiff)
            spiff.eval()
        return spiff

    def _get_model(self) -> torch.nn.Module:
        model_factory = models.LinearModelFactory(
            self.cfg.model_layers,
            models.ActivationFuncFactory(
                self.cfg.model_activation, *self.cfg.model_activation_args
            ),
            self.cfg.model_use_batch_norm,
        )
        return model_factory(self.cfg.spiff_config.latent_size, 1)

    @abstractmethod
    def get_datamodule(self) -> BaselineDatamodule:
        pass

    @abstractmethod
    def get_experiment(self) -> experiments.BaselineExperiment:
        pass


class QM9Factory(BaselineExperimentFactory):
    def get_datamodule(self) -> BaselineDatamodule:
        return BaselineDatamodule(
            datasets.QM9Data(self.featurizer), self.cfg.batch_size
        )

    def get_experiment(self) -> experiments.BaselineExperiment:
        return experiments.RegressionExperiment(
            self._get_spiff(),
            self._get_model(),
            self.cfg.learning_rate,
            not self.is_spiff_frozen(),
        )


class BACEFactory(BaselineExperimentFactory):
    def get_datamodule(self) -> BaselineDatamodule:
        return BaselineDatamodule(
            datasets.BACEData(self.featurizer), self.cfg.batch_size
        )

    def get_experiment(self) -> experiments.BaselineExperiment:
        return experiments.RegressionExperiment(
            self._get_spiff(),
            self._get_model(),
            self.cfg.learning_rate,
            not self.is_spiff_frozen(),
        )


class HIVFactory(BaselineExperimentFactory):
    def get_datamodule(self) -> BaselineDatamodule:
        return BaselineDatamodule(
            datasets.HIVData(self.featurizer), self.cfg.batch_size
        )

    def get_experiment(self) -> experiments.BaselineExperiment:
        return experiments.ClassificationExperiment(
            self._get_spiff(),
            self._get_model(),
            self.cfg.learning_rate,
            not self.is_spiff_frozen(),
        )


def get_baseline_factory(cfg: BaselineConfig) -> BaselineExperimentFactory:
    dataset = cfg.dataset.lower()
    if dataset == "qm9":
        return QM9Factory(cfg)
    elif dataset == "bace":
        return BACEFactory(cfg)
    elif dataset == "hiv":
        return HIVFactory(cfg)
    else:
        raise ValueError()
