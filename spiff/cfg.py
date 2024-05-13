"""Default experiment parameters."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

"""
The module defines a set of dataclasses that extend a base Config class.

The ExperimentConfig class is intended to be used as a top-level configuration.

The config class defines 'override' method that allows overwriting its fields' values
with values taken from a dict, that comes from a parsed JSON file.
The keys in the JSON should have exactly the same signature as fields of the Config
subclass, whose object is calling the override method. If a field's type is another
subclass of Config, the value for the respective key should be a JSON object, that is
appropriate the given subclass.

It is not necessary to overwrite all the fields.
"""


def default_chem_features():
    """Define the default chemical features to be extracted from atoms."""
    return [
        "atomicnum",
        "degree",
        "formalcharge",
        "hybridization",
        "isaromatic",
        "mass",
        "numimpliciths",
    ]


class Config(ABC):
    """Abstract base class that defines the override method."""

    def override(self, cfg: Dict[str, Union[str, int, float, dict]]) -> None:
        for key, val in cfg.items():
            if key not in self.__dict__:
                raise ValueError(f"key {key} is invalid for {self.__class__.__name__}")
            if isinstance(getattr(self, key), Config):
                getattr(self, key).override(val)
            else:
                setattr(self, key, val)


@dataclass(eq=False)
class ModelConfig(Config):
    gnn: str = "gin"
    gnn_layers: int = 5
    gnn_activation: str = "ReLU"
    gnn_activation_args: List = field(default_factory=list)
    readout: str = "mean"
    readout_args: List = field(default_factory=list)
    hidden_size: int = 512
    intermediate_size: int = 512
    linear_layer_sizes: List[int] = field(default_factory=lambda: [512] * 3)
    linear_activation: str = "LeakyReLU"
    linear_activation_args: List = field(default_factory=lambda: [0.2])
    latent_size: int = 256
    projection_head_size: Optional[int] = 128


@dataclass(eq=False)
class SystemConfig(Config):
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    dataset_path: str = "dataset/zinc_small.smi"
    results_dir: str = "results"


@dataclass(eq=False)
class ExperimentConfig(Config):
    learning_rate: float = 1e-4
    batch_size: int = 3 * 512
    epochs: int = 1000
    margin: float = 1.0
    use_force_field: bool = True
    chem_features: List[str] = field(default_factory=default_chem_features)
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig())
    system_config: SystemConfig = field(default_factory=lambda: SystemConfig())

    def validate(self, using_wandb: bool) -> None:
        if using_wandb and (
            not self.system_config.wandb_entity or not self.system_config.wandb_project
        ):
            raise ValueError("using wandb, but either entity or project is not set")
        if self.batch_size % 3 != 0:
            raise ValueError("setting batch size not divisible by 3")
