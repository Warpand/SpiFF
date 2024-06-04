from dataclasses import dataclass, field
from typing import List, Optional

import spiff.cfg

TYPES = ["frozen", "tuned", "random", "trained"]
DATASETS = ["HIV", "BACE", "QM9"]


@dataclass(eq=False)
class BaselineSystemConfig(spiff.cfg.Config):
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    results_dir: str = "results"
    checkpoint_path: Optional[str] = None


@dataclass(eq=False)
class BaselineConfig(spiff.cfg.Config):
    spiff_config: spiff.cfg.ModelConfig = field(
        default_factory=lambda: spiff.cfg.ModelConfig()
    )
    system_config: BaselineSystemConfig = field(
        default_factory=lambda: BaselineSystemConfig()
    )
    learning_rate: float = 1e-4
    batch_size: int = 1024
    epochs: int = 100
    chem_features: List[str] = field(default_factory=spiff.cfg.default_chem_features)
    dataset: str = "bace"
    type: str = "frozen"
    model_layers: List[int] = field(default_factory=lambda: [256, 256])
    model_activation: str = "ReLU"
    model_activation_args: List = field(default_factory=list)
    model_use_batch_norm: bool = True

    def validate(self, using_wandb: bool) -> None:
        if using_wandb and (
            not self.system_config.wandb_entity or not self.system_config.wandb_project
        ):
            raise ValueError("using wandb, but either entity or project is not set")
        if self.dataset.lower() not in map(str.lower, DATASETS):
            raise ValueError(f"unknown dataset: {self.dataset}")
        type_case_insensitive = self.type.lower()
        if type_case_insensitive not in TYPES:
            raise ValueError(f"unknown baseline type: {self.type}")
        if (
            type_case_insensitive == "frozen" or type_case_insensitive == "tuned"
        ) and self.system_config.checkpoint_path is None:
            raise ValueError(
                f"Baseline type is {self.type}, but checkpoint path is not specified"
            )
