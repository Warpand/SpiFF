import os

import pytorch_lightning
import torch.utils.data
import torch_geometric.loader

from data.datasets import ZincDataset
from data.featurizer import Featurizer


class ZincDatamodule(pytorch_lightning.LightningDataModule):
    """PyTorch Lightning data module class for ZincDataset."""

    def __init__(
        self,
        data_path: str | os.PathLike,
        featurizer: Featurizer,
        batch_size: int,
        num_workers: int = 8,
    ) -> None:
        """
        Construct the ZincDataModule.

        :param data_path: path to the dataset file.
        :param featurizer: featurizer to extract features from molecules.
        :param batch_size: batch size for a Dataloader.
        :param num_workers: number of subprocesses which are used in DataLoader.
        """

        super().__init__()

        self.data = None
        self.data_path = data_path
        self.featurizer = featurizer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        self.data = ZincDataset(self.data_path, self.featurizer)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Get the train DataLoader for ZincDateset.

        :returns: the train DataLoader.
        """

        return torch_geometric.loader.DataLoader(
            self.data,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )
