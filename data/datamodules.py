import os

import pytorch_lightning
import pytorch_lightning.utilities.types as pl_types
import torch.utils.data
import torch_geometric.loader

import data.datasets as datasets
from data.featurizer import Featurizer


class ZincDatamodule(pytorch_lightning.LightningDataModule):
    """PyTorch Lightning data module class for ZincDataset."""

    def __init__(
        self,
        data_path: str | os.PathLike,
        featurizer: Featurizer,
        batch_size: int,
        dataloader_num_workers: int = 4,
    ) -> None:
        """
        Construct the ZincDataModule.

        :param data_path: path to the dataset file.
        :param featurizer: featurizer to extract features from molecules.
        :param batch_size: batch size Dataloaders.
        :param dataloader_num_workers: number of subprocesses used in DataLoader.
        """

        super().__init__()
        self.data = datasets.ZincDataset(data_path, featurizer)
        self.data.generate_conformations()
        self.batch_size = batch_size
        self.num_workers = dataloader_num_workers

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
            pin_memory=True,
        )

    @property
    def dataset(self):
        return self.data


class BaselineDatamodule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        data: datasets.BaselineDatasetSplit,
        batch_size: int,
        dataloader_num_workers: int = 4,
    ) -> None:
        """
        Construct the DataModule.

        :param data: object with train and test datasets.
        :param batch_size: batch size for Dataloaders.
        :param dataloader_num_workers: number of subprocesses used in DataLoaders.
        """

        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = dataloader_num_workers

    def train_dataloader(self) -> pl_types.TRAIN_DATALOADERS:
        """
        Get the train DataLoader for a baseline experiment.

        :return: the train dataloader.
        """

        return torch_geometric.loader.DataLoader(
            self.data.train,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> pl_types.EVAL_DATALOADERS:
        """
        Get the test DataLoader for a baseline experiment.

        Used as pl_lightning validation dataloader, so the test epoch can be more
        frequently.

        :return: the test dataloader.
        """

        return torch_geometric.loader.DataLoader(
            self.data.test,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        return self.data.train

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        return self.data.test
