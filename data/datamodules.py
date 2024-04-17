from pathlib import Path

import pytorch_lightning
import torch.utils.data
from datasets import ZincDataset
from featurizer import FeaturizerFactory
from torch_geometric.loader import DataLoader


class ZincDatamodule(pytorch_lightning.LightningDataModule):
    """
    Data module class for ZincDataset.
    """

    def __init__(
        self,
        data_path: Path,
        featurizer_factory: FeaturizerFactory,
        batch_size: int,
        num_workers: int = 8,
    ) -> None:
        """
        Constructs ZincDataModule.

        :param data_path: path to dataset file.
        :param featurizer_factory: factory creating some featurizer for molecule data.
        :param batch_size: batch size for Dataloader.
        :param num_workers: number of subprocesses which are used in Dataloader.
        """

        super().__init__()

        self.data = ZincDataset(data_path, featurizer_factory())
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns train DataLoader for ZincDateset.
        """

        return DataLoader(
            self.data,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
