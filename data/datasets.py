import logging
import multiprocessing
import os
import pathlib
from typing import Iterable, List, SupportsIndex, Tuple

import pandas as pd
import torch
import torch_geometric.data
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import chem.utils
from data.featurizer import Featurizer, GraphFeatures

logger = logging.getLogger(__name__)


class ZincDataset(Dataset):
    """Dataset representing molecules from Zinc database."""

    def __init__(self, data_path: str | os.PathLike, featurizer: Featurizer) -> None:
        """
        Construct the ZincDataset.

        :param data_path: path to the dataset file.
        :param featurizer: featurizer to extract features from molecules.
        """

        super().__init__()
        logger.info(f"Extracting chemical features from {data_path}.")
        df = pd.read_csv(data_path, sep=" ")
        self.molecules = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"]]
        self.feature_data = [featurizer.extract_features(mol) for mol in self.molecules]

    def __getitem__(self, index: int) -> Tuple[int, torch_geometric.data.Data]:
        """
        Get the features for a molecule at the given index and the index itself.

        The index is returned, so it can be used to extract molecule info.

        :param index: index of the data sample.
        :returns: data sample at the index and the index.
        """

        data_sample: GraphFeatures = self.feature_data[index]
        return (
            index,
            torch_geometric.data.Data(
                data_sample.node_features, data_sample.edge_index
            ),
        )

    def __len__(self) -> int:
        """
        Calculate the size of the dataset.

        :returns: the size of the dataset.
        """

        return len(self.feature_data)

    def get_molecule(self, index: SupportsIndex | int) -> Chem.Mol:
        """
        Get the molecule at the given index.

        :param index: index of the molecule.
        :returns: rdkit object representing a chemical molecule.
        """

        return self.molecules[index]

    def get_molecules(self, indexes: Iterable[SupportsIndex | int]) -> List[Chem.Mol]:
        """
        Get a list of molecules at the given indexes.

        :param indexes: iterable of indexes of the molecules.
        :return: list of rdkit objects representing chemical molecules.
        """

        return [self.molecules[i] for i in indexes]

    def generate_conformations(self, num_processes: int = 8) -> None:
        """
        Generate conformations for the set of molecules.

        Parallelized with multiple processes.

        :param num_processes: number of processes used to parallelize the computation.
        """

        logger.info("Generating conformations.")
        with multiprocessing.Pool(num_processes) as p:
            self.molecules = p.map(chem.utils.generate_conformation, self.molecules)


class BaselineDataset(Dataset):
    """Utility class for datasets used in baselines."""

    def __init__(self, features: List[GraphFeatures], labels: pd.Series) -> None:
        """
        Construct the dataset.

        The type of the dataset (classification vs. regression) is inferred from the
        data type of labels parameter.

        :param features: features of the molecules in the dataset.
        :param labels: labels of the molecules - class indexes for classification
        or values for regression.
        """

        assert len(features) == len(labels)
        self.features = features
        if pd.api.types.is_integer_dtype(labels):
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = torch.Tensor(labels)

    def __len__(self) -> int:
        """
        Calculate the size of the dataset.

        :returns: the size of the dataset.
        """

        return len(self.features)

    def __getitem__(self, index) -> Tuple[torch_geometric.data.Data, torch.Tensor]:
        """
        Get the features for a molecule at the given index and an associated label.

        :param index: index of the data sample.
        :returns: data sample at the index and a label.
        """

        data_sample: GraphFeatures = self.features[index]
        return (
            torch_geometric.data.Data(
                data_sample.node_features, data_sample.edge_index
            ),
            self.labels[index],
        )


class BaselineDatasetSplit:
    def __init__(
        self,
        data_path: str | os.PathLike,
        featurizer: Featurizer,
        label: str,
        *,
        test_size: float = 0.2,
        seed: int = 42,
        stratify: bool = False,
    ) -> None:
        """
        Construct the dataset.

        :param data_path: path to the dataset file.
        :param featurizer: featurizer to extract features from molecules.
        :param label: name of the column with the label value.
        :param test_size: fraction of the dataset that will be used as test set.
        :param seed: random used during train-test splitting.
        :param stratify: whether to use labels to stratify the split.
        """

        super().__init__()
        logger.info(f"Extracting chemical features from {data_path}.")
        df = pd.read_csv(data_path)
        molecules = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"]]
        feature_data = [featurizer.extract_features(mol) for mol in molecules]
        labels = df[label]
        features_train, features_test, labels_train, labels_test = train_test_split(
            feature_data,
            labels,
            random_state=seed,
            test_size=test_size,
            stratify=labels if stratify else None,
        )
        self._train = BaselineDataset(features_train, labels_train)
        self._test = BaselineDataset(features_test, labels_test)

    @property
    def train(self) -> BaselineDataset:
        return self._train

    @property
    def test(self) -> BaselineDataset:
        return self._test


class HIVData(BaselineDatasetSplit):
    """Split dataset for HIV activity baseline (classification)."""

    def __init__(self, featurizer) -> None:
        super().__init__(
            pathlib.Path(__file__).parent.resolve() / "dataset/HIV.csv",
            featurizer,
            "HIV_active",
            stratify=True,
        )


class BACEData(BaselineDatasetSplit):
    """Split dataset for beta-secretase pIC50 (regression)."""

    def __init__(self, featurizer) -> None:
        super().__init__(
            pathlib.Path(__file__).parent.resolve() / "dataset/bace.csv",
            featurizer,
            "pIC50",
        )


class QM9Data(BaselineDatasetSplit):
    """Slit dataset for quantum properties prediction (regression)."""

    def __init__(self, featurizer) -> None:
        super().__init__(
            pathlib.Path(__file__).parent.resolve() / "dataset/QM9.csv",
            featurizer,
            "mu",
        )
