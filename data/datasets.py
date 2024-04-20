import os
from typing import Iterable, List, SupportsIndex, Tuple

import pandas as pd
import torch_geometric.data
from rdkit import Chem
from torch.utils.data import Dataset

from data.featurizer import Featurizer, GraphFeatures


class ZincDataset(Dataset):
    """Dataset representing molecules from Zinc database."""

    def __init__(self, data_path: str | os.PathLike, featurizer: Featurizer) -> None:
        """
        Construct the ZincDataset.

        :param data_path: path to the dataset file.
        :param featurizer: featurizer to extract features from molecules.
        """

        super().__init__()
        df = pd.read_csv(data_path, sep=" ")
        self.molecules = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"]]
        self.feature_data = [featurizer.extract_features(mol) for mol in self.molecules]

    def __getitem__(self, index: int) -> Tuple[int, torch_geometric.data.Data]:
        """
        Get features for molecule at given index and the index itself.

        The index is returned, so it can be used to extract molecule info.

        :param index: index of data sample.
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
        Calculate size of the dataset.

        :returns: size of the dataset.
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
