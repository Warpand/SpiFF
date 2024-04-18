import os

import pandas as pd
import rdkit.Chem
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

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

    def __getitem__(self, index: int) -> (GraphFeatures, int):
        """
        Get features for molecule at given index and the index itself.

        :param index: index of data sample.
        :returns: data sample at the index and the index.
        """

        data_sample = self.feature_data[index]
        return Data(data_sample.node_features, data_sample.edge_index), index

    def __len__(self) -> int:
        """
        Calculate size of the dataset.

        :returns: size of the dataset.
        """

        return len(self.feature_data)

    def get_molecule(self, index: int) -> rdkit.Chem.Mol:
        """
        Get the molecule at given index.

        :param index: index of the molecule.
        :returns: a chemical molecule as an rdkit object.
        """

        return self.molecules[index]
