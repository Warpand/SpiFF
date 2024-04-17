from pathlib import Path

import pandas as pd
import rdkit.Chem
from featurizer import Featurizer, GraphFeatures
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data


class ZincDataset(Dataset):
    """
    Dataset representing molecules from Zinc database.
    """

    def __init__(self, data_path: Path, featurizer: Featurizer) -> None:
        """
        Constructs ZincDataset.

        :param data_path: path to dataset file.
        :param featurizer: featurizer which can be used to
        extract features from molecules.
        """

        super().__init__()
        df = pd.read_csv(data_path, sep=" ")
        self.molecules = [Chem.MolFromSmiles(smiles) for smiles in df["smiles"]]
        self.feature_data = [featurizer.extract_features(mol) for mol in self.molecules]

    def __getitem__(self, index: int) -> (GraphFeatures, int):
        """
        Returns features for molecule at given index and its index.

        :param index: index of data sample.
        """

        data_sample = self.feature_data[index]
        return Data(data_sample.node_features, data_sample.edge_index), index

    def __len__(self) -> int:
        """
        Returns size of the dataset.

        :return: size of the dataset.
        """

        return len(self.feature_data)

    def get_molecule(self, index: int) -> rdkit.Chem.Mol:
        """
        Returns molecule at given index.

        :param index: index of molecule.
        """

        return self.molecules[index]
