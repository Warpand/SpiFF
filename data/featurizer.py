from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Union

import torch
from rdkit import Chem


@dataclass
class GraphFeatures:
    """Stores tensors representing a graph in PyTorch Geometric compatible formats."""

    node_features: torch.Tensor
    edge_index: torch.Tensor


class Featurizer(ABC):
    """Extracts features that can be used by machine learning models from molecules."""

    @abstractmethod
    def num_features(self) -> int:
        """
        Calculate length of feature tensor this class produces.

        :returns: length of tensors or length of atom_features field in GraphFeatures
        that extract_features method returns.
        """
        pass

    @abstractmethod
    def extract_features(
        self, molecule: Chem.rdchem.Mol
    ) -> Union[GraphFeatures, torch.Tensor]:
        """
        Calculate features for a given molecule.

        :param molecule: the molecule for which the features will be calculated.
        :returns: GraphFeatures object with calculated features or features tensor.
        """
        pass


class FeaturizerFactory(ABC):
    @abstractmethod
    def __call__(self) -> Featurizer:
        pass


class GraphFeaturizer(Featurizer):
    """
    Featurizer for graph neural networks.

    Encodes atom types as one-hot and adds additional features.
    """

    def __init__(
        self,
        atom_symbols: List[str],
        atom_features: List[Callable[[Chem.rdchem.Atom], Union[float, int]]],
        add_self_loops: bool = True,
    ) -> None:
        """
        Construct the featurizer.

        :param atom_symbols: symbols of atoms that will be one-hot encoded.
        Atoms not in the list will be encoded as unknown.
        :param atom_features: functions used to extract additional features from atoms.
        :param add_self_loops: whether to add self loops as edges.
        """
        super().__init__()
        self.atom_codes = {symbol: nr for nr, symbol in enumerate(atom_symbols)}
        self.atom_features = atom_features
        self.add_self_loops = add_self_loops

    def num_features(self) -> int:
        return len(self.atom_codes) + 1 + len(self.atom_features)

    def extract_features(self, molecule: Chem.rdchem.Mol) -> GraphFeatures:
        unknown_code = len(self.atom_codes)
        codes = [
            self.atom_codes.get(atom.GetSymbol(), unknown_code)
            for atom in molecule.GetAtoms()
        ]
        atom_type_features = torch.nn.functional.one_hot(
            torch.LongTensor(codes), unknown_code + 1
        )
        atom_features_values = torch.Tensor(
            [
                [feature(atom) for feature in self.atom_features]
                for atom in molecule.GetAtoms()
            ]
        )
        atom_features = torch.cat((atom_type_features, atom_features_values), dim=1)

        edges = []
        for bond in molecule.GetBonds():
            edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        if self.add_self_loops:
            edges.extend([(i, i) for i in range(molecule.GetNumAtoms())])
        edges_begin, edges_end = zip(*edges)
        edge_index = torch.vstack((torch.Tensor(edges_begin), torch.Tensor(edges_end)))

        return GraphFeatures(atom_features, edge_index)


class DefaultFeaturizerFactory(FeaturizerFactory):
    def __call__(self) -> Featurizer:
        return GraphFeaturizer(
            ["C", "H", "O", "N", "S", "F", "Cl", "Br", "I"],
            [Chem.rdchem.Atom.GetAtomicNum],
        )

    # TO DO add atom features
