import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
from rdkit import Chem

logger = logging.getLogger(__name__)


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

        :param molecule: the molecule for which the features are calculated.
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

    FEATURES_KEY = "features"
    SYMBOLS_KEY = "atoms"

    def __init__(
        self,
        atom_symbols: List[str],
        atom_features: List[Callable[[Chem.rdchem.Atom], Union[float, int]]],
        add_self_loops: bool = True,
    ) -> None:
        """
        Construct the featurizer.

        :param atom_symbols: symbols of atoms that will be one-hot encoded.
        Atoms not in the list are encoded as unknown.
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

        return GraphFeatures(atom_features, edge_index.type(torch.int64))


class GraphFeaturizerFactory(FeaturizerFactory):
    def __init__(
        self, atom_feature_names: List[str], atom_symbols: Optional[List[str]] = None
    ):
        """
        Construct the factory.

        :param atom_feature_names: names of features to be extracted by
        a GraphFeaturizer produced by this factory. Names should be the same as names
        of methods from rdkit.Chem.rdchem.Atom, without leading get (lettercase is
        ignored).
        :param atom_symbols: list of symbols of atoms used by a produced featurizer.
        If None, a default one is used.
        :raises ValueError: if an unsupported atom feature name is passed in
        atom_feature_names.
        """

        self.atom_features = list(
            map(GraphFeaturizerFactory._string_to_atom_feature, atom_feature_names)
        )
        if atom_symbols is None:
            self.atom_symbols = ["C", "H", "O", "N", "S", "F", "Cl", "Br", "I"]
        else:
            self.atom_symbols = atom_symbols

    @staticmethod
    def _string_to_atom_feature(
        feature_name: str,
    ) -> Callable[[Chem.rdchem.Atom], Union[float, int]]:
        name = feature_name.lower()
        match name:
            case "atomicnum":
                return Chem.rdchem.Atom.GetAtomicNum
            case "degree":
                return Chem.rdchem.Atom.GetDegree
            case "formalcharge":
                return Chem.rdchem.Atom.GetFormalCharge
            case "hybridization":
                return Chem.rdchem.Atom.GetHybridization
            case "isaromatic":
                return Chem.rdchem.Atom.GetIsAromatic
            case "mass":
                return Chem.rdchem.Atom.GetMass
            case "numimpliciths":
                return Chem.rdchem.Atom.GetNumImplicitHs
            case _:
                logger.error("Unexpected atom feature name: " + feature_name)
                raise ValueError()

    def __call__(self) -> Featurizer:
        return GraphFeaturizer(self.atom_symbols, self.atom_features)
