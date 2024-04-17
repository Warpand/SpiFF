from abc import ABC, abstractmethod

from rdkit import Chem


class MoleculeSimilarity(ABC):
    """A measure of similarity between two molecules."""

    @abstractmethod
    def __call__(self, mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol) -> float:
        """
        Calculates the similarity measure between the two molecules.

        The higher the value, the more similar the molecules.

        :param mol1: the first molecule of interest.
        :param mol2: the second molecule of interest.

        :returns: value of the similarity measure.
        """
        pass
