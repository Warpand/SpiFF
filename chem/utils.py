from rdkit import Chem
from rdkit.Chem import AllChem

EMBEDDING_SEED = 42


def generate_conformation(
    mol: Chem.rdchem.Mol, seed: int = EMBEDDING_SEED
) -> Chem.rdchem.Mol:
    """
    Generate a conformation for the given molecule.

    Returns a new molecule with added implicit hydrogen atoms and the generated
    conformation. Conformations are optimized by using a force field.

    :param mol: the molecule of interest.
    :param seed: random seed used while embedding the molecule.
    :return: molecule with generated conformation.
    """
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol
