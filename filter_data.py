import os
import sys

import pandas as pd
import rdkit.Chem.rdmolops as rdmolops
from rdkit.Chem import AllChem, rdForceFieldHelpers
from tqdm import tqdm

SEED = 42
SEP = " "


def test_mol(smiles: str) -> bool:
    mol = rdmolops.AddHs(AllChem.MolFromSmiles(smiles))
    embed_success = AllChem.EmbedMolecule(mol, randomSeed=SEED) == 0
    ff_success = rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol)
    return embed_success and ff_success


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Usage: python3 {os.path.basename(__file__)} "
            f"[PATH_TO_DATASET] [FILTERED_DATASET_SAVE_PATH]"
        )
        sys.exit(1)

    data = pd.read_csv(sys.argv[1], sep=SEP)
    old_size = len(data)

    ok_indexes = [
        i for i, sm in tqdm(enumerate(data["smiles"]), total=old_size) if test_mol(sm)
    ]

    new_size = len(ok_indexes)
    deleted = old_size - new_size

    print(
        f"Deleted {deleted} molecules from the dataset ({100.0 * deleted / old_size}%)."
    )

    data.iloc[ok_indexes].to_csv(sys.argv[2], sep=SEP)
