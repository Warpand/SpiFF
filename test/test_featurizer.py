from unittest import TestCase

import torch
from rdkit import Chem

from data.featurizer import GraphFeaturizer


class TestGraphFeaturizer(TestCase):
    def test_num_features(self):
        featurizer = GraphFeaturizer(["C", "H", "O"], [Chem.rdchem.Atom.GetAtomicNum])
        self.assertEqual(5, featurizer.num_features())

    def test_extract_features(self):
        featurizer = GraphFeaturizer(["C", "H", "O"], [Chem.rdchem.Atom.GetAtomicNum])
        mol = Chem.MolFromSmiles("CCO")
        features = featurizer.extract_features(mol)
        expected_atom = torch.Tensor(
            [[1, 0, 0, 0, 6], [1, 0, 0, 0, 6], [0, 0, 1, 0, 8]]
        )
        expected_edge = torch.Tensor([[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]])
        self.assertTrue(torch.all(expected_atom == features.node_features))
        self.assertTrue(torch.all(expected_edge == features.edge_index))
