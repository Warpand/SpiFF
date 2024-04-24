from unittest import TestCase
from unittest.mock import Mock

import torch

from spiff.mining import TripletMiner


class TestTripletMiner(TestCase):
    def test_mine(self):
        values = {
            ("mol0", "mol1"): 0.99,
            ("mol1", "mol0"): 0.99,
            ("mol0", "mol2"): 0.6,
            ("mol2", "mol0"): 0.6,
            ("mol1", "mol2"): 0.45,
            ("mol2", "mol1"): 0.45,
        }
        similarity_measure = Mock()
        similarity_measure.side_effect = lambda m1, m2: values[(m1, m2)]
        miner = TripletMiner(similarity_measure)
        molecules = ["mol0", "mol1", "mol2"]
        triplets, distances = miner.mine(molecules)  # type: ignore
        self.assertTrue(torch.all(triplets.anchor_indexes == torch.LongTensor([1])))
        self.assertTrue(torch.all(triplets.positive_indexes == torch.LongTensor([0])))
        self.assertTrue(torch.all(triplets.negative_indexes == torch.LongTensor([2])))
        self.assertTrue(
            torch.allclose(distances, torch.FloatTensor([0.99, 0.6, 0.45]))
            or torch.allclose(distances, torch.FloatTensor([0.99, 0.45, 0.6]))
            or torch.allclose(distances, torch.FloatTensor([0.6, 0.99, 0.45]))
            or torch.allclose(distances, torch.FloatTensor([0.6, 0.45, 0.99]))
            or torch.allclose(distances, torch.FloatTensor([0.45, 0.99, 0.6]))
            or torch.allclose(distances, torch.FloatTensor([0.45, 0.6, 0.99]))
        )
