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
        # fmt: off
        molecules = [
            "mol0", "mol1", "mol2",
            "mol0", "mol2", "mol1",
            "mol1", "mol0", "mol2",
            "mol1", "mol2", "mol0",
            "mol2", "mol0", "mol1",
            "mol2", "mol1", "mol0",
        ]
        # fmt: on
        triplets, distances = miner.mine(molecules)  # type: ignore
        self.assertTrue(
            torch.all(triplets.anchor_indexes == torch.LongTensor([1, 5, 6, 9, 14, 16]))
        )
        self.assertTrue(
            torch.all(
                triplets.positive_indexes == torch.LongTensor([0, 3, 7, 11, 13, 17])
            )
        )
        self.assertTrue(
            torch.all(
                triplets.negative_indexes == torch.LongTensor([2, 4, 8, 10, 12, 15])
            )
        )

        for i in range(0, len(distances), 3):
            dist = distances[i : i + 3]
            self.assertTrue(
                torch.allclose(dist, torch.FloatTensor([0.99, 0.6, 0.45]))
                or torch.allclose(dist, torch.FloatTensor([0.99, 0.45, 0.6]))
                or torch.allclose(dist, torch.FloatTensor([0.6, 0.99, 0.45]))
                or torch.allclose(dist, torch.FloatTensor([0.6, 0.45, 0.99]))
                or torch.allclose(dist, torch.FloatTensor([0.45, 0.99, 0.6]))
                or torch.allclose(dist, torch.FloatTensor([0.45, 0.6, 0.99]))
            )
