from unittest import TestCase
from unittest.mock import Mock

import torch

from spiff.mining import TripletMiner


class TestTripletMiner(TestCase):
    def setUp(self):
        values = {
            ("mol0", "mol1"): 0.99,
            ("mol1", "mol0"): 0.99,
            ("mol0", "mol2"): 0.6,
            ("mol2", "mol0"): 0.6,
            ("mol1", "mol2"): 0.45,
            ("mol2", "mol1"): 0.45,
        }
        self.similarity_measure = Mock()
        self.similarity_measure.side_effect = lambda m1, m2: values[(m1, m2)]
        # fmt: off
        self.molecules = [
            "mol0", "mol1", "mol2",
            "mol0", "mol2", "mol1",
            "mol1", "mol0", "mol2",
            "mol1", "mol2", "mol0",
            "mol2", "mol0", "mol1",
            "mol2", "mol1", "mol0",
        ]
        # fmt: on

    def test_mine_triplets(self):
        miner = TripletMiner(self.similarity_measure)
        triplets, _ = miner.mine(self.molecules)  # type: ignore

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

    def test_mine_similarity(self):
        miner = TripletMiner(self.similarity_measure)
        _, sim = miner.mine(self.molecules)  # type: ignore

        # fmt: off
        expected_similarity = torch.FloatTensor(
            [
                0.99, 0.45, 0.6,
                0.6, 0.45, 0.99,
                0.99, 0.6, 0.45,
                0.45, 0.6, 0.99,
                0.6, 0.99, 0.45,
                0.45, 0.99, 0.6,
            ]
        )
        # fmt: on

        self.assertTrue(torch.allclose(expected_similarity, sim))
