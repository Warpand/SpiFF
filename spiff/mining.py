import logging
import sys
from dataclasses import dataclass
from typing import List, Tuple

if sys.version_info <= (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import torch
from rdkit import Chem

import spiff.utils as utils
from chem.sim import MoleculeSimilarity

logger = logging.getLogger(__name__)


@dataclass
class TripleIndexes:
    anchor_indexes: torch.LongTensor
    positive_indexes: torch.LongTensor
    negative_indexes: torch.LongTensor

    def to(self, device: str | torch.device) -> Self:
        self.anchor_indexes.to(device)
        self.positive_indexes.to(device)
        self.negative_indexes.to(device)
        return self


class TripletMiner:
    """
    Class dividing a batch of molecules into anchor-positive-negative triples based
    on a similarity measure.
    """

    def __init__(self, similarity_measure: MoleculeSimilarity) -> None:
        """
        Construct the miner.

        :param similarity_measure: similarity measure of molecules used by the miner.
        """

        self.similarity_measure = similarity_measure

    # noinspection DuplicatedCode
    def mine(
        self, molecules: List[Chem.rdchem.Mol]
    ) -> Tuple[TripleIndexes, torch.Tensor]:
        """
        Divide a batch of molecules into anchor-positive-negative triples.

        Also returns a tensor with values of calculated similarity measures.
        The order of values in the returned similarities tensor is intentional.

        :param molecules: batch of molecules to divide.
        :return: TripleIndexes object containing indexes of molecules belonging
        respectively to anchors, positives and negatives and a tensor containing all the
        calculated similarity measure values.
        :raises ValueError: if the batch of received molecules has size not divisible
        by 3.
        """

        if len(molecules) % 3 != 0:
            logger.error("Triplet Miner received a batch of size not divisible by 3.")
            raise ValueError()
        anchors, positives, negatives = [], [], []
        similarities = []
        for (i1, mol1), (i2, mol2), (i3, mol3) in utils.triple_wise(
            enumerate(molecules)
        ):
            sim_1_2 = self.similarity_measure(mol1, mol2)
            sim_2_3 = self.similarity_measure(mol2, mol3)
            sim_1_3 = self.similarity_measure(mol1, mol3)
            similarities.extend((sim_1_2, sim_2_3, sim_1_3))
            # oof
            if sim_1_2 >= sim_2_3:
                if sim_1_3 >= sim_1_2:
                    anchors.append(i3)
                    positives.append(i1)
                    negatives.append(i2)
                elif sim_1_3 >= sim_2_3:
                    anchors.append(i2)
                    positives.append(i1)
                    negatives.append(i3)
                else:
                    anchors.append(i1)
                    positives.append(i2)
                    negatives.append(i3)
            else:
                if sim_1_3 >= sim_2_3:
                    anchors.append(i1)
                    positives.append(i3)
                    negatives.append(i2)
                elif sim_1_3 >= sim_1_2:
                    anchors.append(i2)
                    positives.append(i3)
                    negatives.append(i1)
                else:
                    anchors.append(i3)
                    positives.append(i2)
                    negatives.append(i1)
        return (
            TripleIndexes(
                torch.LongTensor(anchors),
                torch.LongTensor(positives),
                torch.LongTensor(negatives),
            ),
            torch.Tensor(similarities),
        )
