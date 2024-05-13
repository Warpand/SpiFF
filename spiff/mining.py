import logging
import multiprocessing
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import List, Tuple

if sys.version_info < (3, 11):
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
    anchor_indexes: torch.Tensor
    positive_indexes: torch.Tensor
    negative_indexes: torch.Tensor

    def to(self, device: str | torch.device) -> Self:
        return TripleIndexes(
            self.anchor_indexes.to(device),
            self.positive_indexes.to(device),
            self.negative_indexes.to(device),
        )


# noinspection DuplicatedCode
def _divide(
    molecules: List[Chem.rdchem.Mol],
    similarity_measure: MoleculeSimilarity,
    start_index: int,
) -> Tuple[TripleIndexes, torch.Tensor]:
    anchors, positives, negatives = [], [], []
    similarities = []
    for (i1, mol1), (i2, mol2), (i3, mol3) in utils.triple_wise(enumerate(molecules)):
        i1 += start_index
        i2 += start_index
        i3 += start_index
        sim_1_2 = similarity_measure(mol1, mol2)
        sim_2_3 = similarity_measure(mol2, mol3)
        sim_1_3 = similarity_measure(mol1, mol3)
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


class TripletMiner:
    """
    Class dividing a batch of molecules into anchor-positive-negative triples based
    on a similarity measure.

    Uses multiprocessing to parallelize the mining process.
    The terminate_process_pool method must be called after finishing
    working with this object, so the resources used by the processes might be freed.
    """

    def __init__(
        self, similarity_measure: MoleculeSimilarity, num_processes: int = 8
    ) -> None:
        """
        Construct the miner.

        :param similarity_measure: similarity measure of molecules used by the miner.
        :param num_processes: number of processes used while mining.
        """

        self.similarity_measure = similarity_measure
        self.num_processes = num_processes
        self.pool = multiprocessing.Pool(num_processes)

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

        size = len(molecules) // 3
        args = []
        for i in range(self.num_processes):
            start_index = 3 * ((i * size) // self.num_processes)
            end_index = 3 * (((i + 1) * size) // self.num_processes)
            args.append(
                (molecules[start_index:end_index], self.similarity_measure, start_index)
            )

        outputs = self.pool.starmap(_divide, args)

        return (
            TripleIndexes(
                torch.cat([output[0].anchor_indexes for output in outputs]),
                torch.cat([output[0].positive_indexes for output in outputs]),
                torch.cat([output[0].negative_indexes for output in outputs]),
            ),
            torch.cat([output[1] for output in outputs]),
        )

    def terminate_process_pool(self) -> None:
        """
        Terminates the process pool held by this object, allowing for the resources
        to be freed.
        """
        self.pool.terminate()


class TripletMinerContextManager(TripletMiner, AbstractContextManager):
    """
    Wrapper around the TripletMiner class that works as a context manager and
    automatically calls the terminate_process_pool method.
    """

    def __exit__(self, __exc_type, __exc_value, __traceback):
        self.terminate_process_pool()
