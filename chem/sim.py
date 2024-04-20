import logging
import os
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import rdkit.Chem.rdmolops as rdmolops
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdMolAlign, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps

logger = logging.getLogger(__name__)


class MoleculeSimilarity(ABC):
    """A measure of similarity between two molecules."""

    @abstractmethod
    def __call__(self, mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol) -> float:
        """
        Calculate the similarity measure between the two molecules.

        The higher the value, the more similar the molecules.

        :param mol1: the first molecule of interest.
        :param mol2: the second molecule of interest.

        :returns: value of the similarity measure.
        """
        pass


class SCSimilarity(MoleculeSimilarity):
    """
    Shape and Color similarity score.

    Mixes a measure based on overlap of pharmacophoric features with a volumetric
    comparison.

    Adapted from https://github.com/fimrie/DeLinker with changes.
    Original code licensed under the 3-Clause BSD License.

    License is in legal/DeLinker_License.

    Due to non-deterministic nature of the measure and high variance of its value,
    the measure is calculated several times.
    """

    _FEATURE_FACTORY = AllChem.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )
    _FM_PARAMS = {
        family: FeatMaps.FeatMapParams()
        for family in _FEATURE_FACTORY.GetFeatureFamilies()
    }
    _KEEP = (
        "Donor",
        "Acceptor",
        "NegIonizable",
        "PosIonizable",
        "ZnBinder",
        "Aromatic",
        "Hydrophobe",
        "LumpedHydrophobe",
    )

    def __init__(
        self,
        num_tries: int = 5,
        shape_score_weight: float = 0.5,
        color_score_weight: float = 0.5,
        reduce: Literal["max", "min", "mean"] = "max",
    ) -> None:
        """
        Construct the class.

        :param num_tries: number of times the measure will be calculated.
        :param reduce: specifies how the final results is derived from the multiple
        tries.
        :param shape_score_weight: weight of the shape compound of the score
        (volumetric comparison).
        :param color_score_weight: weight of the color compound of the
        score (pharmacophoric features).

        :raises ValueError: if an inappropriate string is supplied as reduce parameter.
        """

        self.num_tries = num_tries
        self.shape_weight = shape_score_weight
        self.color_weight = color_score_weight
        if reduce == "max":
            self.reduce_func = np.max
        elif reduce == "min":
            self.reduce_func = np.min
        elif reduce == "mean":
            self.reduce_func = np.mean
        else:
            logger.error(
                f"Passed unsupported '{reduce}' as reduce parameter in SCSimilarity."
            )
            raise ValueError()

    @staticmethod
    def _embed(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
        mol = rdmolops.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        return mol

    @staticmethod
    def _get_fmap_score(mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol) -> float:
        feat_lists = [
            [
                feat
                for feat in SCSimilarity._FEATURE_FACTORY.GetFeaturesForMol(mol)
                if feat.GetFamily() in SCSimilarity._KEEP
            ]
            for mol in (mol1, mol2)
        ]

        fms = FeatMaps.FeatMap(
            feats=feat_lists[0],
            weights=[1] * len(feat_lists[0]),
            params=SCSimilarity._FM_PARAMS,
        )
        fms.scoreMode = FeatMaps.FeatMapScoreMode.Best
        score_feats = fms.ScoreFeats(feat_lists[1])

        return score_feats / min(fms.GetNumFeatures(), len(feat_lists[1]))

    @staticmethod
    def _get_protrude_dist(mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol) -> float:
        return rdShapeHelpers.ShapeProtrudeDist(mol1, mol2, allowReordering=True)

    def _calc_sc_score(self, mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol) -> float:
        # noinspection PyBroadException
        try:
            mol1 = SCSimilarity._embed(mol1)
            mol2 = SCSimilarity._embed(mol2)

            rdMolAlign.GetO3A(mol1, mol2).Align()

            fmap_score = SCSimilarity._get_fmap_score(mol1, mol2)
            protrude_dist = SCSimilarity._get_protrude_dist(mol1, mol2)
            return self.color_weight * fmap_score + self.shape_weight * (
                1.0 - protrude_dist
            )
        except Exception:
            logger.warning("Could not calculate the sc score.")
            return 0.0

    def __call__(self, mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol) -> float:
        values = [self._calc_sc_score(mol1, mol2) for _ in range(self.num_tries)]

        return float(self.reduce_func(values))
