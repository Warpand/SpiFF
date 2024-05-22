import logging
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pytorch_lightning
import torch
import torch_geometric.data.batch
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from data.featurizer import Featurizer, GraphFeaturizerFactory
from spiff.metrics import Histogram
from spiff.mining import TripletMiner
from spiff.models import SPiFF
from spiff.utils import figure_to_wandb

logger = logging.getLogger(__name__)


class SPiFFModule(pytorch_lightning.LightningModule):
    """PyTorch Lightning module for the SPiFF model."""

    FEATURIZER_CHECKPOINT_KEY = "featurizer"

    def __init__(
        self,
        model: SPiFF,
        loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        miner: TripletMiner,
        lr: float,
        using_wandb: bool,
    ) -> None:
        """
        Construct the SPiFFModule.

        :param model: the SPiFF model.
        :param loss_fn: loss function.
        :param miner: miner dividing batches into triplets.
        :param lr: the optimizer learning rate.
        :param using_wandb: whether Weights & Biases is used.
        """

        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.miner = miner
        self.lr = lr

        num_bins = 10
        self.hist_metric = Histogram(num_bins)

        self.using_wandb = using_wandb

        self._featurizer_factory: Optional[GraphFeaturizerFactory] = None

    def training_step(
        self,
        batch: List[Union[torch.Tensor, torch_geometric.data.batch.Batch]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Perform a training step.

        :param batch: data batch.
        :param batch_idx: index of the current batch.
        :returns: computed loss.
        """

        molecule_indexes, batch_data = batch
        x: torch.Tensor = batch_data.x
        edge_indexes: torch.Tensor = batch_data.edge_index
        batch_indexes: torch.Tensor = batch_data.batch

        embeddings = self(x, edge_indexes, batch_indexes)
        molecules = self.trainer.train_dataloader.dataset.get_molecules(
            molecule_indexes
        )

        triple_indexes, similarity_values = self.miner.mine(molecules)
        triple_indexes = triple_indexes.to(self.device)
        similarity_values = similarity_values.to(self.device)

        anchor = embeddings[triple_indexes.anchor_indexes]
        positive = embeddings[triple_indexes.positive_indexes]
        negative = embeddings[triple_indexes.negative_indexes]

        loss = self.loss_fn(anchor, positive, negative)

        self.hist_metric.update(similarity_values)
        self.log(
            "loss", loss, on_epoch=True, on_step=False, batch_size=len(molecule_indexes)
        )

        return loss

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        """
        Perform the SpiFF model forward pass.

        :param x: batch of molecule node features.
        :param edge_index: graph connectivity tensor.
        :param batch: assigns each node to a specific example.
        :returns: representation of a chemical molecule graph.
        """
        return self.model(x, edge_index, batch)

    def on_train_epoch_end(self) -> None:
        """Log metrics to wandb."""

        if self.using_wandb:
            wandb_logger = self.logger.experiment  # type: ignore
            hist = self.hist_metric.compute()
            hist /= torch.sum(hist)  # normalize to [0.1]

            fig, ax = plt.subplots()

            labels = [
                "{:.2f}".format(bin_label.item())
                for bin_label in self.hist_metric.bins[:-1]
            ]

            plt.bar(labels, hist.cpu().numpy())
            ax.set_ylabel("probability")
            ax.set_title("Similarity Distribution")

            wandb_logger.log(
                {"histogram": figure_to_wandb(fig)}, step=self.current_epoch
            )

            plt.close(fig)

        self.hist_metric.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure the optimizer and lr scheduler for the experiment.

        :returns: optimizer-scheduler pair.
        """

        optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        try:
            key = self.FEATURIZER_CHECKPOINT_KEY
            checkpoint[key] = self.trainer.datamodule.featurizer.save()  # type: ignore
        except AttributeError:
            logger.warning("Error while saving featurizer to checkpoint.")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        try:
            self._featurizer_factory = GraphFeaturizerFactory.load(
                checkpoint[self.FEATURIZER_CHECKPOINT_KEY]
            )
        except KeyError:
            logger.warning("Error while loading featurizer from checkpoint.")

    def get_compatible_featurizer(self) -> Featurizer:
        """
        Get a GraphFeaturizer compatible with the model.

        Data necessary to construct the featurizer is extracted from a PyTorch Lightning
        checkpoint.

        :return: a compatible featurizer.
        :raises ValueError: if appropriate data was not loaded from the checkpoint.
        """

        if self._featurizer_factory is not None:
            return self._featurizer_factory()
        else:
            logger.error("Unable to create a compatible FeaturizerFactory.")
            raise ValueError()

    @property
    def latent_size(self) -> int:
        return self.model.latent_size
