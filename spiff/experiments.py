from typing import Callable, List, Union

import pytorch_lightning
import torch
import torch_geometric.data.batch
import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from spiff.metrics import Histogram
from spiff.mining import TripletMiner
from spiff.models import SPiFF


class SPiFFModule(pytorch_lightning.LightningModule):
    """PyTorch Lightning module for the SPiFF model."""

    def __init__(
        self,
        model: SPiFF,
        loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        miner: TripletMiner,
        lr: float,
    ) -> None:
        """
        Construct the SPiFFModule.

        :param model: the SPiFF model
        :param loss_fn: loss function.
        :param miner: miner dividing batches into triplets
        :param lr: the optimizer learning rate.
        """

        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.miner = miner
        self.lr = lr

        step = 0.1
        bins = torch.arange(0.0, 1.0 + step, step)
        self.hist_metric = Histogram(bins)

        self.using_wandb = isinstance(self.logger, pl_loggers.WandbLogger)

        self.save_hyperparameters()

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
        triple_indexes.to(self.device)
        similarity_values.to(self.device)

        anchor = embeddings[triple_indexes.anchor_indexes]
        positive = embeddings[triple_indexes.positive_indexes]
        negative = embeddings[triple_indexes.negative_indexes]

        loss = self.loss_fn(anchor, positive, negative)

        self.hist_metric.update(similarity_values)
        self.log("loss", loss, on_epoch=True, on_step=False)

        return loss

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        """
        Perform the SpiFF model forward pass.

        :param x: batch of molecule node features.
        :param edge_index: graph connectivity tensor.
        :param batch: assigns each node to a specific example.
        :returns: representation of a chemical molecule graph.
        """
        return self.model(x)

    def on_train_epoch_end(self) -> None:
        """Log metrics to wandb."""

        if self.using_wandb:
            wandb_logger = self.logger.experiment  # type: ignore
            hist = self.hist_metric.compute()
            hist /= torch.sum(hist)  # normalize to [0.1]
            data = [
                [str(bin_label.item()), bin_val]
                for bin_label, bin_val in zip(self.hist_metric.bins.cpu(), hist)
            ]
            table = wandb.Table(data=data, columns=["bin", "probability"])
            wandb_logger.log(
                {
                    "histogram": wandb.plot.bar(
                        table, "bin", "probability", title="Similarity Distribution"
                    )
                },
                step=self.current_epoch,
            )

        self.hist_metric.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure the optimizer and lr scheduler for the experiment.

        :returns: optimizer-scheduler pair.
        """

        optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        return [optimizer], [scheduler]
