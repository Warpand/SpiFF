from typing import List

import pytorch_lightning
import torch
import torch_geometric.data.batch
import wandb

from spiff.metrics import Histogram
from spiff.mining import TripletMiner
from spiff.models import SPiFF


class SPiFFModule(pytorch_lightning.LightningModule):
    """PyTorch Lightning module class for SPiFF model."""

    def __init__(
        self,
        model: SPiFF,
        loss_fn: torch.nn.TripletMarginLoss,
        miner: TripletMiner,
        lr: float,
    ) -> None:
        """
        Construct the SPiFFModule.

        :param model: SPiFF model
        :param loss_fn: loss function for training phase.
        :param miner: miner dividing batch into triplets
        :param lr: optimizer learning rate
        """

        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.miner = miner
        self.lr = lr

        step = 0.05
        self.bins = torch.arange(0.0, 1 + step, step)
        self.hist_metric = Histogram(self.bins)

    def on_fit_start(self) -> None:
        """Set the model to training mode."""

        self.model.train()

    def on_fit_end(self) -> None:
        """Set the model to evaluation mode."""

        self.model.eval()

    def training_step(
        self,
        batch: List[torch.Tensor, torch_geometric.data.batch.Batch],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Perform SPiFF training step with TripletMargin as loss function.

        Returns computed loss.

        :param batch: data batch.
        :param batch_idx: index of current batch.
        """

        molecule_indexes, batch_data = batch
        x = batch_data.x
        edge_indexes = batch_data.edge_index
        batch_indexes = batch_data.batch

        embeddings = self.model(x, edge_indexes, batch_indexes)
        molecules = self.train_dataloader().dataset.get_molecules(molecule_indexes)

        triple_indexes, similarity_values = self.miner.mine(molecules)
        anchor = embeddings[triple_indexes.anchor_indexes]
        positive = embeddings[triple_indexes.positive_indexes]
        negative = embeddings[triple_indexes.negative_indexes]
        main_loss = self.loss_fn(anchor, positive, negative)

        self.hist_metric.update(similarity_values)
        self.log("loss", main_loss, on_epoch=True, on_step=False)

        return main_loss

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        """
        Perform SpiFF model forward pass.

        :param x: batch of molecule node features.
        :param edge_index: graph connectivity tensor.
        :param batch: assigns each node to a specific example.
        :returns: representation of a chemical molecule graph.
        """
        return self.model(x)

    def on_train_epoch_end(self) -> None:
        """Perform wandb logging."""

        hist = self.hist_metric.compute()
        data = [
            [str(bin_name.item()), bin_count]
            for bin_name, bin_count in zip(self.bins[:-1], hist)
        ]
        table = wandb.Table(data=data, columns=["bin_name", "bin_count"])
        wandb.log(
            {
                "histogram": wandb.plot.bar(
                    table, "bins", "counts", title="Similarity values histogram"
                )
            },
            step=self.current_epoch,
        )

        self.hist_metric.reset()

    def configure_optimizers(self) -> torch.optim.optimizer:
        """
        Return AdamW optimizer.

        :return: used optimizer.
        """

        optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)
        return optimizer
