import itertools
from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Union

import pytorch_lightning
import torch
import torch_geometric.data.batch
import torchmetrics
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from spiff.models import SPiFF


class BaselineExperiment(ABC, pytorch_lightning.LightningModule):
    def __init__(
        self,
        spiff: SPiFF,
        model: torch.nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lr: float,
        spiff_trainable: bool,
        train_metrics: torch.nn.ModuleList,
        test_metrics: torch.nn.ModuleList,
    ) -> None:
        pytorch_lightning.LightningModule.__init__(self)
        self.spiff = spiff
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.spiff_trainable = spiff_trainable
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.spiff_trainable:
            params = itertools.chain(self.spiff.parameters(), self.model.parameters())
        else:
            params = self.model.parameters()
        return torch.optim.Adam(params, self.lr)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        latent = self.spiff(x, edge_index, batch)
        return self.model(latent)

    @abstractmethod
    def compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, phase: Literal["train", "test"]
    ) -> None:
        pass

    def _step(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        y: torch.Tensor,
        phase: Literal["train", "test"],
    ) -> torch.Tensor:
        pred = self(x, edge_index, batch)
        loss = self.loss_fn(pred, y)

        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True)

        self.compute_metrics(pred, y, phase)

        return loss

    def training_step(
        self,
        batch: List[Union[torch.Tensor, torch_geometric.data.batch.Batch]],
        batch_idx: int,
    ) -> torch.Tensor:
        batch_data, target = batch
        x: torch.Tensor = batch_data.x
        edge_indexes: torch.Tensor = batch_data.edge_index
        batch_indexes: torch.Tensor = batch_data.batch
        return self._step(x, edge_indexes, batch_indexes, target, "train")

    def validation_step(
        self,
        batch: List[Union[torch.Tensor, torch_geometric.data.batch.Batch]],
        batch_idx: int,
    ) -> None:
        batch_data, target = batch
        x: torch.Tensor = batch_data.x
        edge_indexes: torch.Tensor = batch_data.edge_index
        batch_indexes: torch.Tensor = batch_data.batch
        self._step(x, edge_indexes, batch_indexes, target, "train")

    def on_train_epoch_end(self) -> None:
        for metric in self.train_metrics:
            self.log(f"train/{metric.__class__.__name__}", metric.compute())
            metric.reset()

    def on_validation_epoch_end(self) -> None:
        for metric in self.test_metrics:
            self.log(f"test/{metric.__class__.__name__}", metric.compute())
            metric.reset()


class ClassificationExperiment(BaselineExperiment):
    def __init__(
        self, spiff: SPiFF, model: torch.nn.Module, lr: float, spiff_trainable: bool
    ) -> None:
        super().__init__(
            spiff,
            model,
            torch.nn.BCEWithLogitsLoss(),
            lr,
            spiff_trainable,
            torch.nn.ModuleList(
                [
                    torchmetrics.classification.BinaryAccuracy(),
                    torchmetrics.classification.BinaryAUROC(),
                    torchmetrics.classification.BinaryF1Score(),
                    torchmetrics.classification.BinaryRecall(),
                ]
            ),
            torch.nn.ModuleList(
                [
                    torchmetrics.classification.BinaryAccuracy(),
                    torchmetrics.classification.BinaryAUROC(),
                    torchmetrics.classification.BinaryF1Score(),
                    torchmetrics.classification.BinaryRecall(),
                ]
            ),
        )

    def compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, phase: Literal["train", "test"]
    ) -> None:
        match phase:
            case "train":
                metrics = self.train_metrics
            case "test":
                metrics = self.test_metrics
            case _:
                raise ValueError()
        labels = pred >= 0.0
        for metric in metrics:
            metric(labels, target)


class RegressionExperiment(BaselineExperiment):
    def __init__(
        self, spiff: SPiFF, model: torch.nn.Module, lr: float, spiff_trainable: bool
    ) -> None:
        super().__init__(
            spiff,
            model,
            torch.nn.MSELoss(),
            lr,
            spiff_trainable,
            torch.nn.ModuleList(
                [
                    torchmetrics.regression.MeanAbsoluteError(),
                    torchmetrics.regression.MeanSquaredError(),
                ]
            ),
            torch.nn.ModuleList(
                [
                    torchmetrics.regression.MeanAbsoluteError(),
                    torchmetrics.regression.MeanSquaredError(),
                ]
            ),
        )

    def compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor, phase: Literal["train", "test"]
    ) -> None:
        match phase:
            case "train":
                metrics = self.train_metrics
            case "test":
                metrics = self.test_metrics
            case _:
                raise ValueError()
        for metric in metrics:
            metric(pred, target)
