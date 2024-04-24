import logging
from abc import ABC, abstractmethod
from typing import Callable, List

import torch
import torch_geometric.nn.aggr
import torch_geometric.nn.models

logger = logging.getLogger(__name__)


class ModelFactory(ABC):
    """Interface for a factory producing neural networks."""

    @abstractmethod
    def __call__(self, input_size: int, output_size: int) -> torch.nn.Module:
        """
        Produces the model.

        :param input_size: the dimensionality of the input of the model.
        :param output_size: the dimensionality of the output of the model.
        :returns: the produced model.
        """
        pass


class FuncFactory(ABC):
    """
    Interface for a factory producing functions, such as activation functions and
    readout functions.
    """

    @abstractmethod
    def __call__(self) -> torch.nn.Module:
        pass


class SPiFF(torch.nn.Module):
    """Model creating latent size representations chemical molecules."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        latent_size: int,
        gnn_factory: ModelFactory,
        linear_factory: ModelFactory,
        readout_factory: FuncFactory,
        projection_head_size: int,
    ) -> None:

        """
        Construct the SpiFF model.

        :param input_size: size of a model input sample.
        :param intermediate_size: size of the intermediate representation after the gnn
        and before the linear layers
        :param latent_size: size of model output (dimensionality of representations).
        :param gnn_factory: factory producing GNN models molecule graph representations.
        :param readout_factory: factory producing readout used after the GNN.
        :param linear_factory: factory producing the linear part of the model.
        :param projection_head_size: size of linear layer used as a projection head
        during the training phase.
        """

        super().__init__()

        self.gnn = gnn_factory(input_size, intermediate_size)
        self.readout_function = readout_factory()
        self.mlp = linear_factory(intermediate_size, latent_size)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(latent_size, projection_head_size),
            torch.nn.ReLU(),
            torch.nn.Linear(projection_head_size, projection_head_size),
        )

        self._latent_size = latent_size

    @property
    def latent_size(self) -> int:
        return self._latent_size

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        The SpiFF model forward pass.

        :param x: batch of molecule node features.
        :param edge_index: graph connectivity tensor.
        :param batch: assigns each node to a specific example.
        :returns: representation of a chemical molecule graph.
        """

        x = self.gnn(x, edge_index)
        x = self.readout_function(x, batch)
        x = self.mlp(x)
        if self.training:
            x = self.projection_head(x)
        return x


class LinearModelFactory(ModelFactory):
    """Factory producing mlp models."""

    def __init__(
        self,
        layer_sizes: List[int],
        activation_factory: FuncFactory,
        use_batch_norm: bool = True,
    ) -> None:
        """
        Construct the factory.

        :param layer_sizes: sizes of the hidden layers of the produced models.
        :param activation_factory: factory producing activation functions used in the
        model.
        :param use_batch_norm: whether to use batch norm between the layers.
        """

        self.layer_sizes = layer_sizes
        self.activation_factory = activation_factory
        self.use_batch_norm = use_batch_norm

    def __call__(self, input_size: int, output_size: int) -> torch.nn.Module:
        layers = []
        for in_features, out_features in zip(
            [input_size] + self.layer_sizes, self.layer_sizes
        ):
            layers.append(torch.nn.Linear(in_features, out_features))
            if self.use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(out_features))
            layers.append(self.activation_factory())
        layers.append(torch.nn.Linear(self.layer_sizes[-1], output_size))
        return torch.nn.Sequential(*layers)


class GNNFactory(ModelFactory):
    """Factory producing gnn models."""

    def __init__(
        self,
        name: str,
        hidden_size: int,
        num_layers: int,
        activation_factory: FuncFactory = lambda: torch.nn.ReLU(),
    ) -> None:
        """
        Construct the factory.

        Maps gnn names given as strings to torch_geometric classes.

        :param name: string representing the name of the gnn this object produces.
        :param hidden_size: size of each hidden sample of the gnn.
        :param num_layers: number of message passing layers.
        :param activation_factory: factory producing the activation function used by
        the gnn.
        :raises ValueError: if unsupported name is supplied.
        """
        match name.lower():
            case "sage":
                self.gnn_class = torch_geometric.nn.models.GraphSAGE
            case "gat":
                self.gnn_class = torch_geometric.nn.models.GAT
            case "gin":
                self.gnn_class = torch_geometric.nn.models.GIN
            case _:
                logger.error(f"Passed unsupported {name} as gnn type.")
                raise ValueError()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation_factory = activation_factory

    def __call__(self, input_size: int, output_size: int) -> torch.nn.Module:
        return self.gnn_class(
            input_size,
            self.hidden_size,
            self.num_layers,
            output_size,
            act=self.activation_factory(),
        )


class ActivationFuncFactory(FuncFactory):
    """
    Factory producing activation functions.

    Maps function names given as strings to torch classes.
    """

    def __init__(self, name: str, *args):
        """
        Construct the factory.

        :param name: string representing the name of the activation function this object
        produces.
        :param args: additional arguments that may be passed to functions' constructors.
        :raises ValueError: if unsupported name is supplied.
        """
        match name.lower():
            case "elu":
                self.func_type = torch.nn.ELU
            case "leakyrelu":
                self.func_type = torch.nn.LeakyReLU
            case "relu":
                self.func_type = torch.nn.ReLU
            case "sigmoid":
                self.func_type = torch.nn.Sigmoid
            case _:
                logger.error(
                    f"Passed unsupported {name} as an activation function name."
                )
                raise ValueError()
        self.args = args

    def __call__(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.func_type(*self.args)


class ReadoutFuncFactory(FuncFactory):
    """
    Factory producing readout functions.

    Maps readout names given as strings to torch_geometric classes.
    """

    def __init__(self, name: str, *args):
        """
        Construct the factory.

        :param name: string representing the name of the readout function this object
        produces.
        :param args: additional arguments that may be passed to functions' constructors.
        :raises ValueError: if unsupported name is supplied.
        """
        match name.lower():
            case "max":
                self.readout = torch_geometric.nn.aggr.MaxAggregation
            case "mean":
                self.readout = torch_geometric.nn.aggr.MeanAggregation
            case "sum":
                self.readout = torch_geometric.nn.aggr.SumAggregation
            case _:
                logger.error(f"Passed unsupported {name} as readout name.")
                raise ValueError()
        self.args = args

    def __call__(self) -> torch.nn.Module:
        return self.readout(*self.args)
