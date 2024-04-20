from abc import ABC, abstractmethod
from typing import Callable, List, Type

import torch
import torch_geometric.nn.models
import torch_geometric.nn.models.basic_gnn as basic_gnn


class ModelFactory(ABC):
    @abstractmethod
    def __call__(self, *args) -> torch.nn.Module:
        pass


class SPiFF(torch.nn.Module):
    """Model creating latent size representations chemical molecules."""

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        gnn_class: Type[basic_gnn.BasicGNN],
        hidden_size: int,
        num_gnn_layers: int,
        readout_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        linear_layer_sizes: List[int],
        linear_activation_func: Type[torch.nn.Module],
        projection_head_size: int,
    ) -> None:

        """
        Construct the SpiFF model.

        :param input_size: size of a model input sample.
        :param latent_size: size of model output (dimensionality of representations).
        :param gnn_class: GNN model class used for molecule graph representations.
        :param hidden_size: size of the GNN hidden layers.
        :param num_gnn_layers: number of layers used in GNN model.
        :param readout_function: readout function used after the GNN.
        :param linear_layer_sizes: list of the linear layer sizes.
        :param linear_activation_func: type of the activation function between the
        linear layers.
        :param projection_head_size: size of linear layer used as a projection head
        during the training phase.
        """

        super().__init__()

        self.gnn = gnn_class(input_size, hidden_size, num_gnn_layers)
        self.readout_function = readout_function

        linear_layer_sizes.insert(0, hidden_size)
        linear_layer_sizes.append(latent_size)

        mlp_layers = []
        for dim_in, dim_out in zip(linear_layer_sizes[:-1], linear_layer_sizes[1:]):
            mlp_layers.append(torch.nn.Linear(dim_in, dim_out))
            mlp_layers.append(linear_activation_func())

        self.mlp = torch.nn.Sequential(*mlp_layers)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(latent_size, projection_head_size),
            linear_activation_func(),
            torch.nn.Linear(projection_head_size, projection_head_size),
        )

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


class SPiFFFactory(ModelFactory):
    def __call__(
        self,
        input_size: int,
        latent_size: int,
        gnn_type: str,
        hidden_size: int,
        num_gnn_layers: int,
        readout_function_type: str,
        linear_layer_sizes: List[int],
        linear_activation_func_type: str,
        projection_head_size: int,
    ) -> SPiFF:
        """
        Create new SPiFF model.

        :param input_size: size of the model input sample.
        :param latent_size: size of model output (dimensionality of representation).
        :param gnn_type: string representing the type of GNN model used
        for molecule graph representation.
        :param hidden_size: size of GNN hidden layers.
        :param num_gnn_layers: number of layers used in the GNN.
        :param readout_function_type: string representing the type of the readout
        function be used after the GNN.
        :param linear_layer_sizes: list of linear layer sizes.
        :param linear_activation_func_type: string representing the type of the
        activation function used between the linear layers.
        :param projection_head_size: size of linear layer used in the projection head
        during the training phase.
        :returns: new instance of the SPiFF class.
        :raises ValueError: if unsupported values are passed as gnn_type or
        linear_activation_func_type.
        """
        
        match gnn_type:
            case "SAGE":
                gnn_class = torch_geometric.nn.models.GraphSAGE
            case "GAT":
                gnn_class = torch_geometric.nn.models.GAT
            case _:
                raise ValueError(
                    f"{gnn_type} is unexpected for gnn_class_type argument"
                )

        match readout_function_type:
            case "mean":
                readout_function = torch_geometric.nn.global_mean_pool
            case "max":
                readout_function = torch_geometric.nn.global_max_pool
            case "add":
                readout_function = torch_geometric.nn.global_add_pool
            case _:
                raise ValueError(
                    f"{readout_function_type} is unexpected "
                    f"for readout_function_type argument"
                )

        match linear_activation_func_type:
            case "relu":
                linear_activation_func = torch.nn.ReLU
            case "elu":
                linear_activation_func = torch.nn.ELU
            case "sigmoid":
                linear_activation_func = torch.nn.Sigmoid
            case _:
                raise ValueError(
                    f"{linear_activation_func_type} "
                    f"is unexpected for linear_activation_func_type argument"
                )

        return SPiFF(
            input_size,
            latent_size,
            gnn_class,
            hidden_size,
            num_gnn_layers,
            readout_function,
            linear_layer_sizes,
            linear_activation_func,
            projection_head_size,
        )
