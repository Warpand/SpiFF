from abc import ABC, abstractmethod
from typing import Callable, List, Type

import torch
import torch_geometric.nn.models
import torch_geometric.nn.models.basic_gnn as basic_gnn


class SPiFF(torch.nn.Module):
    """
    Model for creating latent size representation of a chemical molecule graph.
    """

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        gnn_class: Type[basic_gnn.BasicGNN],
        hidden_size: int,
        num_gnn_layers: int,
        readout_function: Callable[
            [torch.Tensor, torch.Tensor | None, int | None], torch.Tensor
        ],
        linear_layer_sizes: List[int],
        linear_activation_func: Type[torch.nn.Module],
        reprojection_head_size: int,
    ) -> None:

        """
        Construct SPiFF model.

        :param input_size: size of model input.
        :param latent_size: size of model output (dimension of representation).
        :param gnn_class: GNN model
         which will be used for molecule graph representation.
        :param hidden_size: size of GNN hidden layers.
        :param num_gnn_layers: number of layers used in GNN model.
        :param readout_function: type of readout function which will be used after GNN.
        :param linear_layer_sizes: list of linear layer sizes.
        :param linear_activation_func: type of activation function
         which will be used along with linear layers.
        :param reprojection_head_size: size of linear layer
         which will be used in reprojection head during training phase.
        """

        super().__init__()

        self.gnn_model = gnn_class(input_size, hidden_size, num_gnn_layers)
        self.readout_function = readout_function

        linear_layer_sizes.insert(0, hidden_size)
        linear_layer_sizes.append(latent_size)

        mlp_layers = []
        for dim_in, dim_out in zip(linear_layer_sizes[:-1], linear_layer_sizes[1:]):
            mlp_layers.append(torch.nn.Linear(dim_in, dim_out))
            mlp_layers.append(linear_activation_func())

        self.mlp_model = torch.nn.Sequential(*mlp_layers)
        self.reprojection_head = torch.nn.Sequential(
            torch.nn.Linear(latent_size, reprojection_head_size),
            linear_activation_func(),
            torch.nn.Linear(reprojection_head_size, reprojection_head_size),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Represent definition of SpiFF forward pass.

        :param x: batch of molecule node features.
        :param edge_index: graph connectivity tensor.
        :param batch: assigns each node to a specific example.
        :returns: representation of a chemical molecule graph.
        """

        x = self.gnn_model(x, edge_index)
        x = self.readout_function(x, batch, None)
        x = self.mlp_model(x)
        x = self.reprojection_head(x) if self.training else x
        return x


class SPiFFFactory(ABC):
    @abstractmethod
    def __call__(self, *args) -> SPiFF:
        pass


class DefaultSPiFFFactory(SPiFFFactory):
    def __call__(
        self,
        input_size: int,
        latent_size: int,
        gnn_class_type: str,
        hidden_size: int,
        num_gnn_layers: int,
        readout_function_type: str,
        linear_layer_sizes: List[int],
        linear_activation_func_type: str,
        reprojection_head_size: int,
    ) -> SPiFF:
        """
        Create new SPiFF model.

        :param input_size: size of model input.
        :param latent_size: size of model output
         (dimension of representation).
        :param gnn_class_type: type of GNN model
         which will be used for molecule graph representation.
        :param hidden_size: size of GNN hidden layers.
        :param num_gnn_layers: number of layers used in GNN model.
        :param readout_function_type: type of readout function
         which will be used after GNN.
        :param linear_layer_sizes: list of linear layer sizes.
        :param linear_activation_func_type: type of activation function
                which will be used along with linear layers.
        :param reprojection_head_size: size of linear layer
         which will be used in reprojection head during training phase.
        :returns: new SPiFF model
        :raises ValueError: if gnn_class_type is other than 'SAGE' or 'GAT'.
         if readout_function_type is other than 'mean', 'max' or 'add'.
         if linear_activation_func_type is other than 'relu', 'elu' or 'sigmoid'.
        """
        match gnn_class_type:
            case "SAGE":
                gnn_class = torch_geometric.nn.models.GraphSAGE
            case "GAT":
                gnn_class = torch_geometric.nn.models.GAT
            case _:
                raise ValueError(
                    f"{gnn_class_type} is unexpected for gnn_class_type argument"
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
            reprojection_head_size,
        )
