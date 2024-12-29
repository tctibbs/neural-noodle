"""
QNet Model Module

This module defines the QNet model, a neural network used for Q-learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A simple feedforward neural network for Q-learning.
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_layers: list[int]
    ) -> None:
        """
        Initialize the QNet.

        Args:
            input_dim: Number of input features.
            output_dim: Number of output actions.
            hidden_layers: List specifying number of nodes in each hidden layer.
        """
        super(Model, self).__init__()

        layers = []
        prev_dim = input_dim

        # Create hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor representing Q-values for each action.
        """
        return self.network(x)

    def _initialize_weights(self) -> None:
        """
        Initializes weights for the network using Xavier initialization.
        """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self._validate_weights()

    def _validate_weights(self) -> None:
        """
        Validates that weights and biases are properly initialized.
        Prints diagnostics for debugging purposes.
        """
        for name, param in self.named_parameters():
            if param.isnan().any():
                raise ValueError(
                    f"Parameter {name} contains NaN after initialization!"
                )
            if param.isinf().any():
                raise ValueError(
                    f"Parameter {name} contains Inf after initialization!"
                )

    @staticmethod
    def from_config(config: dict, input_dim: int, output_dim: int) -> Model:
        """
        Create a Model instance from a configuration dictionary.

        Args:
            config: Dictionary containing model configuration.
            input_dim: Number of input features.
            output_dim: Number of output actions.

        Returns:
            An instance of Model.
        """
        hidden_layers = config["hidden_layers"]
        return Model(input_dim, output_dim, hidden_layers)

    @staticmethod
    def clone_with_same_settings(existing_model: Model) -> Model:
        """
        Create a new Model instance with the same settings as an existing model.

        Args:
            existing_model: The model to clone.

        Returns:
            A new Model with the same architecture as the given model.
        """
        # Retrieve settings from the existing model
        input_dim = existing_model.network[0].in_features
        output_dim = existing_model.network[-1].out_features
        hidden_layers = [
            layer.out_features
            for layer in existing_model.network
            if isinstance(layer, nn.Linear)
            and layer is not existing_model.network[-1]
        ]

        # Create and return a new model with the same settings
        return Model(input_dim, output_dim, hidden_layers)
