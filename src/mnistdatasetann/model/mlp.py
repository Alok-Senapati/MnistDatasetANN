"""Multilayer perceptron classifier used for MNIST digits."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """A small feed-forward network for digit classification.

    The model consists of a list of hidden fully connected layers, optional batch
    normalization, ReLU activations, and a final classification head.
    """

    def __init__(
        self,
        *,
        hidden: list[int],
        in_dim: int,
        out_dim: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        """Build the network architecture from the provided configuration.

        Args:
            hidden: Hidden-layer sizes for each fully connected layer.
            in_dim: Number of input features for the first layer.
            out_dim: Number of output classes for the final prediction head.
            dropout: Dropout probability applied after each hidden activation.
            use_batchnorm: Whether to insert batch normalization after each hidden layer.
        """
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = in_dim

        for hidden_dim in hidden:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.head = nn.Linear(previous_dim, out_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and batch-normalization parameters in a stable way.

        Args:
            module: A submodule created by the network that needs parameter initialization.

        Returns:
            None. The function mutates the module parameters in place.
        """
        if isinstance(module, nn.Linear):
            if module is self.head:
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Run the network on an input batch.

        Args:
            X: Input tensor of shape ``(batch_size, in_dim)`` or
                spatial ``(batch_size, 1, 28, 28)``.

        Returns:
            Unnormalized class logits for each sample in the batch.
        """
        if X.ndim > 2:
            X = X.flatten(1)
        hidden_output = self.hidden_layers(X)
        return self.head(hidden_output)

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for each sample in the batch.

        Args:
            X: Input tensor of shape ``(batch_size, in_dim)`` or
                spatial ``(batch_size, 1, 28, 28)``.

        Returns:
            A probability distribution over output classes for each row in ``X``.
        """
        if X.ndim > 2:
            X = X.flatten(1)
        logits = self.head(self.hidden_layers(X))
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return the highest-probability class index for each row in ``X``.

        Args:
            X: Input tensor of shape ``(batch_size, in_dim)`` or
                spatial ``(batch_size, 1, 28, 28)``.

        Returns:
            Class indices selected by the model for each sample in the batch.
        """
        if X.ndim > 2:
            X = X.flatten(1)
        logits = self.head(self.hidden_layers(X))
        return torch.argmax(logits, dim=1)
