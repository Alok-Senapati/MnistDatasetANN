"""Convolutional Neural Network (CNN) classifier for MNIST handwritten digit recognition."""

from __future__ import annotations

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    """Convolutional Neural Network classifier for 2D image classification.

    This model consists of a sequence of 2D convolutional blocks followed by a fully
    connected classification head. Each convolutional block applies a 3x3 convolution
    with padding, 2D batch normalization, in-place ReLU activation, and 2x2 max pooling.
    """

    def __init__(
        self,
        in_dims: tuple[int, int, int] = (1, 28, 28),
        conv_channels: list[int] | tuple[int, ...] = (32, 64),
        fc_hidden: int = 128,
        num_classes: int = 10,
        dropout: float = 0.25,
    ) -> None:
        """Initialize the CNN architecture with dynamic feature map scaling.

        Args:
            in_dims: Dimensions of input tensor as `(in_channels, height, width)`.
                Defaults to `(1, 28, 28)`.
            conv_channels: List or tuple of output channel depths for each convolutional block.
                Defaults to `(32, 64)`.
            fc_hidden: Number of hidden units in the dense classification head.
                Defaults to `128`.
            num_classes: Number of discrete target output classes.
                Defaults to `10`.
            dropout: Dropout probability applied before the final classification layer.
                Defaults to `0.25`.
        """
        super().__init__()
        conv_layers: dict[str, nn.Module] = {}
        self.in_dims = in_dims
        self.num_classes = num_classes

        in_channels = in_dims[0]
        image_dims = in_dims[1:]

        # Build dynamic 2D convolutional feature extraction blocks
        for idx, channels in enumerate(conv_channels, start=1):
            conv_layers[f"conv{idx}"] = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            in_channels = channels
            image_dims = tuple(dim // 2 for dim in image_dims)

        self.conv_layers = nn.ModuleDict(conv_layers)

        # Compute total flattened feature size: out_channels * height * width
        feature_dims = in_channels * reduce(lambda i, j: i * j, image_dims)

        # Fully connected classification head
        self.classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dims, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, num_classes),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using activation-matched schemes (Kaiming Normal, Xavier Uniform).

        Args:
            module: Sub-module instance traversed by `self.apply()`.
        """
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            if module.out_features == self.num_classes:
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the forward pass through convolutional blocks and classification head.

        Args:
            x: Input tensor of shape `(Batch, in_channels, Height, Width)` or
                flattened `(Batch, Features)`.

        Returns:
            Raw unnormalized logit predictions of shape `(Batch, num_classes)`.
        """
        # Automatically reshape flattened 1D features (B, 784) into 4D spatial tensors (B, C, H, W)
        if x.ndim == 2:
            x = x.reshape((x.shape[0], *self.in_dims))

        conv_out = x

        for layer in self.conv_layers.values():
            conv_out = layer(conv_out)

        return self.classifier_layer(conv_out)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute softmax class probabilities for the given input samples.

        Args:
            x: Input tensor of shape `(Batch, in_channels, H, W)` or `(Batch, Features)`.

        Returns:
            Normalized probability distribution of shape `(Batch, num_classes)`.
        """
        layer_output = self.forward(x)
        return F.softmax(layer_output, dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict discrete class index for each input sample.

        Args:
            x: Input tensor of shape `(Batch, in_channels, H, W)` or `(Batch, Features)`.

        Returns:
            1D tensor of predicted class indices of shape `(Batch, )`.
        """
        layer_output = self.forward(x)
        return torch.argmax(layer_output, dim=1)

    @torch.no_grad()
    def get_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract intermediate activation feature maps from each convolutional block.

        Args:
            x: Input tensor of shape `(Batch, in_channels, H, W)` or `(Batch, Features)`.

        Returns:
            Dictionary mapping layer names (e.g. `"conv1"`, `"conv2"`) to their
            output activation tensors of shape `(Batch, Channels, Height, Width)`.
        """
        if x.ndim == 2:
            x = x.reshape(x.shape[0], *self.in_dims)

        feature_maps: dict[str, torch.Tensor] = {}
        conv_out: torch.Tensor = x

        for name, layer in self.conv_layers.items():
            conv_out = layer(conv_out)
            feature_maps[name] = conv_out.detach().cpu()

        return feature_maps
