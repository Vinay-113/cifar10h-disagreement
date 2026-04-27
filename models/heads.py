"""Prediction heads that map 512-dimensional features to CIFAR-10 logits."""

from __future__ import annotations

import torch
from torch import nn

from config import MLP_DROPOUT, MLP_HIDDEN_DIM, NUM_CLASSES, TEMPERATURE_INIT


class LinearHead(nn.Module):
    """Single linear classifier head for soft-label prediction."""

    def __init__(self, in_features: int = 512, num_classes: int = NUM_CLASSES) -> None:
        """Initialize the linear classification head.

        Args:
            in_features: Feature dimensionality from the backbone.
            num_classes: Number of output classes.
        """

        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project backbone features directly to class logits.

        Args:
            features: Tensor of shape `(B, in_features)`.

        Returns:
            Logits tensor of shape `(B, num_classes)`.
        """

        return self.classifier(features)


class MLPHead(nn.Module):
    """Two-layer MLP head with dropout for higher-capacity predictions."""

    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = MLP_HIDDEN_DIM,
        dropout: float = MLP_DROPOUT,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Initialize the MLP classification head.

        Args:
            in_features: Feature dimensionality from the backbone.
            hidden_dim: Hidden layer width.
            dropout: Dropout probability between the hidden and output layers.
            num_classes: Number of output classes.
        """

        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map features to logits through a small MLP.

        Args:
            features: Tensor of shape `(B, in_features)`.

        Returns:
            Logits tensor of shape `(B, num_classes)`.
        """

        return self.classifier(features)


class TemperatureScaledHead(nn.Module):
    """Linear head with a learnable scalar temperature."""

    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = NUM_CLASSES,
        initial_temperature: float = TEMPERATURE_INIT,
    ) -> None:
        """Initialize the temperature-scaled head.

        Args:
            in_features: Feature dimensionality from the backbone.
            num_classes: Number of output classes.
            initial_temperature: Initial value of the learnable temperature.
        """

        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(initial_temperature))))

    @property
    def temperature(self) -> torch.Tensor:
        """Return the positive scalar temperature used to scale logits."""

        return self.log_temperature.exp().clamp_min(1e-4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to logits and divide by the learnable temperature.

        Args:
            features: Tensor of shape `(B, in_features)`.

        Returns:
            Logits tensor of shape `(B, num_classes)`.
        """

        return self.classifier(features) / self.temperature


def build_head(head_name: str, in_features: int = 512) -> nn.Module:
    """Construct one of the supported disagreement-prediction heads.

    Args:
        head_name: One of `linear`, `mlp`, or `temperature`.
        in_features: Backbone feature dimension.

    Returns:
        Instantiated PyTorch module.
    """

    if head_name == "linear":
        return LinearHead(in_features=in_features)
    if head_name == "mlp":
        return MLPHead(in_features=in_features)
    if head_name == "temperature":
        return TemperatureScaledHead(in_features=in_features)
    raise ValueError(f"Unsupported head '{head_name}'.")
