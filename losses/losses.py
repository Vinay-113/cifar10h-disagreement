"""Soft-label losses for matching human annotator distributions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from config import COMPOSITE_LOSS_LAMBDA, EPS


def entropy_from_probabilities(probabilities: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Compute Shannon entropy in bits for each row of a probability tensor.

    Args:
        probabilities: Tensor of shape `(B, C)` containing class probabilities.
        eps: Numerical stability constant.

    Returns:
        Tensor of shape `(B,)` with entropy values in bits.
    """

    clipped = probabilities.clamp_min(eps)
    return -(clipped * (clipped.log() / math.log(2.0))).sum(dim=1)


class KLDivLoss(nn.Module):
    """KL divergence from the true annotator distribution to the model distribution."""

    def __init__(self) -> None:
        """Initialize the KL divergence loss."""

        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits: torch.Tensor, true_soft_labels: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence using model logits and target soft labels.

        Args:
            logits: Unnormalized model logits of shape `(B, C)`.
            true_soft_labels: Ground-truth soft labels of shape `(B, C)`.

        Returns:
            Scalar KL divergence loss.
        """

        log_predictions = F.log_softmax(logits, dim=1)
        targets = true_soft_labels.clamp_min(EPS)
        return self.criterion(log_predictions, targets)


class JSDivLoss(nn.Module):
    """Jensen-Shannon divergence between the target and predicted distributions."""

    def forward(self, logits: torch.Tensor, true_soft_labels: torch.Tensor) -> torch.Tensor:
        """Compute the Jensen-Shannon divergence.

        Args:
            logits: Unnormalized model logits of shape `(B, C)`.
            true_soft_labels: Ground-truth soft labels of shape `(B, C)`.

        Returns:
            Scalar Jensen-Shannon divergence.
        """

        predictions = torch.softmax(logits, dim=1).clamp_min(EPS)
        targets = true_soft_labels.clamp_min(EPS)
        mixture = 0.5 * (predictions + targets)

        target_term = (targets * (targets.log() - mixture.log())).sum(dim=1)
        prediction_term = (predictions * (predictions.log() - mixture.log())).sum(dim=1)
        return 0.5 * (target_term + prediction_term).mean()


class CosineLoss(nn.Module):
    """Cosine-distance loss between predicted and target probability vectors."""

    def forward(self, logits: torch.Tensor, true_soft_labels: torch.Tensor) -> torch.Tensor:
        """Compute `1 - cosine_similarity` averaged over the batch.

        Args:
            logits: Unnormalized model logits of shape `(B, C)`.
            true_soft_labels: Ground-truth soft labels of shape `(B, C)`.

        Returns:
            Scalar cosine-distance loss.
        """

        predictions = torch.softmax(logits, dim=1)
        similarity = F.cosine_similarity(predictions, true_soft_labels, dim=1)
        return (1.0 - similarity).mean()


class CustomCompositeLoss(nn.Module):
    """KL divergence with an entropy-matching penalty.

    This objective is designed for disagreement prediction rather than standard
    classification. KL divergence encourages the model to place mass on the same
    classes as the annotators, while the entropy penalty explicitly encourages
    the model to match how uncertain humans were about the image overall. That
    makes the loss sensitive not only to distribution shape, but also to the
    level of ambiguity reflected by the annotator pool.
    """

    def __init__(self, lambda_entropy: float = COMPOSITE_LOSS_LAMBDA) -> None:
        """Initialize the composite loss.

        Args:
            lambda_entropy: Weight assigned to the squared entropy mismatch term.
        """

        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.kl_loss = KLDivLoss()

    def forward(self, logits: torch.Tensor, true_soft_labels: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence plus a squared entropy mismatch penalty.

        Args:
            logits: Unnormalized model logits of shape `(B, C)`.
            true_soft_labels: Ground-truth soft labels of shape `(B, C)`.

        Returns:
            Scalar composite loss value.
        """

        predictions = torch.softmax(logits, dim=1)
        kl_value = self.kl_loss(logits, true_soft_labels)
        true_entropy = entropy_from_probabilities(true_soft_labels)
        pred_entropy = entropy_from_probabilities(predictions)
        entropy_penalty = (true_entropy - pred_entropy).pow(2).mean()
        return kl_value + self.lambda_entropy * entropy_penalty


def build_loss(loss_name: str) -> nn.Module:
    """Instantiate one of the supported loss modules.

    Args:
        loss_name: One of `kl`, `js`, `cosine`, or `composite`.

    Returns:
        Instantiated PyTorch loss module.
    """

    if loss_name == "kl":
        return KLDivLoss()
    if loss_name == "js":
        return JSDivLoss()
    if loss_name == "cosine":
        return CosineLoss()
    if loss_name == "composite":
        return CustomCompositeLoss()
    raise ValueError(f"Unsupported loss '{loss_name}'.")
