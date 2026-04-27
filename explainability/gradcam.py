"""Grad-CAM analysis for clear and ambiguous CIFAR-10H test images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CIFAR10_CLASSES,
    EXPLAINABILITY_DIR,
    GRADCAM_HIGH_COUNT,
    GRADCAM_LOW_COUNT,
    RAW_DATA_DIR,
    SEED,
    build_run_name,
    ensure_project_dirs,
    get_device,
    seed_everything,
)
from data.dataset import CIFAR10HSoftLabelDataset, build_transforms, denormalize_image
from evaluate import load_model_from_checkpoint, resolve_checkpoint_path
from models.backbone import get_last_conv_layer


class GradCAM:
    """Minimal Grad-CAM implementation for CNN backbones."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        """Register hooks needed to compute Grad-CAM heatmaps.

        Args:
            model: Trained disagreement model.
            target_layer: Last convolutional layer to visualize.
        """

        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        """Store feature maps from the target layer."""

        self.activations = output.detach()

    def _save_gradients(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Store gradients flowing back to the target feature maps."""

        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate a normalized Grad-CAM map and predicted probabilities.

        Args:
            image_tensor: Normalized input tensor of shape `(1, 3, 32, 32)`.
            class_index: Optional class index to backpropagate from.

        Returns:
            Tuple `(heatmap, predicted_probabilities)`.
        """

        logits = self.model(image_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        logits[:, class_index].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(heatmap, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = heatmap[0, 0]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max().clamp_min(1e-12)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        return heatmap.detach().cpu().numpy(), probabilities

    def close(self) -> None:
        """Remove the registered forward and backward hooks."""

        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Blend a Grad-CAM heatmap with the underlying RGB image.

    Args:
        image: RGB image array in `[0, 1]`.
        heatmap: Heatmap array in `[0, 1]`.
        alpha: Blend weight for the heatmap.

    Returns:
        RGB overlay image in `[0, 1]`.
    """

    colored = plt.get_cmap("jet")(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * colored, 0.0, 1.0)


def _plot_distribution(ax: plt.Axes, probabilities: np.ndarray, title: str, color: str) -> None:
    """Draw a class-probability bar chart on an axis.

    Args:
        ax: Destination axis.
        probabilities: Probability vector of shape `(10,)`.
        title: Axis title.
        color: Bar color.
    """

    ax.bar(np.arange(len(probabilities)), probabilities, color=color)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(len(probabilities)))
    ax.set_xticklabels([name[:3] for name in CIFAR10_CLASSES], rotation=45, fontsize=8)
    ax.set_title(title, fontsize=10)


def sanitize_descriptor(text: str) -> str:
    """Convert a short text description into a filesystem-safe token."""

    return text.lower().replace(" ", "_").replace("/", "_")


def save_row_panel(
    output_path: Path,
    original_image: np.ndarray,
    overlay: np.ndarray,
    true_probs: np.ndarray,
    pred_probs: np.ndarray,
    true_entropy: float,
    pred_entropy: float,
) -> None:
    """Save one four-panel Grad-CAM summary row for a single image.

    Args:
        output_path: Destination PNG path.
        original_image: RGB image array in `[0, 1]`.
        overlay: RGB overlay image in `[0, 1]`.
        true_probs: Ground-truth soft-label vector.
        pred_probs: Predicted probability vector.
        true_entropy: True entropy in bits.
        pred_entropy: Predicted entropy in bits.
    """

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    axes[0].imshow(original_image)
    axes[0].axis("off")
    axes[0].set_title(f"Original\nH_true={true_entropy:.2f}")

    axes[1].imshow(overlay)
    axes[1].axis("off")
    axes[1].set_title("Grad-CAM overlay")

    _plot_distribution(axes[2], true_probs, "True distribution", "#2a9d8f")
    _plot_distribution(axes[3], pred_probs, f"Pred distribution\nH_pred={pred_entropy:.2f}", "#e76f51")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Grad-CAM generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", default="kl", choices=["kl", "js", "cosine", "composite"])
    parser.add_argument(
        "--backbone_init",
        default="cifar10_pretrained",
        choices=["random", "cifar10_pretrained", "imagenet_pretrained"],
    )
    parser.add_argument("--head", default="mlp", choices=["linear", "mlp", "temperature"])
    return parser.parse_args()


def main() -> None:
    """Generate Grad-CAM figures for the easiest and hardest test images."""

    args = parse_args()
    seed_everything()
    ensure_project_dirs()

    run_name = build_run_name(args.loss, args.backbone_init, args.head)
    output_dir = EXPLAINABILITY_DIR / "gradcam" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    checkpoint_path = resolve_checkpoint_path(args.loss, args.backbone_init, args.head, checkpoint_path=None)
    model, _ = load_model_from_checkpoint(checkpoint_path, device, args.loss, args.backbone_init, args.head)
    model.eval()

    _, eval_transform = build_transforms()
    dataset = CIFAR10HSoftLabelDataset(split="test", transform=eval_transform, data_dir=RAW_DATA_DIR, seed=SEED)
    test_entropy = dataset.entropies[dataset.indices]
    lowest = np.argsort(test_entropy)[:GRADCAM_LOW_COUNT]
    highest = np.argsort(test_entropy)[-GRADCAM_HIGH_COUNT:][::-1]
    selected = [("low", int(index)) for index in lowest] + [("high", int(index)) for index in highest]

    gradcam = GradCAM(model, get_last_conv_layer(model))
    grid_fig, grid_axes = plt.subplots(len(selected), 4, figsize=(15, 3.1 * len(selected)))
    grid_axes = np.atleast_2d(grid_axes)

    for row_index, (group_name, local_index) in enumerate(selected):
        image_tensor, _ = dataset[local_index]
        metadata = dataset.get_metadata(local_index)
        input_tensor = image_tensor.unsqueeze(0).to(device)
        heatmap, pred_probs = gradcam.generate(input_tensor)
        original_image = denormalize_image(image_tensor)
        overlay = overlay_heatmap(original_image, heatmap)
        true_probs = metadata["soft_label"]
        pred_entropy = float(-(pred_probs * (np.log2(np.clip(pred_probs, 1e-12, 1.0)))).sum())
        descriptor = sanitize_descriptor(
            f"{group_name}_{row_index:02d}_true_{CIFAR10_CLASSES[int(np.argmax(true_probs))]}_pred_{CIFAR10_CLASSES[int(np.argmax(pred_probs))]}"
        )
        save_row_panel(
            output_dir / f"{descriptor}.png",
            original_image,
            overlay,
            true_probs,
            pred_probs,
            metadata["entropy"],
            pred_entropy,
        )

        row_axes = grid_axes[row_index]
        row_axes[0].imshow(original_image)
        row_axes[0].axis("off")
        row_axes[0].set_title(f"{group_name.title()} example\nidx={metadata['global_index']}")

        row_axes[1].imshow(overlay)
        row_axes[1].axis("off")
        row_axes[1].set_title("Grad-CAM overlay")

        _plot_distribution(row_axes[2], true_probs, f"True\nH={metadata['entropy']:.2f}", "#2a9d8f")
        _plot_distribution(row_axes[3], pred_probs, f"Pred\nH={pred_entropy:.2f}", "#e76f51")

    grid_fig.tight_layout()
    grid_path = output_dir / "gradcam_summary_grid.png"
    grid_fig.savefig(grid_path, dpi=220)
    plt.close(grid_fig)
    gradcam.close()

    print(f"Saved Grad-CAM outputs to {output_dir}.")
    print(f"Saved summary grid to {grid_path}.")


if __name__ == "__main__":
    main()
