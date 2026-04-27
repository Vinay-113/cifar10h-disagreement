"""Robustness B: analyze prediction entropy under simple image corruptions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BATCH_SIZE,
    NUM_WORKERS,
    OOD_SEVERITIES,
    RAW_DATA_DIR,
    ROBUSTNESS_DIR,
    SEED,
    build_run_name,
    ensure_project_dirs,
    get_device,
    seed_everything,
)
from data.dataset import CIFAR10HSoftLabelDataset, build_transforms
from evaluate import entropy_from_probabilities, load_model_from_checkpoint, resolve_checkpoint_path


def apply_corruption(
    image: np.ndarray,
    corruption_type: str,
    severity: int,
    sample_index: int,
) -> Image.Image:
    """Apply a requested corruption to a CIFAR-10 image.

    Args:
        image: Raw RGB image array of shape `(32, 32, 3)`.
        corruption_type: One of `gaussian_noise`, `gaussian_blur`, or `contrast`.
        severity: Integer severity level from 1 to 5.
        sample_index: Dataset index used to make the corruption deterministic per image.

    Returns:
        Corrupted PIL image.
    """

    pil_image = Image.fromarray(image)
    if corruption_type == "gaussian_noise":
        std = severity * 0.05
        noisy = np.asarray(pil_image).astype(np.float32) / 255.0
        rng = np.random.default_rng(SEED + severity * 10_000 + sample_index)
        noisy = np.clip(noisy + rng.normal(0.0, std, size=noisy.shape), 0.0, 1.0)
        return Image.fromarray((255.0 * noisy).astype(np.uint8))
    if corruption_type == "gaussian_blur":
        sigma = severity * 0.5
        return pil_image.filter(ImageFilter.GaussianBlur(radius=sigma))
    if corruption_type == "contrast":
        factor = max(0.05, 1.0 - severity * 0.15)
        return ImageEnhance.Contrast(pil_image).enhance(factor)
    raise ValueError(f"Unsupported corruption type '{corruption_type}'.")


class CorruptedCIFAR10HDataset(Dataset):
    """Dataset wrapper that serves corrupted versions of the CIFAR-10H test images."""

    def __init__(self, corruption_type: str, severity: int) -> None:
        """Initialize the corrupted dataset.

        Args:
            corruption_type: Corruption identifier.
            severity: Integer severity level from 1 to 5.
        """

        super().__init__()
        _, self.eval_transform = build_transforms()
        self.base_dataset = CIFAR10HSoftLabelDataset(split="test", transform=None, data_dir=RAW_DATA_DIR, seed=SEED)
        self.corruption_type = corruption_type
        self.severity = severity

    def __len__(self) -> int:
        """Return the number of test examples."""

        return len(self.base_dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return a normalized, corrupted image tensor.

        Args:
            index: Split-relative sample index.

        Returns:
            Corrupted image tensor of shape `(3, 32, 32)`.
        """

        metadata = self.base_dataset.get_metadata(index)
        corrupted = apply_corruption(
            metadata["raw_image"],
            self.corruption_type,
            self.severity,
            sample_index=index,
        )
        return self.eval_transform(corrupted)


def compute_mean_entropy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute the mean predictive entropy over a corrupted dataset.

    Args:
        model: Trained disagreement model.
        dataloader: Dataloader yielding corrupted images.
        device: Device on which to execute inference.

    Returns:
        Mean entropy in bits.
    """

    model.eval()
    entropy_values: list[np.ndarray] = []
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            entropy_values.append(entropy_from_probabilities(probabilities))
    return float(np.concatenate(entropy_values, axis=0).mean())


def plot_corruption_curves(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save predicted entropy vs severity for each corruption type.

    Args:
        results_df: Dataframe with `corruption`, `severity`, and `mean_entropy`.
        output_path: Destination PNG path.
    """

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=results_df, x="severity", y="mean_entropy", hue="corruption", marker="o", linewidth=2.5, ax=ax)
    ax.set_title("Predicted Entropy Under OOD Corruptions")
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Mean predicted entropy (bits)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for corruption robustness."""

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
    """Evaluate entropy changes under OOD corruptions and report the trend."""

    args = parse_args()
    seed_everything()
    ensure_project_dirs()

    run_name = build_run_name(args.loss, args.backbone_init, args.head)
    output_dir = ROBUSTNESS_DIR / "ood_corruptions" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    checkpoint_path = resolve_checkpoint_path(args.loss, args.backbone_init, args.head, checkpoint_path=None)
    model, _ = load_model_from_checkpoint(checkpoint_path, device, args.loss, args.backbone_init, args.head)

    rows: list[dict[str, float | int | str]] = []
    corruption_types = ["gaussian_noise", "gaussian_blur", "contrast"]
    for corruption_type in corruption_types:
        for severity in OOD_SEVERITIES:
            dataset = CorruptedCIFAR10HDataset(corruption_type=corruption_type, severity=severity)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            mean_entropy = compute_mean_entropy(model, dataloader, device)
            rows.append(
                {
                    "corruption": corruption_type,
                    "severity": severity,
                    "mean_entropy": mean_entropy,
                }
            )

    results_df = pd.DataFrame(rows)
    csv_path = output_dir / "ood_corruption_entropy.csv"
    plot_path = output_dir / "ood_corruption_entropy.png"
    results_df.to_csv(csv_path, index=False)
    plot_corruption_curves(results_df, plot_path)

    print(results_df.to_string(index=False))
    for corruption_type in corruption_types:
        curve = results_df[results_df["corruption"] == corruption_type].sort_values("severity")
        diffs = np.diff(curve["mean_entropy"].to_numpy())
        if np.all(diffs >= -1e-8):
            print(f"{corruption_type}: predicted entropy increased monotonically with severity.")
        else:
            print(f"{corruption_type}: predicted entropy did NOT increase monotonically with severity.")

    print(f"Saved robustness table to {csv_path}.")
    print(f"Saved plot to {plot_path}.")


if __name__ == "__main__":
    main()
