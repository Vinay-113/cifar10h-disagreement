"""Dataset utilities, deterministic splits, sanity checks, and visualizations."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BATCH_SIZE,
    CIFAR10_CLASSES,
    CIFAR10_DOWNLOAD_SUBDIR,
    CIFAR10H_COUNTS_FILENAME,
    CIFAR10H_DOWNLOAD_SUBDIR,
    CIFAR10_IMAGE_SIZE,
    CIFAR10_MEAN,
    CIFAR_RANDOM_CROP_PADDING,
    CIFAR10_STD,
    DATA_DIR,
    ENTROPY_TOLERANCE,
    EPS,
    NUM_CLASSES,
    NUM_WORKERS,
    PRETRAIN_VAL_SIZE,
    RAW_DATA_DIR,
    RESULTS_DIR,
    SEED,
    SPLIT_SIZES,
    ensure_project_dirs,
    seed_everything,
)


def shannon_entropy(probabilities: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Compute Shannon entropy in bits for each distribution row.

    Args:
        probabilities: Array of shape `(N, C)` containing categorical probabilities.
        eps: Numerical stability constant.

    Returns:
        Array of shape `(N,)` with entropy values in bits.
    """

    clipped = np.clip(probabilities, eps, 1.0)
    return -(clipped * (np.log(clipped) / math.log(2.0))).sum(axis=1)


def normalize_counts_to_probabilities(counts: np.ndarray) -> np.ndarray:
    """Convert raw annotator counts to row-normalized soft labels.

    Args:
        counts: Count matrix of shape `(N, C)`.

    Returns:
        Row-normalized probability matrix of shape `(N, C)`.
    """

    counts = counts.astype(np.float32)
    row_sums = counts.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every CIFAR-10H count row must contain at least one vote.")
    return counts / row_sums


def load_cifar10h_counts(data_dir: Path = RAW_DATA_DIR) -> np.ndarray:
    """Load the CIFAR-10H annotator count matrix.

    Args:
        data_dir: Root data directory containing the CIFAR-10H subfolder.

    Returns:
        Count matrix of shape `(10000, 10)`.
    """

    counts_path = data_dir / CIFAR10H_DOWNLOAD_SUBDIR / CIFAR10H_COUNTS_FILENAME
    if not counts_path.exists():
        raise FileNotFoundError(
            f"Missing CIFAR-10H counts file at {counts_path}. Run python data/download.py first."
        )

    counts = np.load(counts_path)
    if counts.shape != (10000, NUM_CLASSES):
        raise ValueError(f"Expected CIFAR-10H count matrix to have shape (10000, {NUM_CLASSES}).")
    return counts.astype(np.float32)


def load_cifar10_test_images(data_dir: Path = RAW_DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 test images and hard labels.

    Args:
        data_dir: Root data directory containing the CIFAR-10 subfolder.

    Returns:
        Tuple of `(images, hard_labels)` with shapes `(10000, 32, 32, 3)` and `(10000,)`.
    """

    cifar_root = data_dir / CIFAR10_DOWNLOAD_SUBDIR
    cifar10 = datasets.CIFAR10(root=str(cifar_root), train=False, download=False)
    return cifar10.data.copy(), np.asarray(cifar10.targets, dtype=np.int64)


def load_cifar10_train_images(data_dir: Path = RAW_DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 training images and hard labels for backbone pretraining.

    Args:
        data_dir: Root data directory containing the CIFAR-10 subfolder.

    Returns:
        Tuple of `(images, hard_labels)` with shapes `(50000, 32, 32, 3)` and `(50000,)`.
    """

    cifar_root = data_dir / CIFAR10_DOWNLOAD_SUBDIR
    cifar10 = datasets.CIFAR10(root=str(cifar_root), train=True, download=False)
    return cifar10.data.copy(), np.asarray(cifar10.targets, dtype=np.int64)


def load_cifar10h_soft_labels(data_dir: Path = RAW_DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10H counts and normalized soft labels.

    Args:
        data_dir: Root data directory containing the CIFAR-10H subfolder.

    Returns:
        Tuple `(counts, soft_labels)` where both arrays have shape `(10000, 10)`.
    """

    counts = load_cifar10h_counts(data_dir)
    soft_labels = normalize_counts_to_probabilities(counts)
    return counts, soft_labels


def run_alignment_checks(
    images: np.ndarray,
    hard_labels: np.ndarray,
    counts: np.ndarray,
    soft_labels: np.ndarray,
) -> None:
    """Validate CIFAR-10 test image and CIFAR-10H annotation alignment.

    CIFAR-10H annotates the CIFAR-10 test set in the same canonical order. This
    check verifies the one-to-one row alignment assumptions used by every split
    and reports the majority-vote agreement with the original CIFAR-10 labels as
    a secondary diagnostic rather than a hard assertion.

    Args:
        images: CIFAR-10 test images of shape `(10000, 32, 32, 3)`.
        hard_labels: Original CIFAR-10 test labels of shape `(10000,)`.
        counts: CIFAR-10H count matrix of shape `(10000, 10)`.
        soft_labels: Normalized CIFAR-10H soft-label matrix of shape `(10000, 10)`.
    """

    expected_shape = (10000, NUM_CLASSES)
    if counts.shape != expected_shape or soft_labels.shape != expected_shape:
        raise AssertionError(f"CIFAR-10H matrices must both have shape {expected_shape}.")
    if images.shape[0] != counts.shape[0] or hard_labels.shape[0] != counts.shape[0]:
        raise AssertionError("CIFAR-10 test images, hard labels, and CIFAR-10H rows must align 1:1.")
    if images.shape[1:] != (CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE, 3):
        raise AssertionError("CIFAR-10 test images must be 32x32 RGB arrays.")
    if not np.all((hard_labels >= 0) & (hard_labels < NUM_CLASSES)):
        raise AssertionError("CIFAR-10 hard labels must be valid class indices.")

    majority_labels = np.argmax(soft_labels, axis=1)
    majority_agreement = float(np.mean(majority_labels == hard_labels))
    print(
        "ASSERTION PASSED: CIFAR-10H rows align 1:1 with CIFAR-10 test images "
        f"(N={counts.shape[0]})."
    )
    print(f"Alignment diagnostic: CIFAR-10 label vs CIFAR-10H majority agreement = {majority_agreement:.4f}")


def run_sanity_checks(soft_labels: np.ndarray) -> np.ndarray:
    """Validate CIFAR-10H soft labels and print entropy statistics.

    Args:
        soft_labels: Normalized probability matrix of shape `(10000, 10)`.

    Returns:
        Entropy vector of shape `(10000,)`.
    """

    row_sums = soft_labels.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=ENTROPY_TOLERANCE):
        raise AssertionError("Soft-label rows must sum to 1.0 within tolerance.")
    if np.isnan(soft_labels).any():
        raise AssertionError("Soft labels contain NaN values.")

    max_deviation = float(np.max(np.abs(row_sums - 1.0)))
    print(
        "ASSERTION PASSED: every soft-label vector sums to 1.0 "
        f"within +/- {ENTROPY_TOLERANCE:g} (max deviation={max_deviation:.2e})."
    )
    print("ASSERTION PASSED: no NaN values found in the CIFAR-10H soft labels.")

    entropies = shannon_entropy(soft_labels)
    print("CIFAR-10H entropy statistics (bits):")
    print(f"  min    : {entropies.min():.4f}")
    print(f"  mean   : {entropies.mean():.4f}")
    print(f"  median : {np.median(entropies):.4f}")
    print(f"  std    : {entropies.std(ddof=0):.4f}")
    print(f"  max    : {entropies.max():.4f}")
    return entropies


def compute_split_entropy_stats(soft_labels: np.ndarray, seed: int = SEED) -> pd.DataFrame:
    """Compute per-split entropy summary statistics.

    Args:
        soft_labels: Normalized CIFAR-10H soft-label matrix of shape `(10000, 10)`.
        seed: Seed used to reproduce the deterministic train/val/test split.

    Returns:
        DataFrame with one row per split and columns for count, min, mean,
        median, standard deviation, and maximum entropy.
    """

    entropies = shannon_entropy(soft_labels)
    rows = []
    for split_name, indices in build_split_indices(seed=seed).items():
        split_entropies = entropies[indices]
        rows.append(
            {
                "split": split_name,
                "count": int(len(indices)),
                "entropy_min": float(split_entropies.min()),
                "entropy_mean": float(split_entropies.mean()),
                "entropy_median": float(np.median(split_entropies)),
                "entropy_std": float(split_entropies.std(ddof=0)),
                "entropy_max": float(split_entropies.max()),
            }
        )
    return pd.DataFrame(rows)


def print_split_entropy_stats(
    soft_labels: np.ndarray,
    output_path: Path | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Print and optionally save per-split entropy statistics.

    Args:
        soft_labels: Normalized CIFAR-10H soft-label matrix of shape `(10000, 10)`.
        output_path: Optional CSV path for the split entropy log.
        seed: Seed used to reproduce the deterministic train/val/test split.

    Returns:
        DataFrame containing the logged per-split entropy statistics.
    """

    stats = compute_split_entropy_stats(soft_labels=soft_labels, seed=seed)
    formatters = {
        column: "{:.4f}".format
        for column in stats.columns
        if column.startswith("entropy_")
    }
    print("Per-split entropy statistics (bits):")
    print(stats.to_string(index=False, formatters=formatters))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(output_path, index=False)
        print(f"Saved split entropy stats to {output_path}")
    return stats


def build_split_indices(seed: int = SEED) -> dict[str, np.ndarray]:
    """Create deterministic train/validation/test splits over the 10k CIFAR-10H examples.

    Args:
        seed: Random seed used to permute the dataset once.

    Returns:
        Dictionary mapping split name to global dataset indices.
    """

    total = sum(SPLIT_SIZES.values())
    if total != 10000:
        raise ValueError("CIFAR-10H split sizes must sum to 10000.")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total)
    train_end = SPLIT_SIZES["train"]
    val_end = train_end + SPLIT_SIZES["val"]
    return {
        "train": permutation[:train_end],
        "val": permutation[train_end:val_end],
        "test": permutation[val_end:],
    }


def build_pretrain_split_indices(seed: int = SEED) -> dict[str, np.ndarray]:
    """Create a deterministic train/validation split over CIFAR-10 hard labels.

    Args:
        seed: Random seed used for the permutation.

    Returns:
        Dictionary with `train` and `val` index arrays.
    """

    total = 50000
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total)
    return {
        "train": permutation[PRETRAIN_VAL_SIZE:],
        "val": permutation[:PRETRAIN_VAL_SIZE],
    }


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Create training and evaluation transforms for CIFAR-10 images.

    Training uses only two label-preserving augmentations:
    `RandomHorizontalFlip` and `RandomCrop(32, padding=4)`. Validation and test
    use normalization only. No class-changing or disagreement-changing transforms
    such as rotations, color jitter, cutout, mixup, or cutmix are used.

    Returns:
        Tuple of `(train_transform, eval_transform)`.
    """

    normalization = [
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ]
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(CIFAR10_IMAGE_SIZE, padding=CIFAR_RANDOM_CROP_PADDING),
            *normalization,
        ]
    )
    eval_transform = transforms.Compose(normalization)
    return train_transform, eval_transform


def print_augmentation_policy() -> None:
    """Print the augmentation policy used for soft-label training.

    The policy is intentionally conservative because CIFAR-10H labels represent
    human uncertainty over the original image; aggressive transforms could change
    the semantic ambiguity being modeled.
    """

    print("Data augmentation policy:")
    print(
        "  train   : RandomHorizontalFlip + "
        f"RandomCrop(size={CIFAR10_IMAGE_SIZE}, padding={CIFAR_RANDOM_CROP_PADDING}) + normalization"
    )
    print("  val/test: normalization only")
    print("  excluded: rotations, color jitter, cutout, mixup, cutmix, and other class-changing transforms")


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor back to a NumPy RGB image in `[0, 1]`.

    Args:
        image_tensor: Tensor of shape `(3, 32, 32)`.

    Returns:
        Image array of shape `(32, 32, 3)`.
    """

    mean = torch.tensor(CIFAR10_MEAN, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    image = image_tensor.detach().cpu() * std.cpu() + mean.cpu()
    return image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()


class CIFAR10HSoftLabelDataset(Dataset):
    """PyTorch dataset that pairs CIFAR-10 test images with CIFAR-10H soft labels."""

    def __init__(
        self,
        split: str,
        transform: transforms.Compose | None = None,
        return_metadata: bool = False,
        data_dir: Path = RAW_DATA_DIR,
        seed: int = SEED,
    ) -> None:
        """Initialize the dataset for a deterministic CIFAR-10H split.

        Args:
            split: One of `train`, `val`, or `test`.
            transform: Optional image transform to apply.
            return_metadata: Whether `__getitem__` should return a metadata dictionary.
            data_dir: Root directory containing raw CIFAR-10 and CIFAR-10H assets.
            seed: Seed controlling the split permutation.
        """

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'.")

        images, hard_labels = load_cifar10_test_images(data_dir)
        counts, soft_labels = load_cifar10h_soft_labels(data_dir)
        entropies = shannon_entropy(soft_labels)

        if len(images) != len(soft_labels):
            raise AssertionError("CIFAR-10 test images and CIFAR-10H labels must have identical length.")

        self.images = images
        self.hard_labels = hard_labels
        self.counts = counts
        self.soft_labels = soft_labels
        self.entropies = entropies
        self.indices = build_split_indices(seed=seed)[split]
        self.transform = transform
        self.return_metadata = return_metadata
        self.split = split

    def __len__(self) -> int:
        """Return the number of examples in the active split."""

        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | dict[str, Any]:
        """Return a sample from the dataset.

        Args:
            index: Split-relative sample index.

        Returns:
            By default, `(image_tensor, soft_label_tensor)`. When `return_metadata=True`,
            returns a dictionary that includes global index, hard label, counts, and entropy.
        """

        global_index = int(self.indices[index])
        image = Image.fromarray(self.images[global_index])
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        soft_label_tensor = torch.from_numpy(self.soft_labels[global_index]).float()
        if not self.return_metadata:
            return image_tensor, soft_label_tensor

        return {
            "image": image_tensor,
            "soft_label": soft_label_tensor,
            "hard_label": int(self.hard_labels[global_index]),
            "global_index": global_index,
            "counts": torch.from_numpy(self.counts[global_index]).float(),
            "entropy": float(self.entropies[global_index]),
        }

    def get_metadata(self, index: int) -> dict[str, Any]:
        """Return split-relative sample metadata without altering `__getitem__` behavior.

        Args:
            index: Split-relative sample index.

        Returns:
            Dictionary with raw image, labels, counts, entropy, and global index.
        """

        global_index = int(self.indices[index])
        return {
            "raw_image": self.images[global_index],
            "soft_label": self.soft_labels[global_index].copy(),
            "hard_label": int(self.hard_labels[global_index]),
            "counts": self.counts[global_index].copy(),
            "entropy": float(self.entropies[global_index]),
            "global_index": global_index,
        }


class CIFAR10HardLabelDataset(Dataset):
    """PyTorch dataset for hard-label CIFAR-10 pretraining."""

    def __init__(
        self,
        split: str,
        transform: transforms.Compose | None = None,
        data_dir: Path = RAW_DATA_DIR,
        seed: int = SEED,
    ) -> None:
        """Initialize a deterministic CIFAR-10 pretraining split.

        Args:
            split: One of `train` or `val`.
            transform: Optional image transform to apply.
            data_dir: Root directory containing the CIFAR-10 download.
            seed: Seed controlling the train/validation split.
        """

        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split '{split}'.")

        images, hard_labels = load_cifar10_train_images(data_dir)
        self.images = images
        self.hard_labels = hard_labels
        self.indices = build_pretrain_split_indices(seed=seed)[split]
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of examples in the active split."""

        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an augmented image tensor and hard class label.

        Args:
            index: Split-relative sample index.

        Returns:
            Tuple of `(image_tensor, hard_label_tensor)`.
        """

        global_index = int(self.indices[index])
        image = Image.fromarray(self.images[global_index])
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        hard_label_tensor = torch.tensor(self.hard_labels[global_index], dtype=torch.long)
        return image_tensor, hard_label_tensor


def build_cifar10h_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    data_dir: Path = RAW_DATA_DIR,
    seed: int = SEED,
) -> dict[str, DataLoader]:
    """Build data loaders for the CIFAR-10H train, validation, and test splits.

    Args:
        batch_size: Mini-batch size for all loaders.
        num_workers: Number of worker processes used by `DataLoader`.
        data_dir: Root directory containing raw data.
        seed: Seed controlling split construction and training shuffling.

    Returns:
        Dictionary mapping split name to `DataLoader`.
    """

    train_transform, eval_transform = build_transforms()
    generator = torch.Generator()
    generator.manual_seed(seed)

    datasets_by_split = {
        "train": CIFAR10HSoftLabelDataset(
            split="train",
            transform=train_transform,
            return_metadata=False,
            data_dir=data_dir,
            seed=seed,
        ),
        "val": CIFAR10HSoftLabelDataset(
            split="val",
            transform=eval_transform,
            return_metadata=False,
            data_dir=data_dir,
            seed=seed,
        ),
        "test": CIFAR10HSoftLabelDataset(
            split="test",
            transform=eval_transform,
            return_metadata=False,
            data_dir=data_dir,
            seed=seed,
        ),
    }

    pin_memory = torch.cuda.is_available()
    return {
        split: DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets_by_split.items()
    }


def build_cifar10_pretrain_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    data_dir: Path = RAW_DATA_DIR,
    seed: int = SEED,
) -> dict[str, DataLoader]:
    """Build data loaders for CIFAR-10 hard-label backbone pretraining.

    Args:
        batch_size: Mini-batch size for all loaders.
        num_workers: Number of worker processes used by `DataLoader`.
        data_dir: Root directory containing raw data.
        seed: Seed controlling split construction and training shuffling.

    Returns:
        Dictionary with `train` and `val` loaders.
    """

    train_transform, eval_transform = build_transforms()
    generator = torch.Generator()
    generator.manual_seed(seed)

    datasets_by_split = {
        "train": CIFAR10HardLabelDataset(
            split="train",
            transform=train_transform,
            data_dir=data_dir,
            seed=seed,
        ),
        "val": CIFAR10HardLabelDataset(
            split="val",
            transform=eval_transform,
            data_dir=data_dir,
            seed=seed,
        ),
    }

    pin_memory = torch.cuda.is_available()
    return {
        split: DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets_by_split.items()
    }


def _style_plot() -> None:
    """Apply a consistent plotting style used across dataset figures."""

    sns.set_theme(style="whitegrid", context="talk")


def plot_entropy_histogram(entropies: np.ndarray, output_path: Path) -> None:
    """Save a histogram of Shannon entropy over all CIFAR-10H examples.

    Args:
        entropies: Entropy vector of shape `(10000,)`.
        output_path: Destination PNG path.
    """

    _style_plot()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(entropies, bins=40, color="#2a9d8f", edgecolor="black", alpha=0.85)
    ax.set_title("CIFAR-10H Shannon Entropy Distribution")
    ax.set_xlabel("Entropy (bits)")
    ax.set_ylabel("Number of images")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_class_average_entropy(
    entropies: np.ndarray,
    hard_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Save a per-class bar chart of average annotator entropy.

    Args:
        entropies: Entropy vector of shape `(10000,)`.
        hard_labels: CIFAR-10 hard labels aligned to the test images.
        output_path: Destination PNG path.
    """

    _style_plot()
    class_means = [float(entropies[hard_labels == class_index].mean()) for class_index in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(CIFAR10_CLASSES, class_means, color="#e76f51")
    ax.set_title("Average Human Disagreement Entropy by CIFAR-10 Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Average entropy (bits)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_majority_vote_distribution_matrix(
    soft_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Save a confusion-style matrix of average annotator distributions by majority class.

    Args:
        soft_labels: Soft-label matrix of shape `(10000, 10)`.
        output_path: Destination PNG path.
    """

    _style_plot()
    majority_labels = np.argmax(soft_labels, axis=1)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)

    for class_index in range(NUM_CLASSES):
        class_mask = majority_labels == class_index
        matrix[class_index] = soft_labels[class_mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="mako",
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        ax=ax,
    )
    ax.set_title("Average Annotator Distribution by Majority-Vote Class")
    ax.set_xlabel("Predicted class mass in human distribution")
    ax.set_ylabel("Majority-vote class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _draw_distribution_inset(ax: plt.Axes, probabilities: np.ndarray) -> None:
    """Draw a compact bar chart inset inside an image subplot.

    Args:
        ax: Parent image axis.
        probabilities: Probability vector of shape `(10,)`.
    """

    inset = inset_axes(ax, width="46%", height="42%", loc="lower right", borderpad=0.5)
    colors = ["#457b9d"] * NUM_CLASSES
    colors[int(np.argmax(probabilities))] = "#e63946"
    inset.bar(np.arange(NUM_CLASSES), probabilities, color=colors)
    inset.set_ylim(0.0, 1.0)
    inset.set_xticks([])
    inset.set_yticks([0.0, 1.0])
    inset.tick_params(axis="both", labelsize=6)
    inset.set_title("p(y|x)", fontsize=7)


def plot_entropy_extremes_grid(
    images: np.ndarray,
    soft_labels: np.ndarray,
    entropies: np.ndarray,
    output_path: Path,
) -> None:
    """Save a 4x4 grid of the lowest- and highest-entropy images.

    Args:
        images: Image array of shape `(10000, 32, 32, 3)`.
        soft_labels: Soft-label matrix of shape `(10000, 10)`.
        entropies: Entropy vector of shape `(10000,)`.
        output_path: Destination PNG path.
    """

    _style_plot()
    lowest = np.argsort(entropies)[:8]
    highest = np.argsort(entropies)[-8:][::-1]
    selected_indices = np.concatenate([lowest, highest])

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for panel_index, (ax, global_index) in enumerate(zip(axes.flat, selected_indices)):
        ax.imshow(images[global_index])
        ax.axis("off")
        label_prefix = "Low" if panel_index < 8 else "High"
        majority_class = CIFAR10_CLASSES[int(np.argmax(soft_labels[global_index]))]
        ax.set_title(
            f"{label_prefix} entropy\nH={entropies[global_index]:.2f} | {majority_class}",
            fontsize=10,
        )
        _draw_distribution_inset(ax, soft_labels[global_index])

    fig.suptitle("Lowest-Entropy (top half) and Highest-Entropy (bottom half) CIFAR-10H Images", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def generate_dataset_visualizations(
    output_dir: Path | None = None,
    data_dir: Path = RAW_DATA_DIR,
) -> dict[str, Path]:
    """Generate all required dataset visualizations.

    Args:
        output_dir: Directory where PNG files should be written.
        data_dir: Root directory containing raw data.

    Returns:
        Mapping from visualization name to its saved path.
    """

    ensure_project_dirs()
    output_dir = output_dir or (RESULTS_DIR / "dataset_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    images, hard_labels = load_cifar10_test_images(data_dir)
    counts, soft_labels = load_cifar10h_soft_labels(data_dir)
    run_alignment_checks(images=images, hard_labels=hard_labels, counts=counts, soft_labels=soft_labels)
    entropies = run_sanity_checks(soft_labels)
    print_split_entropy_stats(soft_labels=soft_labels, output_path=output_dir / "split_entropy_stats.csv")
    print_augmentation_policy()

    paths = {
        "entropy_histogram": output_dir / "entropy_histogram.png",
        "per_class_entropy": output_dir / "per_class_average_entropy.png",
        "majority_vote_matrix": output_dir / "majority_vote_distribution_matrix.png",
        "entropy_extremes_grid": output_dir / "entropy_extremes_grid.png",
    }

    plot_entropy_histogram(entropies, paths["entropy_histogram"])
    plot_per_class_average_entropy(entropies, hard_labels, paths["per_class_entropy"])
    plot_majority_vote_distribution_matrix(soft_labels, paths["majority_vote_matrix"])
    plot_entropy_extremes_grid(images, soft_labels, entropies, paths["entropy_extremes_grid"])
    return paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "dataset_analysis",
        help="Directory where dataset visualizations should be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run sanity checks and generate the required dataset visualizations."""

    seed_everything()
    ensure_project_dirs()
    args = parse_args()

    paths = generate_dataset_visualizations(output_dir=args.output_dir, data_dir=RAW_DATA_DIR)

    print("Created dataset visualizations:")
    for name, path in paths.items():
        print(f"  {name:24s} -> {path}")
    print("Deterministic split sizes:")
    for split_name, indices in build_split_indices().items():
        print(f"  {split_name:5s}: {len(indices)}")


if __name__ == "__main__":
    main()
