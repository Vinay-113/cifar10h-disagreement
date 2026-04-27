"""Robustness A: evaluate model fit under subsampled annotator counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_DIR,
    ROBUSTNESS_DIR,
    SEED,
    SUBSAMPLED_ANNOTATOR_LEVELS,
    build_run_name,
    ensure_project_dirs,
    get_device,
    seed_everything,
)
from data.dataset import CIFAR10HSoftLabelDataset, build_cifar10h_dataloaders
from evaluate import (
    collect_predictions,
    kl_divergence,
    load_model_from_checkpoint,
    resolve_checkpoint_path,
)


def subsample_soft_labels(counts: np.ndarray, num_annotators: int, seed: int = SEED) -> np.ndarray:
    """Resample annotator counts into a smaller synthetic annotation pool.

    Args:
        counts: Full count matrix of shape `(N, C)`.
        num_annotators: Number of annotators to sample per image.
        seed: Random seed for deterministic resampling.

    Returns:
        Soft-label matrix of shape `(N, C)` induced by the subsampled counts.
    """

    rng = np.random.default_rng(seed + num_annotators)
    sampled = np.zeros_like(counts, dtype=np.float32)

    for row_index, count_row in enumerate(counts):
        probabilities = count_row / count_row.sum()
        sampled_counts = rng.multinomial(num_annotators, probabilities)
        sampled[row_index] = sampled_counts / max(sampled_counts.sum(), 1)

    return sampled


def plot_subsampling_curve(results_df: pd.DataFrame, output_path: Path, baseline_kl: float) -> None:
    """Save the mean KL divergence as a function of annotator count.

    Args:
        results_df: Dataframe containing `num_annotators` and `mean_kl`.
        output_path: Destination PNG path.
        baseline_kl: Mean KL against the full original soft labels.
    """

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(data=results_df, x="num_annotators", y="mean_kl", marker="o", linewidth=2.5, ax=ax)
    ax.axhline(baseline_kl, linestyle="--", color="black", label=f"Full-label baseline ({baseline_kl:.3f})")
    ax.set_title("Model KL vs Number of Resampled Annotators")
    ax.set_xlabel("Number of annotators used")
    ax.set_ylabel("Mean KL divergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for annotator subsampling robustness."""

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
    """Measure how model KL changes when annotator pools are downsampled."""

    args = parse_args()
    seed_everything()
    ensure_project_dirs()

    run_name = build_run_name(args.loss, args.backbone_init, args.head)
    output_dir = ROBUSTNESS_DIR / "annotator_subsampling" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    checkpoint_path = resolve_checkpoint_path(args.loss, args.backbone_init, args.head, checkpoint_path=None)
    model, _ = load_model_from_checkpoint(checkpoint_path, device, args.loss, args.backbone_init, args.head)

    dataloaders = build_cifar10h_dataloaders(data_dir=RAW_DATA_DIR, seed=SEED)
    pred_probs, true_probs = collect_predictions(model, dataloaders["test"], device)
    baseline_kl = float(kl_divergence(true_probs, pred_probs).mean())

    dataset = CIFAR10HSoftLabelDataset(split="test", transform=None, data_dir=RAW_DATA_DIR, seed=SEED)
    test_counts = dataset.counts[dataset.indices]

    rows: list[dict[str, float | int]] = []
    for num_annotators in SUBSAMPLED_ANNOTATOR_LEVELS:
        sampled_soft_labels = subsample_soft_labels(test_counts, num_annotators=num_annotators, seed=SEED)
        kl_values = kl_divergence(sampled_soft_labels, pred_probs)
        rows.append(
            {
                "num_annotators": num_annotators,
                "mean_kl": float(kl_values.mean()),
                "std_kl": float(kl_values.std(ddof=0)),
            }
        )

    results_df = pd.DataFrame(rows)
    csv_path = output_dir / "annotator_subsampling.csv"
    plot_path = output_dir / "annotator_subsampling_curve.png"
    results_df.to_csv(csv_path, index=False)
    plot_subsampling_curve(results_df, plot_path, baseline_kl=baseline_kl)

    print(results_df.to_string(index=False))
    print(f"Full original-label baseline KL: {baseline_kl:.4f}")
    print(f"Saved robustness table to {csv_path}.")
    print(f"Saved plot to {plot_path}.")


if __name__ == "__main__":
    main()
