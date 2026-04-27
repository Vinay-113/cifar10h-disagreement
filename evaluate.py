"""Evaluate a trained CIFAR-10H model and generate report-ready artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr, spearmanr

from config import (
    CHECKPOINT_DIR,
    CIFAR10_CLASSES,
    EPS,
    NUM_CLASSES,
    QUALITATIVE_BUCKET_SIZE,
    RAW_DATA_DIR,
    RESULTS_DIR,
    SEED,
    TOP_K_VALUES,
    build_run_name,
    ensure_project_dirs,
    get_device,
    seed_everything,
)
from data.dataset import CIFAR10HSoftLabelDataset, build_cifar10h_dataloaders
from models.backbone import build_disagreement_model


EVALUATION_ROOT = RESULTS_DIR / "evaluations"
SUMMARY_CSV_PATH = RESULTS_DIR / "evaluation_summary.csv"


def entropy_from_probabilities(probabilities: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Compute Shannon entropy in bits for each probability distribution.

    Args:
        probabilities: Array of shape `(N, C)`.
        eps: Numerical stability constant.

    Returns:
        Entropy vector of shape `(N,)`.
    """

    clipped = np.clip(probabilities, eps, 1.0)
    return -(clipped * (np.log(clipped) / math.log(2.0))).sum(axis=1)


def kl_divergence(true_probs: np.ndarray, pred_probs: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Compute per-example KL divergence `KL(true || pred)`.

    Args:
        true_probs: Ground-truth probability matrix of shape `(N, C)`.
        pred_probs: Predicted probability matrix of shape `(N, C)`.
        eps: Numerical stability constant.

    Returns:
        KL divergence vector of shape `(N,)`.
    """

    p = np.clip(true_probs, eps, 1.0)
    q = np.clip(pred_probs, eps, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(axis=1)


def js_divergence(true_probs: np.ndarray, pred_probs: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Compute per-example Jensen-Shannon divergence.

    Args:
        true_probs: Ground-truth probability matrix of shape `(N, C)`.
        pred_probs: Predicted probability matrix of shape `(N, C)`.
        eps: Numerical stability constant.

    Returns:
        JS divergence vector of shape `(N,)`.
    """

    p = np.clip(true_probs, eps, 1.0)
    q = np.clip(pred_probs, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (np.log(p) - np.log(m))).sum(axis=1) + (q * (np.log(q) - np.log(m))).sum(axis=1))


def cosine_similarity(true_probs: np.ndarray, pred_probs: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Compute per-example cosine similarity between target and predicted distributions.

    Args:
        true_probs: Ground-truth probability matrix of shape `(N, C)`.
        pred_probs: Predicted probability matrix of shape `(N, C)`.
        eps: Numerical stability constant.

    Returns:
        Cosine similarity vector of shape `(N,)`.
    """

    numerator = (true_probs * pred_probs).sum(axis=1)
    denominator = np.linalg.norm(true_probs, axis=1) * np.linalg.norm(pred_probs, axis=1)
    return numerator / np.clip(denominator, eps, None)


def precision_at_k(true_scores: np.ndarray, pred_scores: np.ndarray, k: int) -> float:
    """Compute Precision@K between two descending rankings.

    Args:
        true_scores: Scores that define the ground-truth ranking.
        pred_scores: Scores that define the predicted ranking.
        k: Number of top elements to retain.

    Returns:
        Fraction of true top-K items recovered by the predicted top-K set.
    """

    k = min(k, len(true_scores))
    true_top = set(np.argsort(-true_scores)[:k].tolist())
    pred_top = set(np.argsort(-pred_scores)[:k].tolist())
    return len(true_top.intersection(pred_top)) / float(k)


def compute_metric_arrays(true_probs: np.ndarray, pred_probs: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-example evaluation statistics.

    Args:
        true_probs: Ground-truth probability matrix of shape `(N, C)`.
        pred_probs: Predicted probability matrix of shape `(N, C)`.

    Returns:
        Dictionary containing per-example metric vectors.
    """

    true_entropy = entropy_from_probabilities(true_probs)
    pred_entropy = entropy_from_probabilities(pred_probs)
    return {
        "kl": kl_divergence(true_probs, pred_probs),
        "jsd": js_divergence(true_probs, pred_probs),
        "cosine": cosine_similarity(true_probs, pred_probs),
        "true_entropy": true_entropy,
        "pred_entropy": pred_entropy,
        "entropy_abs_error": np.abs(true_entropy - pred_entropy),
    }


def compute_summary_metrics(
    true_probs: np.ndarray,
    pred_probs: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Aggregate all required test metrics for one model.

    Args:
        true_probs: Ground-truth probability matrix of shape `(N, C)`.
        pred_probs: Predicted probability matrix of shape `(N, C)`.

    Returns:
        Tuple of `(summary_dict, per_example_metric_dict)`.
    """

    arrays = compute_metric_arrays(true_probs, pred_probs)

    if np.std(arrays["true_entropy"]) == 0 or np.std(arrays["pred_entropy"]) == 0:
        pearson_value = 0.0
        spearman_value = 0.0
    else:
        pearson_value = float(pearsonr(arrays["true_entropy"], arrays["pred_entropy"]).statistic)
        spearman_value = float(spearmanr(arrays["true_entropy"], arrays["pred_entropy"]).statistic)

    summary = {
        "kl_mean": float(arrays["kl"].mean()),
        "kl_std": float(arrays["kl"].std(ddof=0)),
        "jsd_mean": float(arrays["jsd"].mean()),
        "jsd_std": float(arrays["jsd"].std(ddof=0)),
        "cosine_mean": float(arrays["cosine"].mean()),
        "cosine_std": float(arrays["cosine"].std(ddof=0)),
        "entropy_pearson": pearson_value,
        "entropy_spearman": spearman_value,
        "precision_at_100": precision_at_k(arrays["true_entropy"], arrays["pred_entropy"], TOP_K_VALUES[0]),
        "precision_at_200": precision_at_k(arrays["true_entropy"], arrays["pred_entropy"], TOP_K_VALUES[1]),
        "precision_at_500": precision_at_k(arrays["true_entropy"], arrays["pred_entropy"], TOP_K_VALUES[2]),
    }
    return summary, arrays


def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a data loader and collect predictions and targets.

    Args:
        model: Trained disagreement model.
        dataloader: Non-shuffled data loader yielding `(images, soft_labels)`.
        device: Device on which to execute inference.

    Returns:
        Tuple `(predicted_probabilities, true_probabilities)`.
    """

    model.eval()
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, soft_labels in dataloader:
            images = images.to(device)
            logits = model(images)
            predictions = torch.softmax(logits, dim=1).cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(soft_labels.numpy())

    return np.concatenate(all_predictions, axis=0), np.concatenate(all_targets, axis=0)


def resolve_checkpoint_path(
    loss_name: str,
    backbone_init: str,
    head_name: str,
    checkpoint_path: Path | None,
) -> Path:
    """Resolve the checkpoint path from explicit or implicit run information.

    Args:
        loss_name: Loss identifier.
        backbone_init: Backbone initialization identifier.
        head_name: Head identifier.
        checkpoint_path: Optional explicit checkpoint path.

    Returns:
        Resolved checkpoint path.
    """

    if checkpoint_path is not None:
        return checkpoint_path
    run_name = build_run_name(loss_name, backbone_init, head_name)
    return CHECKPOINT_DIR / f"{run_name}_best.pt"


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    loss_name: str,
    backbone_init: str,
    head_name: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a trained model checkpoint and rebuild the corresponding model.

    Args:
        checkpoint_path: Path to the model checkpoint.
        device: Device on which to place the model.
        loss_name: Loss identifier, used for metadata fallback.
        backbone_init: Backbone initialization identifier.
        head_name: Head identifier.

    Returns:
        Tuple `(model, checkpoint_payload)`.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    model = build_disagreement_model(
        head_name=metadata.get("head", head_name),
        initialization="random",
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint.setdefault("metadata", {})
    checkpoint["metadata"].setdefault("loss", loss_name)
    checkpoint["metadata"].setdefault("backbone_init", backbone_init)
    checkpoint["metadata"].setdefault("head", head_name)
    return model, checkpoint


def _style_plot() -> None:
    """Apply a consistent plotting style for evaluation figures."""

    sns.set_theme(style="whitegrid", context="talk")


def plot_entropy_scatter(
    true_entropy: np.ndarray,
    pred_entropy: np.ndarray,
    output_path: Path,
) -> None:
    """Save a scatter plot comparing true and predicted entropy.

    Args:
        true_entropy: Ground-truth entropy vector.
        pred_entropy: Predicted entropy vector.
        output_path: Destination PNG path.
    """

    _style_plot()
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(true_entropy, pred_entropy, c=true_entropy, cmap="viridis", alpha=0.75, s=24)
    min_val = min(true_entropy.min(), pred_entropy.min())
    max_val = max(true_entropy.max(), pred_entropy.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1.5)
    ax.set_title("Predicted vs True Entropy")
    ax.set_xlabel("True entropy (bits)")
    ax.set_ylabel("Predicted entropy (bits)")
    fig.colorbar(scatter, ax=ax, label="True entropy (bits)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_metrics_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Save a grouped bar chart comparing metrics across evaluated losses.

    Args:
        summary_df: Summary table with one row per evaluated run.
        output_path: Destination PNG path.
    """

    _style_plot()
    metric_columns = [
        "kl_mean",
        "jsd_mean",
        "cosine_mean",
        "entropy_pearson",
        "entropy_spearman",
        "precision_at_100",
        "precision_at_200",
        "precision_at_500",
    ]
    plot_df = summary_df.loc[:, ["loss", *metric_columns]].copy()
    plot_df = plot_df.melt(id_vars="loss", var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=plot_df, x="metric", y="value", hue="loss", ax=ax)
    ax.set_title("Evaluation Metrics Across Loss Functions")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _draw_distribution_pair(ax: plt.Axes, true_probs: np.ndarray, pred_probs: np.ndarray) -> None:
    """Draw compact true and predicted distribution bar charts inside one axis.

    Args:
        ax: Axis showing the underlying image.
        true_probs: Ground-truth distribution vector.
        pred_probs: Predicted distribution vector.
    """

    inset_true = inset_axes(ax, width="42%", height="33%", loc="lower left", borderpad=0.35)
    inset_pred = inset_axes(ax, width="42%", height="33%", loc="lower right", borderpad=0.35)

    x_positions = np.arange(NUM_CLASSES)
    inset_true.bar(x_positions, true_probs, color="#2a9d8f")
    inset_pred.bar(x_positions, pred_probs, color="#e76f51")

    for inset, title in ((inset_true, "True"), (inset_pred, "Pred")):
        inset.set_ylim(0.0, 1.0)
        inset.set_xticks([])
        inset.set_yticks([0.0, 1.0])
        inset.tick_params(axis="both", labelsize=6)
        inset.set_title(title, fontsize=7)


def plot_qualitative_entropy_grid(
    dataset: CIFAR10HSoftLabelDataset,
    pred_probs: np.ndarray,
    output_path: Path,
) -> None:
    """Save a 3x3 qualitative panel of low-, mid-, and high-disagreement images.

    Args:
        dataset: Test dataset with metadata-aligned raw images.
        pred_probs: Predicted probability matrix in dataset order.
        output_path: Destination PNG path.
    """

    _style_plot()
    true_entropy = dataset.entropies[dataset.indices]
    order = np.argsort(true_entropy)
    low_indices = order[:QUALITATIVE_BUCKET_SIZE]
    mid_start = len(order) // 2 - (QUALITATIVE_BUCKET_SIZE // 2)
    mid_indices = order[mid_start: mid_start + QUALITATIVE_BUCKET_SIZE]
    high_indices = order[-QUALITATIVE_BUCKET_SIZE:]

    grid_indices = np.concatenate([low_indices, mid_indices, high_indices])
    row_labels = ["Low disagreement", "Mid disagreement", "High disagreement"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for row in range(3):
        for col in range(3):
            local_index = int(grid_indices[row * 3 + col])
            metadata = dataset.get_metadata(local_index)
            ax = axes[row, col]
            ax.imshow(metadata["raw_image"])
            ax.axis("off")
            predicted_label = CIFAR10_CLASSES[int(np.argmax(pred_probs[local_index]))]
            ax.set_title(
                f"{row_labels[row]}\nH_true={metadata['entropy']:.2f} | Pred={predicted_label}",
                fontsize=10,
            )
            _draw_distribution_pair(ax, metadata["soft_label"], pred_probs[local_index])

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def update_summary_csv(summary_row: dict[str, Any], summary_csv_path: Path = SUMMARY_CSV_PATH) -> pd.DataFrame:
    """Append or replace one evaluation row inside the project summary table.

    Args:
        summary_row: Dictionary containing metadata and metrics for one run.
        summary_csv_path: Destination summary CSV path.

    Returns:
        Updated summary dataframe.
    """

    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_csv_path.exists():
        summary_df = pd.read_csv(summary_csv_path)
        summary_df = summary_df[summary_df["run_name"] != summary_row["run_name"]]
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])

    summary_df = summary_df.sort_values("run_name").reset_index(drop=True)
    summary_df.to_csv(summary_csv_path, index=False)
    return summary_df


def print_summary(summary_row: dict[str, Any]) -> None:
    """Print a compact metrics summary to stdout.

    Args:
        summary_row: Dictionary containing evaluation metrics for one run.
    """

    print(f"Run: {summary_row['run_name']}")
    print("Distribution matching:")
    print(f"  KL Divergence        : {summary_row['kl_mean']:.4f} ± {summary_row['kl_std']:.4f}")
    print(f"  Jensen-Shannon Div. : {summary_row['jsd_mean']:.4f} ± {summary_row['jsd_std']:.4f}")
    print(f"  Cosine Similarity   : {summary_row['cosine_mean']:.4f} ± {summary_row['cosine_std']:.4f}")
    print("Entropy prediction:")
    print(f"  Pearson             : {summary_row['entropy_pearson']:.4f}")
    print(f"  Spearman            : {summary_row['entropy_spearman']:.4f}")
    print("Precision@K:")
    print(f"  P@100               : {summary_row['precision_at_100']:.4f}")
    print(f"  P@200               : {summary_row['precision_at_200']:.4f}")
    print(f"  P@500               : {summary_row['precision_at_500']:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=["kl", "js", "cosine", "composite"], required=True)
    parser.add_argument(
        "--backbone_init",
        choices=["random", "cifar10_pretrained", "imagenet_pretrained"],
        required=True,
    )
    parser.add_argument("--head", choices=["linear", "mlp", "temperature"], required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path. Defaults to the canonical checkpoint name.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a trained checkpoint, evaluate it on the test split, and save artifacts."""

    args = parse_args()
    seed_everything()
    ensure_project_dirs()

    run_name = build_run_name(args.loss, args.backbone_init, args.head)
    run_dir = EVALUATION_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    checkpoint_path = resolve_checkpoint_path(args.loss, args.backbone_init, args.head, args.checkpoint)
    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        loss_name=args.loss,
        backbone_init=args.backbone_init,
        head_name=args.head,
    )

    dataloaders = build_cifar10h_dataloaders(data_dir=RAW_DATA_DIR, seed=SEED)
    pred_probs, true_probs = collect_predictions(model, dataloaders["test"], device)
    summary_metrics, metric_arrays = compute_summary_metrics(true_probs, pred_probs)

    summary_row: dict[str, Any] = {
        "run_name": run_name,
        "loss": args.loss,
        "backbone_init": args.backbone_init,
        "head": args.head,
        "checkpoint_path": str(checkpoint_path),
        **summary_metrics,
    }
    summary_df = update_summary_csv(summary_row)

    pd.DataFrame([summary_row]).to_csv(run_dir / "metrics.csv", index=False)
    np.save(run_dir / "predicted_probabilities.npy", pred_probs)
    np.save(run_dir / "true_probabilities.npy", true_probs)

    plot_entropy_scatter(metric_arrays["true_entropy"], metric_arrays["pred_entropy"], run_dir / "entropy_scatter.png")
    filtered_summary_df = summary_df[
        (summary_df["backbone_init"] == args.backbone_init) & (summary_df["head"] == args.head)
    ].copy()
    plot_metrics_comparison(filtered_summary_df, run_dir / "loss_metrics_grouped_bar.png")

    qualitative_dataset = CIFAR10HSoftLabelDataset(
        split="test",
        transform=None,
        return_metadata=False,
        data_dir=RAW_DATA_DIR,
        seed=SEED,
    )
    plot_qualitative_entropy_grid(
        qualitative_dataset,
        pred_probs=pred_probs,
        output_path=run_dir / "qualitative_entropy_grid.png",
    )

    print_summary(summary_row)
    print(f"Saved evaluation artifacts to {run_dir}.")
    print(f"Updated summary CSV at {SUMMARY_CSV_PATH}.")


if __name__ == "__main__":
    main()
