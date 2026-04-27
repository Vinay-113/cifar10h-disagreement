"""Identify and visualize the largest entropy-mismatch failures on the test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CIFAR10_CLASSES,
    EXPLAINABILITY_DIR,
    FAILURE_ANALYSIS_LIMIT,
    FAILURE_ENTROPY_GAP_BITS,
    RAW_DATA_DIR,
    SEED,
    build_run_name,
    ensure_project_dirs,
    get_device,
    seed_everything,
)
from data.dataset import CIFAR10HSoftLabelDataset
from evaluate import (
    collect_predictions,
    compute_metric_arrays,
    entropy_from_probabilities,
    load_model_from_checkpoint,
    resolve_checkpoint_path,
)


def plot_distribution(ax: plt.Axes, probabilities: np.ndarray, title: str, color: str) -> None:
    """Draw a probability bar chart.

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


def save_failure_panel(
    output_path: Path,
    raw_image: np.ndarray,
    true_probs: np.ndarray,
    pred_probs: np.ndarray,
    true_entropy: float,
    pred_entropy: float,
    entropy_gap: float,
) -> None:
    """Save a detailed panel for one failure case.

    Args:
        output_path: Destination PNG path.
        raw_image: Raw RGB image array.
        true_probs: Ground-truth soft-label vector.
        pred_probs: Predicted probability vector.
        true_entropy: True entropy in bits.
        pred_entropy: Predicted entropy in bits.
        entropy_gap: Absolute entropy difference in bits.
    """

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].imshow(raw_image)
    axes[0].axis("off")
    axes[0].set_title(f"Image\n|ΔH|={entropy_gap:.2f}")
    plot_distribution(axes[1], true_probs, f"True\nH={true_entropy:.2f}", "#2a9d8f")
    plot_distribution(axes[2], pred_probs, f"Pred\nH={pred_entropy:.2f}", "#e76f51")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a dataframe as a simple markdown table without extra dependencies.

    Args:
        df: Dataframe to render.

    Returns:
        Markdown table string.
    """

    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header_line, separator_line, *rows])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for failure analysis."""

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
    """Find and visualize the top entropy-mismatch failures."""

    args = parse_args()
    seed_everything()
    ensure_project_dirs()

    run_name = build_run_name(args.loss, args.backbone_init, args.head)
    output_dir = EXPLAINABILITY_DIR / "failure_analysis" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    checkpoint_path = resolve_checkpoint_path(args.loss, args.backbone_init, args.head, checkpoint_path=None)
    model, _ = load_model_from_checkpoint(checkpoint_path, device, args.loss, args.backbone_init, args.head)

    dataset = CIFAR10HSoftLabelDataset(split="test", transform=None, data_dir=RAW_DATA_DIR, seed=SEED)
    from data.dataset import build_cifar10h_dataloaders

    dataloaders = build_cifar10h_dataloaders(data_dir=RAW_DATA_DIR, seed=SEED)
    pred_probs, true_probs = collect_predictions(model, dataloaders["test"], device)
    metric_arrays = compute_metric_arrays(true_probs, pred_probs)

    candidate_indices = np.where(metric_arrays["entropy_abs_error"] > FAILURE_ENTROPY_GAP_BITS)[0]
    ranked_indices = candidate_indices[np.argsort(-metric_arrays["entropy_abs_error"][candidate_indices])]
    top_indices = ranked_indices[:FAILURE_ANALYSIS_LIMIT]

    rows: list[dict[str, object]] = []
    grid_cols = 4
    grid_rows = int(np.ceil(max(len(top_indices), 1) / grid_cols))
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 4.2 * grid_rows))
    axes = np.atleast_2d(axes)

    for plot_index, local_index in enumerate(top_indices):
        metadata = dataset.get_metadata(int(local_index))
        true_entropy = float(metric_arrays["true_entropy"][local_index])
        pred_entropy = float(metric_arrays["pred_entropy"][local_index])
        entropy_gap = float(metric_arrays["entropy_abs_error"][local_index])
        true_top = CIFAR10_CLASSES[int(np.argmax(true_probs[local_index]))]
        pred_top = CIFAR10_CLASSES[int(np.argmax(pred_probs[local_index]))]

        save_failure_panel(
            output_dir / f"failure_rank_{plot_index + 1:02d}_idx_{metadata['global_index']}.png",
            metadata["raw_image"],
            true_probs[local_index],
            pred_probs[local_index],
            true_entropy,
            pred_entropy,
            entropy_gap,
        )

        ax = axes.flat[plot_index]
        ax.imshow(metadata["raw_image"])
        ax.axis("off")
        ax.set_title(
            f"Rank {plot_index + 1}\n|ΔH|={entropy_gap:.2f}\n{true_top}→{pred_top}",
            fontsize=10,
        )

        rows.append(
            {
                "rank": plot_index + 1,
                "global_index": metadata["global_index"],
                "true_top": true_top,
                "pred_top": pred_top,
                "true_entropy": round(true_entropy, 3),
                "pred_entropy": round(pred_entropy, 3),
                "abs_entropy_gap": round(entropy_gap, 3),
                "kl": round(float(metric_arrays["kl"][local_index]), 3),
                "jsd": round(float(metric_arrays["jsd"][local_index]), 3),
            }
        )

    for ax in axes.flat[len(top_indices):]:
        ax.axis("off")

    summary_grid_path = output_dir / "failure_summary_grid.png"
    fig.tight_layout()
    fig.savefig(summary_grid_path, dpi=220)
    plt.close(fig)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "failure_statistics.csv", index=False)

    print(f"Failure threshold: |H(true) - H(pred)| > {FAILURE_ENTROPY_GAP_BITS:.1f} bit")
    print(f"Number of qualifying failures: {len(candidate_indices)}")
    print(dataframe_to_markdown(results_df))
    print(f"Saved failure analysis outputs to {output_dir}.")
    print(f"Saved summary grid to {summary_grid_path}.")


if __name__ == "__main__":
    main()
