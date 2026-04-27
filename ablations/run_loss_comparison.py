"""Ablation B: compare loss functions under a shared backbone and head."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ABLATION_DIR, RAW_DATA_DIR, SEED, build_run_name, ensure_project_dirs, get_device, seed_everything
from data.dataset import build_cifar10h_dataloaders
from evaluate import (
    SUMMARY_CSV_PATH,
    collect_predictions,
    compute_summary_metrics,
    load_model_from_checkpoint,
    plot_metrics_comparison,
    resolve_checkpoint_path,
    update_summary_csv,
)


def fetch_summary_row(loss_name: str, backbone_init: str, head_name: str) -> dict[str, Any]:
    """Load one run's evaluation summary, evaluating the checkpoint if needed.

    Args:
        loss_name: Loss identifier.
        backbone_init: Backbone initialization identifier.
        head_name: Head identifier.

    Returns:
        Dictionary containing metrics and metadata for the requested run.
    """

    run_name = build_run_name(loss_name, backbone_init, head_name)
    if SUMMARY_CSV_PATH.exists():
        summary_df = pd.read_csv(SUMMARY_CSV_PATH)
        match = summary_df[summary_df["run_name"] == run_name]
        if not match.empty:
            return match.iloc[0].to_dict()

    device = get_device()
    dataloaders = build_cifar10h_dataloaders(data_dir=RAW_DATA_DIR, seed=SEED)
    checkpoint_path = resolve_checkpoint_path(loss_name, backbone_init, head_name, checkpoint_path=None)
    model, _ = load_model_from_checkpoint(checkpoint_path, device, loss_name, backbone_init, head_name)
    pred_probs, true_probs = collect_predictions(model, dataloaders["test"], device)
    summary, _ = compute_summary_metrics(true_probs, pred_probs)
    row = {
        "run_name": run_name,
        "loss": loss_name,
        "backbone_init": backbone_init,
        "head": head_name,
        "checkpoint_path": str(checkpoint_path),
        **summary,
    }
    update_summary_csv(row)
    return row


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the loss comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone_init",
        default="cifar10_pretrained",
        choices=["random", "cifar10_pretrained", "imagenet_pretrained"],
    )
    parser.add_argument("--head", default="mlp", choices=["linear", "mlp", "temperature"])
    return parser.parse_args()


def main() -> None:
    """Run the loss-function ablation and save outputs."""

    args = parse_args()
    seed_everything()
    ensure_project_dirs()

    losses = ["kl", "js", "cosine", "composite"]
    rows = [fetch_summary_row(loss_name, args.backbone_init, args.head) for loss_name in losses]
    results_df = pd.DataFrame(rows).sort_values("loss").reset_index(drop=True)

    output_dir = ABLATION_DIR / "loss_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "loss_comparison_summary.csv"
    plot_path = output_dir / "loss_comparison_metrics.png"

    results_df.to_csv(csv_path, index=False)
    plot_metrics_comparison(results_df, plot_path)

    print(results_df.to_string(index=False))
    print(f"Saved ablation table to {csv_path}.")
    print(f"Saved plot to {plot_path}.")


if __name__ == "__main__":
    main()
