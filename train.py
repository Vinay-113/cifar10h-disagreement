"""Train a deep model to predict CIFAR-10H human annotator distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LEARNING_RATE,
    LOG_DIR,
    MAX_EPOCHS,
    PATIENCE,
    RAW_DATA_DIR,
    SEED,
    WEIGHT_DECAY,
    build_run_name,
    ensure_project_dirs,
    get_device,
    seed_everything,
)
from data.dataset import build_cifar10h_dataloaders
from evaluate import compute_metric_arrays
from losses.losses import build_loss
from models.backbone import build_disagreement_model


def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one training or evaluation epoch.

    Args:
        model: Disagreement prediction model.
        dataloader: Data loader yielding `(images, soft_labels)`.
        criterion: Loss function for the active run.
        device: Torch device used for computation.
        optimizer: Optimizer for training, or `None` for validation.

    Returns:
        Tuple `(mean_loss, predicted_probabilities, true_probabilities)`.
    """

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    progress = tqdm(dataloader, desc="train" if is_training else "eval", leave=False)
    for images, soft_labels in progress:
        images = images.to(device)
        soft_labels = soft_labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, soft_labels)
            if is_training:
                loss.backward()
                optimizer.step()

        probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        targets = soft_labels.detach().cpu().numpy()
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_predictions.append(probabilities)
        all_targets.append(targets)

    mean_loss = total_loss / max(total_examples, 1)
    return mean_loss, np.concatenate(all_predictions, axis=0), np.concatenate(all_targets, axis=0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=["kl", "js", "cosine", "composite"], required=True)
    parser.add_argument(
        "--backbone_init",
        choices=["random", "cifar10_pretrained", "imagenet_pretrained"],
        required=True,
    )
    parser.add_argument("--head", choices=["linear", "mlp", "temperature"], required=True)
    return parser.parse_args()


def main() -> None:
    """Train one CIFAR-10H model configuration with early stopping."""

    args = parse_args()
    seed_everything()
    ensure_project_dirs()

    device = get_device()
    dataloaders = build_cifar10h_dataloaders(batch_size=BATCH_SIZE, data_dir=RAW_DATA_DIR, seed=SEED)
    model = build_disagreement_model(head_name=args.head, initialization=args.backbone_init).to(device)
    criterion = build_loss(args.loss)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    run_name = build_run_name(args.loss, args.backbone_init, args.head)
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    log_path = LOG_DIR / f"{run_name}.csv"

    history: list[dict[str, float | int]] = []
    best_val_kl = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    header = (
        f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | "
        f"{'Val KL':>8} | {'Val JSD':>8} | {'Val Cos':>8}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _ = run_epoch(model, dataloaders["train"], criterion, device, optimizer=optimizer)
        val_loss, val_pred_probs, val_true_probs = run_epoch(model, dataloaders["val"], criterion, device)
        scheduler.step()

        metric_arrays = compute_metric_arrays(val_true_probs, val_pred_probs)
        val_kl = float(metric_arrays["kl"].mean())
        val_jsd = float(metric_arrays["jsd"].mean())
        val_cosine = float(metric_arrays["cosine"].mean())

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_kl": val_kl,
            "val_jsd": val_jsd,
            "val_cosine_sim": val_cosine,
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(log_path, index=False)

        print(
            f"{epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f} | "
            f"{val_kl:8.4f} | {val_jsd:8.4f} | {val_cosine:8.4f}"
        )

        if val_kl < best_val_kl:
            best_val_kl = val_kl
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metadata": {
                        "loss": args.loss,
                        "backbone_init": args.backbone_init,
                        "head": args.head,
                        "seed": SEED,
                    },
                    "best_val_metrics": {
                        "val_loss": val_loss,
                        "val_kl": val_kl,
                        "val_jsd": val_jsd,
                        "val_cosine_sim": val_cosine,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"Best checkpoint saved to {checkpoint_path} (epoch {best_epoch}, val KL {best_val_kl:.4f}).")
    print(f"Epoch logs saved to {log_path}.")


if __name__ == "__main__":
    main()
