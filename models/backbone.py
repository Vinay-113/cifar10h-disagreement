"""ResNet-18 backbone adapted for CIFAR-10 and optional hard-label pretraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models as tv_models
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHECKPOINT_DIR,
    LEARNING_RATE,
    LOG_DIR,
    MAX_EPOCHS,
    NUM_CLASSES,
    PATIENCE,
    PRETRAIN_BATCH_SIZE,
    PRETRAIN_LEARNING_RATE,
    PRETRAIN_MAX_EPOCHS,
    PRETRAIN_PATIENCE,
    PRETRAIN_WEIGHT_DECAY,
    RAW_DATA_DIR,
    SEED,
    WEIGHT_DECAY,
    ensure_project_dirs,
    get_device,
    seed_everything,
)
from data.dataset import build_cifar10_pretrain_dataloaders
from models.heads import build_head


PRETRAIN_CHECKPOINT_PATH = CHECKPOINT_DIR / "cifar10_pretrained_backbone.pt"
PRETRAIN_LOG_PATH = LOG_DIR / "cifar10_pretraining.csv"


class CIFARResNet18Backbone(nn.Module):
    """ResNet-18 feature extractor adapted for 32x32 CIFAR images."""

    def __init__(self) -> None:
        """Initialize a CIFAR-style ResNet-18 backbone with a 3x3 stem."""

        super().__init__()
        base_model = tv_models.resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()
        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.out_features = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a 512-dimensional feature vector for each image.

        Args:
            x: Input tensor of shape `(B, 3, 32, 32)`.

        Returns:
            Feature tensor of shape `(B, 512)`.
        """

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class CIFAR10PretrainModel(nn.Module):
    """Hard-label classifier used to pretrain the ResNet-18 backbone."""

    def __init__(self) -> None:
        """Initialize the backbone and a temporary hard-label classifier."""

        super().__init__()
        self.backbone = CIFARResNet18Backbone()
        self.classifier = nn.Linear(self.backbone.out_features, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predict hard-label logits from an input batch of images.

        Args:
            images: Input tensor of shape `(B, 3, 32, 32)`.

        Returns:
            Logits tensor of shape `(B, 10)`.
        """

        return self.classifier(self.backbone(images))


class DisagreementPredictor(nn.Module):
    """Backbone plus prediction head for CIFAR-10H soft-label modeling."""

    def __init__(self, backbone: CIFARResNet18Backbone, head_name: str) -> None:
        """Initialize the end-to-end disagreement model.

        Args:
            backbone: Feature extractor producing 512-dimensional image features.
            head_name: Prediction head type to attach.
        """

        super().__init__()
        self.backbone = backbone
        self.head = build_head(head_name=head_name, in_features=backbone.out_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predict class logits for a batch of images.

        Args:
            images: Tensor of shape `(B, 3, 32, 32)`.

        Returns:
            Logits tensor of shape `(B, 10)`.
        """

        features = self.backbone(images)
        return self.head(features)


def _load_imagenet_adapted_weights(backbone: CIFARResNet18Backbone) -> None:
    """Load ImageNet-pretrained ResNet-18 weights into the CIFAR backbone.

    Args:
        backbone: Backbone instance to initialize.
    """

    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
    imagenet_model = tv_models.resnet18(weights=weights)
    source_state = imagenet_model.state_dict()
    target_state = backbone.state_dict()

    conv1_weight = source_state["conv1.weight"]
    resized_conv1 = F.interpolate(conv1_weight, size=(3, 3), mode="bicubic", align_corners=True)
    target_state["stem.0.weight"] = resized_conv1

    mapping = {
        "bn1.weight": "stem.1.weight",
        "bn1.bias": "stem.1.bias",
        "bn1.running_mean": "stem.1.running_mean",
        "bn1.running_var": "stem.1.running_var",
        "bn1.num_batches_tracked": "stem.1.num_batches_tracked",
    }
    for source_key, target_key in mapping.items():
        target_state[target_key] = source_state[source_key]

    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        for key, value in source_state.items():
            if key.startswith(layer_name):
                target_state[key] = value

    backbone.load_state_dict(target_state, strict=True)


def build_backbone(
    initialization: str = "random",
    checkpoint_path: Path = PRETRAIN_CHECKPOINT_PATH,
) -> CIFARResNet18Backbone:
    """Build the CIFAR-10H backbone under a requested initialization scheme.

    Args:
        initialization: One of `random`, `cifar10_pretrained`, or `imagenet_pretrained`.
        checkpoint_path: Path to the saved CIFAR-10 pretraining checkpoint.

    Returns:
        Initialized backbone model.
    """

    if initialization not in {"random", "cifar10_pretrained", "imagenet_pretrained"}:
        raise ValueError(f"Unsupported backbone initialization '{initialization}'.")

    backbone = CIFARResNet18Backbone()
    if initialization == "imagenet_pretrained":
        _load_imagenet_adapted_weights(backbone)
    elif initialization == "cifar10_pretrained":
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing pretrained checkpoint at {checkpoint_path}. Run python models/backbone.py first."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        backbone_state = checkpoint["backbone_state_dict"]
        backbone.load_state_dict(backbone_state, strict=True)
    return backbone


def build_disagreement_model(
    head_name: str,
    initialization: str = "random",
    checkpoint_path: Path = PRETRAIN_CHECKPOINT_PATH,
) -> DisagreementPredictor:
    """Construct the full disagreement model from backbone and head choices.

    Args:
        head_name: Head type to attach to the backbone.
        initialization: Backbone initialization mode.
        checkpoint_path: Optional path to the CIFAR-10 pretrained backbone checkpoint.

    Returns:
        Fully assembled soft-label prediction model.
    """

    backbone = build_backbone(initialization=initialization, checkpoint_path=checkpoint_path)
    return DisagreementPredictor(backbone=backbone, head_name=head_name)


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last convolutional layer used for Grad-CAM.

    Args:
        model: Full disagreement model.

    Returns:
        Last residual block module from `layer4`.
    """

    return model.backbone.layer4[-1].conv2


def _run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one hard-label training or validation epoch.

    Args:
        model: Pretraining model.
        dataloader: Data loader over hard-label CIFAR-10 data.
        criterion: Classification criterion.
        optimizer: Optimizer for training, or `None` for evaluation.
        device: Device on which to execute the epoch.

    Returns:
        Tuple of `(mean_loss, accuracy_percent)`.
    """

    is_training = optimizer is not None
    model.train(is_training)
    running_loss = 0.0
    running_correct = 0
    total_examples = 0

    progress = tqdm(dataloader, desc="train" if is_training else "val", leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        running_correct += int((predictions == labels).sum().item())
        total_examples += batch_size

    mean_loss = running_loss / max(total_examples, 1)
    accuracy = 100.0 * running_correct / max(total_examples, 1)
    return mean_loss, accuracy


def pretrain_on_cifar10(
    output_checkpoint: Path = PRETRAIN_CHECKPOINT_PATH,
    log_path: Path = PRETRAIN_LOG_PATH,
) -> Path:
    """Pretrain the adapted ResNet-18 backbone on hard-label CIFAR-10.

    Args:
        output_checkpoint: Destination path for the best pretraining checkpoint.
        log_path: Destination CSV path for epoch logs.

    Returns:
        Path to the saved checkpoint.
    """

    seed_everything()
    ensure_project_dirs()
    device = get_device()
    dataloaders = build_cifar10_pretrain_dataloaders(
        batch_size=PRETRAIN_BATCH_SIZE,
        data_dir=RAW_DATA_DIR,
        seed=SEED,
    )

    model = CIFAR10PretrainModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=PRETRAIN_LEARNING_RATE,
        weight_decay=PRETRAIN_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=PRETRAIN_MAX_EPOCHS)

    best_val_accuracy = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>10} | "
        f"{'Val Loss':>8} | {'Val Acc':>8}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, PRETRAIN_MAX_EPOCHS + 1):
        train_loss, train_acc = _run_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_acc = _run_epoch(model, dataloaders["val"], criterion, None, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)
        print(
            f"{epoch:5d} | {train_loss:10.4f} | {train_acc:10.2f} | "
            f"{val_loss:8.4f} | {val_acc:8.2f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "backbone_state_dict": model.backbone.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "seed": SEED,
                },
                output_checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PRETRAIN_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"Saved best pretrained backbone to {output_checkpoint} (epoch {best_epoch}, val acc {best_val_accuracy:.2f}%).")
    print(f"Saved pretraining log to {log_path}.")
    return output_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse optional command-line arguments for the pretraining script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=PRETRAIN_CHECKPOINT_PATH,
        help="Destination path for the best backbone checkpoint.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=PRETRAIN_LOG_PATH,
        help="Destination CSV path for pretraining logs.",
    )
    return parser.parse_args()


def main() -> None:
    """Pretrain the backbone on CIFAR-10 hard labels."""

    args = parse_args()
    pretrain_on_cifar10(output_checkpoint=args.output_checkpoint, log_path=args.log_path)


if __name__ == "__main__":
    main()
