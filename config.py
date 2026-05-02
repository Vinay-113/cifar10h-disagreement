"""Central configuration for the CIFAR-10H disagreement project."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SEED = 42

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"
LOG_DIR = RESULTS_DIR / "logs"
ABLATION_DIR = RESULTS_DIR / "ablations"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
EXPLAINABILITY_DIR = RESULTS_DIR / "explainability"

CIFAR10H_COUNTS_URL = (
    "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-counts.npy"
)
CIFAR10H_COUNTS_FILENAME = "cifar10h-counts.npy"
CIFAR10_DOWNLOAD_SUBDIR = "cifar10"
CIFAR10H_DOWNLOAD_SUBDIR = "cifar10h"

NUM_CLASSES = 10
TRAIN_SPLIT = 6000
VAL_SPLIT = 2000
TEST_SPLIT = 2000
SPLIT_SIZES = {
    "train": TRAIN_SPLIT,
    "val": VAL_SPLIT,
    "test": TEST_SPLIT,
}

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
NUM_WORKERS = 4

PRETRAIN_BATCH_SIZE = 128
PRETRAIN_LEARNING_RATE = 1e-3
PRETRAIN_WEIGHT_DECAY = 1e-4
PRETRAIN_MAX_EPOCHS = 30
PRETRAIN_PATIENCE = 5
PRETRAIN_VAL_SIZE = 5000

MLP_HIDDEN_DIM = 256
MLP_DROPOUT = 0.3
TEMPERATURE_INIT = 1.5
COMPOSITE_LOSS_LAMBDA = 0.5
EPS = 1e-12
ENTROPY_TOLERANCE = 1e-6
FAILURE_ENTROPY_GAP_BITS = 1.0
TOP_K_VALUES = (100, 200, 500)
SUBSAMPLED_ANNOTATOR_LEVELS = (5, 10, 20, 50)
OOD_SEVERITIES = (1, 2, 3, 4, 5)
GRADCAM_LOW_COUNT = 5
GRADCAM_HIGH_COUNT = 5
FAILURE_ANALYSIS_LIMIT = 20
QUALITATIVE_BUCKET_SIZE = 3

CIFAR10_IMAGE_SIZE = 32
CIFAR_RANDOM_CROP_PADDING = 4
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def ensure_project_dirs() -> None:
    """Create project output directories if they do not already exist."""

    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        CHECKPOINT_DIR,
        RESULTS_DIR,
        VISUALIZATION_DIR,
        LOG_DIR,
        ABLATION_DIR,
        ROBUSTNESS_DIR,
        EXPLAINABILITY_DIR,
        RAW_DATA_DIR / CIFAR10_DOWNLOAD_SUBDIR,
        RAW_DATA_DIR / CIFAR10H_DOWNLOAD_SUBDIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def build_run_name(loss_name: str, backbone_init: str, head_name: str) -> str:
    """Create a canonical run name from the main experiment choices."""

    return f"{loss_name}_{backbone_init}_{head_name}"


def get_device() -> torch.device:
    """Return the preferred torch device for training or evaluation."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
