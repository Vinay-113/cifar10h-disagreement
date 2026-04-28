"""Gradio demo app for CIFAR-10H disagreement prediction."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from config import (
    CHECKPOINT_DIR,
    CIFAR10_CLASSES,
    CIFAR10_IMAGE_SIZE,
    CIFAR10_MEAN,
    CIFAR10_STD,
    EPS,
    build_run_name,
    get_device,
)
from evaluate import entropy_from_probabilities, load_model_from_checkpoint


DEFAULT_RUN_NAME = build_run_name("kl", "cifar10_pretrained", "mlp")
DEFAULT_CHECKPOINT_PATH = CHECKPOINT_DIR / f"{DEFAULT_RUN_NAME}_best.pt"


def resolve_checkpoint_path() -> Path:
    """Resolve the checkpoint path for the demo model.

    The app first checks `MODEL_CHECKPOINT_PATH`. If that file is missing and
    `MODEL_CHECKPOINT_URL` is defined, the checkpoint is downloaded to
    `checkpoints/demo_model.pt`. Otherwise the app falls back to the default
    baseline checkpoint path expected from the training scripts.

    Returns:
        Path to the checkpoint that should be loaded by the demo.
    """

    configured_path = os.getenv("MODEL_CHECKPOINT_PATH")
    if configured_path:
        candidate = Path(configured_path)
        if candidate.exists():
            return candidate

    checkpoint_url = os.getenv("MODEL_CHECKPOINT_URL")
    if checkpoint_url:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        downloaded_path = CHECKPOINT_DIR / "demo_model.pt"
        if not downloaded_path.exists():
            urllib.request.urlretrieve(checkpoint_url, downloaded_path)
        return downloaded_path

    return DEFAULT_CHECKPOINT_PATH


def build_inference_transform() -> transforms.Compose:
    """Create the image transform used by the public demo.

    Returns:
        Torchvision transform that resizes to CIFAR resolution and normalizes
        using the dataset statistics from the training pipeline.
    """

    return transforms.Compose(
        [
            transforms.Resize((CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def load_demo_model() -> tuple[torch.nn.Module | None, str]:
    """Load the trained model used by the Space.

    Returns:
        Tuple of `(model_or_none, status_message)`.
    """

    try:
        checkpoint_path = resolve_checkpoint_path()
        if not checkpoint_path.exists():
            return None, (
                "No checkpoint found. Set `MODEL_CHECKPOINT_PATH` to a local `.pt` file "
                "or `MODEL_CHECKPOINT_URL` to a downloadable checkpoint."
            )

        device = get_device()
        model, checkpoint = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            loss_name="kl",
            backbone_init="cifar10_pretrained",
            head_name="mlp",
        )
        metadata = checkpoint.get("metadata", {})
        status = (
            f"Loaded checkpoint: `{checkpoint_path}`\n\n"
            f"- loss: `{metadata.get('loss', 'unknown')}`\n"
            f"- backbone_init: `{metadata.get('backbone_init', 'unknown')}`\n"
            f"- head: `{metadata.get('head', 'unknown')}`\n"
            f"- device: `{device}`"
        )
        return model, status
    except Exception as exc:
        return None, f"Checkpoint load failed: `{exc}`"


MODEL, MODEL_STATUS = load_demo_model()
DEVICE = get_device()
INFERENCE_TRANSFORM = build_inference_transform()


def predict_distribution(image: Image.Image | None) -> tuple[pd.DataFrame, plt.Figure, str]:
    """Predict the CIFAR-10H label distribution for an uploaded image.

    Args:
        image: Uploaded RGB image.

    Returns:
        Tuple containing a dataframe of class probabilities, a bar plot figure,
        and a markdown summary string.
    """

    if image is None:
        empty_df = pd.DataFrame({"class": CIFAR10_CLASSES, "probability": [0.0] * len(CIFAR10_CLASSES)})
        return empty_df, _plot_distribution(np.zeros(len(CIFAR10_CLASSES), dtype=np.float32)), "Upload an image to begin."

    if MODEL is None:
        empty_df = pd.DataFrame({"class": CIFAR10_CLASSES, "probability": [0.0] * len(CIFAR10_CLASSES)})
        return empty_df, _plot_distribution(np.zeros(len(CIFAR10_CLASSES), dtype=np.float32)), MODEL_STATUS

    image = image.convert("RGB")
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    entropy = float(entropy_from_probabilities(probabilities[None, :])[0])
    top_index = int(np.argmax(probabilities))
    top_class = CIFAR10_CLASSES[top_index]

    summary_lines = [
        f"**Top class:** `{top_class}`",
        f"**Predicted probability:** `{probabilities[top_index]:.4f}`",
        f"**Predicted entropy:** `{entropy:.4f}` bits",
    ]
    if entropy < 1.0:
        summary_lines.append("**Interpretation:** low disagreement prediction.")
    elif entropy < 2.0:
        summary_lines.append("**Interpretation:** moderate disagreement prediction.")
    else:
        summary_lines.append("**Interpretation:** high disagreement prediction.")

    result_df = pd.DataFrame(
        {
            "class": CIFAR10_CLASSES,
            "probability": probabilities,
        }
    ).sort_values("probability", ascending=False, ignore_index=True)
    return result_df, _plot_distribution(probabilities), "\n\n".join(summary_lines)


def _plot_distribution(probabilities: np.ndarray) -> plt.Figure:
    """Create a bar chart of predicted class probabilities.

    Args:
        probabilities: Vector of class probabilities of shape `(10,)`.

    Returns:
        Matplotlib figure visualizing the distribution.
    """

    clipped = np.clip(probabilities, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#457b9d"] * len(CIFAR10_CLASSES)
    if clipped.sum() > EPS:
        colors[int(np.argmax(clipped))] = "#e63946"
    ax.bar(CIFAR10_CLASSES, clipped, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title("Predicted Annotator Distribution")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def build_app() -> gr.Blocks:
    """Create the Gradio interface used for deployment.

    Returns:
        Configured Gradio `Blocks` application.
    """

    with gr.Blocks(title="CIFAR-10H Disagreement Predictor") as demo:
        gr.Markdown(
            """
            # CIFAR-10H Disagreement Predictor

            Upload an image and the model will predict a **10-way human annotator distribution**
            rather than a single hard class. The output approximates `q(y|x)`, the predicted
            distribution of labels that a pool of human annotators might assign.
            """
        )
        gr.Markdown(MODEL_STATUS)

        with gr.Row():
            input_image = gr.Image(type="pil", label="Input image")
            output_plot = gr.Plot(label="Predicted distribution")

        with gr.Row():
            output_table = gr.Dataframe(
                headers=["class", "probability"],
                datatype=["str", "number"],
                label="Class probabilities",
                row_count=(10, "fixed"),
                col_count=(2, "fixed"),
            )
            output_summary = gr.Markdown()

        predict_button = gr.Button("Predict distribution", variant="primary")

        predict_button.click(
            fn=predict_distribution,
            inputs=input_image,
            outputs=[output_table, output_plot, output_summary],
        )

    return demo


APP = build_app()


if __name__ == "__main__":
    APP.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
