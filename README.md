---
title: CIFAR-10H Disagreement Predictor
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
python_version: 3.10
app_file: app.py
fullWidth: true
header: default
short_description: Predict full human annotator label distributions for CIFAR-10 style images.
suggested_hardware: cpu-basic
---

# CIFAR-10H Disagreement Prediction

This project trains deep neural networks to predict the full human annotator label distribution for a CIFAR-10 image using the CIFAR-10H dataset. Instead of outputting a single hard class, the model outputs a 10-dimensional distribution `q(y|x)` that approximates the empirical human distribution `p(y|x)`.

## Project Overview

The repository covers the full research workflow:

- download and align CIFAR-10 with CIFAR-10H soft labels
- run sanity checks and dataset visualizations
- pretrain a CIFAR-adapted ResNet-18 backbone on hard-label CIFAR-10
- train soft-label models with multiple heads and multiple losses
- evaluate distribution matching and entropy prediction quality
- run ablations over losses, backbones, and heads
- test robustness to annotator subsampling and image corruptions
- inspect model behavior with Grad-CAM and failure-case analysis

## Repository Structure

```text
cifar10h-disagreement/
├── README.md
├── requirements.txt
├── config.py
├── data/
│   ├── download.py
│   └── dataset.py
├── models/
│   ├── backbone.py
│   └── heads.py
├── losses/
│   └── losses.py
├── train.py
├── evaluate.py
├── ablations/
│   ├── run_backbone_init.py
│   ├── run_loss_comparison.py
│   └── run_head_comparison.py
├── robustness/
│   ├── annotator_subsampling.py
│   └── ood_corruptions.py
└── explainability/
    ├── gradcam.py
    └── failure_analysis.py
```

## Environment Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Hugging Face Demo Deployment

This repository now includes a Gradio app at [app.py](/Users/vinaypatil/Documents/Playground/cifar10h-disagreement/app.py) so you can deploy it as a Hugging Face Space.

### What the demo needs

The Space code is ready, but it still needs a trained checkpoint. The app looks for a model in this order:

1. `MODEL_CHECKPOINT_PATH`
2. `MODEL_CHECKPOINT_URL`
3. `checkpoints/kl_cifar10_pretrained_mlp_best.pt`

The simplest path is:

1. Train the baseline model locally or on a GPU machine.
2. Upload the resulting checkpoint somewhere downloadable.
3. Set `MODEL_CHECKPOINT_URL` in your Space variables.

### Recommended deployment flow

1. Push this repository to GitHub.
2. Create a new Hugging Face Space and choose the `Gradio` SDK.
3. Upload this repository to the Space, or connect it via GitHub sync.
4. Add a Space variable named `MODEL_CHECKPOINT_URL` that points to your trained checkpoint.
5. Let the Space rebuild automatically.

### Local demo run

If you already have a trained checkpoint locally:

```bash
MODEL_CHECKPOINT_PATH=checkpoints/kl_cifar10_pretrained_mlp_best.pt python app.py
```

If your checkpoint is hosted remotely:

```bash
MODEL_CHECKPOINT_URL="https://YOUR_PUBLIC_CHECKPOINT_URL/model.pt" python app.py
```

Then open `http://127.0.0.1:7860`.

### Optional GitHub to Space sync

After the repo is on GitHub, you can either:

- push directly to the Hugging Face Space repo, or
- set up GitHub Actions to sync to the Space on every push

See the official Hugging Face docs linked at the end of this README.

## Data

`data/download.py` downloads:

- CIFAR-10 via `torchvision.datasets.CIFAR10`
- CIFAR-10H annotator counts from the official public release

The project stores raw assets under `data/raw/` and writes outputs under:

- `checkpoints/`
- `results/logs/`
- `results/evaluations/`
- `results/ablations/`
- `results/robustness/`
- `results/explainability/`

## Training Design

- Backbone: ResNet-18 adapted for `32x32` images with a `3x3` stride-1 stem and no initial max-pooling.
- Backbone initialization modes:
  - `random`
  - `cifar10_pretrained`
  - `imagenet_pretrained`
- Heads:
  - `linear`
  - `mlp`
  - `temperature`
- Losses:
  - `kl`
  - `js`
  - `cosine`
  - `composite`

The `composite` loss adds an entropy-matching penalty to KL divergence so the model is trained to match both class mass and overall human uncertainty.

## How To Run

Run the following commands from the repository root.

### 1. Download data

```bash
python data/download.py
```

Expected outputs:

- CIFAR-10 files under `data/raw/cifar10/`
- `data/raw/cifar10h/cifar10h-counts.npy`

### 2. Build dataset visuals and sanity checks

```bash
python data/dataset.py
```

Expected outputs in `results/dataset_analysis/`:

- `entropy_histogram.png`
- `per_class_average_entropy.png`
- `majority_vote_distribution_matrix.png`
- `entropy_extremes_grid.png`

### 3. Pretrain the backbone on CIFAR-10 hard labels

```bash
python models/backbone.py
```

Expected outputs:

- `checkpoints/cifar10_pretrained_backbone.pt`
- `results/logs/cifar10_pretraining.csv`

### 4. Train soft-label models

Example:

```bash
python train.py --loss kl --backbone_init cifar10_pretrained --head mlp
```

Expected outputs:

- `checkpoints/{loss}_{backbone_init}_{head}_best.pt`
- `results/logs/{loss}_{backbone_init}_{head}.csv`

### 5. Evaluate trained checkpoints

Example:

```bash
python evaluate.py --loss kl --backbone_init cifar10_pretrained --head mlp
```

Expected outputs in `results/evaluations/{run_name}/`:

- `metrics.csv`
- `predicted_probabilities.npy`
- `true_probabilities.npy`
- `entropy_scatter.png`
- `loss_metrics_grouped_bar.png`
- `qualitative_entropy_grid.png`

Also updates:

- `results/evaluation_summary.csv`

### 6. Run ablations

```bash
python ablations/run_backbone_init.py
python ablations/run_loss_comparison.py
python ablations/run_head_comparison.py
```

Expected outputs:

- summary CSV tables in `results/ablations/...`
- comparison plots in `results/ablations/...`

### 7. Run robustness analyses

```bash
python robustness/annotator_subsampling.py
python robustness/ood_corruptions.py
```

Expected outputs:

- `results/robustness/annotator_subsampling/...`
- `results/robustness/ood_corruptions/...`

### 8. Run explainability analyses

```bash
python explainability/gradcam.py
python explainability/failure_analysis.py
```

Expected outputs:

- per-image Grad-CAM panels and `gradcam_summary_grid.png`
- failure-case panels, `failure_summary_grid.png`, and `failure_statistics.csv`

## Required Execution Order

1. `python data/download.py`
2. `python data/dataset.py`
3. `python models/backbone.py`
4. `python train.py --loss kl --backbone_init cifar10_pretrained --head mlp`
5. `python train.py --loss js --backbone_init cifar10_pretrained --head mlp`
6. `python train.py --loss cosine --backbone_init cifar10_pretrained --head mlp`
7. `python train.py --loss composite --backbone_init cifar10_pretrained --head mlp`
8. `python evaluate.py --loss kl --backbone_init cifar10_pretrained --head mlp`
9. `python evaluate.py --loss js --backbone_init cifar10_pretrained --head mlp`
10. `python evaluate.py --loss cosine --backbone_init cifar10_pretrained --head mlp`
11. `python evaluate.py --loss composite --backbone_init cifar10_pretrained --head mlp`
12. `python ablations/run_backbone_init.py`
13. `python ablations/run_loss_comparison.py`
14. `python ablations/run_head_comparison.py`
15. `python robustness/annotator_subsampling.py`
16. `python robustness/ood_corruptions.py`
17. `python explainability/gradcam.py`
18. `python explainability/failure_analysis.py`

## Metrics Reported

The evaluation pipeline computes:

- KL divergence mean and standard deviation
- Jensen-Shannon divergence mean and standard deviation
- cosine similarity mean and standard deviation
- Pearson correlation between true and predicted entropy
- Spearman correlation between true and predicted entropy
- Precision@100, Precision@200, Precision@500 for top-entropy retrieval

## Notes

- All seeds are centralized in `config.py` and default to `42`.
- CIFAR-10H splits are deterministic: `6000 / 2000 / 2000`.
- All training scripts use early stopping on validation KL divergence.
- If predictive entropy under OOD corruption does not increase with severity, `robustness/ood_corruptions.py` reports that result honestly.

## Deployment Checklist

Use this exact sequence if you want both GitHub and a public demo:

1. `cd /Users/vinaypatil/Documents/Playground/cifar10h-disagreement`
2. `gh auth login -h github.com`
3. `gh repo create cifar10h-disagreement --public --source=. --remote=origin --push`
4. Train a model and keep the checkpoint `checkpoints/kl_cifar10_pretrained_mlp_best.pt`
5. Upload that checkpoint to a public URL or a Hugging Face model repo
6. Create a new Hugging Face Space with `Gradio`
7. Copy this repo into the Space or sync it from GitHub
8. Add `MODEL_CHECKPOINT_URL` in the Space settings
9. Wait for the automatic rebuild

## Official References

- [GitHub CLI auth login](https://cli.github.com/manual/gh_auth_login)
- [GitHub CLI repo create](https://cli.github.com/manual/gh_repo_create)
- [Add an existing project to GitHub](https://docs.github.com/en/github/importing-your-projects-to-github/adding-an-existing-project-to-github-using-the-command-line)
- [Hugging Face Spaces overview](https://huggingface.co/docs/hub/main/en/spaces-overview)
- [Hugging Face Spaces config reference](https://huggingface.co/docs/hub/main/spaces-config-reference)
- [Hugging Face Gradio Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio)
- [Hugging Face GitHub sync for Spaces](https://huggingface.co/docs/hub/main/spaces-github-actions)
