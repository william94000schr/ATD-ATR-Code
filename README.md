# ATD-ATR-SAR — Automatic Target Detection and Recognition in SAR Images

Automatic Target Detection and Recognition (ATD/ATR) in Synthetic Aperture Radar (SAR) images using **Faster R-CNN ResNet-50 FPN v2**.

---

## Overview

This project trains and evaluates an object detection model capable of simultaneously **localizing and classifying ground vehicles** in SAR images — a significantly harder task than single-target classification, as the model must handle full scenes with no prior knowledge of object location.

---

## Results

The model was trained and evaluated on the **NUDT4MSTAR SOC** dataset (~68 000 images, COCO format). Class groupings were applied iteratively to reduce inter-class confusion between similar vehicle sub-types.

| Configuration    | mAP@50 |
|-----------------|--------|
| 40 classes      | 0.8976 |
| 22 classes      | 0.9158 |
| **17 classes**  | **0.9352** |

The 17-class model **surpasses the Faster R-CNN baseline reported in the reference paper** (mAP@50 = 0.883), while using a stricter train/val split that reduces available training data.

The delivered model detects the following 17 vehicle categories:

> Mini Car · Car · SUV · Small Bus · Medium Bus · Large Bus · Pickup ·
> Heavy DT · Heavy ST · Light PV · Heavy FT · Medium TT · Mixter Truck ·
> Forklift · Ambulance · ECV · Construction

---

## Repository Structure

```
ATR_SAR/
├── data/                          # Dataset (images + COCO annotations)
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       └── test.json
├── experiments/
│   ├── config/
│   │   ├── config.yaml            # Training hyperparameters and data paths
│   │   └── classes.json           # Class index → name mapping
│   ├── models/                    # Saved model checkpoints (.pt)
│   └── outputs/                   # Training logs and evaluation results
├── src/
│   ├── data/
│   │   ├── dataset.py             # SAR_ATR_Dataset (VisionDataset + COCO)
│   │   └── transforms.py          # CocoToFasterRCNN transform
│   ├── models/
│   │   └── model.py               # get_model() — builds Faster R-CNN
│   └── visualization/
│       ├── gradcam.py             # FasterRCNNGradCAM
│       └── visualization.py       # Drawing and saving prediction figures
├── scripts/
│   ├── train.py                   # Training only
│   ├── train_val.py               # Training + validation split
│   ├── validation.py              # Evaluation on test set (mAP via torchmetrics)
│   └── predict.py                 # Inference on a single image
├── notebooks/                     # Exploratory notebooks
├── requirements.txt
└── pyproject.toml                 # Ruff linting configuration
```

---

## Installation

**Python 3.11+ is required.**

```bash
# Clone the repository
git clone <repo-url>
cd ATR_SAR

# Install dependencies (with uv)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

---

## Dataset Setup

The dataset must follow the COCO format. Place images and annotation files under `data/` according to the paths defined in `experiments/config/config.yaml`:

```
data/
├── images/
│   ├── train/   ← training images
│   └── test/    ← test images
└── annotations/
    ├── train.json
    └── test.json
```

Data paths, batch size, and learning rate schedule can all be adjusted in `experiments/config/config.yaml`.

---

## Pre-trained Weights & Provided Data

Processed datasets and trained model weights are available for download via Google Drive:

📁 [Download datasets and weights](https://drive.google.com/drive/u/0/folders/1yu1Cb0r8uCBfdTcCfNXY8aBreFWryY-Q)

The Drive folder contains one sub-folder per configuration, each including the dataset (`.tar` archive), the `config.json` file mapping class IDs to names, and — where available — the corresponding trained weights (`.pt`). 

Pre-trained weights are provided for the **SOC 40-class** and **SOC 17-class** configurations.

### Setup instructions

**1. Extract the dataset**
```bash
tar -xf dataset.tar
```

Move the extracted contents so that the repository layout matches:
```
data/
├── images/
│   ├── train/
│   └── test/
└── annotations/
    ├── train.json
    └── test.json
```

**2. Place the model weights and config**

Copy the `config.json` file to:
```
experiments/config/classes.json
```

Copy the `.pt` file to:
```
experiments/models/faster_rcnn.pt
```

Once both are in place, you can run evaluation or inference directly — no training required.

---

## Training

**Train only** (saves checkpoints every 2 epochs and a final model):

```bash
python scripts/train.py \
    --num_classes 17 \
    --num_epochs 15 \
    --proportion 1.0
```

**Train + validation split** (80/20 split, tracks validation loss):

```bash
python scripts/train_val.py \
    --num_classes 17 \
    --num_epochs 15 \
    --proportion 1.0
```

| Argument | Default | Description |
|---|---|---|
| `--num_classes` | `10` | Number of target classes (excluding background) |
| `--num_epochs` | `5` | Number of training epochs |
| `--proportion` | `1.0` | Fraction of the dataset to use (e.g. `0.1` for a quick test) |

The trained model is saved to `experiments/models/faster_rcnn.pt`.
Training metrics are saved to `experiments/outputs/train_results.json`.

### Default Hyperparameters

| Parameter | Value |
|---|---|
| Batch size | 16 |
| Optimizer | SGD (momentum=0.9, weight\_decay=1e-4) |
| LR schedule | 0.01 (epochs 0–9) → 0.001 (10–12) → 0.0001 (13–14) |
| Backbone | ResNet-50 FPN v2, all layers trainable |
| Input size | 128×128 |
| Mixed precision | Enabled automatically on compatible GPUs (CUDA capability ≥ 7) |

---

## Evaluation

Evaluate the trained model on the test set and compute mAP (via `torchmetrics`):

```bash
python scripts/validation.py \
    --num_classes 17 \
    --proportion 1.0
```

Results are saved to `experiments/outputs/test_results.json`, including global mAP metrics and per-image detailed predictions.

---

## Inference

Run prediction on a single image:

```bash
python scripts/predict.py \
    --image_path path/to/image.png \
    --threshold 0.5
```

With GradCAM explainability overlay:

```bash
python scripts/predict.py \
    --image_path path/to/image.png \
    --threshold 0.5 \
    --explainability
```

| Argument | Default | Description |
|---|---|---|
| `--image_path` | *(required)* | Path to the input image |
| `--threshold` | `0.5` | Minimum confidence score to keep a detection |
| `--explainability` | `False` | Enable GradCAM heatmap visualization |

Outputs are saved to `experiments/outputs/predictions/`.
Predictions are drawn in **red**, ground truth (when available) in **green**.

> **Note on the confidence threshold:** a higher threshold reduces false positives but may miss targets; a lower threshold increases recall at the cost of precision.

---

## Explainability (GradCAM)

The `--explainability` flag activates a GradCAM module that highlights the image regions the model attends to when making each detection. The target layer is `backbone.body.layer4.2.conv3` (last convolutional layer of the ResNet-50 backbone, before the FPN).

GradCAM on detection models is an active research area and results are not always conclusive — this feature is provided as an exploratory tool rather than a fully validated explainability method.

---

## Model Architecture

- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN v2) — captures multi-scale features for robust detection of objects at varying sizes.
- **RPN**: Region Proposal Network — proposes candidate regions using sliding-window anchors.
- **RoI Pooling**: Extracts fixed-size feature crops for each proposal.
- **Detection heads**: Classification head (vehicle category) + bounding box regression head (localization refinement).

The classification head is replaced to match the target number of classes via `FastRCNNPredictor`. Pre-trained COCO/ImageNet weights are used as initialization; **all backbone layers are fine-tuned**.
