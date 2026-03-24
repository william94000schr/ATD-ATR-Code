# Scene Generator Usage Guide

## Overview

The Scene Generator creates synthetic multi-target SAR scenes from existing image chips and COCO annotations. It is designed to produce:

- Scene images with multiple targets.
- A COCO annotation file for the generated scenes.
- Optional visual samples for quick quality checks.

Main entry points:

- `scene_generator/generate.py`: scene creation pipeline.
- `scene_generator/verify.py`: output validation and sample rendering.

## Prerequisites

1. Python environment with required dependencies installed (at minimum: `Pillow`, `numpy`).
2. Input data available in this structure under `/data`:

```text
data/
  images/
    test/
  annotations/
    test.json
```

3. `test.json` must follow COCO format and include:
   - `images` with `id` and `file_name`
   - `annotations` with `image_id`, `bbox`, `category_id`
   - `categories` with `id`, `name`

## Quick Start

Run commands from the `ATR_SAR/scene_generator` directory.

```bash
cd ATR_SAR/scene_generator
python generate.py
```

Default execution creates 50 scenes of size 128x128 and writes output to `ATR_SAR/data/scenes`.

Validate generated data:

```bash
python verify.py
```

## CLI Parameters (`generate.py`)

- `--num_scenes` (int, default: `50`): number of scenes to generate.
- `--canvas_size` (int, default: `128`): output scene width and height in pixels.
- `--min_targets` (int, default: `2`): minimum targets per scene.
- `--max_targets` (int, default: `5`): maximum targets per scene.
- `--margin` (int, default: `4`): extra pixels around each extracted target chip.
- `--min_gap` (int, default: `4`): minimum separation between placed chips.
- `--output_dir` (str, default: `../data/scenes`): output base directory.
- `--seed` (int, default: `42`): random seed for reproducibility.

Example:

```bash
python generate.py \
  --num_scenes 200 \
  --canvas_size 160 \
  --min_targets 2 \
  --max_targets 6 \
  --margin 6 \
  --min_gap 6 \
  --seed 123
```

## Output Structure

Generated files are organized as follows:

```text
data/scenes/
  images/
    scene_0000.png
    scene_0001.png
    ...
  annotations/
    scenes.json
  samples/
    sample_0001.png
    sample_0002.png
    ...
```

- `images/`: synthetic scenes.
- `annotations/scenes.json`: COCO labels for generated scenes.
- `samples/`: quick visual checks with bounding boxes (created by `verify.py`).

## Operational Notes

1. Use the correct working directory.
   - `generate.py` imports local modules with relative imports, so run it from `scene_generator`.

2. Keep parameter ranges realistic.
   - Very high target counts or very small canvas values can increase failed placements.

3. Tune `margin` and `min_gap` carefully.
   - `margin` affects local context around each object chip.
   - `min_gap` controls overlap avoidance and annotation clarity.

4. Verify every generated batch.
   - `verify.py` checks structure, image references, category IDs, and bbox bounds.

## Minimal Production Workflow

1. Prepare `data/images/test` and `data/annotations/test.json`.
2. Run `python generate.py` with target parameters.
3. Run `python verify.py`.
4. Use `data/scenes/images` and `data/scenes/annotations/scenes.json` in downstream training or evaluation.
