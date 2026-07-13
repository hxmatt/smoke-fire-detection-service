# Smoke & Fire Detection Service

[![CI](https://github.com/hxmatt/smoke-fire-detection-service/actions/workflows/ci.yml/badge.svg)](https://github.com/hxmatt/smoke-fire-detection-service/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A containerized service that detects smoke and fire in images, with a notebook comparing the three models behind it. The repository has two parts: a FastAPI service that serves any of the 
models on request, and an analysis notebook that characterizes the dataset and evaluates the models on the test set.

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [How it works](#how-it-works)
- [Running the service](#running-the-service)
    - [Fetch the model weights](#fetch-the-model-weights)
    - [Pull the prebuilt image](#pull-the-prebuilt-image)
    - [Build the image](#build-the-image)
        - [Start the container](#start-the-container)
    - [The API](#the-api)
- [Dataset](#dataset)
- [The analysis notebook](#the-analysis-notebook)
    - [Results](#results)
    - [Regenerating the notebook](#regenerating-the-notebook)
    - [Exporting to HTML](#exporting-to-html)
- [Development](#development)
    - [Dependencies](#dependencies)
    - [PyTorch](#pytorch)
    - [Training](#training)
    - [Container image](#container-image)

## Overview

Three models are fine-tuned on a smoke/fire dataset and served through one interface. Pick one by id at request time:

| id           | model         | architecture                                       |
| ------------ | ------------- | -------------------------------------------------- |
| `yolo11n`    | YOLOv11 Nano  | One-stage anchor-free CNN detector                 |
| `rtdetr`     | RT-DETR       | End-to-end transformer detector (DETR)             |
| `fasterrcnn` | Faster R-CNN  | Two-stage CNN detector, MobileNetV3-L FPN backbone |

The service exposes them behind `GET /models` and `POST /predict`, with a web UI for uploading an image and seeing the boxes. The notebook covers the analysis: dataset characterization
(21,527 images and 26,539 labeled boxes across the two classes; roughly 46% of the images are background, with no boxes), the standard detection metrics, and a breakdown of accuracy by 
object size across the three models.

## Repository layout

```
src/            FastAPI service
  main.py         app entrypoint; mounts the API and the static UI
  api/            routes and request/response schemas
  models/         model registry and one wrapper per detector
  frontend/       static upload UI (index.html, style.css, app.js)
scripts/        analysis and training helpers (dataset, training, viz)
notebooks/      summary.ipynb, the comparative analysis
Dockerfile      builds the service image
```

`src/` holds the service and the model code it shares. `scripts/` holds the analysis and training code, which the running service never imports.

## How it works

The service is a FastAPI app with two routes, `GET /models` and `POST /predict`, and the static UI mounted at `/`. Both routes reach the models through a registry, injected as a dependency.

The registry maps each model id to a wrapper and its weights file, and loads a wrapper the first time that id is requested (it stays in memory afterwards). Every wrapper exposes the same 
calls (`load`, `predict`, `get_info`) and returns detections in one shape (`label`, `confidence`, and an `[x, y, width, height]` box), so `/predict` runs the chosen model without knowing 
which one it is.

Adding a model means a new wrapper and one registry entry; the routes never change. A prediction decodes the upload to RGB, runs the wrapper, and returns the detections 
with its measured inference time.

The UI is a plain static page (hand-written HTML, CSS, and JS) served by the same process at `/`, with no frontend framework or build step. It is only an API client: it posts an image to 
`/predict` and draws the returned boxes, so the service stays one deployable app and the API stays the real interface.

## Running the service

The service runs from the container image; you can build the image yourself or pull the prebuilt one. Either solution requires you to fetch the model weights first, as described below.

The commands below use `docker`; Podman works the same, with one note for SELinux hosts at the end.

### Fetch the model weights

The weights live outside the image and outside git, published as assets on the repository's [latest GitHub release](https://github.com/hxmatt/smoke-fire-detection-service/releases/latest).
That keeps the image code-only and lets the weights change without a rebuild.

The repository is private, so fetching the assets needs an authenticated [GitHub CLI](https://cli.github.com). Run `gh auth login` once, then pull the three `.pt` files into a local 
`weights/` directory (same on macOS, Linux, and Windows):

```bash
gh release download --repo hxmatt/smoke-fire-detection-service --dir weights --pattern '*.pt'
```

### Pull the prebuilt image

A prebuilt image is published to the repository's GitHub Container Registry, as an alternative to building from source. The package is private, so pulling needs read access to the 
repository or package, with Docker logged in to GHCR (`docker login ghcr.io`):

```bash
docker pull ghcr.io/hxmatt/smoke-fire-detection-service:latest
```

The pulled image is code-only, like a local build, so the [weights step](#fetch-the-model-weights) below still applies. Once the weights are in `weights/`, this runs the pulled image in full:

```bash
docker run --rm -p 8000:8000 -v "$PWD/weights:/weights" ghcr.io/hxmatt/smoke-fire-detection-service:latest
```

or

```bash
podman run --rm -p 8000:8000 -v "$PWD/weights:/weights:Z" ghcr.io/hxmatt/smoke-fire-detection-service:latest
```

The API is then served at `http://localhost:8000`, with the upload UI at `/`. Building the image yourself is the alternative below.

### Build the image

```bash
docker build -t smoke-fire-detection-service .
```

## Start the container

Mount `weights/` at `/weights`, where the service reads them from:

```bash
docker run --rm -p 8000:8000 -v "$PWD/weights:/weights" smoke-fire-detection-service
```

The API is then served at `http://localhost:8000`, with the upload UI at `/`.

On SELinux hosts (Fedora, RHEL) using Podman, add the `:Z` relabel suffix so the container can read the mounted weights:

```bash
podman run --rm -p 8000:8000 -v "$PWD/weights:/weights:Z" smoke-fire-detection-service
```

### The API

`GET /models` lists the available models and their metadata:

```bash
curl -s http://localhost:8000/models
```

`POST /predict` takes the `image` file, a `model_id` from the list above, and an optional `conf` confidence threshold (default `0.25`). It returns the model used, the detections, and the 
inference time:

```bash
curl -s http://localhost:8000/predict \
    -F image=@sample.jpg \
    -F model_id=yolo11n \
    -F conf=0.25
```

Each detection carries a `label` (`smoke` or `fire`), a `confidence`, and a `bbox` given as `[x, y, width, height]` in pixels.

## Dataset

The models are trained and evaluated on the public [Smoke-Fire Detection YOLO dataset](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) (21,527 images across the two 
classes). It is not stored in the repo; download it with:

```bash
uv run python scripts/download_dataset.py
```

The download comes from the public Kaggle mirror, so no account or API key is needed. It extracts under `data/smoke-fire-detection-yolo/`; pass a target directory as the first argument, 
and `--force` re-downloads over an existing copy.

The [notebook](#regenerating-the-notebook) fetches the dataset automatically on its first run, and [training](#training) expects it on disk beforehand. The service does not use the 
dataset, but its labeled images are handy as test inputs for `/predict` and the upload UI.

## The analysis notebook

The committed `notebooks/summary.ipynb` has its outputs stripped, so the figures and tables are not in the repo. A rendered `summary.html` of a full execution is available under the assets 
of the [latest release](https://github.com/hxmatt/smoke-fire-detection-service/releases/latest); start there to read the analysis without running anything.

### Results

All three models are scored on the same 4,306-image test set. YOLOv11 Nano is the deployed default: highest accuracy, and also the smallest and fastest.

| model        | mAP@50 | mAP@50–95 | latency (CPU) | size    |
| ------------ | ------ | --------- | ------------- | ------- |
| YOLOv11 Nano | 0.754  | 0.430     | 27.2 ms       | 5.5 MB  |
| RT-DETR      | 0.597  | 0.278     | 262.9 ms      | 66.2 MB |
| Faster R-CNN | 0.619  | 0.306     | 206.7 ms      | 76.0 MB |

YOLO's lead is the most precise on small objects (COCO size buckets): 0.166 AP against 0.059 and 0.058, roughly three times the other two. Fire boxes also skew small relative to the frame, 
usually under 1% of it.

RT-DETR and Faster R-CNN are still served through the API, but neither justifies its extra size on this data, and RT-DETR early-stopped and is under-trained. Latency was measured on one 
CPU-only machine; on a GPU the ordering itself could change. See `summary.html` for the per-class curves and the size breakdown.

### Regenerating the notebook

The notebook is analysis-only but loads the trained models, so it needs the `analysis` and `training` groups and the weights on disk:

```bash
uv sync --group analysis --group training
gh release download --repo hxmatt/smoke-fire-detection-service --dir weights --pattern '*.pt'
```

Then run it top to bottom:

```bash
uv run jupyter lab notebooks/summary.ipynb
```

A full run takes a while: the evaluation scores each model over the entire test set, which dominates the runtime and varies with the machine. On CPU, YOLO is quick, while RT-DETR and 
Faster R-CNN are both far slower.

The dataset is fetched automatically on the first cell that needs it and cached under `data/` for later runs (see [Dataset](#dataset)).

### Exporting to HTML

The notebook is delivered as HTML rather than PDF: a PDF export (nbconvert's `webpdf`) clips the wide results table at the page margin, while HTML lets the browser scroll it. 

`jupyter nbconvert` writes a single file that opens in any browser, with figures and tables embedded:

```bash
uv run jupyter nbconvert --to html notebooks/summary.ipynb
```

This writes `notebooks/summary.html`, and needs no headless browser or LaTeX toolchain.

## Development

### Dependencies

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/) and split into groups, so each workflow installs only what it needs.

| Group      | Purpose                                                                   | Install                                  |
| ---------- | ------------------------------------------------------------------------- | ---------------------------------------- |
| *(base)*   | Shared deps (config and utilities used across the repo)                   | `uv sync`                                |
| `dev`      | Tooling: Ruff, pre-commit, Jupyter                                        | included by `uv sync`                    |
| `analysis` | Notebook analysis (`notebooks/summary.ipynb`): dataframes, plots, metrics | `uv sync --group analysis`               |
| `training` | Model training and evaluation                                             | `uv sync --group training --extra cu128` |
| `service`  | FastAPI inference service                                                 | `uv sync --group service --extra cpu`    |

`uv sync` creates `.venv` (if needed) and installs the base and `dev` groups. Add a group to pull in its extras:

```bash
uv sync --group analysis
uv sync --group service
```

Groups combine:

```bash
uv sync --group analysis --group training
```

Run project commands through `uv run`, which uses the synced environment without activating it manually.

To skip `dev` (e.g. a lean service install):

```bash
uv sync --no-dev --group service --extra cpu
```

### PyTorch

`torch` and `torchvision` come from one of two mutually exclusive extras, so each install picks its accelerator explicitly:

- `cpu`: CPU-only wheels, used by the container and fine for local dev.
- `cu128`: CUDA 12.8 wheels, used for GPU training.

Only one may be active at a time (uv enforces this). For example, `uv sync --group service --extra cpu` for the service, `uv sync --group training --extra cu128` for GPU training. On macOS 
both extras resolve to the same CPU/MPS wheels. Resolved versions are pinned in `uv.lock`.

### Training

The served weights are fine-tuned, not stock: the stock checkpoints are trained on COCO, which has no smoke or fire class. `scripts/train.py` fine-tunes each model from its pretrained 
checkpoint and writes the best epoch into `weights/`. YOLOv11 Nano and RT-DETR start from their Ultralytics weights; Faster R-CNN starts from the COCO-pretrained torchvision MobileNetV3-L 
FPN with its backbone frozen, so only the FPN and head train. Training reads the dataset from `data/` and stops with a reminder if it is missing, so download it first (see [Dataset](#dataset)).

```bash
uv sync --group training --extra cu128
uv run python scripts/download_dataset.py      # fetch the dataset first
uv run python scripts/train.py                 # all three
uv run python scripts/train.py --models rtdetr # just one
```

Each model trains up to 60 epochs with early stopping. A few settings (learning rate, batch, warmup) are set per model, since the three don't share one recipe. The script skips models 
whose weights already exist, and `--force` overrides that.

Training needs a CUDA GPU; the published weights came from a rented RTX A6000 and are the ones attached to the release, so you only need to retrain to reproduce or change them. RT-DETR 
early-stopped at epoch 12 (its accuracy peaked around epoch 2, against YOLO's full 60), so it is under-trained, and scores below the other two.

### Container image

The `Dockerfile` builds the service image. It starts from `python:3.13-slim` and lets `uv` install the `service` group from the lockfile, so the image has exactly the pinned versions and 
nothing from the notebook or training workflows.

Torch comes from the `cpu` extra, since the service runs inference on CPU and the CUDA build would only add weight it never uses. The image also pulls in `libgl1` and `libglib2.0-0`, the 
system libraries OpenCV loads at import time (OpenCV arrives indirectly through ultralytics).

The image is built locally with the `docker build` command above and pushed to the repository's container registry by hand; there is no CI/CD that publishes it. The service reads the 
weights at runtime from the directory `WEIGHTS_DIR` points at, which the image sets to `/weights`.
