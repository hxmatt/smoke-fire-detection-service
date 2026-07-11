## Dependencies

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/) and split into groups, so each workflow installs only what it needs.

| Group      | Purpose                                                          | Install                    |
| ---------- | --------------------------------------------------------------- | -------------------------- |
| *(base)*   | Shared deps (config and utilities used across the repo)         | `uv sync`                  |
| `dev`      | Tooling: Ruff, pre-commit, Jupyter                              | included by `uv sync`      |
| `analysis` | Notebook analysis (`summary.ipynb`): dataframes, plots, metrics | `uv sync --group analysis` |
| `training` | Model training and evaluation                                   | `uv sync --group training --extra cu128` |
| `service`  | FastAPI inference service                                       | `uv sync --group service --extra cpu`  |

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

## PyTorch

`torch` and `torchvision` come from one of two mutually exclusive extras, so each install picks its accelerator explicitly:

- `cpu`: CPU-only wheels, used by the container and fine for local dev.
- `cu128`: CUDA 12.8 wheels, used for GPU training.

Only one may be active at a time (uv enforces this). For example, `uv sync --group service --extra cpu` for the service, `uv sync --group training --extra cu128` for GPU training. On macOS both extras resolve to the same CPU/MPS wheels. Resolved versions are pinned in `uv.lock`.

## Container Image

The `Dockerfile` builds a self-contained image of the inference service. It starts from `python:3.13-slim` and lets `uv` install the `service` group from the lockfile, so the image contains exactly the pinned versions and nothing from the notebook or training workflows. 

Torch is installed from the `cpu` extra, since the service runs inference on CPU and the CUDA build would only add weight it never uses. It also pulls in `libgl1` and `libglib2.0-0`, the system libraries OpenCV loads at import time (OpenCV arrives indirectly through ultralytics). The commands below use `docker`, though Podman works the same way.

Building the image takes a single command:

```bash
docker build -t smoke-fire-detection-service .
```

The model weights are kept out of the image on purpose. They are gitignored and published instead as assets on the repository's [GitHub release](https://github.com/hxmatt/smoke-fire-detection-service/releases/latest), which keeps the image code-only and lets the weights change independently of the code that serves them. The service reads them at runtime from the directory `WEIGHTS_DIR` points at, which the image sets to `/weights`. 

Since the repository is private, fetching the assets needs an authenticated [GitHub CLI](https://cli.github.com), so run `gh auth login` once beforehand. After that, pulling the three `.pt` files into a local `weights/` directory is a single command, and it behaves the same on macOS, Linux, and Windows:

```bash
gh release download --repo hxmatt/smoke-fire-detection-service --dir weights --pattern '*.pt'
```

With the weights in place, start the container and mount that directory at `/weights`:

```bash
docker run --rm -p 8000:8000 -v "$PWD/weights:/weights" smoke-fire-detection-service
```

The API is then served at `http://localhost:8000`, with the detection UI at `/`.

One caveat for Podman on SELinux hosts (Fedora, RHEL): add the `:Z` relabel suffix to the volume so the container is allowed to read the mounted weights:

```bash
podman run --rm -p 8000:8000 -v "$PWD/weights:/weights:Z" smoke-fire-detection-service
```
