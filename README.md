## Dependencies

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/) and split into groups, so each workflow installs only what it needs.

| Group      | Purpose                                                          | Install                    |
| ---------- | --------------------------------------------------------------- | -------------------------- |
| *(base)*   | Shared deps (config and utilities used across the repo)         | `uv sync`                  |
| `dev`      | Tooling: Ruff, pre-commit, Jupyter                              | included by `uv sync`      |
| `analysis` | Notebook analysis (`summary.ipynb`): dataframes, plots, metrics | `uv sync --group analysis` |
| `training` | Model training and evaluation                                   | `uv sync --group training` |
| `service`  | FastAPI inference service                                       | `uv sync --group service`  |

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
uv sync --no-dev --group service
```

## PyTorch

On Linux, `torch` and `torchvision` resolve to the CUDA 12.8 wheels; on macOS, to the default PyPI wheels. The resolved versions are pinned in `uv.lock`.
