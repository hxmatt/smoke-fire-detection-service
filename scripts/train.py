"""Fine-tune the three detectors from the CLI, one at a time.

Runs on a CUDA GPU; lower ``--batch`` if you hit out-of-memory. Skips models
whose weights already exist unless ``--force``.

Examples:
    python scripts/train.py                      # all three, full run
    python scripts/train.py --quick              # fast end-to-end sanity pass
    python scripts/train.py --models rtdetr      # retrain a single model
    python scripts/train.py --imgsz 768 --batch 12
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm.auto import tqdm
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parents[1]  # run from the repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fasterrcnn_data import loader  # noqa: E402
from src.models.fasterrcnn import build_model  # noqa: E402

ALL_MODELS = ("yolo11n", "rtdetr", "fasterrcnn")

# Base training settings; per-model overrides tuned from the local 20-epoch runs
BASE = {"epochs": 20, "patience": 10, "batch": 16, "lr0": 0.01}
MODEL_DEFAULTS = {
    "yolo11n": {"epochs": 60, "patience": 15},
    "rtdetr": {"epochs": 60, "lr0": 1e-4, "batch": 8, "warmup_epochs": 3.0},
    "fasterrcnn": {"epochs": 60},
}


def write_local_yaml(data_root: Path) -> Path:
    """Rewrite data.yaml with an absolute path.

    Args:
        data_root: Dataset root holding the ``data`` directory.

    Returns:
        Path to the written ``data_local.yaml``.
    """
    out = data_root / "data_local.yaml"

    out.write_text(
        yaml.safe_dump(
            {
                "path": str(data_root.resolve()),
                "train": "data/train/images",
                "val": "data/val/images",
                "test": "data/test/images",
                "names": {0: "smoke", 1: "fire"},
            },
            sort_keys=False,
        )
    )

    return out


def train_ultralytics(
    name: str, s: dict, data_yaml: Path, weights: Path, runs: Path
) -> dict:
    """Fine-tune an Ultralytics model, save best weights, return val metrics.

    Args:
        name: Model name (``yolo11n`` or ``rtdetr``); also names the weights.
        s: Resolved settings (epochs, batch, imgsz, lr0, ...).
        data_yaml: Dataset YAML passed to Ultralytics.
        weights: Directory to copy the best weights into.
        runs: Directory for Ultralytics run artifacts.

    Returns:
        ``{"model", "mAP50", "mAP50-95"}`` for the run.
    """
    out_name = f"{name}_best.pt"
    model = YOLO("yolo11n.pt") if name == "yolo11n" else RTDETR("rtdetr-l.pt")
    # warmup_epochs only when set (RT-DETR); otherwise Ultralytics' default applies
    extra = {"warmup_epochs": s["warmup_epochs"]} if "warmup_epochs" in s else {}

    # RT-DETR's grid_sampler has no deterministic CUDA backward
    if name != "yolo11n":
        extra["deterministic"] = False

    model.train(
        data=str(data_yaml),
        epochs=s["epochs"],
        patience=s["patience"],
        imgsz=s["imgsz"],
        batch=s["batch"],
        fraction=s["fraction"],
        lr0=s["lr0"],
        cache=False,
        seed=s["seed"],
        project=str(runs),
        name=name,
        exist_ok=True,
        **extra,
    )

    val = model.val(data=str(data_yaml), imgsz=s["imgsz"], split="val")

    shutil.copy(model.trainer.best, weights / out_name)
    print(f"saved {out_name}  mAP50={val.box.map50:.3f}  mAP50-95={val.box.map:.3f}")

    del model
    torch.cuda.empty_cache()

    return {
        "model": out_name,
        "mAP50": float(val.box.map50),
        "mAP50-95": float(val.box.map),
    }


def train_one_epoch(
    model: torch.nn.Module,
    data: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    """Run one training epoch, returning the mean batch loss.

    Args:
        model: Detection model in training mode.
        data: Training DataLoader yielding ``(images, targets)``.
        optimizer: Optimizer to step.
        scaler: AMP gradient scaler.
        device: CUDA device to move batches to.

    Returns:
        Mean per-batch loss over the epoch.
    """
    model.train()
    total = 0.0

    for images, targets in tqdm(data, desc="train", leave=False):
        images = [i.to(device, non_blocking=True) for i in images]
        targets = [
            {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
        ]
        with torch.autocast("cuda"):
            loss = torch.stack(list(model(images, targets).values())).sum()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()

    return total / max(len(data), 1)


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module, data: DataLoader, device: torch.device, name: str
) -> dict:
    """Compute validation mAP over a split with torchmetrics.

    Args:
        model: Detection model to evaluate.
        data: Validation DataLoader yielding ``(images, targets)``.
        device: CUDA device to move images to.
        name: Weights filename to record in the result.

    Returns:
        ``{"model", "mAP50", "mAP50-95"}`` for the split.
    """
    metric = MeanAveragePrecision(box_format="xyxy")
    model.eval()

    for images, targets in tqdm(data, desc="map", leave=False):
        images = [i.to(device, non_blocking=True) for i in images]
        preds = [{k: v.cpu() for k, v in p.items()} for p in model(images)]
        metric.update(preds, targets)
    m = metric.compute()

    return {"model": name, "mAP50": float(m["map_50"]), "mAP50-95": float(m["map"])}


def train_fasterrcnn(
    s: dict, data_root: Path, device: torch.device, weights: Path
) -> dict:
    """Short AMP training loop for torchvision Faster R-CNN; returns val metrics.

    Args:
        s: Resolved settings (epochs, batch, imgsz, fraction, ...).
        data_root: Dataset root holding the ``data`` directory.
        device: CUDA device to train on.
        weights: Directory to save ``fasterrcnn_best.pt`` into.

    Returns:
        ``{"model", "mAP50", "mAP50-95"}`` for the run.
    """
    out = weights / "fasterrcnn_best.pt"
    epochs = s["epochs"]
    patience = s["patience"]
    model = build_model().to(device)

    train_data = loader(
        data_root,
        "train",
        True,
        s["imgsz"],
        s["fraction"],
        s["seed"],
        s["batch"],
        s["num_workers"],
    )

    val_data = loader(
        data_root,
        "val",
        False,
        s["imgsz"],
        s["fraction"],
        s["seed"],
        s["batch"],
        s["num_workers"],
    )

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-3,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    # Early stopping on val mAP@50-95: keep the best weights, stop after
    # `patience` epochs without improvement
    best = -1.0
    best_fr: dict = {}
    stale = 0

    for e in range(epochs):
        tr = train_one_epoch(model, train_data, optimizer, scaler, device)
        scheduler.step()
        fr = evaluate_map(model, val_data, device, out.name)
        improved = fr["mAP50-95"] > best
        marker = " *" if improved else ""
        print(
            f"epoch {e + 1}/{epochs}  train_loss={tr:.4f}  "
            f"mAP50={fr['mAP50']:.3f}  mAP50-95={fr['mAP50-95']:.3f}{marker}"
        )
        if improved:
            best, best_fr, stale = fr["mAP50-95"], fr, 0
            torch.save(model.state_dict(), out)
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {e + 1} (no gain in {patience} epochs)")
                break

    print(
        f"saved {out.name}  mAP50={best_fr['mAP50']:.3f}  "
        f"mAP50-95={best_fr['mAP50-95']:.3f}"
    )

    del model
    torch.cuda.empty_cache()

    return best_fr


def report(results: dict[str, dict], weights: Path) -> None:
    """Print the saved weight files and a validation mAP table.

    Args:
        results: Per-model ``{"model", "mAP50", "mAP50-95"}`` metrics.
        weights: Directory holding the saved ``.pt`` files.
    """
    print("\nweights in ./weights/:")

    for f in sorted(weights.glob("*.pt")):
        print(f"  {f.name:24s} {f.stat().st_size / 1e6:6.1f} MB")

    if results:
        print("\nValidation mAP:")
        print(f"  {'model':10s} {'mAP@50':>8s} {'mAP@50-95':>10s}")

        for r in results.values():
            n = r["model"].replace("_best.pt", "")
            print(f"  {n:10s} {r['mAP50']:8.3f} {r['mAP50-95']:10.3f}")


def run_training(args: argparse.Namespace) -> dict[str, dict]:
    """Train the requested models end to end, returning per-model val metrics.

    Args:
        args: Parsed CLI options (models, quick, epochs, batch, imgsz, ...).

    Returns:
        Per-model ``{"model", "mAP50", "mAP50-95"}`` metrics, keyed by name.
    """
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. Run this on a GPU pod, not CPU.")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)

    print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")
    print(
        f"mode={'QUICK' if args.quick else 'FULL'}  imgsz={args.imgsz}"
        f"  models={args.models}"
    )

    weights = ROOT / "weights"
    weights.mkdir(exist_ok=True)
    found = list((ROOT / "data" / "smoke-fire-detection-yolo").rglob("data.yaml"))

    if not found:
        raise SystemExit(
            "Dataset not found under data/smoke-fire-detection-yolo.\n"
            "Run `python scripts/download_dataset.py` first."
        )

    data_root = found[0].parent
    data_yaml = write_local_yaml(data_root)
    print("dataset:", data_root)

    results: dict[str, dict] = {}

    for name in args.models:
        if not args.force and (weights / f"{name}_best.pt").exists():
            print(
                f"skip {name}: weights already in ./weights/ (use --force to retrain)"
            )
            continue
        s = {
            **BASE,
            **MODEL_DEFAULTS.get(name, {}),
            "imgsz": args.imgsz,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "fraction": 0.1 if args.quick else 1.0,
        }
        if args.quick:  # short pass regardless of per-model defaults
            s["epochs"], s["patience"] = 3, 3
        if args.epochs is not None:
            s["epochs"] = args.epochs
        if args.batch is not None:
            s["batch"] = args.batch

        warmup = f" warmup={s['warmup_epochs']}" if "warmup_epochs" in s else ""
        print(
            f"\n=== training {name} ===  epochs={s['epochs']} batch={s['batch']} "
            f"lr0={s['lr0']}{warmup}"
        )

        if name == "fasterrcnn":
            results[name] = train_fasterrcnn(s, data_root, device, weights)
        else:
            results[name] = train_ultralytics(
                name, s, data_yaml, weights, ROOT / "runs"
            )

    report(results, weights)

    return results


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(ALL_MODELS),
        help="which models to train (default: all three)",
    )
    p.add_argument("--quick", action="store_true", help="3 epochs on 10%% of data")
    # None = use the per-model tuned default in MODEL_DEFAULTS.
    p.add_argument("--epochs", type=int, default=None, help="override epoch count")
    p.add_argument(
        "--batch", type=int, default=None, help="override batch (default 16)"
    )
    p.add_argument("--imgsz", type=int, default=640, help="image size (default 640)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--force", action="store_true", help="retrain even if weights already exist"
    )
    run_training(p.parse_args())


if __name__ == "__main__":
    main()
