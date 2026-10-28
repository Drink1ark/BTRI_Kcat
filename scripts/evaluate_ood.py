"""Evaluate saved BTRIKcat PMAK checkpoints without retraining.

Defaults use the last-epoch files produced by train/train_BTRIKcat_inPmakDB.py.
Override checkpoint, feature and output paths with command-line arguments.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Defaults correspond to train_BTRIKcat_inPmakDB.py's last-epoch checkpoints.
CHECKPOINT_ROOT = PROJECT_DIR / "results/PMAK-DB/PMAK-DB_BTRIKcat_last_epoch"
SEEDS = None  # None: use the saved summary, then discover available last checkpoints.
CHECKPOINT_PATTERN = "seed_{seed}/last_model.pth"
CHECKPOINTS = []
FEATURE_DIR = None  # None: use the selected checkpoint's recorded feature root.
TASK_NAME = None
TRAIN_SPLIT = None
OOD_SPLIT = "OOD"
OUTPUT_DIR = None  # None: derive from the selected checkpoint run.
MODEL_MODULE = "model.BTRIKcat"
MODEL_CLASS = None  # None: use checkpoint metadata (including ablation class).
DEVICE = None

from train.train_BTRIKcat_inCatpredDB import (  # noqa: E402
    LabelProcessor,
    TokenDataset,
    calc_metrics,
    exact_overlap_report,
    load_checkpoint,
    make_loader,
    predict_processed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained BTRI-Kcat checkpoint(s) on an OOD split."
    )
    parser.add_argument(
        "--checkpoint",
        required=False,
        nargs="+",
        type=Path,
        default=None,
        help="One or more last_model.pth/best_model.pth files. Overrides CHECKPOINTS above.",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=None,
        help="Feature root. Overrides FEATURE_DIR above.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Task directory. Overrides TASK_NAME above.",
    )
    parser.add_argument("--train-split", default=None, help="Training split. Overrides TRAIN_SPLIT above.")
    parser.add_argument("--ood-split", default=None, help="OOD split. Overrides OOD_SPLIT above.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Overrides OUTPUT_DIR above.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None, choices=("cuda", "cpu"))
    parser.add_argument(
        "--model-module",
        default=None,
        help="Python module containing the model class. Overrides MODEL_MODULE above.",
    )
    parser.add_argument(
        "--model-class",
        default=None,
        help="Model class name. Overrides MODEL_CLASS above.",
    )
    parser.add_argument("--print-config", action="store_true",
                        help="Read checkpoint metadata and print resolved settings without inference")
    return parser.parse_args()


def _load_model(checkpoint: Dict[str, Any], model_cls: type, device: str) -> torch.nn.Module:
    model = model_cls(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _make_eval_cfg(checkpoint: Dict[str, Any], args: argparse.Namespace, device: str):
    saved_cfg = dict(checkpoint.get("cfg", {}))
    cfg = SimpleNamespace(**saved_cfg)
    cfg.device = device
    cfg.batch_size = args.batch_size if args.batch_size is not None else int(saved_cfg.get("batch_size", 32))
    cfg.num_workers = (
        args.num_workers if args.num_workers is not None else int(saved_cfg.get("num_workers", 0))
    )
    cfg.pin_memory = bool(saved_cfg.get("pin_memory", True))
    cfg.use_amp = bool(saved_cfg.get("use_amp", True))
    cfg.max_protein_residues_per_batch = saved_cfg.get("max_protein_residues_per_batch")
    if cfg.batch_size < 1 or cfg.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    return cfg


def _task_split(checkpoint: Dict[str, Any], key: str, fallback: str) -> str:
    tasks = checkpoint.get("cfg", {}).get("tasks", [])
    if tasks and isinstance(tasks[0], dict):
        value = tasks[0].get(key)
        if value:
            return str(value)
    return fallback


def _task_name(checkpoint: Dict[str, Any], fallback: str = "datasets") -> str:
    tasks = checkpoint.get("cfg", {}).get("tasks", [])
    if tasks and isinstance(tasks[0], dict) and tasks[0].get("task_name"):
        return str(tasks[0]["task_name"])
    return fallback


def _configured_checkpoints() -> List[Path]:
    if CHECKPOINTS:
        return [Path(path) for path in CHECKPOINTS]
    seeds = SEEDS
    summary_path = CHECKPOINT_ROOT / "summary.json"
    if seeds is None and summary_path.is_file():
        seeds = json.loads(summary_path.read_text()).get("ensemble_seeds")
    if seeds is not None:
        return [CHECKPOINT_ROOT / CHECKPOINT_PATTERN.format(seed=seed) for seed in seeds]
    return sorted(CHECKPOINT_ROOT.glob("seed_*/last_model.pth"))


def main() -> None:
    args = parse_args()
    configured_checkpoints = args.checkpoint or _configured_checkpoints()
    if not configured_checkpoints:
        raise ValueError("No checkpoint configured; edit CHECKPOINTS or pass --checkpoint.")
    checkpoint_paths = [Path(path).expanduser().resolve() for path in configured_checkpoints]
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    first_checkpoint = load_checkpoint(checkpoint_paths[0], map_location="cpu")
    saved_cfg = first_checkpoint.get("cfg", {})
    feature_dir = args.feature_dir or FEATURE_DIR
    if feature_dir is None:
        saved_feature_dir = str(saved_cfg.get("base_feature_dir", "")).strip()
        if not saved_feature_dir:
            raise ValueError("Could not determine feature directory; pass --feature-dir.")
        feature_dir = Path(saved_feature_dir)
    if not str(feature_dir):
        raise ValueError("Could not determine feature directory; pass --feature-dir.")
    feature_dir = feature_dir.expanduser().resolve()

    task_name = args.task_name or TASK_NAME or _task_name(first_checkpoint)
    train_split = args.train_split or TRAIN_SPLIT or _task_split(
        first_checkpoint, "train_split", "kcat-random_train"
    )
    ood_split = args.ood_split or OOD_SPLIT
    task_dir = feature_dir / task_name
    # Accept a task directory directly as a convenience when the caller
    # passes .../token_features_.../datasets instead of its feature root.
    if not task_dir.is_dir() and (feature_dir / ood_split).is_dir():
        task_dir = feature_dir
    ood_dir = task_dir / ood_split
    train_dir = task_dir / train_split
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"PMAK training feature split not found: {train_dir}. "
            "Generate BTRIKcat features from PMAK train.csv first."
        )
    if not ood_dir.is_dir():
        raise FileNotFoundError(
            f"PMAK OOD feature split not found: {ood_dir}. "
            "Generate BTRIKcat features from PMAK OOD.csv first."
        )

    device = args.device or DEVICE or str(saved_cfg.get("device", "cuda"))
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device = "cpu"

    model_module_name = args.model_module or MODEL_MODULE
    model_class_name = args.model_class or MODEL_CLASS or first_checkpoint.get("model_class", "BTRIKcat")
    model_module = importlib.import_module(model_module_name)
    try:
        model_cls = getattr(model_module, model_class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"{model_class_name!r} was not found in {model_module_name!r}"
        ) from exc

    output_dir = args.output_dir or OUTPUT_DIR
    if output_dir is None:
        first_parent = checkpoint_paths[0].parent
        output_dir = (first_parent.parent if first_parent.name.startswith("seed_") else first_parent) / "ood_eval"
    output_dir = output_dir.expanduser().resolve()
    cfg = _make_eval_cfg(first_checkpoint, args, device)
    print(json.dumps({"evaluation_script": str(Path(__file__).resolve()),
                      "checkpoints": [str(path) for path in checkpoint_paths],
                      "feature_dir": str(feature_dir), "task_name": task_name,
                      "train_split": train_split, "ood_split": ood_split,
                      "model_class": model_class_name, "model_module": model_module_name,
                      "output_dir": str(output_dir), "cfg": vars(cfg)},
                     ensure_ascii=False, indent=2), flush=True)
    if args.print_config:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    # The scaler is fitted on the training labels and restored from each
    # checkpoint.  OOD labels must never be used to fit or update it.
    label_processors: List[LabelProcessor] = []
    models: List[torch.nn.Module] = []
    checkpoints: List[Dict[str, Any]] = [first_checkpoint]
    for path in checkpoint_paths[1:]:
        checkpoints.append(load_checkpoint(path, map_location="cpu"))

    for checkpoint in checkpoints:
        processor = LabelProcessor()
        processor.load_state_dict(checkpoint["label_processor"])
        label_processors.append(processor)
        models.append(_load_model(checkpoint, model_cls, device))

    # The loader only needs the labels for shape and collate compatibility;
    # predictions are inverse-transformed with each checkpoint's own scaler.
    ood_ds = TokenDataset(ood_dir, label_processor=label_processors[0], fit_label_processor=False)
    loader = make_loader(ood_ds, cfg, shuffle=False)
    labels_raw = np.asarray(ood_ds.labels, dtype=np.float32).reshape(-1)

    train_ds = TokenDataset(train_dir, label_processor=label_processors[0], fit_label_processor=False)
    overlap = exact_overlap_report(train_ds.meta, ood_ds.meta)
    with (output_dir / f"overlap_train_vs_{ood_split}.json").open("w", encoding="utf-8") as handle:
        json.dump(overlap, handle, ensure_ascii=False, indent=2)

    predictions: List[np.ndarray] = []
    per_checkpoint: List[Dict[str, Any]] = []
    for path, model, processor in zip(checkpoint_paths, models, label_processors):
        preds_proc = predict_processed(model, loader, device, cfg)
        preds_raw = processor.inverse_transform(preds_proc)
        predictions.append(preds_raw)
        metrics = calc_metrics(labels_raw, preds_raw)
        per_checkpoint.append({"checkpoint": str(path), "metrics": metrics})
        checkpoint_pred_df = ood_ds.meta.copy()
        checkpoint_pred_df["y_true"] = labels_raw
        checkpoint_pred_df["y_pred"] = preds_raw
        checkpoint_pred_df["error"] = checkpoint_pred_df["y_pred"] - checkpoint_pred_df["y_true"]
        checkpoint_pred_df["abs_error"] = checkpoint_pred_df["error"].abs()
        checkpoint_pred_df.to_csv(
            output_dir / f"predictions_{ood_split}_{path.parent.name}.csv", index=False
        )
        print(
            f"{path} | r2={metrics['r2']:.4f} | pcc={metrics['pcc']:.4f} | "
            f"spearman={metrics['spearman']:.4f} | rmse={metrics['rmse']:.4f} | mae={metrics['mae']:.4f}"
        )

    stacked = np.stack(predictions, axis=0)
    ensemble_pred = np.mean(stacked, axis=0)
    ensemble_metrics = calc_metrics(labels_raw, ensemble_pred)
    pred_df = ood_ds.meta.copy()
    pred_df["y_true"] = labels_raw
    pred_df["y_pred"] = ensemble_pred
    pred_df["error"] = pred_df["y_pred"] - pred_df["y_true"]
    pred_df["abs_error"] = pred_df["error"].abs()
    for index, path in enumerate(checkpoint_paths):
        pred_df[f"pred_{path.parent.name}"] = stacked[index]
    pred_df.to_csv(output_dir / f"predictions_{ood_split}.csv", index=False)

    payload = {
        "method": "mean_prediction_ensemble" if len(models) > 1 else "single_checkpoint",
        "checkpoints": [str(path) for path in checkpoint_paths],
        "feature_dir": str(feature_dir),
        "ood_split": ood_split,
        "num_samples": int(len(labels_raw)),
        "metrics": ensemble_metrics,
        "per_checkpoint": per_checkpoint,
        "overlap": overlap,
    }
    with (output_dir / f"metrics_{ood_split}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(
        f"Ensemble ({len(models)} checkpoint(s)) | r2={ensemble_metrics['r2']:.4f} | "
        f"pcc={ensemble_metrics['pcc']:.4f} | spearman={ensemble_metrics['spearman']:.4f} | "
        f"rmse={ensemble_metrics['rmse']:.4f} | mae={ensemble_metrics['mae']:.4f}"
    )
    print(f"Saved OOD results to: {output_dir}")


if __name__ == "__main__":
    main()
