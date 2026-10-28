"""Train the PMAK CatPred split for fixed epochs and evaluate last weights.

Only the training split is used during optimization. The test and OOD splits
are evaluated once after the final-epoch checkpoint has been saved.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_FEATURE_ROOT = PROJECT_DIR / "features" / "P128_C32"
DEFAULT_SAVE_ROOT = PROJECT_DIR / "results" / "PMAK-DB"
DEFAULT_SEEDS = tuple(range(37, 47))
TASK_NAME = "PMAK-DB"
TRAIN_SPLIT = "train"
EVAL_SPLITS = ("test", "OOD")


from train import train_BTRIKcat_inCatpredDB as shared_training


@dataclass
class CFG(shared_training.CFG):
    """Edit dataset-specific settings here; explicit CLI options take priority."""
    base_feature_dir: str = str(DEFAULT_FEATURE_ROOT)
    save_root: str = str(DEFAULT_SAVE_ROOT)
    epochs: int = 100
    ensemble_seeds: tuple[int, ...] = DEFAULT_SEEDS
    run_name: str = "BTRIKcat_last_epoch"
    model_variant: str = "full"
    early_stopping_patience: int = 0


def parse_args(argv=None) -> argparse.Namespace:
    defaults = CFG()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path(defaults.base_feature_dir))
    parser.add_argument("--save-root", type=Path, default=Path(defaults.save_root))
    parser.add_argument("--seeds", type=shared_training.parse_seed_list, default=defaults.ensemble_seeds)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--print-config", action="store_true",
                        help="Print effective configuration without loading data or training")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive and num-workers non-negative")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must contain unique integer IDs")
    return args


def validate_splits(feature_root: Path) -> None:
    missing = [
        feature_root / TASK_NAME / split
        for split in (TRAIN_SPLIT, *EVAL_SPLITS)
        if not (feature_root / TASK_NAME / split).is_dir()
    ]
    if missing:
        raise FileNotFoundError("Missing PMAK feature directories:\n" + "\n".join(map(str, missing)))


def train_seed(training: Any, cfg: Any, seed: int) -> dict[str, Any]:
    import pandas as pd
    import torch

    training.seed_everything(seed)
    task_dir = Path(cfg.base_feature_dir) / TASK_NAME
    seed_dir = training.task_save_dir(cfg, TASK_NAME) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    logger = training.setup_logger(seed_dir, TASK_NAME)
    logger.info(
        "Fixed-epoch PMAK run | seed=%d | epochs=%d | checkpoint_policy=last_epoch",
        seed, cfg.epochs,
    )
    logger.info("Training split: %s", task_dir / TRAIN_SPLIT)
    logger.info("Evaluation splits (used only after training): %s", EVAL_SPLITS)

    label_processor = training.LabelProcessor(cfg.label_method)
    train_ds = training.TokenDataset(
        task_dir / TRAIN_SPLIT,
        label_processor=label_processor,
        fit_label_processor=True,
    )
    train_loader = training.make_loader(train_ds, cfg, shuffle=True)
    model, model_config = training.build_model(train_ds, cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = training.make_criterion(cfg)
    amp_enabled = bool(cfg.use_amp and cfg.device == "cuda" and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history: list[dict[str, float | int]] = []
    start_time = time.time()
    for epoch in range(cfg.epochs):
        learning_rate = training.cosine_lr(optimizer, epoch, cfg)
        train_loss, _, _ = training.run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            cfg.device, cfg, train=True,
        )
        history.append({"epoch": epoch + 1, "lr": learning_rate, "train_loss": train_loss})
        logger.info(
            "Epoch %03d/%03d | lr=%.7g | train_loss=%.6f",
            epoch + 1, cfg.epochs, learning_rate, train_loss,
        )

    elapsed = time.time() - start_time
    torch.save({
        "epoch": cfg.epochs,
        "checkpoint_policy": "last_epoch",
        "model_name": "BTRIKcat",
        "model_class": type(model).__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "label_processor": train_ds.label_processor.state_dict(),
        "cfg": training.cfg_to_dict(cfg),
        "model_config": model_config,
        "feature_shapes": train_ds.feature_shapes,
        "seed": seed,
    }, seed_dir / "last_model.pth")
    pd.DataFrame(history).to_csv(seed_dir / "history.csv", index=False)

    eval_results = {}
    for split in EVAL_SPLITS:
        eval_results[split] = training.evaluate_split(
            model=model,
            split_dir=task_dir / split,
            split_name=split,
            train_ds=train_ds,
            criterion=criterion,
            cfg=cfg,
            save_dir=seed_dir,
            logger=logger,
        )

    seed_summary = {
        "task_name": TASK_NAME,
        "seed": seed,
        "checkpoint_policy": "last_epoch",
        "checkpoint_epoch": cfg.epochs,
        "evaluation_used_for_selection": False,
        "train_split": TRAIN_SPLIT,
        "eval_splits": list(EVAL_SPLITS),
        "eval_results": eval_results,
        "hours": elapsed / 3600,
        "cfg": training.cfg_to_dict(cfg),
        "model_config": model_config,
        "feature_shapes": train_ds.feature_shapes,
    }
    (seed_dir / "summary.json").write_text(
        json.dumps(seed_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved final-epoch checkpoint: %s", seed_dir / "last_model.pth")

    feature_shapes = train_ds.feature_shapes
    del model, optimizer, train_loader, train_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "seed": seed,
        "seed_dir": str(seed_dir),
        "eval_results": eval_results,
        "hours": elapsed / 3600,
        "feature_shapes": feature_shapes,
        "model_config": model_config,
    }


def ensemble_split(training: Any, cfg: Any, split: str,
                   seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    frames = [
        pd.read_csv(Path(result["seed_dir"]) / f"predictions_{split}.csv")
        for result in seed_results
    ]
    reference = frames[0]
    for frame in frames[1:]:
        if len(frame) != len(reference) or not np.array_equal(
            frame["y_true"].to_numpy(), reference["y_true"].to_numpy()
        ):
            raise ValueError(f"Seed predictions are not aligned for PMAK {split}")

    seed_predictions = [frame["y_pred"].to_numpy(dtype=float) for frame in frames]
    mean_prediction = np.mean(np.stack(seed_predictions, axis=0), axis=0)
    labels = reference["y_true"].to_numpy(dtype=float)
    metrics = training.calc_metrics(labels, mean_prediction)

    checkpoint = training.load_checkpoint(
        Path(seed_results[0]["seed_dir"]) / "last_model.pth",
        map_location="cpu",
    )
    label_processor = training.LabelProcessor(cfg.label_method)
    label_processor.load_state_dict(checkpoint["label_processor"])
    metrics["loss"] = float(F.smooth_l1_loss(
        torch.as_tensor(label_processor.transform(mean_prediction)),
        torch.as_tensor(label_processor.transform(labels)),
        beta=cfg.smoothl1_beta,
    ).item())

    output_dir = training.task_save_dir(cfg, TASK_NAME)
    ensemble_frame = reference.copy()
    ensemble_frame["y_pred"] = mean_prediction
    ensemble_frame["error"] = mean_prediction - labels
    ensemble_frame["abs_error"] = np.abs(ensemble_frame["error"])
    for result, prediction in zip(seed_results, seed_predictions):
        ensemble_frame[f"pred_seed_{result['seed']}"] = prediction
    ensemble_frame.to_csv(output_dir / f"ensemble_predictions_{split}.csv", index=False)

    payload = {
        "method": "mean_prediction_ensemble",
        "checkpoint_policy": "last_epoch",
        "evaluation_used_for_selection": False,
        "seeds": [result["seed"] for result in seed_results],
        "metrics": metrics,
        "per_seed_metrics": [
            {"seed": result["seed"], **result["eval_results"][split]}
            for result in seed_results
        ],
    }
    (output_dir / f"ensemble_metrics_{split}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def make_run_config(args: argparse.Namespace) -> Any:
    cfg = CFG()
    cfg.base_feature_dir = str(args.feature_root.expanduser().resolve())
    cfg.save_root = str(args.save_root.expanduser().resolve())
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    if args.device is not None:
        cfg.device = shared_training.resolve_device(args.device)
    cfg.seed = args.seeds[0]
    cfg.ensemble_seeds = args.seeds
    cfg.fusion_definition_version = "BTRIKcat_F4_last_epoch"
    cfg.tasks = ({
        "task_name": TASK_NAME,
        "train_split": TRAIN_SPLIT,
        "val_split": "",
        "eval_splits": EVAL_SPLITS,
    },)

    if cfg.model_variant != "full":
        raise ValueError("PMAK trains full BTRIKcat; run model ablations through the CatPred entry point")
    shared_training.validate_training_config(cfg, shared_training.MODEL_VARIANTS["full"])
    return cfg


def main() -> None:
    args = parse_args()
    seeds = args.seeds

    from train import train_BTRIKcat_inCatpredDB as training
    from model.BTRIKcat import BTRIKcat

    training.KcatBiCrossAttention = BTRIKcat
    cfg = make_run_config(args)
    print(json.dumps({"training_script": str(Path(__file__).resolve()),
                      "checkpoint_policy": "last_epoch", "cfg": training.cfg_to_dict(cfg)},
                     ensure_ascii=False, indent=2), flush=True)
    if args.print_config:
        return
    validate_splits(args.feature_root)

    seed_results = [train_seed(training, cfg, seed) for seed in seeds]
    ensemble_results = {
        split: ensemble_split(training, cfg, split, seed_results)
        for split in EVAL_SPLITS
    }
    summary = {
        "task_name": TASK_NAME,
        "train_split": TRAIN_SPLIT,
        "val_split": None,
        "eval_splits": list(EVAL_SPLITS),
        "checkpoint_policy": "last_epoch",
        "evaluation_used_for_selection": False,
        "ensemble_method": "mean_prediction",
        "ensemble_seeds": list(seeds),
        "seed_results": seed_results,
        "ensemble_results": ensemble_results,
        "cfg": training.cfg_to_dict(cfg),
    }
    save_dir = training.task_save_dir(cfg, TASK_NAME)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
