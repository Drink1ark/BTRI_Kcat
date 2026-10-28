"""Fit the cold-start random training pool and evaluate its independent test.

Optional CV folds use validation R2 to select a checkpoint. The final model
always trains for a fixed number of epochs and saves its last checkpoint.
Final runs support multiple seeds and report their metrics as mean/std.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DATASETS = {
    "cold_enzyme": {
        "train_pattern": "kcat_train_fold_{fold}_en_with_sub.csv",
        "eval_pattern": "kcat_val_fold_{fold}_en_with_sub.csv",
        "pool_split": "kcat-random_train",
        "test_split": "kcat-random_test",
    },
    "cold_reaction": {
        "train_pattern": "kcat_train_fold_{fold}_with_sub.csv",
        "eval_pattern": "kcat_val_fold_{fold}_with_sub.csv",
        "pool_split": "kcat-random_train",
        "test_split": "kcat-random_test",
    },
}

DEFAULT_FEATURE_ROOT = PROJECT_DIR / "features" / "P128_C32" / "TurnUp"
DEFAULT_SAVE_ROOT = PROJECT_DIR / "results" / "TurnUp_cold_protocol"
DEFAULT_FOLDS = (1, 2, 3, 4, 5)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


from train import train_BTRIKcat_inCatpredDB as shared_training


@dataclass
class CFG(shared_training.CFG):
    """Edit dataset-specific settings here; explicit CLI options take priority."""
    base_feature_dir: str = str(DEFAULT_FEATURE_ROOT)
    save_root: str = str(DEFAULT_SAVE_ROOT)
    epochs: int = 100
    ensemble_seeds: tuple[int, ...] = DEFAULT_SEEDS
    run_name: str = "BTRIKcat_full"
    model_variant: str = "full"
    early_stopping_patience: int = 0
    dataset: str = "both"
    folds: tuple[int, ...] = DEFAULT_FOLDS
    phase: str = "final"  # final: random pool/test only; cv or both: add five folds
    final_epochs: int | None = None


def parse_folds(value: str) -> tuple[int, ...]:
    if value.lower() == "all":
        return DEFAULT_FOLDS
    folds = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not folds or any(fold < 1 or fold > 5 for fold in folds):
        raise argparse.ArgumentTypeError("folds must be 1-5 or 'all'")
    return folds


def parse_args(argv=None) -> argparse.Namespace:
    defaults = CFG()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path(defaults.base_feature_dir))
    parser.add_argument("--save-root", type=Path, default=Path(defaults.save_root))
    parser.add_argument("--seeds", type=shared_training.parse_seed_list, default=defaults.ensemble_seeds,
                        help="Final training seeds; CV uses one seed (the first in phase both)")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--print-config", action="store_true",
                        help="Print effective configuration without loading data or training")
    parser.add_argument("--dataset", choices=[*DATASETS, "both"], default=defaults.dataset)
    parser.add_argument("--folds", type=parse_folds, default=defaults.folds)
    parser.add_argument("--phase", choices=("both", "cv", "final"), default=defaults.phase)
    parser.add_argument("--final-epochs", type=int, default=defaults.final_epochs,
                        help="Full-pool epochs; default is --epochs")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive and num-workers non-negative")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must contain unique integer IDs")
    if any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds):
        parser.error("seeds must be between 0 and 4294967295")
    if args.phase == "cv" and len(args.seeds) != 1:
        parser.error("CV-only training uses one seed per fold; pass --seeds 42")
    if args.final_epochs is not None and args.final_epochs < 1:
        parser.error("--final-epochs must be positive")
    if args.dataset not in {*DATASETS, "both"}:
        parser.error("dataset must be cold_enzyme, cold_reaction or both")
    if not args.folds or len(set(args.folds)) != len(args.folds) or any(fold not in DEFAULT_FOLDS for fold in args.folds):
        parser.error("folds must contain unique IDs from 1 to 5")
    return args


def split_names(dataset: str, fold: int) -> tuple[str, str]:
    names = DATASETS[dataset]
    train_split = Path(names["train_pattern"].format(fold=fold)).stem
    eval_split = Path(names["eval_pattern"].format(fold=fold)).stem
    return train_split, eval_split


def validate_fold_dirs(feature_root: Path, dataset: str, fold: int) -> tuple[Path, Path]:
    train_split, eval_split = split_names(dataset, fold)
    base = feature_root / dataset
    train_dir, eval_dir = base / train_split, base / eval_split
    missing = [path for path in (train_dir, eval_dir) if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing fold feature directories:\n" + "\n".join(map(str, missing)))
    return train_dir, eval_dir


def validate_final_dirs(feature_root: Path, dataset: str) -> tuple[Path, Path]:
    names = DATASETS[dataset]
    base = feature_root / dataset
    train_dir, test_dir = base / names["pool_split"], base / names["test_split"]
    missing = [path for path in (train_dir, test_dir) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing full-pool/test token features; preprocess the independent "
            "splits before final training:\n" + "\n".join(map(str, missing))
        )
    return train_dir, test_dir


def train_last_epoch_seed(
    training: Any,
    cfg: Any,
    dataset: str,
    fold: int | None,
    seed: int,
    train_split: str,
    eval_split: str,
) -> dict[str, Any]:
    import pandas as pd
    import torch

    training.seed_everything(seed)
    task_dir = Path(cfg.base_feature_dir) / dataset
    train_dir, eval_dir = task_dir / train_split, task_dir / eval_split
    seed_dir = training.task_save_dir(cfg, dataset) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    logger = training.setup_logger(seed_dir, dataset)
    logger.info(
        "Fixed-epoch final run | dataset=%s fold=%s seed=%d epochs=%d",
        dataset, fold, seed, cfg.epochs,
    )
    logger.info("Training split: %s", train_dir)
    logger.info("Evaluation split (used only after training): %s", eval_dir)

    label_processor = training.LabelProcessor(cfg.label_method)
    train_ds = training.TokenDataset(
        train_dir, label_processor=label_processor, fit_label_processor=True
    )
    train_loader = training.make_loader(train_ds, cfg, shuffle=True)
    model, model_config = training.build_model(train_ds, cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = training.make_criterion(cfg)
    amp_enabled = bool(cfg.use_amp and cfg.device == "cuda" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

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
    checkpoint = {
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
    }
    torch.save(checkpoint, seed_dir / "last_model.pth")
    pd.DataFrame(history).to_csv(seed_dir / "history.csv", index=False)

    eval_results = training.evaluate_split(
        model=model,
        split_dir=eval_dir,
        split_name=eval_split,
        train_ds=train_ds,
        criterion=criterion,
        cfg=cfg,
        save_dir=seed_dir,
        logger=logger,
    )
    seed_summary = {
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "checkpoint_policy": "last_epoch",
        "checkpoint_epoch": cfg.epochs,
        "train_split": train_split,
        "evaluation_split": eval_split,
        "evaluation_used_for_selection": False,
        "eval_results": {eval_split: eval_results},
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



def make_run_config(args: argparse.Namespace, dataset: str, fold: int) -> Any:
    train_split, eval_split = split_names(dataset, fold)
    cfg = CFG()
    cfg.base_feature_dir = str(args.feature_root.expanduser().resolve())
    cfg.save_root = str((args.save_root / dataset).expanduser().resolve())
    cfg.tasks = ({
        "task_name": dataset,
        "train_split": train_split,
        "val_split": eval_split,
        "eval_splits": (eval_split,),
    },)
    cfg.ensemble_seeds = (args.seeds[0],)
    cfg.seed = args.seeds[0]
    cfg.dataset = dataset
    cfg.folds = args.folds
    cfg.phase = args.phase
    cfg.final_epochs = args.final_epochs
    cfg.run_name = f"cv_fold_{fold}"
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    if args.device is not None:
        cfg.device = shared_training.resolve_device(args.device)
    cfg.early_stopping_patience = cfg.epochs + 1

    if cfg.model_variant != "full":
        raise ValueError("TurnUp trains full BTRIKcat; run model ablations through the CatPred entry point")
    shared_training.validate_training_config(cfg, shared_training.MODEL_VARIANTS["full"])
    return cfg


def run_fold(training: Any, args: argparse.Namespace, dataset: str, fold: int,
             seeds: tuple[int, ...]) -> dict[str, Any]:
    validate_fold_dirs(args.feature_root, dataset, fold)
    train_split, eval_split = split_names(dataset, fold)
    cfg = make_run_config(args, dataset, fold)
    result = training.train_one_seed(cfg, dataset, train_split, eval_split,
                                     (eval_split,), seeds[0])
    metrics = dict(result["eval_results"][eval_split])
    metrics["mse"] = metrics["rmse"] ** 2
    fold_summary = {
        "dataset": dataset,
        "fold": fold,
        "seed": seeds[0],
        "evaluation_split": eval_split,
        "method": "single_seed_cv",
        "checkpoint_policy": "best_validation_r2",
        "best_epoch": result["best_epoch"],
        "metrics": metrics,
        "seed_result": result,
        "cfg": training.cfg_to_dict(cfg),
    }
    fold_dir = training.task_save_dir(cfg, dataset)
    (fold_dir / "summary.json").write_text(
        json.dumps(fold_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "dataset": dataset,
        "fold": fold,
        "seed": seeds[0],
        "best_epoch": result["best_epoch"],
        **metrics,
    }


def summarize_dataset(args: argparse.Namespace, dataset: str,
                      fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import pandas as pd

    result_root = args.save_root / dataset
    result_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(fold_rows)
    frame.to_csv(result_root / "five_fold_metrics.csv", index=False)
    metric_names = ("r2", "pcc", "spearman", "mse", "rmse", "mae", "loss")
    aggregate = {
        metric: {
            "mean": float(frame[metric].mean()),
            "std": float(frame[metric].std(ddof=0)),
        }
        for metric in metric_names if metric in frame
    }
    final_epochs = args.final_epochs if args.final_epochs is not None else args.epochs
    summary = {
            "dataset": dataset,
            "checkpoint_policy": "best_validation_r2",
            "seed_policy": "one_seed_per_fold",
            "seeds": list(dict.fromkeys(row["seed"] for row in fold_rows)),
            "folds": fold_rows,
            "aggregate": aggregate,
            "final_epochs": final_epochs,
            "final_epoch_policy": ("final_epochs_override" if args.final_epochs is not None
                                   else "fixed_training_epochs"),
            "independent_test_used_in_cv": False,
    }
    (result_root / "five_fold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def make_final_config(args: argparse.Namespace, dataset: str, epochs: int) -> Any:
    cfg = CFG()
    cfg.base_feature_dir = str(args.feature_root.expanduser().resolve())
    cfg.save_root = str((args.save_root / dataset).expanduser().resolve())
    names = DATASETS[dataset]
    cfg.tasks = ({"task_name": dataset, "train_split": names["pool_split"],
                  "val_split": "", "eval_splits": (names["test_split"],)},)
    cfg.ensemble_seeds = args.seeds
    cfg.seed = args.seeds[0]
    cfg.dataset = dataset
    cfg.folds = args.folds
    cfg.phase = args.phase
    cfg.final_epochs = epochs
    cfg.run_name = "final"
    cfg.epochs = epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.early_stopping_patience = 0
    if args.device is not None:
        cfg.device = shared_training.resolve_device(args.device)
    shared_training.validate_training_config(cfg, shared_training.MODEL_VARIANTS["full"])
    return cfg


def run_final(training: Any, args: argparse.Namespace, dataset: str,
              epochs: int) -> dict[str, Any]:
    import pandas as pd

    validate_final_dirs(args.feature_root, dataset)
    names = DATASETS[dataset]
    cfg = make_final_config(args, dataset, epochs)
    root = args.save_root / dataset
    root.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for seed in args.seeds:
        cfg.seed = seed
        print(f"Final training | dataset={dataset} seed={seed} epochs={epochs}", flush=True)
        result = train_last_epoch_seed(training, cfg, dataset, None, seed,
                                       names["pool_split"], names["test_split"])
        test_metrics = dict(result["eval_results"])
        test_metrics["mse"] = test_metrics["rmse"] ** 2
        seed_rows.append({
            "seed": seed, "epochs": epochs, **test_metrics,
            "checkpoint": str(Path(result["seed_dir"]) / "last_model.pth"),
        })
        pd.DataFrame(seed_rows).to_csv(root / "final_seed_metrics.csv", index=False)
    metric_names = ("r2", "pcc", "spearman", "mse", "rmse", "mae", "loss")
    aggregate = {
        metric: {
            "mean": statistics.mean(row[metric] for row in seed_rows),
            "std": (statistics.stdev(row[metric] for row in seed_rows)
                    if len(seed_rows) > 1 else 0.0),
        }
        for metric in metric_names if all(metric in row for row in seed_rows)
    }
    summary = {
        "dataset": dataset, "seeds": list(args.seeds), "train_split": names["pool_split"],
        "test_split": names["test_split"], "checkpoint_policy": "last_epoch",
        "epochs": epochs, "method": "independent_seed_runs",
        "per_seed": seed_rows, "aggregate": aggregate,
        "std_ddof": 1 if len(seed_rows) > 1 else 0,
        "final_epoch_policy": ("final_epochs_override" if args.final_epochs is not None
                               else "fixed_training_epochs"),
        "independent_test_used_for_selection": False,
    }
    if len(args.seeds) == 1:
        summary.update(seed=args.seeds[0], test=test_metrics,
                       checkpoint=seed_rows[0]["checkpoint"])
    (root / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved final seed metrics: {root / 'final_seed_metrics.csv'}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    seeds = args.seeds

    from train import train_BTRIKcat_inCatpredDB as training
    from model.BTRIKcat import BTRIKcat

    training.KcatBiCrossAttention = BTRIKcat

    datasets = list(DATASETS) if args.dataset == "both" else [args.dataset]
    configs = ([make_run_config(args, dataset, fold)
                for dataset in datasets for fold in args.folds]
               if args.phase in ("both", "cv") else [])
    print(json.dumps({"training_script": str(Path(__file__).resolve()),
                      "phase": args.phase,
                      "cv_checkpoint_policy": "best_validation_r2",
                      "final_checkpoint_policy": "last_epoch",
                      "final_seeds": list(seeds) if args.phase in ("both", "final") else [],
                      "final_epochs": (args.final_epochs if args.final_epochs is not None else args.epochs)
                                      if args.phase in ("both", "final") else None,
                      "runs": [training.cfg_to_dict(cfg) for cfg in configs]},
                     ensure_ascii=False, indent=2), flush=True)
    if args.print_config:
        return
    for dataset in datasets:
        if args.phase == "both":
            validate_final_dirs(args.feature_root, dataset)
        if args.phase in ("both", "cv"):
            cv_seeds = (seeds[0],)
            fold_rows = [run_fold(training, args, dataset, fold, cv_seeds)
                         for fold in args.folds]
            summarize_dataset(args, dataset, fold_rows)
        if args.phase in ("both", "final"):
            final_epochs = args.final_epochs if args.final_epochs is not None else args.epochs
            run_final(training, args, dataset, final_epochs)


if __name__ == "__main__":
    main()
