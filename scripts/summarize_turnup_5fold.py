#!/usr/bin/env python3
"""Summarize fixed TurnUp folds, treating seeds as repeats within each fold.

The five validation splits are the reporting units. Seed scores within a fold
are averaged first; the overall mean and standard deviation then use the five
fold means. These seed-score averages are not prediction ensembles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results" / "TurnUp"
DATASETS = ("cold_enzyme", "cold_reaction")
METRICS = ("r2", "pcc", "spearman", "rmse", "mae", "loss")
FOLDS = (1, 2, 3, 4, 5)
EXCLUDED_SEEDS = ()  # Include every saved seed unless explicitly excluded.
DDOF = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to <result-root>/summary_seed_stats.",
    )
    parser.add_argument(
        "--exclude-seeds", type=int, nargs="*", default=EXCLUDED_SEEDS,
        help="Seed IDs to omit; default includes all seeds.",
    )
    parser.add_argument("--ddof", type=int, choices=(0, 1), default=DDOF,
                        help="Standard deviation convention: 1 sample (default), 0 population.")
    return parser.parse_args()


def mean_std(values: list[float], ddof: int) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = statistics.mean(values)
    if len(values) <= ddof:
        return mean, 0.0 if len(values) else math.nan
    std = statistics.stdev(values) if ddof else statistics.pstdev(values)
    return mean, std


def get_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("eval_results", {}) or payload.get("ensemble_results", {})
    if not results:
        # Per-seed metrics files sometimes contain the metric dictionary directly.
        results = {"metrics": payload}
    result = next(iter(results.values()))
    values = result.get("metrics", result)
    return {
        metric: float(value)
        for metric, value in values.items()
        if metric in METRICS and isinstance(value, (int, float)) and math.isfinite(value)
    }


def metric_file(seed_dir: Path) -> Path | None:
    candidates = sorted(seed_dir.glob("metrics_*.json"))
    return candidates[0] if candidates else None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_dataset(dataset: str, result_root: Path, output_dir: Path,
                      excluded: set[int], ddof: int) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for fold in FOLDS:
        fold_dir = result_root / dataset / f"{dataset}_fold_{fold}"
        for seed_dir in sorted(fold_dir.glob("seed_*")):
            try:
                seed = int(seed_dir.name.removeprefix("seed_"))
            except ValueError:
                continue
            if seed in excluded:
                continue
            source = metric_file(seed_dir)
            if source is None:
                missing.append(str(seed_dir))
                continue
            metrics = get_metrics(source)
            run_rows.append({"dataset": dataset, "fold": fold, "seed": seed,
                             **metrics, "source": str(source)})

    if not run_rows:
        raise FileNotFoundError(f"No seed metrics found for {dataset} under {result_root}")

    run_rows.sort(key=lambda row: (row["fold"], row["seed"]))
    fold_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        fold_runs = [row for row in run_rows if row["fold"] == fold]
        if not fold_runs:
            continue
        row: dict[str, Any] = {"dataset": dataset, "fold": fold, "n_seeds": len(fold_runs),
                               "seeds": ",".join(str(item["seed"]) for item in fold_runs)}
        for metric in METRICS:
            vals = [item[metric] for item in fold_runs if metric in item]
            if vals:
                mean, std = mean_std(vals, ddof)
                row[metric] = f"{mean:.6f} ± {std:.6f}"
        fold_rows.append(row)

    # Each fold is one data split. Seeds repeat training on that same split,
    # so 5 folds x 5 seeds must not be summarized as 25 independent splits.
    overall_rows: list[dict[str, Any]] = []
    overall: dict[str, Any] = {"dataset": dataset,
                               "sampling_unit": "fold",
                               "n_folds": len(fold_rows), "ddof": ddof,
                               "excluded_seeds": sorted(excluded)}
    for metric in METRICS:
        fold_means = [
            statistics.mean(item[metric] for item in run_rows
                            if item["fold"] == row["fold"] and metric in item)
            for row in fold_rows
            if any(item["fold"] == row["fold"] and metric in item for item in run_rows)
        ]
        if not fold_means:
            continue
        fold_mean, fold_std = mean_std(fold_means, ddof)
        overall[metric] = {
            "fold_means": {"n": len(fold_means), "mean": fold_mean, "std": fold_std},
        }
        overall_rows.append({"dataset": dataset, "metric": metric,
                             "n_folds": len(fold_means),
                             "fold_mean±std": f"{fold_mean:.6f} ± {fold_std:.6f}"})

    dataset_out = output_dir / dataset
    write_csv(dataset_out / "per_seed_metrics.csv", run_rows,
              ["dataset", "fold", "seed", *METRICS, "source"])
    write_csv(dataset_out / "per_fold_summary.csv", fold_rows,
              ["dataset", "fold", "n_seeds", "seeds", *METRICS])
    write_csv(dataset_out / "overall_summary.csv", overall_rows,
              ["dataset", "metric", "n_folds", "fold_mean±std"])
    report = {"runs": run_rows, "folds": fold_rows, "overall": overall,
              "missing_metrics": missing}
    (dataset_out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[{dataset}] {len(fold_rows)} folds, {len(run_rows)} seed runs; excluded seeds: {sorted(excluded)}")
    for row in overall_rows:
        print(f"  {row['metric']}: fold means {row['fold_mean±std']}")
    if missing:
        print(f"  Missing metrics: {len(missing)} seed directories")
    print(f"  Saved: {dataset_out}")
    return report


def main() -> int:
    args = parse_args()
    result_root = args.result_root.resolve()
    output_dir = (args.output_dir or result_root / "summary_seed_stats").resolve()
    excluded = set(args.exclude_seeds)
    reports = {
        dataset: summarize_dataset(dataset, result_root, output_dir, excluded, args.ddof)
        for dataset in DATASETS
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "all_datasets_summary.json").write_text(
        json.dumps({"excluded_seeds": sorted(excluded), "ddof": args.ddof,
                    "datasets": reports}, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
