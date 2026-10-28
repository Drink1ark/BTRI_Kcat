#!/usr/bin/env python3
"""Summarize BCAKcat random-test metrics and per-seed training durations.

Only kcat-random_test is exported. Training hours come from summary.json and
cover the training/validation loop, excluding subsequent test evaluations.
The final row is the saved mean-prediction ensemble, distinct from the mean
and population standard deviation of the individual seeds (the saved
BCAKcat aggregation uses ddof=0).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results/CatPred-DB/CatPred-DB_BTRIKcat_full"
SPLIT = "kcat-random_test"
METRICS = ("r2", "pcc", "spearman", "rmse", "mae")
FIELDS = (
    "seed", "result_type", "split", *METRICS, "mse", "loss", "hours",
    "duration_seconds", "duration_hms", "best_epoch", "best_val_r2",
    "epochs_completed", "num_seeds", "time_source",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_data_dir(root: Path) -> Path:
    candidates = [root] + sorted(path for path in root.iterdir() if path.is_dir())
    matches = [path for path in candidates
               if (path / f"ensemble_metrics_{SPLIT}.json").is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one random-test result directory under {root}; found {matches}")
    return matches[0]


def duration_fields(hours: float) -> dict:
    seconds = hours * 3600
    whole_hours, rest = divmod(round(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return {"hours": hours, "duration_seconds": round(seconds, 3),
            "duration_hms": f"{whole_hours:02d}:{minutes:02d}:{secs:02d}"}


def build_rows(data_dir: Path) -> list[dict]:
    ensemble = read_json(data_dir / f"ensemble_metrics_{SPLIT}.json")
    seeds = sorted(ensemble["seeds"])
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds in ensemble metadata")
    recorded = {row["seed"]: row for row in ensemble["per_seed_metrics"]}
    if set(recorded) != set(seeds):
        raise ValueError("Ensemble seed list and per-seed metrics do not match")
    rows = []
    for seed in seeds:
        directory = data_dir / f"seed_{seed}"
        summary = read_json(directory / "summary.json")
        metrics = read_json(directory / f"metrics_{SPLIT}.json")
        if summary["seed"] != seed:
            raise ValueError(f"Seed mismatch in {directory}")
        for metric in METRICS:
            # Ensemble aggregation reads saved CSV predictions; float precision
            # can differ slightly from the metrics calculated during evaluation.
            if not math.isclose(metrics[metric], recorded[seed][metric],
                                rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(f"Stale ensemble metadata: seed {seed}, metric {metric}")
        with (directory / "history.csv").open(newline="", encoding="utf-8-sig") as handle:
            epochs = [int(row["epoch"]) for row in csv.DictReader(handle)]
        if not epochs or summary["best_epoch"] not in epochs:
            raise ValueError(f"Missing checkpoint epoch in {directory / 'history.csv'}")
        hours = float(summary["hours"])
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError(f"Invalid training duration in {directory}")
        row = {"seed": seed, "result_type": "seed", "split": SPLIT,
               **{metric: recorded[seed][metric] for metric in METRICS},
               "mse": recorded[seed]["rmse"] ** 2,
               "loss": metrics.get("loss"), "best_epoch": summary["best_epoch"],
               "best_val_r2": summary["best_r2"], "epochs_completed": max(epochs),
               "num_seeds": 1, "time_source": "summary.json:hours"}
        row.update(duration_fields(hours))
        rows.append(row)
    individual_rows = tuple(rows)
    for kind, aggregate in (("mean", statistics.mean), ("std", statistics.pstdev)):
        row = {"seed": kind, "result_type": f"individual_{kind}",
               "split": SPLIT, "num_seeds": len(seeds)}
        for field in (*METRICS, "mse", "loss"):
            values = [item[field] for item in individual_rows]
            if all(value is not None for value in values):
                row[field] = aggregate(values)
        row.update(duration_fields(aggregate([item["hours"] for item in individual_rows])))
        rows.append(row)
    row = {"seed": "ensemble", "result_type": ensemble["method"],
           "split": SPLIT, "num_seeds": len(seeds),
           **{metric: ensemble["metrics"][metric] for metric in METRICS},
           "mse": ensemble["metrics"]["rmse"] ** 2}
    # The ensemble JSON does not record SmoothL1 loss or a separate fit time.
    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path,
                        help="Default: results_dir/test_results_summary.csv")
    args = parser.parse_args()
    root = args.results_dir.resolve()
    data_dir = resolve_data_dir(root)
    output = args.output or root / "test_results_summary.csv"
    rows = build_rows(data_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} random-test rows to {output}")


if __name__ == "__main__":
    main()
