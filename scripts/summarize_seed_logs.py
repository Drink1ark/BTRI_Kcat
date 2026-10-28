#!/usr/bin/env python3
"""Summarize evaluation metrics across model-specific random-seed logs.

Edit ``MODEL_LOG_GLOBS`` below to configure the per-seed log files for each
model. The script reads lines such as ``Eval test | r2=0.61 | rmse=1.04`` and
keeps the last evaluation for each split in each log. Use ``--seeds`` to
summarize only selected seeds; without it, all matching seeds are included.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import statistics
from pathlib import Path


EVAL_RE = re.compile(r"\bEval\s+(.+?)\s*\|\s*(.*)$", re.IGNORECASE)
PAIR_RE = re.compile(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
SEED_RE = re.compile(r"\bseed\s*[:=]\s*(\d+)", re.IGNORECASE)
SEED_PATH_RE = re.compile(r"(?:^|[/_])seed[_-]?(\d+)(?:$|[/_.-])", re.IGNORECASE)

# Edit this list to add or remove model runs. Each glob should match only the
# per-seed logs for one model; model groups are summarized independently below.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = PROJECT_ROOT / "results/seed_metrics_summary.csv"
RESULT_ROOT = PROJECT_ROOT / "results/CatPred-DB"
MODEL_LOG_GLOBS = None  # None: discover runs when main() is called, not at import.
EVAL_SPLITS = None
SELECTED_SEEDS = None
DDOF = 1


def collect_logs(pattern: str) -> list[Path]:
    return sorted(Path(p).resolve() for p in glob.glob(pattern, recursive=True) if Path(p).is_file())


def find_seed(path: Path, lines: list[str]) -> str:
    seed = None
    for line in lines:
        match = SEED_RE.search(line)
        if match:
            seed = match.group(1)
    if seed is not None:
        return seed
    match = SEED_PATH_RE.search(path.as_posix())
    return match.group(1) if match else path.stem


def parse_log(path: Path) -> tuple[str, dict[str, dict[str, float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    seed = find_seed(path, lines)
    evaluations: dict[str, dict[str, float]] = {}
    for line in lines:
        match = EVAL_RE.search(line)
        if not match:
            continue
        split = match.group(1).strip()
        metrics = {
            key.lower(): float(value)
            for key, value in PAIR_RE.findall(match.group(2))
        }
        metrics = {key: value for key, value in metrics.items() if math.isfinite(value)}
        if metrics:
            evaluations[split] = metrics
    return seed, evaluations


def parse_selected_seeds(values: list[str] | None) -> set[str] | None:
    """Parse ``--seeds`` values, accepting both spaces and commas."""
    if not values:
        return None
    selected = {
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    }
    if not selected:
        raise ValueError("--seeds must contain at least one seed")
    return selected


def seed_is_selected(seed: str, selected_seeds: set[str] | None) -> bool:
    if selected_seeds is None:
        return True
    if seed in selected_seeds:
        return True
    # Treat numeric spellings such as ``01`` and ``1`` as the same seed.
    if seed.isdigit():
        return str(int(seed)) in {
            str(int(item)) for item in selected_seeds if item.isdigit()
        }
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize metrics from per-seed evaluation logs."
    )
    parser.add_argument(
        "--seeds",
        "--seed",
        nargs="+",
        metavar="SEED",
        help="Only summarize these seeds; provide values separated by spaces or commas.",
    )
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--ddof", type=int, choices=(0, 1), default=DDOF)
    args = parser.parse_args()
    try:
        ddof = args.ddof
        if ddof not in (0, 1):
            raise ValueError("DDOF must be 0 or 1")
        configured_seeds = (
            [str(seed) for seed in SELECTED_SEEDS]
            if SELECTED_SEEDS is not None
            else None
        )
        selected_seeds = parse_selected_seeds(
            args.seeds if args.seeds is not None else configured_seeds
        )
        splits = args.splits if args.splits is not None else EVAL_SPLITS
        selected_splits = {split.lower() for split in splits} if splits else None
        result_root = args.result_root or RESULT_ROOT
        model_patterns = MODEL_LOG_GLOBS
        if model_patterns is None or args.result_root is not None:
            model_patterns = [(directory.name, str(directory / "seed_*" / "train_CatPred-DB.log"))
                              for directory in sorted(result_root.glob("*"))
                              if directory.is_dir() and any(directory.glob("seed_*/train_CatPred-DB.log"))]
        if not model_patterns:
            raise ValueError(f"No model log directories found under {result_root}")
        output_csv = args.output if args.output is not None else OUTPUT_CSV
        rows = []
        print(f"Standard deviation: {'sample' if ddof else 'population'} (ddof={ddof})")
        if selected_seeds is not None:
            ordered_seeds = sorted(
                selected_seeds,
                key=lambda seed: (0, int(seed)) if seed.isdigit() else (1, seed),
            )
            print(f"Selected seeds: {', '.join(ordered_seeds)}")
        for model, pattern in model_patterns:
            logs = collect_logs(pattern)
            if not logs:
                print(f"\n[{model}] No logs match: {pattern}")
                continue

            # Each seed log contributes its last evaluation for each split.
            samples: dict[tuple[str, str], list[tuple[str, float]]] = {}
            selected_log_count = 0
            for path in logs:
                seed, evaluations = parse_log(path)
                if not seed_is_selected(seed, selected_seeds):
                    continue
                selected_log_count += 1
                for split, metrics in evaluations.items():
                    if selected_splits is not None and split.lower() not in selected_splits:
                        continue
                    for metric, value in metrics.items():
                        samples.setdefault((split, metric), []).append((seed, value))

            if selected_seeds is None:
                print(f"\n[{model}] Logs found: {len(logs)}")
            else:
                print(
                    f"\n[{model}] Logs found: {len(logs)}; "
                    f"selected: {selected_log_count}"
                )
            if not samples:
                print("No matching 'Eval <split> | metric=value' entries found")
                continue
            metrics = sorted({metric for _, metric in samples})
            wide_rows = []
            for split in sorted({split for split, _ in samples}):
                split_samples = {
                    metric: samples[(split, metric)]
                    for metric in metrics
                    if (split, metric) in samples
                }
                seed_ids = sorted(
                    {seed for values in split_samples.values() for seed, _ in values},
                    key=lambda seed: (0, int(seed)) if seed.isdigit() else (1, seed),
                )
                row = {
                    "model": model,
                    "split": split,
                    "n": len(seed_ids),
                    "seeds": ",".join(seed_ids),
                }
                for metric in metrics:
                    values = split_samples.get(metric)
                    if not values:
                        row[metric] = ""
                        continue
                    numbers = [value for _, value in values]
                    std = statistics.stdev(numbers) if ddof == 1 and len(numbers) > 1 else statistics.pstdev(numbers)
                    mean = statistics.mean(numbers)
                    row[metric] = f"{mean:.6f} +/- {std:.6f}"
                wide_rows.append(row)
                rows.append(row)

            print(f"Seeds: {', '.join(sorted({seed for values in samples.values() for seed, _ in values}, key=lambda seed: (0, int(seed)) if seed.isdigit() else (1, seed)))}")
            headers = ["Split", "N", *metrics]
            table = [[row["split"], str(row["n"]), *(row[metric] for metric in metrics)] for row in wide_rows]
            widths = [max(len(str(value)) for value in [header, *(line[index] for line in table)]) for index, header in enumerate(headers)]
            print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
            for line in table:
                print("  ".join(value.ljust(widths[index]) for index, value in enumerate(line)))

        if output_csv and rows:
            output = output_csv.expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="") as handle:
                metric_columns = sorted({
                    key for row in rows for key in row
                    if key not in {"model", "split", "n", "seeds"}
                })
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["model", "split", "n", "seeds", *metric_columns],
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved: {output}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
