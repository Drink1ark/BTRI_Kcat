#!/usr/bin/env python3
"""Summarize protein sequence lengths in a CatPred-DB CSV file.

The default unit is one valid sequence record. Use ``--unique-sequences`` to
count each distinct sequence once, which is useful when a protein occurs in
multiple reaction records.

Ranges are non-overlapping:
  <128       : length < 128
  128-512    : 128 <= length < 512
  512-1022   : 512 <= length <= 1022
  >1022      : length > 1022
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Iterable, List


DEFAULT_INPUT = str(Path(__file__).resolve().parents[1] / "data/CatPred-DB/kcat-random_train.csv")
WHITESPACE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_INPUT, help="Input CSV file")
    parser.add_argument("--sequence-column", default="sequence", help="Sequence column name")
    parser.add_argument(
        "--unique-sequences",
        action="store_true",
        help="Count distinct sequence strings instead of all valid CSV records",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the summary table CSV",
    )
    return parser.parse_args()


def normalized_sequence(value: str) -> str:
    """Remove whitespace so line-wrapped sequences have their true length."""
    return WHITESPACE.sub("", value.strip())


def read_lengths(path: Path, sequence_column: str, unique: bool) -> tuple[List[int], int, int]:
    lengths: List[int] = []
    seen = set()
    total_rows = 0
    invalid_rows = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or sequence_column not in reader.fieldnames:
            columns = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"Missing sequence column {sequence_column!r}. Available columns: {columns}"
            )
        for row in reader:
            total_rows += 1
            raw = row.get(sequence_column)
            sequence = normalized_sequence(raw) if raw is not None else ""
            if not sequence:
                invalid_rows += 1
                continue
            if unique:
                if sequence in seen:
                    continue
                seen.add(sequence)
            lengths.append(len(sequence))
    return lengths, total_rows, invalid_rows


def make_summary(lengths: Iterable[int]) -> tuple[dict[str, float], list[dict[str, object]]]:
    values = sorted(int(value) for value in lengths)
    if not values:
        raise ValueError("No valid protein sequences were found")

    def percentile(percent: float) -> float:
        """Linear interpolation, matching NumPy's default percentile method."""
        if len(values) == 1:
            return float(values[0])
        position = (len(values) - 1) * percent / 100.0
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        fraction = position - lower
        return values[lower] + fraction * (values[upper] - values[lower])

    stats = {
        "min": float(values[0]),
        "max": float(values[-1]),
        "mean": float(statistics.mean(values)),
        "median": percentile(50),
        "P25": percentile(25),
        "P75": percentile(75),
        "P90": percentile(90),
        "P95": percentile(95),
        "P99": percentile(99),
    }

    ranges = [
        ("<128", lambda value: value < 128),
        ("128-512", lambda value: 128 <= value < 512),
        ("512-1022", lambda value: 512 <= value <= 1022),
        (">1022", lambda value: value > 1022),
    ]
    table = [
        {
            "Length range": label,
            "# proteins": sum(predicate(value) for value in values),
            "Percentage": sum(predicate(value) for value in values) / len(values) * 100.0,
        }
        for label, predicate in ranges
    ]
    return stats, table


def print_report(
    path: Path,
    lengths: List[int],
    stats: dict[str, float],
    table: list[dict[str, object]],
    total_rows: int,
    invalid_rows: int,
    unique: bool,
) -> None:
    unit = "unique sequences" if unique else "valid sequence records"
    print(f"Input: {path}")
    print(f"Total CSV rows: {total_rows}")
    print(f"Invalid/empty sequences: {invalid_rows}")
    print(f"Counted {unit}: {len(lengths)}")
    print("\nLength statistics (aa)")
    for key in ("min", "max", "mean", "median", "P25", "P75", "P90", "P95", "P99"):
        value = stats[key]
        print(f"{key:>6}: {value:.2f}" if key not in {"min", "max"} else f"{key:>6}: {value:.0f}")

    print("\n| Length range | # proteins | Percentage |")
    print("| ------------ | ---------- | ---------- |")
    for row in table:
        print(f"| {row['Length range']:<12} | {row['# proteins']:>10} | {row['Percentage']:>9.1f}% |")


def main() -> None:
    args = parse_args()
    path = Path(args.csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    lengths, total_rows, invalid_rows = read_lengths(path, args.sequence_column, args.unique_sequences)
    stats, table = make_summary(lengths)
    print_report(path, lengths, stats, table, total_rows, invalid_rows, args.unique_sequences)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Length range", "# proteins", "Percentage"])
            writer.writeheader()
            writer.writerows(table)
        print(f"\nSaved table: {args.output}")


if __name__ == "__main__":
    main()
