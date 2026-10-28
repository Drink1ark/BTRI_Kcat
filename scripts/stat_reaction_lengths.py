#!/usr/bin/env python3
"""Analyze reactant, product, and full-reaction lengths in a reaction CSV.

SMILES character length is always reported. If RDKit is installed, the script
also reports heavy-atom counts, which are a more chemistry-oriented measure of
molecular size. For each reaction, total side length is reactants + products;
full reaction SMILES length additionally includes the ``>>`` separator.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_INPUT = str(Path(__file__).resolve().parents[1] / "data/CatPred-DB/kcat-random_train.csv")
PERCENTILES = (25, 75, 90, 95, 99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_INPUT, help="Input CSV file")
    parser.add_argument("--reactant-column", default="reactant_smiles")
    parser.add_argument("--product-column", default="product_smiles")
    parser.add_argument("--reaction-column", default="reaction_smiles")
    parser.add_argument("--reaction-source", choices=("auto", "reaction", "sides"), default="auto",
                        help="auto prefers a nonempty reaction column, then falls back to reactant/product columns")
    parser.add_argument("--output", type=Path, help="Optional path for the statistics CSV")
    return parser.parse_args()


def clean(value: str | None) -> str:
    return "" if value is None else "".join(value.split())


def percentile(sorted_values: list[int], p: int) -> float:
    """Linear interpolation, matching NumPy's default percentile method."""
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def summarize(values: Iterable[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    result: dict[str, float | int] = {
        "n": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
        "median": percentile(ordered, 50),
    }
    result.update({f"P{p}": percentile(ordered, p) for p in PERCENTILES})
    return result


def read_measurements(args: argparse.Namespace):
    measurements: dict[str, list[int]] = {
        "Reactant side | SMILES characters": [],
        "Product side | SMILES characters": [],
        "Total sides | SMILES characters": [],
        "Full reaction | SMILES characters incl. >>": [],
    }
    invalid_reaction_rows = 0
    missing_reactant_rows = 0
    missing_product_rows = 0

    try:
        from rdkit import Chem
    except ImportError:
        Chem = None
    if Chem is not None:
        measurements.update({
            "Reactant side | heavy atoms": [],
            "Product side | heavy atoms": [],
            "Total sides | heavy atoms": [],
        })
    atom_parse_failures = {"reactant": 0, "product": 0}

    def heavy_atom_count(side: str, side_name: str) -> int | None:
        if Chem is None:
            return None
        if not side:
            return 0
        total = 0
        for smiles in side.split("."):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                atom_parse_failures[side_name] += 1
                return None
            total += molecule.GetNumHeavyAtoms()
        return total

    with Path(args.csv_path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        has_sides = args.reactant_column in fields and args.product_column in fields
        if args.reaction_source == "sides" and not has_sides:
            raise ValueError("Selected reactant/product columns are missing")
        if args.reaction_source == "reaction" and args.reaction_column not in fields:
            raise ValueError(f"Selected reaction column is missing: {args.reaction_column}")
        if not has_sides and args.reaction_column not in fields:
            raise ValueError("CSV must contain reaction or reactant/product columns")

        for row in reader:
            reaction = clean(row.get(args.reaction_column)) if args.reaction_source != "sides" else ""
            if ">>" in reaction:
                reactants, products = reaction.split(">>", 1)
            elif has_sides and args.reaction_source != "reaction":
                reactants = clean(row.get(args.reactant_column))
                products = clean(row.get(args.product_column))
                reaction = f"{reactants}>>{products}"
            else:
                invalid_reaction_rows += 1
                continue
            if not reactants:
                missing_reactant_rows += 1
            if not products:
                missing_product_rows += 1

            reactant_chars = len(reactants)
            product_chars = len(products)
            measurements["Reactant side | SMILES characters"].append(reactant_chars)
            measurements["Product side | SMILES characters"].append(product_chars)
            measurements["Total sides | SMILES characters"].append(reactant_chars + product_chars)
            measurements["Full reaction | SMILES characters incl. >>"].append(len(reaction))

            reactant_atoms = heavy_atom_count(reactants, "reactant")
            product_atoms = heavy_atom_count(products, "product")
            if reactant_atoms is not None:
                measurements["Reactant side | heavy atoms"].append(reactant_atoms)
            if product_atoms is not None:
                measurements["Product side | heavy atoms"].append(product_atoms)
            if reactant_atoms is not None and product_atoms is not None:
                measurements["Total sides | heavy atoms"].append(reactant_atoms + product_atoms)

    metadata = {
        "rows": sum(len(values) for key, values in measurements.items() if key == "Reactant side | SMILES characters"),
        "invalid_reaction_rows": invalid_reaction_rows,
        "missing_reactant_rows": missing_reactant_rows,
        "missing_product_rows": missing_product_rows,
        "rdkit_available": Chem is not None,
        "atom_parse_failures": atom_parse_failures,
    }
    return measurements, metadata


def main() -> None:
    args = parse_args()
    path = Path(args.csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    measurements, metadata = read_measurements(args)

    print(f"Input: {path}")
    print(f"Valid reaction rows: {metadata['rows']}")
    print(f"Rows without reactants: {metadata['missing_reactant_rows']}")
    print(f"Rows without products: {metadata['missing_product_rows']}")
    print(f"Rows with invalid reaction format: {metadata['invalid_reaction_rows']}")
    if not metadata["rdkit_available"]:
        print("RDKit unavailable; heavy-atom statistics skipped.")
    elif any(metadata["atom_parse_failures"].values()):
        print(f"RDKit component parse failures: {metadata['atom_parse_failures']}")

    rows = []
    for metric, values in measurements.items():
        stats = summarize(values)
        if not stats["n"]:
            continue
        row = {"Metric": metric, **stats}
        rows.append(row)
        print(f"\n{metric} (n={stats['n']})")
        for key in ("min", "max", "mean", "median", "P25", "P75", "P90", "P95", "P99"):
            value = stats.get(key)
            if value is not None:
                print(f"{key:>6}: {value:.2f}" if key not in {"min", "max"} else f"{key:>6}: {value}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        columns = ["Metric", "n", "min", "max", "mean", "median", "P25", "P75", "P90", "P95", "P99"]
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved statistics: {args.output}")


if __name__ == "__main__":
    main()
