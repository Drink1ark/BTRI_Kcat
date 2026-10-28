"""Audit TurnUp features and predictions against their recorded source CSV rows.

Writes a split-count report, row manifest, validation predictions with stable
record keys, and fold-overlap counts. No pretrained models are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "features/P128_C32/TurnUp"
TASKS = {
    "cold_enzyme": ("turnup/cold_enzyme", "_en_with_sub"),
    "cold_reaction": ("turnup/cold_reaction", "_with_sub"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def text_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    require(column in frame, f"Missing {column} column")
    return frame[column].fillna("").astype(str).to_numpy()


def resolved_reaction(row: pd.Series) -> str:
    reaction = row.get("reaction_smiles")
    if isinstance(reaction, str) and reaction.strip() and reaction.strip().lower() != "nan":
        return reaction.strip()
    reactants = row.get("reactant_smiles")
    products = row.get("product_smiles")
    left = reactants.strip() if isinstance(reactants, str) else ""
    right = products.strip() if isinstance(products, str) else ""
    require(bool(left or right), "Cannot reconstruct empty source reaction")
    return f"{left}>>{right}"


def source_signature(frame: pd.DataFrame) -> list[tuple]:
    columns = ("Reaction ID", "Sequence ID", "sequence", "log10kcat_max")
    require(all(column in frame for column in columns), "Source signature columns are missing")
    return list(frame.loc[:, columns].itertuples(index=False, name=None))


def audit_split(
    task: str, fold: int, phase: str, source_root: Path,
    feature_root: Path, results_root: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    fallback_subdir, suffix = TASKS[task]
    stem = f"kcat_{phase}_fold_{fold}{suffix}"
    feature_dir = feature_root / task / stem
    require(feature_dir.is_dir(), f"Missing feature split: {feature_dir}")
    stats = json.loads((feature_dir / "stats.json").read_text(encoding="utf-8"))
    recorded_tasks = [entry for entry in stats.get("config", {}).get("tasks", ())
                      if entry.get("task_name") == task]
    require(len(recorded_tasks) <= 1, f"{stem}: ambiguous source task configuration")
    source_task = recorded_tasks[0] if recorded_tasks else {}
    csv_subdir = source_task.get("csv_subdir", fallback_subdir)
    if "files" in source_task:
        require(f"{stem}.csv" in source_task["files"], f"{stem}: absent from recorded source files")
    csv_path = source_root / csv_subdir / f"{stem}.csv"
    require(csv_path.is_file(), f"Missing source CSV: {csv_path}")
    source = pd.read_csv(csv_path, low_memory=False)
    meta = pd.read_parquet(feature_dir / "meta.parquet")
    labels = np.load(feature_dir / "labels.npy", mmap_mode="r")
    n = len(meta)
    require(stats.get("input_rows") == len(source), f"{stem}: source count mismatch")
    require(stats.get("kept_rows") == n == stats.get("num_samples"), f"{stem}: retained count mismatch")
    dropped = len(source) - n
    require(stats.get("dropped_rows") == dropped, f"{stem}: exclusion count mismatch")
    failures = stats.get("fail_counts", {})
    require(sum(failures.values()) == dropped, f"{stem}: exclusion reasons do not sum to dropped rows")
    require(list(labels.shape) == [n], f"{stem}: label shape mismatch")

    shapes = {}
    for name in ("protein", "rxnfp", "chemberta"):
        arr = np.load(feature_dir / f"{name}_tokens.npy", mmap_mode="r")
        shapes[name] = list(arr.shape)
        require(arr.shape[0] == n, f"{stem}: {name} row count mismatch")
        require(shapes[name] == stats[f"{name}_tokens_shape"], f"{stem}: {name} stats shape mismatch")
        del arr

    require("sample_idx" in meta, f"{stem}: no sample_idx")
    sample_idx = meta["sample_idx"].to_numpy(dtype=np.int64)
    require(np.array_equal(sample_idx, np.arange(n)), f"{stem}: sample_idx order mismatch")
    if "source_row_index" in meta:
        source_idx = meta["source_row_index"].to_numpy(dtype=np.int64)
    elif dropped == 0:
        source_idx = np.arange(n, dtype=np.int64)
    else:
        # Legacy metadata did not persist source row numbers. These four
        # unmodified source columns must uniquely identify every raw row;
        # otherwise it would be unsound to guess which duplicate was retained.
        raw_signatures = source_signature(source)
        require(len(set(raw_signatures)) == len(source),
                f"{stem}: duplicate source signatures prevent unique row mapping")
        raw_lookup = {signature: index for index, signature in enumerate(raw_signatures)}
        kept_signatures = source_signature(meta)
        require(all(signature in raw_lookup for signature in kept_signatures),
                f"{stem}: retained record absent from source")
        source_idx = np.asarray([raw_lookup[signature] for signature in kept_signatures],
                                dtype=np.int64)
    require(np.all((source_idx >= 0) & (source_idx < len(source))), f"{stem}: source row out of bounds")
    require(np.all(np.diff(source_idx) > 0), f"{stem}: source rows are not unique and ordered")
    selected = source.iloc[source_idx].reset_index(drop=True)

    for column in ("Reaction ID", "Sequence ID", "sequence"):
        require(np.array_equal(text_column(selected, column), text_column(meta, column)),
                f"{stem}: {column} differs from source")
    reactions = np.asarray([resolved_reaction(row) for _, row in selected.iterrows()])
    require(np.array_equal(reactions, text_column(meta, "reaction_smiles")),
            f"{stem}: metadata reaction differs from source/fallback")
    source_labels = pd.to_numeric(selected["log10kcat_max"], errors="raise").to_numpy(dtype=np.float32)
    meta_labels = pd.to_numeric(meta["log10kcat_max"], errors="raise").to_numpy(dtype=np.float32)
    require(np.array_equal(source_labels, np.asarray(labels)), f"{stem}: feature labels differ from source")
    require(np.array_equal(meta_labels, np.asarray(labels)), f"{stem}: metadata labels differ from features")

    keys = np.asarray([f"{task}/{stem}/{idx}" for idx in source_idx])
    require(len(set(keys)) == n, f"{stem}: duplicate record keys")
    if "record_key" in meta:
        require(np.array_equal(keys, text_column(meta, "record_key")),
                f"{stem}: stored record keys differ from source rows")
    source_sha = sha256_file(csv_path)
    if "source_csv_sha256" in stats:
        require(source_sha == stats["source_csv_sha256"],
                f"{stem}: source CSV content changed since preprocessing")
    manifest = pd.DataFrame({
        "task": task, "fold": fold, "phase": phase, "split": stem,
        "source_csv_sha256": source_sha,
        "record_key": keys, "source_row_index": source_idx,
        "sample_idx": sample_idx,
        "protein_row": sample_idx, "rxnfp_row": sample_idx,
        "chemberta_row": sample_idx, "label_row": sample_idx,
        "Reaction ID": text_column(meta, "Reaction ID"),
        "Sequence ID": text_column(meta, "Sequence ID"),
        "sequence_sha256": [sha256_text(x) for x in text_column(meta, "sequence")],
        "reaction_sha256": [sha256_text(x) for x in reactions],
        "label": np.asarray(labels),
    })

    enriched = None
    prediction_rows = ""
    if phase == "val":
        pred_path = results_root / task / f"{task}_fold_{fold}" / f"ensemble_predictions_{stem}.csv"
        require(pred_path.is_file(), f"Missing prediction output: {pred_path}")
        pred = pd.read_csv(pred_path, low_memory=False)
        prediction_rows = len(pred)
        require(prediction_rows == n, f"{stem}: prediction count mismatch")
        require(np.array_equal(pred["sample_idx"].to_numpy(dtype=np.int64), sample_idx),
                f"{stem}: prediction sample_idx mismatch")
        for column in ("Reaction ID", "Sequence ID", "sequence", "reaction_smiles"):
            require(np.array_equal(text_column(pred, column), text_column(meta, column)),
                    f"{stem}: prediction {column} mismatch")
        y_true = pred["y_true"].to_numpy(dtype=np.float64)
        y_pred = pred["y_pred"].to_numpy(dtype=np.float64)
        require(np.allclose(y_true, labels, rtol=1e-6, atol=1e-6), f"{stem}: prediction labels mismatch")
        require(np.isfinite(y_pred).all(), f"{stem}: nonfinite prediction")
        seed_cols = sorted((c for c in pred if c.startswith("pred_seed_")),
                           key=lambda x: int(x.removeprefix("pred_seed_")))
        if seed_cols:
            seed_preds = pred[seed_cols].to_numpy(dtype=np.float64)
            require(np.isfinite(seed_preds).all(), f"{stem}: nonfinite seed prediction")
            require(np.allclose(y_pred, seed_preds.mean(axis=1), rtol=1e-5, atol=1e-5),
                    f"{stem}: ensemble differs from seed mean")
        enriched = manifest.drop(columns=["phase", "label"]).copy()
        enriched["prediction_row"] = sample_idx
        enriched["y_true"] = y_true
        enriched["y_pred"] = y_pred
        for column in seed_cols:
            enriched[column] = pred[column].to_numpy(dtype=np.float64)

    try:
        source_name = str(csv_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        source_name = str(csv_path.resolve())
    report = {
        "task": task, "fold": fold, "phase": phase, "split": stem,
        "source_csv": source_name,
        "source_csv_sha256": source_sha,
        "source_rows": len(source), "retained_rows": n, "dropped_rows": dropped,
        "exclusion_reasons": json.dumps(failures, sort_keys=True),
        "protein_shape": json.dumps(shapes["protein"]),
        "rxnfp_shape": json.dumps(shapes["rxnfp"]),
        "chemberta_shape": json.dumps(shapes["chemberta"]),
        "prediction_rows": prediction_rows,
        "row_alignment_check": "pass",
    }
    return report, manifest, enriched, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "both"], default="both")
    parser.add_argument("--source-root", type=Path,
                        default=PROJECT_ROOT / "data/PMAK-DB")
    parser.add_argument("--feature-root", type=Path,
                        default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--results-root", type=Path,
                        default=PROJECT_ROOT / "results/TurnUp")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "audit_results/TurnUp")
    args = parser.parse_args()

    reports, manifests, predictions, overlaps = [], [], [], []
    for task in (TASKS if args.task == "both" else (args.task,)):
        for fold in range(1, 6):
            fold_meta = {}
            for phase in ("train", "val"):
                report, manifest, enriched, meta = audit_split(
                    task, fold, phase, args.source_root, args.feature_root, args.results_root
                )
                reports.append(report)
                manifests.append(manifest)
                fold_meta[phase] = meta
                if enriched is not None:
                    predictions.append(enriched)
            train, val = fold_meta["train"], fold_meta["val"]
            overlaps.append({
                "task": task, "fold": fold,
                "shared_reaction_ids": len(set(text_column(train, "Reaction ID")) &
                                           set(text_column(val, "Reaction ID"))),
                "shared_exact_sequences": len(set(text_column(train, "sequence")) &
                                              set(text_column(val, "sequence"))),
                "shared_reaction_smiles": len(set(text_column(train, "reaction_smiles")) &
                                              set(text_column(val, "reaction_smiles"))),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports).to_csv(args.output_dir / "split_audit.csv", index=False)
    pd.concat(manifests, ignore_index=True).to_csv(args.output_dir / "record_manifest.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / "validation_predictions.csv", index=False
    )
    pd.DataFrame(overlaps).to_csv(args.output_dir / "fold_overlap.csv", index=False)
    print(f"Audited {len(reports)} splits, {sum(r['retained_rows'] for r in reports)} "
          f"feature rows, {sum(r['prediction_rows'] or 0 for r in reports)} "
          f"validation predictions -> {args.output_dir}")


if __name__ == "__main__":
    main()
