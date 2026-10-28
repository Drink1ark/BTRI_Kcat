"""
Preprocess features for the bidirectional cross-attention kcat model.

Outputs per split
-----------------
protein_tokens.npy    : [N, P, protein_feature_dim] - ESM2 or ProtT5 tokens
rxnfp_tokens.npy      : [N, 1, 256]
chemberta_tokens.npy  : [N, R, mol_hidden_size] - ChemBERTa tokens
labels.npy            : [N]
meta.parquet          : metadata aligned with saved arrays
stats.json            : filtering stats and tensor shapes

Protein and ChemBERTa representations are pooled to fixed segment counts here.
The downstream F4 model then learns attention and pooling over these features.


Two useful modes
----------------
Edit PreprocessConfig below, then run this file directly.
By default, the task is both TurnUp cold datasets, including five-fold
train/validation files and independent full-pool/test files from PMAK.

1) Full preprocessing from CSVs:
   leave reuse_feature_dir empty.

2) Fast upgrade from existing old token_features:
   set reuse_feature_dir to the old feature directory and output_dir to a
   different new feature directory.

Mode 2 reuses existing protein_tokens.npy, rxnfp_tokens.npy, labels.npy and
meta.parquet, and only runs ChemBERTa forward passes to create
chemberta_tokens.npy. It does not rerun ESM2 or RXNFP.
"""

from __future__ import annotations

import argparse

import gc
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch.nn.functional as F

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DATA_DIR = str(PROJECT_DIR / "data")
DEFAULT_FEATURE_ROOT = str(PROJECT_DIR / "features")
DEFAULT_OUTPUT_DIR = str(PROJECT_DIR / "features" / "P128_C32")
DEFAULT_TURNUP_DATA_ROOT = str(
    PROJECT_DIR.parents[1] / "PMAK-master" / "supplement" / "data" / "turnup"
)
DEFAULT_ESM_MODEL_ID = "facebook/esm2_t36_3B_UR50D"
DEFAULT_PROTEIN_MODEL_ID = "Rostlab/prot_t5_xl_uniref50"
DEFAULT_MOL_MODEL_ID = "DeepChem/ChemBERTa-77M-MLM"


def default_esm_model_name() -> str:
    return DEFAULT_ESM_MODEL_ID


# Edit this catalog to change the CSVs for a task. The active tasks are chosen
# by PreprocessConfig.dataset / turnup_dataset below (default: both cold tasks).
DEFAULT_TASKS = (
    {
        "task_name": "CatPred-DB",
        "csv_subdir": "CatPred-DB",
        "files": (
            "kcat-random_train.csv", "kcat-random_val.csv", "kcat-random_test.csv",
        ),
    },
    {
        "task_name": "PMAK-DB",
        "csv_subdir": "PMAK-DB/catpred",
        "files": ("train.csv", "test.csv", "OOD.csv"),
    },
    {
        "task_name": "TurnUp/cold_enzyme",
        "csv_subdir": str(Path(DEFAULT_TURNUP_DATA_ROOT) / "cold_enzyme"),
        "files": (
            "kcat_train_fold_1_en_with_sub.csv", "kcat_val_fold_1_en_with_sub.csv",
            "kcat_train_fold_2_en_with_sub.csv", "kcat_val_fold_2_en_with_sub.csv",
            "kcat_train_fold_3_en_with_sub.csv", "kcat_val_fold_3_en_with_sub.csv",
            "kcat_train_fold_4_en_with_sub.csv", "kcat_val_fold_4_en_with_sub.csv",
            "kcat_train_fold_5_en_with_sub.csv", "kcat_val_fold_5_en_with_sub.csv",
            "kcat-random_train.csv", "kcat-random_test.csv",
        ),
    },
    {
        "task_name": "TurnUp/cold_reaction",
        "csv_subdir": str(Path(DEFAULT_TURNUP_DATA_ROOT) / "cold_reaction_full"),
        "files": (
            "kcat_train_fold_1_with_sub.csv", "kcat_val_fold_1_with_sub.csv",
            "kcat_train_fold_2_with_sub.csv", "kcat_val_fold_2_with_sub.csv",
            "kcat_train_fold_3_with_sub.csv", "kcat_val_fold_3_with_sub.csv",
            "kcat_train_fold_4_with_sub.csv", "kcat_val_fold_4_with_sub.csv",
            "kcat_train_fold_5_with_sub.csv", "kcat_val_fold_5_with_sub.csv",
            "kcat-random_train.csv", "kcat-random_test.csv",
        ),
    },
)

@dataclass
class PreprocessConfig:
    base_data_dir: str = DEFAULT_BASE_DATA_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    dataset: str = "turnup"  # Edit here: turnup, catpred, pmak, or all
    turnup_dataset: str = "both"
    turnup_data_root: str = DEFAULT_TURNUP_DATA_ROOT

    # ESM2‑3B
    esm_model_name: str = default_esm_model_name()
    esm_load_in_8bit: bool = False
    protein_encoder: str = "esm"
    protein_model_name: str = DEFAULT_PROTEIN_MODEL_ID
    protein_feature_dim: int = 2560

    # ChemBERTa‑2 MLM版本（推荐用于你的kcat项目）
    mol_model_path: str = DEFAULT_MOL_MODEL_ID

    reuse_feature_dir: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    max_seq_len: int = 1022
    protein_long_mode: str = "sliding_window"  # truncate or sliding_window
    protein_window_stride: int = 768
    protein_segments: int = 128
    protein_layer: int = 36
    chemberta_segments: int = 32
    chemberta_input_mode: str = "separate"  # merged or separate
    mol_hidden_size: int = 384
    save_dtype: str = "float16"  # float16 or float32
    mol_max_length: int = 512  # maximum tokens per ChemBERTa window, including special tokens
    mol_chunk_batch_size: int = 8
    label_column: str = "log10kcat_max"
    reaction_column: str = "reaction_smiles"
    sequence_column: str = "sequence"

    use_cache: bool = True
    cache_protein: bool = True
    cache_rxnfp: bool = True
    cache_chemberta: bool = True
    fail_example_limit: int = 10
    task_filter: str = "all"
    skip_existing: bool = True  # Resume compatible feature extraction safely
    check_only: bool = False
    print_config: bool = False
    tasks: Tuple[Dict[str, Any], ...] = field(default_factory=lambda: DEFAULT_TASKS)

    @property
    def np_save_dtype(self):
        if self.save_dtype == "float16":
            return np.float16
        if self.save_dtype == "float32":
            return np.float32
        raise ValueError("save_dtype must be 'float16' or 'float32'")


_model_esm = None
_esm_tokenizer = None
_protein_model = None
_protein_tokenizer = None
_protein_hidden_size = None
_rxnfp_generator = None
_mol_tokenizer = None
_mol_model = None


def _torch_device(cfg: PreprocessConfig) -> torch.device:
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return device


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _jsonable_cfg(cfg: PreprocessConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    # Keep config serialization consistent with selected_tasks(): callers may
    # provide either one task dictionary or a tuple/list of task dictionaries.
    d["tasks"] = selected_tasks(cfg)
    return d


def _check_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")


def validate_config(cfg: PreprocessConfig) -> None:
    _ = cfg.np_save_dtype
    if cfg.esm_load_in_8bit:
        raise ValueError("esm_load_in_8bit is not implemented in this pipeline; set it to False")
    if cfg.dataset not in {"catpred", "pmak", "turnup", "all"}:
        raise ValueError("dataset must be catpred, pmak, turnup or all")
    if cfg.turnup_dataset not in {"cold_enzyme", "cold_reaction", "both"}:
        raise ValueError("turnup_dataset must be cold_enzyme, cold_reaction or both")
    if cfg.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if cfg.protein_long_mode not in {"truncate", "sliding_window"}:
        raise ValueError("protein_long_mode must be 'truncate' or 'sliding_window'")
    if cfg.protein_window_stride <= 0:
        raise ValueError("protein_window_stride must be positive")
    if cfg.protein_window_stride > cfg.max_seq_len:
        raise ValueError("protein_window_stride must be <= max_seq_len")
    if cfg.protein_segments <= 0:
        raise ValueError("protein_segments must be positive")
    if cfg.protein_layer < -1:
        raise ValueError("protein_layer must be -1 or non-negative")
    if cfg.protein_encoder not in {"esm", "prot_t5"}:
        raise ValueError("protein_encoder must be esm or prot_t5")
    if cfg.protein_feature_dim <= 0:
        raise ValueError("protein_feature_dim must be positive")
    if cfg.chemberta_segments <= 0:
        raise ValueError("chemberta_segments must be positive")
    if cfg.chemberta_input_mode not in {"merged", "separate"}:
        raise ValueError("chemberta_input_mode must be 'merged' or 'separate'")
    if cfg.chemberta_input_mode == "separate" and cfg.chemberta_segments % 2 != 0:
        raise ValueError("chemberta_segments must be even when chemberta_input_mode='separate'")
    if cfg.mol_max_length <= 0:
        raise ValueError("mol_max_length must be positive")
    if cfg.mol_chunk_batch_size <= 0:
        raise ValueError("mol_chunk_batch_size must be positive")
    if cfg.mol_hidden_size <= 0:
        raise ValueError("mol_hidden_size must be positive")
    if cfg.reuse_feature_dir and Path(cfg.reuse_feature_dir).resolve() == Path(cfg.output_dir).resolve():
        raise ValueError("reuse_feature_dir and output_dir must be different to avoid overwriting source features")
    selected_tasks(cfg)


def _parse_finite_label(value: Any) -> float:
    if pd.isna(value):
        raise ValueError("label is NaN")
    label = float(value)
    if not np.isfinite(label):
        raise ValueError("label is not finite")
    return label


def _sequence_metadata(sequence: Any, cfg: PreprocessConfig) -> Dict[str, Union[int, bool, str]]:
    raw = str(sequence).strip().upper()
    raw_no_ws = re.sub(r"\s+", "", raw)
    sanitized_full = re.sub(r"[UZOB]", "X", raw_no_ws)
    sanitized_full = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", sanitized_full)
    sanitized_len = len(sanitized_full)
    window_count = _protein_window_count(sanitized_len, cfg)
    if cfg.protein_long_mode == "sliding_window":
        used_len = sanitized_len
        was_truncated = False
    else:
        used_len = min(sanitized_len, cfg.max_seq_len)
        was_truncated = sanitized_len > cfg.max_seq_len
    return {
        "sequence_raw_length": int(len(raw_no_ws)),
        "sequence_sanitized_length": int(sanitized_len),
        "sequence_used_length": int(used_len),
        "sequence_was_truncated": bool(was_truncated),
        "protein_long_mode": cfg.protein_long_mode,
        "protein_window_count": int(window_count),
    }


def _assert_shape(name: str, arr: np.ndarray, expected_shape: Tuple[int, ...]) -> None:
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {tuple(arr.shape)}")


def _duplicate_label_stats(meta_df: pd.DataFrame, cfg: PreprocessConfig) -> Dict[str, Any]:
    keys = [cfg.sequence_column, cfg.reaction_column]
    if cfg.label_column not in meta_df.columns or any(k not in meta_df.columns for k in keys):
        return {}
    grouped = meta_df.groupby(keys, dropna=False)[cfg.label_column]
    group_sizes = grouped.size()
    label_ranges = grouped.agg(lambda s: float(np.max(s.astype(float)) - np.min(s.astype(float))))
    conflict_mask = (group_sizes > 1) & (label_ranges > 1e-6)
    return {
        "duplicate_sequence_reaction_pairs": int((group_sizes > 1).sum()),
        "conflicting_duplicate_labels": int(conflict_mask.sum()),
        "max_duplicate_label_range": float(label_ranges.max()) if len(label_ranges) else 0.0,
    }


def _add_failure(
    fail_counts: Counter,
    fail_examples: Dict[str, List[Dict[str, str]]],
    reason: str,
    row_idx: Any,
    message: str,
    cfg: PreprocessConfig,
) -> None:
    fail_counts[reason] += 1
    if len(fail_examples[reason]) < cfg.fail_example_limit:
        fail_examples[reason].append({"row_index": str(row_idx), "message": str(message)[:500]})


def _sanitize_sequence_full(sequence: Any) -> str:
    seq = str(sequence).strip().upper()
    seq = re.sub(r"\s+", "", seq)
    seq = re.sub(r"[UZOB]", "X", seq)
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", seq)
    return seq


def _sanitize_sequence(sequence: str, max_len: int) -> str:
    return _sanitize_sequence_full(sequence)[:max_len]


def _protein_windows(seq: str, cfg: PreprocessConfig) -> List[str]:
    if len(seq) <= cfg.max_seq_len or cfg.protein_long_mode == "truncate":
        return [seq[:cfg.max_seq_len]]
    windows = []
    start = 0
    while start < len(seq):
        end = min(start + cfg.max_seq_len, len(seq))
        windows.append(seq[start:end])
        if end == len(seq):
            break
        start += cfg.protein_window_stride
    return windows


def _protein_window_count(seq_len: int, cfg: PreprocessConfig) -> int:
    if seq_len <= 0:
        return 0
    if seq_len <= cfg.max_seq_len or cfg.protein_long_mode == "truncate":
        return 1
    return int(np.ceil((seq_len - cfg.max_seq_len) / cfg.protein_window_stride)) + 1


def _sequence_cache_key(sequence: Any, cfg: PreprocessConfig) -> str:
    seq = _sanitize_sequence_full(sequence)
    if cfg.protein_long_mode == "truncate":
        seq = seq[:cfg.max_seq_len]
    return f"{cfg.protein_long_mode}|{cfg.max_seq_len}|{cfg.protein_window_stride}|{seq}"


def _validate_reaction_smiles(reaction_smiles: str) -> str:
    if not isinstance(reaction_smiles, str):
        raise ValueError("reaction_smiles is not a string")
    rxn = reaction_smiles.strip()
    if not rxn or ">>" not in rxn:
        raise ValueError("reaction_smiles must contain '>>'")
    return rxn


def _resolve_row_reaction(row: Any, cfg: PreprocessConfig) -> str:
    """Return a reaction SMILES, rebuilding it when source CSV leaves it blank.

    The TurnUp cold-reaction files store reactants and products in separate
    columns while leaving ``reaction_smiles`` empty.  RXNFP and ChemBERTa need
    the canonical ``reactants>>products`` representation, so construct it at
    preprocessing time and persist it in the output metadata.
    """
    reaction = row.get(cfg.reaction_column) if hasattr(row, "get") else None
    if isinstance(reaction, str) and reaction.strip() and reaction.strip().lower() != "nan":
        return reaction.strip()

    reactants = row.get("reactant_smiles") if hasattr(row, "get") else None
    products = row.get("product_smiles") if hasattr(row, "get") else None
    reactants = reactants.strip() if isinstance(reactants, str) else ""
    products = products.strip() if isinstance(products, str) else ""
    if reactants or products:
        return f"{reactants}>>{products}"
    raise ValueError(
        f"{cfg.reaction_column} is empty and reactant_smiles/product_smiles "
        "cannot reconstruct a reaction"
    )


def selected_tasks(cfg: PreprocessConfig) -> List[Dict[str, Any]]:
    # Accept a single task dictionary as a convenience.  The canonical form
    # remains a tuple/list of dictionaries, which is useful when preprocessing
    # both cold-enzyme and cold-reaction datasets in one run.
    tasks = cfg.tasks
    if isinstance(tasks, dict):
        tasks = (tasks,)
    if not isinstance(tasks, (tuple, list)):
        raise TypeError(
            "PreprocessConfig.tasks must be a task dict or a tuple/list of task dicts; "
            f"got {type(tasks).__name__}"
        )
    invalid = [type(task).__name__ for task in tasks if not isinstance(task, dict)]
    if invalid:
        raise TypeError(
            "Each item in PreprocessConfig.tasks must be a dictionary with "
            "task_name, csv_subdir, and files; "
            f"got item types: {invalid}"
        )
    if cfg.task_filter == "all":
        return [dict(t) for t in tasks]
    out = [dict(t) for t in tasks if t["task_name"] == cfg.task_filter]
    if not out:
        raise ValueError(f"Unknown task: {cfg.task_filter}")
    return out


# ===================== Model loading =====================
def init_models(cfg: PreprocessConfig, load_esm_rxnfp: bool = True, load_chemberta: bool = True) -> None:
    global _model_esm, _esm_tokenizer, _protein_model, _protein_tokenizer
    global _protein_hidden_size, _rxnfp_generator, _mol_tokenizer, _mol_model

    device = _torch_device(cfg)
    print(f"Using device: {device}")

    if load_esm_rxnfp:
        from transformers import AutoTokenizer, AutoModel
        from rxnfp.transformer_fingerprints import (
            RXNBERTFingerprintGenerator,
            get_default_model_and_tokenizer,
        )
        # ============================================================
        # 1. Protein encoder. Keep the old ESM globals as aliases so helper
        # code copied from the original preprocessing script remains usable.
        model_name = cfg.esm_model_name if cfg.protein_encoder == "esm" else cfg.protein_model_name
        if cfg.protein_encoder == "prot_t5":
            # Loading T5 through AutoTokenizer selects T5TokenizerFast and
            # attempts a slow-to-fast conversion that requires protobuf.
            # ProtT5's slow tokenizer is sufficient for residue encoding and
            # preserves the legacy behavior used by the existing pipeline.
            from transformers import T5Tokenizer

            _protein_tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                do_lower_case=False,
                legacy=True,
            )
        else:
            _protein_tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs = {"torch_dtype": torch.float32}
        if cfg.protein_encoder == "esm":
            if device.type == "cuda":
                load_kwargs["device_map"] = {"": device}
            _protein_model = AutoModel.from_pretrained(model_name, **load_kwargs)
        elif cfg.protein_encoder == "prot_t5":
            # ProtT5 is a seq2seq T5 checkpoint.  AutoModel may return a
            # decoder-capable T5 model, which requires decoder_input_ids even
            # when we only need residue embeddings.  Load the encoder-only
            # class so sequence_to_segment_tokens can call it directly.
            from transformers import T5EncoderModel

            _protein_model = T5EncoderModel.from_pretrained(model_name, **load_kwargs)
        else:
            _protein_model = AutoModel.from_pretrained(model_name, **load_kwargs)
        if cfg.protein_encoder != "esm" or device.type != "cuda":
            _protein_model = _protein_model.to(device)
        _protein_model.eval()
        _protein_model.requires_grad_(False)
        _protein_hidden_size = int(getattr(_protein_model.config, "hidden_size", 0))
        if _protein_hidden_size != cfg.protein_feature_dim:
            raise ValueError(
                f"protein_feature_dim={cfg.protein_feature_dim} does not match "
                f"{model_name} hidden_size={_protein_hidden_size}"
            )
        _model_esm, _esm_tokenizer = _protein_model, _protein_tokenizer
        print(f"Loaded protein encoder {model_name} with hidden_size={_protein_hidden_size}")
        # ============================================================
        # 2. RXNFP
        # ============================================================
        # RXNFP's helper otherwise moves weights to cuda:0 before we can select
        # the requested GPU. Stage on CPU, then move directly to our device.
        model_rxn, tokenizer_rxn = get_default_model_and_tokenizer(force_no_cuda=True)

        model_rxn = model_rxn.to(device)
        model_rxn.eval()
        model_rxn.requires_grad_(False)

        _rxnfp_generator = RXNBERTFingerprintGenerator(
            model_rxn,
            tokenizer_rxn,
            force_no_cuda=device.type != "cuda",
        )
        _rxnfp_generator.device = device
        print("Loaded RXNFP model")

    if load_chemberta:
        from transformers import AutoModel, AutoTokenizer
        _mol_tokenizer = AutoTokenizer.from_pretrained(cfg.mol_model_path)
        if not getattr(_mol_tokenizer, "is_fast", False):
            raise RuntimeError(
                "ChemBERTa long-input encoding requires a fast tokenizer "
                "to create overflow windows without truncating the input"
            )
        _mol_model = AutoModel.from_pretrained(cfg.mol_model_path, torch_dtype=torch.float32).to(device)
        hidden_size = int(getattr(_mol_model.config, "hidden_size", cfg.mol_hidden_size))
        if hidden_size != cfg.mol_hidden_size:
            raise ValueError(f"ChemBERTa hidden size mismatch: config expects {cfg.mol_hidden_size}, model has {hidden_size}")
        _mol_model.eval()
        _mol_model.requires_grad_(False)
        print(f"Loaded molecular encoder with hidden_size={hidden_size}")

    if load_esm_rxnfp:
        print("protein encoder dtype:", next(_protein_model.parameters()).dtype)
        print("protein encoder device:", next(_protein_model.parameters()).device)


def sequence_to_segment_tokens(
    sequence: str,
    cfg: PreprocessConfig
) -> np.ndarray:
    """Encode a protein with the selected Hugging Face residue encoder."""

    if not isinstance(sequence, str) or len(sequence.strip()) == 0:
        raise ValueError("empty protein sequence")

    if _protein_model is None or _protein_tokenizer is None:
        raise RuntimeError("Protein encoder is not initialized")

    device = _torch_device(cfg)

    seq = _sanitize_sequence_full(sequence)

    if len(seq) == 0:
        raise ValueError("protein sequence is empty after sanitization")

    reps_parts = []
    covered_until = 0

    windows = _protein_windows(seq, cfg)

    for window_idx, window in enumerate(windows):

        if not window:
            continue

        # --------------------------------------------------------
        # Calculate original sequence coordinates
        # --------------------------------------------------------
        if (
            cfg.protein_long_mode == "sliding_window"
            and len(seq) > cfg.max_seq_len
        ):
            window_start = window_idx * cfg.protein_window_stride
        else:
            window_start = 0

        window_end = window_start + len(window)

        # --------------------------------------------------------
        # HuggingFace tokenizer
        # --------------------------------------------------------
        tokenizer_text = " ".join(window) if cfg.protein_encoder == "prot_t5" else window
        inputs = _protein_tokenizer(
            tokenizer_text,
            return_tensors="pt",
            add_special_tokens=True,
        )

        model_device = next(_protein_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # --------------------------------------------------------
        # Encoder forward
        # --------------------------------------------------------
        with torch.inference_mode():

            outputs = _protein_model(
                **inputs,
                output_hidden_states=True,
            )

        # hidden_states[0] is the embedding output and the final entry is the
        # last encoder layer.  ESM2 has BOS/EOS around residues; ProtT5 has
        # EOS only.  The selected layer is handled identically afterward.

        hidden_states = outputs.hidden_states
        layer = cfg.protein_layer if cfg.protein_layer >= 0 else -1
        window_reps = hidden_states[layer][0]

        # Remove special-token representations while preserving exactly one
        # vector per input residue. ProtT5's spaced sequence tokenizes as
        # residues + EOS; ESM2 tokenizes as BOS + residues + EOS.
        if cfg.protein_encoder == "prot_t5":
            if window_reps.shape[0] < len(window):
                raise ValueError(
                    f"ProtT5 representation is shorter than the residue window: "
                    f"expected at least {len(window)}, got {window_reps.shape[0]}"
                )
            window_reps = window_reps[:len(window)]
        elif window_reps.shape[0] == len(window) + 2:
            window_reps = window_reps[1 : 1 + len(window)]
        elif window_reps.shape[0] == len(window) + 1:
            window_reps = window_reps[:len(window)]

        # Safety check
        if window_reps.shape[0] != len(window):
            raise ValueError(
                f"ESM residue length mismatch: "
                f"expected {len(window)}, "
                f"got {window_reps.shape[0]}"
            )

        # --------------------------------------------------------
        # Remove overlap between sliding windows
        # --------------------------------------------------------
        keep_start = max(
            window_start,
            covered_until
        )

        if keep_start < window_end:

            local_start = keep_start - window_start

            reps_parts.append(
                window_reps[local_start:]
            )

            covered_until = window_end

        del inputs
        del outputs
        del window_reps

    # ------------------------------------------------------------
    # Check representations
    # ------------------------------------------------------------
    if not reps_parts:
        raise ValueError(
            "ESM produced no residue representations"
        )

    if len(reps_parts) > 1:
        reps = torch.cat(
            reps_parts,
            dim=0
        )
    else:
        reps = reps_parts[0]

    if reps.size(0) == 0:
        raise ValueError(
            "ESM produced zero residue representations"
        )

    # ------------------------------------------------------------
    # Expected:
    # [L, 2560]
    # ------------------------------------------------------------
    if reps.shape[1] != cfg.protein_feature_dim:
        raise ValueError(
            f"Protein hidden dimension mismatch: expected {cfg.protein_feature_dim}, got {reps.shape[1]}"
        )

    # ------------------------------------------------------------
    # Pool residue representations to fixed protein segments
    # ------------------------------------------------------------
    # [L, 2560]
    #      ↓ transpose
    # [2560, L]
    #      ↓
    # [1, 2560, L]
    # ------------------------------------------------------------

    reps = reps.transpose(0, 1).unsqueeze(0)

    mean_pool = F.adaptive_avg_pool1d(
        reps,
        cfg.protein_segments
    )

    max_pool = F.adaptive_max_pool1d(
        reps,
        cfg.protein_segments
    )

    pooled = 0.5 * mean_pool + 0.5 * max_pool

    # [1, 2560, P]
    #      ↓
    # [P, 2560]

    pooled = (
        pooled
        .squeeze(0)
        .transpose(0, 1)
    )

    arr = (
        pooled
        .detach()
        .cpu()
        .numpy()
        .astype(
            cfg.np_save_dtype,
            copy=False
        )
    )

    _check_finite(
        "protein_tokens",
        arr
    )

    return arr


def get_rxnfp_token(reaction_smiles: str, cfg: PreprocessConfig) -> np.ndarray:
    if _rxnfp_generator is None:
        raise RuntimeError("RXNFP generator is not initialized")
    rxn = _validate_reaction_smiles(reaction_smiles)
    feat = _rxnfp_generator.convert(rxn)
    arr = np.asarray(feat, dtype=cfg.np_save_dtype).reshape(-1)
    if arr.shape != (256,):
        raise ValueError(f"rxnfp_token shape mismatch: expected (256,), got {arr.shape}")
    _check_finite("rxnfp_token", arr)
    return arr


def _empty_mol_segments(n_segments: int, cfg: PreprocessConfig) -> np.ndarray:
    return np.zeros((n_segments, cfg.mol_hidden_size), dtype=cfg.np_save_dtype)


def _encode_chemberta_text_to_segments(
    text: str,
    n_segments: int,
    cfg: PreprocessConfig,) -> np.ndarray:
    """
    Encode ChemBERTa tokens and compress them with mean + max dual pooling.
    pooled = 0.5 * avg_pool + 0.5 * max_pool
    This preprocessing step keeps fixed reaction tokens; downstream learnable
    compression or pooling happens inside the training model.
    """
    if _mol_tokenizer is None or _mol_model is None:
        raise RuntimeError("ChemBERTa model is not initialized")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty ChemBERTa input text")

    device = _torch_device(cfg)
    # Tokenizer overflow windows cover the full input instead of dropping tokens
    # after mol_max_length. A zero stride avoids counting overlap tokens twice.
    inputs = _mol_tokenizer(
        text.strip(),
        truncation=True,
        max_length=cfg.mol_max_length,
        stride=0,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    model_input_keys = {"input_ids", "attention_mask", "token_type_ids"}
    n_chunks = inputs["input_ids"].shape[0]
    hidden_parts = []
    for start in range(0, n_chunks, cfg.mol_chunk_batch_size):
        end = min(start + cfg.mol_chunk_batch_size, n_chunks)
        chunk_inputs = {
            k: inputs[k][start:end].to(device)
            for k in model_input_keys
            if k in inputs
        }
        with torch.inference_mode():
            outputs = _mol_model(**chunk_inputs)

        attention_mask = inputs["attention_mask"][start:end].bool()
        special_mask = inputs["special_tokens_mask"][start:end].bool()
        valid_mask = (attention_mask & (~special_mask)).to(device)
        hidden_parts.append(outputs.last_hidden_state[valid_mask].cpu())

    if not hidden_parts or sum(part.size(0) for part in hidden_parts) == 0:
        raise ValueError("ChemBERTa input has no non-special tokens")
    hidden = torch.cat(hidden_parts, dim=0)

    # Mean + max dual pooling to fixed segment tokens.
    hidden = hidden.transpose(0, 1).unsqueeze(0)  # [1, mol_hidden_size, T_valid]
    avg_pool = F.adaptive_avg_pool1d(hidden, n_segments)
    max_pool = F.adaptive_max_pool1d(hidden, n_segments)  # [1, mol_hidden_size, n_segments]
    pooled = 0.5 * avg_pool + 0.5 * max_pool  # [1, mol_hidden_size, n_segments]
    pooled = pooled.squeeze(0).transpose(0, 1)  # [n_segments, mol_hidden_size]
    
    arr = pooled.detach().cpu().numpy().astype(cfg.np_save_dtype, copy=False)
    _check_finite("chemberta_tokens", arr)
    return arr

def get_chemberta_segment_tokens(reaction_smiles: str, cfg: PreprocessConfig) -> np.ndarray:
    """
    Build ChemBERTa token features for a reaction SMILES string.
    """
    rxn = _validate_reaction_smiles(reaction_smiles)
    reactants, products = rxn.split(">>", 1)
    reactants = reactants.strip()
    products = products.strip()
    if not reactants and not products:
        raise ValueError("reaction_smiles has empty reactants and products")

    if cfg.chemberta_input_mode == "merged":
        merged = f"{reactants}.{products}".strip(".")
        return _encode_chemberta_text_to_segments(
            merged,
            cfg.chemberta_segments,
            cfg,
        )

    if cfg.chemberta_input_mode == "separate":
        half_segments = cfg.chemberta_segments // 2
        if reactants:
            left = _encode_chemberta_text_to_segments(
                reactants,
                half_segments,
                cfg,
            )
        else:
            left = _empty_mol_segments(half_segments, cfg)

        if products:
            right = _encode_chemberta_text_to_segments(
                products,
                half_segments,
                cfg,
            )
        else:
            right = _empty_mol_segments(half_segments, cfg)
        arr = np.concatenate([left, right], axis=0).astype(cfg.np_save_dtype, copy=False)
        _check_finite("chemberta_segment_tokens", arr)
        return arr

    raise ValueError("chemberta_input_mode must be 'merged' or 'separate'")
# ===================== Save helpers =====================
def _save_split(
    save_dir: Path,
    protein_tokens_all: Sequence[np.ndarray],
    rxnfp_tokens_all: Sequence[np.ndarray],
    chemberta_tokens_all: Sequence[np.ndarray],
    labels_all: Sequence[float],
    meta_df: pd.DataFrame,
    cfg: PreprocessConfig,
    extra_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    save_dir.mkdir(parents=True, exist_ok=True)

    num_samples = len(labels_all)
    for name, samples in (
        ("protein_tokens", protein_tokens_all),
        ("rxnfp_tokens", rxnfp_tokens_all),
        ("chemberta_tokens", chemberta_tokens_all),
        ("metadata", meta_df),
    ):
        if len(samples) != num_samples:
            raise ValueError(
                f"{name} sample count mismatch: expected {num_samples}, got {len(samples)}"
            )

    protein_shape = (num_samples, cfg.protein_segments, cfg.protein_feature_dim)
    labels_arr = np.asarray(labels_all, dtype=np.float32)
    protein_tokens_arr = np.lib.format.open_memmap(
        save_dir / "protein_tokens.npy",
        mode="w+",
        dtype=cfg.np_save_dtype,
        shape=protein_shape,
    )
    for index, tokens in enumerate(protein_tokens_all):
        _assert_shape("protein_tokens", tokens, protein_shape[1:])
        _check_finite("protein_tokens", tokens)
        protein_tokens_arr[index] = tokens
        # Full preprocessing stores samples in a list; release each source
        # array after copying it so the output does not double peak memory.
        if isinstance(protein_tokens_all, list):
            protein_tokens_all[index] = None  # type: ignore[index, assignment]
    protein_tokens_arr.flush()
    del protein_tokens_arr

    rxnfp_tokens_arr = np.stack(rxnfp_tokens_all, axis=0).astype(cfg.np_save_dtype, copy=False)
    chemberta_tokens_arr = np.stack(chemberta_tokens_all, axis=0).astype(cfg.np_save_dtype, copy=False)

    _assert_shape("rxnfp_tokens", rxnfp_tokens_arr, (len(labels_arr), 1, 256))
    _assert_shape("chemberta_tokens", chemberta_tokens_arr, (len(labels_arr), cfg.chemberta_segments, cfg.mol_hidden_size))
    _check_finite("labels", labels_arr)

    np.save(save_dir / "rxnfp_tokens.npy", rxnfp_tokens_arr)
    np.save(save_dir / "chemberta_tokens.npy", chemberta_tokens_arr)
    np.save(save_dir / "labels.npy", labels_arr)
    meta_df.to_parquet(save_dir / "meta.parquet", index=False)

    stats: Dict[str, Any] = {
        "num_samples": int(len(labels_arr)),
        "protein_tokens_shape": list(protein_shape),
        "rxnfp_tokens_shape": list(rxnfp_tokens_arr.shape),
        "chemberta_tokens_shape": list(chemberta_tokens_arr.shape),
        "labels_shape": list(labels_arr.shape),
        "save_dtype": cfg.save_dtype,
        "label_column": cfg.label_column,
        "reaction_column": cfg.reaction_column,
        "sequence_column": cfg.sequence_column,
        "chemberta_segments": cfg.chemberta_segments,
        "mol_hidden_size": cfg.mol_hidden_size,
        "chemberta_input_mode": cfg.chemberta_input_mode,
        "config": _jsonable_cfg(cfg),
    }
    if extra_stats:
        stats.update(extra_stats)

    with open(save_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def _copy_or_subset_existing_array(src_path: Path, dst_path: Path, kept_indices: List[int]) -> List[int]:
    arr = np.load(src_path, mmap_mode="r")
    shape = list(arr.shape)
    if len(kept_indices) == arr.shape[0] and src_path.resolve() != dst_path.resolve():
        shutil.copy2(src_path, dst_path)
    elif src_path.resolve() == dst_path.resolve() and len(kept_indices) == arr.shape[0]:
        pass
    else:
        idx = np.asarray(kept_indices, dtype=np.int64)
        np.save(dst_path, np.asarray(arr[idx]))
    return shape


# ===================== Processing modes =====================
FEATURE_CONFIG_FIELDS = (
    "protein_encoder", "esm_model_name", "protein_model_name", "protein_feature_dim",
    "protein_segments", "protein_layer", "protein_long_mode", "max_seq_len",
    "protein_window_stride", "mol_model_path", "chemberta_segments", "chemberta_input_mode",
    "mol_hidden_size", "mol_max_length", "save_dtype", "sequence_column",
    "reaction_column", "label_column", "reuse_feature_dir",
)


def _file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reuse_source_signature(src_dir: Path) -> Dict[str, Any]:
    # Stat checks do not read the potentially very large feature arrays.
    result = {}
    for name in ("protein_tokens.npy", "rxnfp_tokens.npy", "labels.npy", "meta.parquet", "stats.json"):
        path = src_dir / name
        stat = path.stat()
        result[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def _validate_reuse_source(src_dir: Path, cfg: PreprocessConfig) -> None:
    _reuse_source_signature(src_dir)
    source = json.loads((src_dir / "stats.json").read_text())
    saved = source.get("config", {})
    fields = ("protein_encoder", "protein_feature_dim", "protein_segments", "protein_layer",
              "protein_long_mode", "max_seq_len", "protein_window_stride", "sequence_column", "label_column")
    fields += ("esm_model_name",) if cfg.protein_encoder == "esm" else ("protein_model_name",)
    changed = [name for name in fields if name not in saved or saved[name] != getattr(cfg, name)]
    if changed:
        raise ValueError(f"Cannot reuse protein features with changed/unrecorded settings {changed}: {src_dir}. "
                         "Use matching settings or run full CSV preprocessing without reuse_feature_dir.")
    protein = np.load(src_dir / "protein_tokens.npy", mmap_mode="r")
    rxnfp = np.load(src_dir / "rxnfp_tokens.npy", mmap_mode="r")
    if protein.ndim != 3 or protein.shape[1:] != (cfg.protein_segments, cfg.protein_feature_dim):
        raise ValueError(f"Reused protein shape does not match requested segments/dimension: {protein.shape}")
    if protein.dtype != np.dtype(cfg.np_save_dtype) or rxnfp.dtype != np.dtype(cfg.np_save_dtype):
        raise ValueError("Reuse copies protein/RXNFP arrays unchanged; match their save_dtype or run full preprocessing")


def _can_skip_existing(save_dir: Path, cfg: PreprocessConfig,
                       source_csv_path: Optional[Path] = None) -> bool:
    if not cfg.skip_existing:
        return False
    required = ("protein_tokens.npy", "rxnfp_tokens.npy", "chemberta_tokens.npy",
                "labels.npy", "meta.parquet", "stats.json")
    if not any((save_dir / name).exists() for name in required):
        return False
    missing = [name for name in required if not (save_dir / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete existing features at {save_dir}: {missing}; cannot skip")
    stats = json.loads((save_dir / "stats.json").read_text())
    saved = stats.get("config", {})
    changed = [name for name in FEATURE_CONFIG_FIELDS if name not in saved or saved[name] != getattr(cfg, name)]
    expected_mode = "reuse_existing_features" if cfg.reuse_feature_dir else "full"
    if stats.get("mode") != expected_mode:
        changed.append("mode")
    if source_csv_path is not None and not cfg.reuse_feature_dir:
        if stats.get("source_csv_sha256") != _file_sha256(source_csv_path):
            changed.append("source_csv_sha256")
    if cfg.reuse_feature_dir:
        # TurnUp has a nested task directory; derive its path from output root.
        src_dir = Path(cfg.reuse_feature_dir) / save_dir.relative_to(Path(cfg.output_dir))
        if stats.get("reuse_source_signature") != _reuse_source_signature(src_dir):
            changed.append("reuse_source_signature")
    if changed:
        raise ValueError(f"Existing features differ from requested settings at {save_dir}: {changed}. "
                         "Use a new output directory or disable skip_existing to regenerate.")
    labels = np.load(save_dir / "labels.npy", mmap_mode="r")
    n = len(labels)
    expected = {"protein_tokens.npy": (n, cfg.protein_segments, cfg.protein_feature_dim),
                "rxnfp_tokens.npy": (n, 1, 256),
                "chemberta_tokens.npy": (n, cfg.chemberta_segments, cfg.mol_hidden_size)}
    for name, shape in expected.items():
        arr = np.load(save_dir / name, mmap_mode="r")
        if arr.shape != shape or arr.dtype != np.dtype(cfg.np_save_dtype):
            raise ValueError(f"Existing feature shape/dtype mismatch: {save_dir / name}")
    if labels.ndim != 1 or len(pd.read_parquet(save_dir / "meta.parquet")) != n:
        raise ValueError(f"Existing feature row alignment mismatch: {save_dir}")
    print(f"Skip compatible existing features: {save_dir}")
    return True


def process_dataframe_full(
    df: pd.DataFrame, task_name: str, split_name: str, cfg: PreprocessConfig,
    source_csv_path: Optional[Path] = None,
) -> None:
    required_cols = [cfg.sequence_column, cfg.reaction_column, cfg.label_column]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{task_name}/{split_name}: missing required column '{col}'")

    save_dir = Path(cfg.output_dir) / task_name / split_name
    if _can_skip_existing(save_dir, cfg, source_csv_path):
        return

    protein_tokens_all: List[np.ndarray] = []
    rxnfp_tokens_all: List[np.ndarray] = []
    chemberta_tokens_all: List[np.ndarray] = []
    labels_all: List[float] = []
    kept_rows: List[Any] = []
    resolved_reactions: List[str] = []
    fail_counts: Counter = Counter()
    fail_examples: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    excluded_rows: List[Dict[str, Any]] = []

    def reject(reason: str, row_idx: Any, error: Exception) -> None:
        message = repr(error)
        _add_failure(fail_counts, fail_examples, reason, row_idx, message, cfg)
        excluded_rows.append({
            "source_row_index": int(row_idx),
            "record_key": f"{task_name}/{split_name}/{int(row_idx)}",
            "reason": reason,
            "error": message[:500],
        })

    protein_cache: Dict[str, np.ndarray] = {}
    rxnfp_cache: Dict[str, np.ndarray] = {}
    chemberta_cache: Dict[str, np.ndarray] = {}

    sequence_meta_all: List[Dict[str, Union[int, bool]]] = []

    iterator = tqdm(df.iterrows(), total=len(df), desc=f"{task_name}/{split_name}")
    for idx, row in iterator:
        try:
            label_value = _parse_finite_label(row[cfg.label_column])
        except Exception as e:
            reject("invalid_label", idx, e)
            continue

        sequence = row[cfg.sequence_column]
        try:
            reaction = _resolve_row_reaction(row, cfg)
        except Exception as e:
            reject("invalid_reaction", idx, e)
            continue
        seq_meta = _sequence_metadata(sequence, cfg)

        try:
            p_key = _sequence_cache_key(sequence, cfg)
            if cfg.use_cache and cfg.cache_protein and p_key in protein_cache:
                protein_tokens = protein_cache[p_key]
            else:
                protein_tokens = sequence_to_segment_tokens(sequence, cfg)
                if cfg.use_cache and cfg.cache_protein:
                    protein_cache[p_key] = protein_tokens
        except Exception as e:
            reject("protein_failed", idx, e)
            continue

        try:
            r_key = str(reaction)
            if cfg.use_cache and cfg.cache_rxnfp and r_key in rxnfp_cache:
                rxnfp_token = rxnfp_cache[r_key]
            else:
                rxnfp_token = get_rxnfp_token(reaction, cfg)
                if cfg.use_cache and cfg.cache_rxnfp:
                    rxnfp_cache[r_key] = rxnfp_token
        except Exception as e:
            reject("rxnfp_failed", idx, e)
            continue

        try:
            r_key = str(reaction)
            if cfg.use_cache and cfg.cache_chemberta and r_key in chemberta_cache:
                chemberta_tokens = chemberta_cache[r_key]
            else:
                chemberta_tokens = get_chemberta_segment_tokens(reaction, cfg)
                if cfg.use_cache and cfg.cache_chemberta:
                    chemberta_cache[r_key] = chemberta_tokens
        except Exception as e:
            reject("chemberta_failed", idx, e)
            continue

        protein_tokens_all.append(protein_tokens)
        rxnfp_tokens_all.append(rxnfp_token[None, :])
        chemberta_tokens_all.append(chemberta_tokens)
        labels_all.append(label_value)
        kept_rows.append(idx)
        resolved_reactions.append(reaction)
        sequence_meta_all.append(seq_meta)

        if len(labels_all) % 200 == 0:
            _cleanup()

    if len(labels_all) == 0:
        print("\n" + "=" * 80)
        print(f"[DEBUG FAIL SUMMARY] task={task_name}, split={split_name}")
        print(f"Input rows: {len(df)}, kept rows: 0")
        print(f"fail_counts: {dict(fail_counts)}")
        print("\nFirst 5 fail examples:")
        for reason, ex_list in fail_examples.items():
            print(f"  [{reason}]")
            for item in ex_list[:5]:
                print(f"    row {item['row_index']}, msg={item['message']}")
        print("=" * 80 + "\n")
        raise ValueError(f"{task_name}/{split_name}: no valid samples after filtering")


    print("\n===== FAIL COUNTS =====")
    print(fail_counts)

    for k, v in fail_examples.items():
        print(f"\n[{k}]")
        for x in v[:5]:
            print(x)

    print("=======================\n")
    meta_df = df.loc[kept_rows].reset_index(drop=True).copy()
    # Store the canonical reaction used by both reaction encoders.  This is
    # essential for TurnUp cold-reaction CSVs, where reaction_smiles is blank
    # and reactant_smiles/product_smiles are populated instead.
    meta_df[cfg.reaction_column] = resolved_reactions
    
    meta_df["sample_idx"] = np.arange(len(meta_df))
    # Preserve the CSV row behind each aligned feature/label row. sample_idx
    # is re-numbered after filtering and is only local to this feature split.
    meta_df["source_row_index"] = [int(i) for i in kept_rows]
    meta_df["record_key"] = [
        f"{task_name}/{split_name}/{int(i)}" for i in kept_rows
    ]
    for key in ("sequence_raw_length", "sequence_sanitized_length", "sequence_used_length", "sequence_was_truncated", "protein_long_mode", "protein_window_count"):
        meta_df[key] = [m[key] for m in sequence_meta_all]
    

    extra_stats = {
        "input_rows": int(len(df)),
        "kept_rows": int(len(labels_all)),
        "dropped_rows": int(len(df) - len(labels_all)),
        "fail_counts": dict(fail_counts),
        "fail_examples": dict(fail_examples),
        "exclusions_file": "exclusions.csv",
        "unique_protein_cache_size": int(len(protein_cache)),
        "unique_rxnfp_cache_size": int(len(rxnfp_cache)),
        "unique_chemberta_cache_size": int(len(chemberta_cache)),
        "cache_protein": bool(cfg.use_cache and cfg.cache_protein),
        "cache_rxnfp": bool(cfg.use_cache and cfg.cache_rxnfp),
        "cache_chemberta": bool(cfg.use_cache and cfg.cache_chemberta),
        "num_truncated_sequences": int(meta_df["sequence_was_truncated"].sum()),
        "max_sequence_raw_length": int(meta_df["sequence_raw_length"].max()),
        "max_sequence_used_length": int(meta_df["sequence_used_length"].max()),
        "max_protein_window_count": int(meta_df["protein_window_count"].max()),
        "protein_long_mode": cfg.protein_long_mode,
        "protein_window_stride": int(cfg.protein_window_stride),
        "mode": "full",
    }
    extra_stats.update(_duplicate_label_stats(meta_df, cfg))
    if source_csv_path is not None:
        import hashlib

        digest = hashlib.sha256()
        with Path(source_csv_path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        extra_stats["source_csv"] = str(source_csv_path)
        extra_stats["source_csv_sha256"] = digest.hexdigest()
    stats = _save_split(save_dir, protein_tokens_all, rxnfp_tokens_all, chemberta_tokens_all, labels_all, meta_df, cfg, extra_stats)
    pd.DataFrame(
        excluded_rows,
        columns=["source_row_index", "record_key", "reason", "error"],
    ).to_csv(save_dir / "exclusions.csv", index=False)
    print(f"Saved {task_name}/{split_name} -> {save_dir}")
    print(json.dumps({k: stats[k] for k in ["num_samples", "protein_tokens_shape", "rxnfp_tokens_shape", "chemberta_tokens_shape"]}, indent=2))


def process_from_existing_features(task_name: str, split_name: str, cfg: PreprocessConfig) -> None:
    src_dir = Path(cfg.reuse_feature_dir) / task_name / split_name
    save_dir = Path(cfg.output_dir) / task_name / split_name
    _validate_reuse_source(src_dir, cfg)
    if _can_skip_existing(save_dir, cfg):
        return

    required = ["protein_tokens.npy", "rxnfp_tokens.npy", "labels.npy", "meta.parquet"]
    for fname in required:
        if not (src_dir / fname).exists():
            raise FileNotFoundError(src_dir / fname)

    meta = pd.read_parquet(src_dir / "meta.parquet")
    labels = np.load(src_dir / "labels.npy", mmap_mode="r")
    protein_src = np.load(src_dir / "protein_tokens.npy", mmap_mode="r")
    rxnfp_src = np.load(src_dir / "rxnfp_tokens.npy", mmap_mode="r")
    if len(meta) != len(labels):
        raise ValueError(f"Length mismatch between meta and labels in {src_dir}")
    if len(meta) != protein_src.shape[0] or len(meta) != rxnfp_src.shape[0]:
        raise ValueError(f"Length mismatch between meta/protein/rxnfp arrays in {src_dir}")
    if protein_src.ndim != 3 or protein_src.shape[2] != cfg.protein_feature_dim:
        raise ValueError(f"source_protein_tokens shape mismatch: expected [N, P, {cfg.protein_feature_dim}], got {tuple(protein_src.shape)}")
    _assert_shape("source_rxnfp_tokens", rxnfp_src, (len(meta), 1, 256))
    _check_finite("source_labels", np.asarray(labels))
    if cfg.reaction_column not in meta.columns:
        raise ValueError(f"{src_dir}/meta.parquet is missing '{cfg.reaction_column}'")

    save_dir.mkdir(parents=True, exist_ok=True)
    chemberta_tokens_all: List[np.ndarray] = []
    kept_indices: List[int] = []
    fail_counts: Counter = Counter()
    fail_examples: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    chemberta_cache: Dict[str, np.ndarray] = {}

    iterator = tqdm(meta.iterrows(), total=len(meta), desc=f"upgrade {task_name}/{split_name}")
    for i, row in iterator:
        reaction = row[cfg.reaction_column]
        try:
            key = str(reaction)
            if cfg.use_cache and cfg.cache_chemberta and key in chemberta_cache:
                chemberta_tokens = chemberta_cache[key]
            else:
                chemberta_tokens = get_chemberta_segment_tokens(reaction, cfg)
                if cfg.use_cache and cfg.cache_chemberta:
                    chemberta_cache[key] = chemberta_tokens
        except Exception as e:
            _add_failure(fail_counts, fail_examples, "chemberta_failed", i, repr(e), cfg)
            continue

        chemberta_tokens_all.append(chemberta_tokens)
        kept_indices.append(int(i))

        if len(chemberta_tokens_all) % 500 == 0:
            _cleanup()

    if len(chemberta_tokens_all) == 0:
        raise ValueError(f"{task_name}/{split_name}: no valid samples after ChemBERTa token extraction")

    original_n = len(meta)
    keep_all = len(kept_indices) == original_n
    kept_meta = meta.iloc[kept_indices].reset_index(drop=True).copy()
    kept_meta["sample_idx"] = np.arange(len(kept_meta))

    src_protein_shape = _copy_or_subset_existing_array(src_dir / "protein_tokens.npy", save_dir / "protein_tokens.npy", kept_indices)
    src_rxnfp_shape = _copy_or_subset_existing_array(src_dir / "rxnfp_tokens.npy", save_dir / "rxnfp_tokens.npy", kept_indices)
    src_labels_shape = _copy_or_subset_existing_array(src_dir / "labels.npy", save_dir / "labels.npy", kept_indices)

    chemberta_tokens_arr = np.stack(chemberta_tokens_all, axis=0).astype(cfg.np_save_dtype, copy=False)
    _assert_shape("chemberta_tokens", chemberta_tokens_arr, (len(kept_indices), cfg.chemberta_segments, cfg.mol_hidden_size))
    np.save(save_dir / "chemberta_tokens.npy", chemberta_tokens_arr)
    kept_meta.to_parquet(save_dir / "meta.parquet", index=False)

    protein_arr = np.load(save_dir / "protein_tokens.npy", mmap_mode="r")
    rxnfp_arr = np.load(save_dir / "rxnfp_tokens.npy", mmap_mode="r")
    labels_arr = np.load(save_dir / "labels.npy", mmap_mode="r")

    stats = {
        "mode": "reuse_existing_features",
        "source_feature_dir": str(src_dir),
        "reuse_source_signature": _reuse_source_signature(src_dir),
        "num_samples": int(len(labels_arr)),
        "original_samples": int(original_n),
        "dropped_by_chemberta": int(original_n - len(labels_arr)),
        "kept_all_existing_samples": bool(keep_all),
        "source_protein_tokens_shape": src_protein_shape,
        "source_rxnfp_tokens_shape": src_rxnfp_shape,
        "source_labels_shape": src_labels_shape,
        "protein_tokens_shape": list(protein_arr.shape),
        "protein_segments_actual": int(protein_arr.shape[1]),
        "rxnfp_tokens_shape": list(rxnfp_arr.shape),
        "chemberta_tokens_shape": list(chemberta_tokens_arr.shape),
        "labels_shape": list(labels_arr.shape),
        "save_dtype": cfg.save_dtype,
        "label_column": cfg.label_column,
        "reaction_column": cfg.reaction_column,
        "chemberta_segments": cfg.chemberta_segments,
        "mol_hidden_size": cfg.mol_hidden_size,
        "chemberta_input_mode": cfg.chemberta_input_mode,
        "fail_counts": dict(fail_counts),
        "fail_examples": dict(fail_examples),
        "unique_chemberta_cache_size": int(len(chemberta_cache)),
        "cache_chemberta": bool(cfg.use_cache and cfg.cache_chemberta),
        "config": _jsonable_cfg(cfg),
    }
    stats.update(_duplicate_label_stats(kept_meta, cfg))
    with open(save_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Saved upgraded {task_name}/{split_name} -> {save_dir}")
    print(json.dumps({k: stats[k] for k in ["num_samples", "protein_tokens_shape", "rxnfp_tokens_shape", "chemberta_tokens_shape"]}, indent=2))


def input_tasks(dataset: str, turnup_dataset: str = "both",
                turnup_data_root: str = DEFAULT_TURNUP_DATA_ROOT) -> Tuple[Dict[str, Any], ...]:
    names = set()
    if dataset in ("catpred", "all"):
        names.add("CatPred-DB")
    if dataset in ("pmak", "all"):
        names.add("PMAK-DB")
    if dataset in ("turnup", "all"):
        kinds = ("cold_enzyme", "cold_reaction") if turnup_dataset == "both" else (turnup_dataset,)
        names.update(f"TurnUp/{kind}" for kind in kinds)
    tasks = []
    for source in DEFAULT_TASKS:
        if source["task_name"] not in names:
            continue
        task = dict(source)
        if task["task_name"].startswith("TurnUp/"):
            kind = task["task_name"].split("/", 1)[1]
            if turnup_data_root:
                if Path(turnup_data_root).resolve() != Path(DEFAULT_TURNUP_DATA_ROOT).resolve():
                    folder = "cold_enzyme" if kind == "cold_enzyme" else "cold_reaction_full"
                    task["csv_subdir"] = str(Path(turnup_data_root) / folder)
            else:
                task["csv_subdir"] = f"PMAK-DB/turnup/{kind}"
                task["files"] = tuple(name for name in task["files"]
                                      if name not in ("kcat-random_train.csv", "kcat-random_test.csv"))
        tasks.append(task)
    return tuple(tasks)


def config_from_args(argv: Optional[Sequence[str]] = None) -> PreprocessConfig:
    cfg = PreprocessConfig()
    # Missing CLI options must not overwrite edits in PreprocessConfig.
    parser = argparse.ArgumentParser(
        description="Create BTRIKcat token features; defaults to both PMAK TurnUp cold datasets.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--dataset", choices=("catpred", "pmak", "turnup", "all"))
    parser.add_argument("--turnup-dataset", choices=("cold_enzyme", "cold_reaction", "both"))
    parser.add_argument("--turnup-data-root", type=Path,
                        help="External TurnUp root containing cold_enzyme and cold_reaction_full CSVs; includes pool/test")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--protein-tokens", type=int)
    parser.add_argument("--chemberta-tokens", type=int)
    parser.add_argument("--protein-encoder", choices=("esm", "prot_t5"))
    parser.add_argument("--protein-model")
    parser.add_argument("--protein-dim", type=int)
    parser.add_argument("--protein-layer", type=int)
    parser.add_argument("--protein-long-mode", choices=("sliding_window", "truncate"))
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--protein-stride", type=int)
    parser.add_argument("--chemberta-model")
    parser.add_argument("--chemberta-dim", type=int)
    parser.add_argument("--chemberta-input-mode", choices=("separate", "merged"))
    parser.add_argument("--mol-max-length", type=int)
    parser.add_argument("--mol-chunk-batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--save-dtype", choices=("float16", "float32"))
    parser.add_argument("--reuse-feature-root", type=Path,
                        help="Reuse matching protein/RXNFP features; regenerate only ChemBERTa")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--check-only", action=argparse.BooleanOptionalAction,
                        help="Check CSVs without loading encoders")
    parser.add_argument("--print-config", action="store_true",
                        help="Print effective configuration without loading CSVs or encoders")
    args = vars(parser.parse_args(argv))
    old_encoder = cfg.protein_encoder
    fields = {
        "dataset": "dataset", "turnup_dataset": "turnup_dataset",
        "turnup_data_root": "turnup_data_root",
        "data_root": "base_data_dir", "output_root": "output_dir",
        "protein_tokens": "protein_segments", "chemberta_tokens": "chemberta_segments",
        "protein_encoder": "protein_encoder", "protein_dim": "protein_feature_dim",
        "protein_layer": "protein_layer", "protein_long_mode": "protein_long_mode",
        "max_seq_len": "max_seq_len", "protein_stride": "protein_window_stride",
        "chemberta_model": "mol_model_path", "chemberta_dim": "mol_hidden_size",
        "chemberta_input_mode": "chemberta_input_mode", "mol_max_length": "mol_max_length",
        "mol_chunk_batch_size": "mol_chunk_batch_size", "device": "device",
        "save_dtype": "save_dtype", "reuse_feature_root": "reuse_feature_dir",
        "skip_existing": "skip_existing", "check_only": "check_only", "print_config": "print_config",
    }
    for option, attr in fields.items():
        if option in args:
            value = args[option]
            setattr(cfg, attr, str(value) if isinstance(value, Path) else value)
    if "protein_model" in args:
        setattr(cfg, "esm_model_name" if cfg.protein_encoder == "esm" else "protein_model_name",
                args["protein_model"])
    # CLI encoder switches keep the established defaults for the two supported
    # encoders. Explicit dimensions/layers always win, including invalid zeroes.
    if cfg.protein_encoder != old_encoder:
        if "protein_dim" not in args:
            cfg.protein_feature_dim = 2560 if cfg.protein_encoder == "esm" else 1024
        if "protein_layer" not in args:
            cfg.protein_layer = 36 if cfg.protein_encoder == "esm" else -1
    automatic_output = (
        cfg.output_dir == DEFAULT_OUTPUT_DIR
        and Path(cfg.output_dir).parent == Path(DEFAULT_FEATURE_ROOT)
        and re.fullmatch(r"P[0-9]+_C[0-9]+", Path(cfg.output_dir).name)
    )
    if "output_root" not in args and (not cfg.output_dir or automatic_output):
        cfg.output_dir = str(Path(DEFAULT_FEATURE_ROOT) / f"P{cfg.protein_segments}_C{cfg.chemberta_segments}")
    cfg.base_data_dir = str(Path(cfg.base_data_dir).expanduser().resolve())
    cfg.output_dir = str(Path(cfg.output_dir).expanduser().resolve())
    if cfg.turnup_data_root:
        cfg.turnup_data_root = str(Path(cfg.turnup_data_root).expanduser().resolve())
    if cfg.reuse_feature_dir:
        cfg.reuse_feature_dir = str(Path(cfg.reuse_feature_dir).expanduser().resolve())
    if cfg.tasks is DEFAULT_TASKS or not cfg.tasks or any(option in args for option in
                            ("dataset", "turnup_dataset", "turnup_data_root")):
        cfg.tasks = input_tasks(cfg.dataset, cfg.turnup_dataset, cfg.turnup_data_root)
    validate_config(cfg)
    return cfg


def main() -> None:
    cfg = config_from_args()
    print(json.dumps({"preprocess_script": str(Path(__file__).resolve()), "cfg": _jsonable_cfg(cfg)},
                     ensure_ascii=False, indent=2), flush=True)
    if cfg.print_config:
        return
    # Check all selected input files before loading large pretrained encoders.
    inputs = []
    for task in selected_tasks(cfg):
        for filename in task["files"]:
            path = Path(cfg.base_data_dir) / task["csv_subdir"] / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            inputs.append((task, path))
    if cfg.check_only:
        for task, path in inputs:
            frame = pd.read_csv(path)
            required = {cfg.sequence_column, cfg.label_column}
            if not required.issubset(frame.columns):
                raise ValueError(f"{path}: missing columns {required - set(frame.columns)}")
            if cfg.reaction_column not in frame.columns and not {"reactant_smiles", "product_smiles"}.issubset(frame.columns):
                raise ValueError(f"{path}: missing reaction inputs")
            print(f"{task['task_name']}/{path.stem}: {len(frame)} rows")
        print(f"Checked {len(inputs)} CSV files; no models were loaded.")
        return
    pending = []
    for task, path in inputs:
        relative = Path(task["task_name"]) / path.stem
        if cfg.reuse_feature_dir:
            _validate_reuse_source(Path(cfg.reuse_feature_dir) / relative, cfg)
        if not _can_skip_existing(Path(cfg.output_dir) / relative, cfg, path):
            pending.append((task, path))
    if not pending:
        print("All selected outputs are compatible; no encoders were loaded.")
        return
    init_models(cfg, load_esm_rxnfp=not bool(cfg.reuse_feature_dir), load_chemberta=True)
    for task, path in pending:
        split = path.stem
        if cfg.reuse_feature_dir:
            process_from_existing_features(task["task_name"], split, cfg)
        else:
            frame = pd.read_csv(path)
            if cfg.reaction_column not in frame.columns:
                frame[cfg.reaction_column] = ""
            process_dataframe_full(frame, task["task_name"], split, cfg, source_csv_path=path)
    print("All preprocessing finished.")


if __name__ == "__main__":
    main()
