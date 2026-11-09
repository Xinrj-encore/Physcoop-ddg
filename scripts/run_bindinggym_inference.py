#!/usr/bin/env python3
"""Run a fixed PhysCoop-DDG ensemble on BindingGYM."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ddg_predictor import DDGPredictor  # noqa: E402
from physcoop.bindinggym_data import (  # noqa: E402
    BindingGymCollate,
    BindingGymDataset,
)
from trainer import CrossValidation, recursive_to  # noqa: E402


FOLD_RUN_RE = re.compile(r"(?:^|_)fold(?P<fold>\d+)$")


def resolve_device(device_name):
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) "
                "are visible."
            )
    return device


def load_model(config_path, checkpoint_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "models" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not a full PhysCoop-DDG checkpoint: missing 'models'."
        )

    num_models = len(checkpoint["models"])
    config["num_cvfolds"] = num_models
    config.setdefault("lr", 1e-4)
    config.setdefault("weight_decay", 0.0)
    model_config = SimpleNamespace(**config)
    cv_manager = CrossValidation(
        config=model_config,
        num_cvfolds=num_models,
        model_factory=DDGPredictor,
    ).to("cpu")
    cv_manager.load_state_dict_inference(checkpoint)
    return model_config, cv_manager, num_models


def parse_folds(value, num_models):
    folds = list(range(num_models)) if value is None else [int(item) for item in value]
    invalid = [fold for fold in folds if fold < 0 or fold >= num_models]
    if invalid:
        raise ValueError(f"Invalid folds {invalid}; checkpoint contains {num_models} models.")
    if not folds:
        raise ValueError("At least one fold must be selected.")
    return folds


def discover_fold_runs(run_dir):
    """Find one final checkpoint directory per trained CV fold."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    fold_runs = {}
    for candidate in sorted(run_dir.iterdir()):
        if not candidate.is_dir():
            continue
        match = FOLD_RUN_RE.search(candidate.name)
        if match is None:
            continue

        fold = int(match.group("fold"))
        config_path = candidate / "train_config.json"
        checkpoint_path = candidate / "checkpoint" / "ddg_model.ckpt"
        if not config_path.is_file() or not checkpoint_path.is_file():
            continue
        if fold in fold_runs:
            raise ValueError(
                f"Found multiple completed runs for fold{fold} in {run_dir}: "
                f"{fold_runs[fold]['run_dir']} and {candidate}"
            )
        fold_runs[fold] = {
            "run_dir": candidate,
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
        }

    if not fold_runs:
        raise ValueError(
            f"No completed fold runs found in {run_dir}. Expected subdirectories "
            "ending in _fold0, _fold1, etc. with train_config.json and "
            "checkpoint/ddg_model.ckpt."
        )
    return fold_runs


def load_fold_models_from_run_dir(run_dir):
    fold_runs = discover_fold_runs(run_dir)
    fold_models = {}
    model_config = None

    for fold, run in sorted(fold_runs.items()):
        config, cv_manager, num_models = load_model(
            run["config_path"],
            run["checkpoint_path"],
        )
        if fold >= num_models:
            raise ValueError(
                f"{run['checkpoint_path']} contains {num_models} model(s), "
                f"so it cannot provide fold{fold}."
            )
        declared_folds = getattr(config, "fold_indices", None)
        if declared_folds is not None and list(declared_folds) != [fold]:
            raise ValueError(
                f"{run['config_path']} declares fold_indices={declared_folds}, "
                f"but its directory is labeled fold{fold}."
            )
        if model_config is None:
            model_config = config
        fold_models[fold] = cv_manager.models[fold]

    return model_config, fold_models, fold_runs


def predict_fold(model, loader, device, fold, n_samples):
    model.to(device)
    model.eval()
    predictions = {}
    start = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(
            loader,
            total=len(loader),
            desc=f"BindingGYM fold{fold}",
            dynamic_ncols=True,
        ):
            sample_ids = batch["sample_id"]
            batch_device = recursive_to(batch, device)
            _loss, output_dict, _log_probs = model(batch_device)
            values = output_dict["ddG_pred"].detach().float().cpu().numpy()
            if len(values) != len(sample_ids):
                raise RuntimeError(
                    f"Model returned {len(values)} predictions for {len(sample_ids)} samples."
                )
            predictions.update(
                {sample_id: float(value) for sample_id, value in zip(sample_ids, values)}
            )
    elapsed = time.perf_counter() - start
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(predictions) != n_samples:
        raise RuntimeError(
            f"Fold {fold} produced {len(predictions)} predictions for {n_samples} samples."
        )
    return predictions, elapsed


def build_output(frame, fold_predictions, preprocess_elapsed, elapsed, peak_gpu_mem_mb):
    result = frame.copy()
    result["binding_score"] = pd.to_numeric(result["dms_score"], errors="coerce")
    for fold, predictions in fold_predictions.items():
        result[f"pred_ddg_fold{fold}"] = result["sample_id"].map(predictions)

    fold_columns = [f"pred_ddg_fold{fold}" for fold in fold_predictions]
    result["pred_ddg"] = result[fold_columns].mean(axis=1)
    result["pred_binding_score"] = -result["pred_ddg"]
    result["fold_count"] = len(fold_columns)
    result["status"] = np.where(result["pred_ddg"].notna(), "ok", "missing_prediction")
    result["preprocess_seconds"] = float(preprocess_elapsed) / max(len(result), 1)
    result["inference_seconds"] = float(elapsed) / max(len(result), 1)
    result["peak_gpu_mem_mb"] = peak_gpu_mem_mb

    columns = [
        "sample_id",
        "assay_id",
        "dms_filename",
        "row_index",
        "poi",
        "complex_id",
        "pdb_file",
        "chain_id",
        "dms_score",
        "binding_score",
        "mutation_pdb",
        "mutation_cleaned",
        "num_muts",
        "num_residues",
        "bindinggym_chain_groups",
        *fold_columns,
        "pred_ddg",
        "pred_binding_score",
        "fold_count",
        "status",
        "preprocess_seconds",
        "inference_seconds",
        "peak_gpu_mem_mb",
    ]
    return result[[column for column in columns if column in result.columns]]


def main():
    parser = argparse.ArgumentParser(
        description="External BindingGYM inference for a fixed PhysCoop-DDG checkpoint."
    )
    parser.add_argument(
        "--run-dir",
        help=(
            "Directory containing one completed *_foldN run per CV fold. The "
            "script uses each run's final checkpoint/ddg_model.ckpt and only "
            "the model matching that fold."
        ),
    )
    parser.add_argument("--config-path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input-dir", default="data/BindingGYM/input")
    parser.add_argument(
        "--manifest",
        default="data/BindingGYM/normalized/bindinggym_manifest.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--assay-id", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PHYSCOOP_NUM_WORKERS", "2")),
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if args.run_dir:
        if args.config_path or args.checkpoint:
            parser.error("--run-dir cannot be combined with --config-path or --checkpoint.")
        if args.folds is not None:
            parser.error("--folds is not needed with --run-dir; all discovered folds are used.")
        model_config, fold_models, fold_runs = load_fold_models_from_run_dir(args.run_dir)
        folds = sorted(fold_models)
        print(
            f"Loaded final checkpoints for folds={folds} from {Path(args.run_dir)}"
        )
    else:
        if not args.config_path or not args.checkpoint:
            parser.error("Provide --run-dir, or both --config-path and --checkpoint.")
        model_config, cv_manager, num_models = load_model(
            args.config_path,
            args.checkpoint,
        )
        folds = parse_folds(args.folds, num_models)
        fold_models = {fold: cv_manager.models[fold] for fold in folds}
        fold_runs = None
        print(f"Loaded checkpoint with {num_models} fold models; evaluating folds={folds}")

    dataset = BindingGymDataset(
        manifest_path=args.manifest,
        input_dir=args.input_dir,
        use_physical_hypergraph=getattr(
            model_config, "use_physical_hypergraph", True
        ),
        phys_contact_cutoff=getattr(model_config, "phys_contact_cutoff", 8.0),
        phys_patch_radius=getattr(model_config, "phys_patch_radius", 10.0),
        phys_max_hyperedges=getattr(model_config, "phys_max_hyperedges", 96),
        phys_max_nodes_per_hyperedge=getattr(
            model_config, "phys_max_nodes_per_hyperedge", 24
        ),
        phys_hypergraph_mode=getattr(model_config, "phys_hypergraph_mode", "full"),
        phys_use_delta_sasa=getattr(model_config, "phys_use_delta_sasa", True),
        phys_use_delta_sasa_attr=getattr(
            model_config, "phys_use_delta_sasa_attr", True
        ),
        assay_ids=args.assay_id,
        max_samples=args.max_samples,
    )
    preprocess_start = time.perf_counter()
    dataset.prepare_structure_metadata()
    preprocess_elapsed = time.perf_counter() - preprocess_start
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.workers)),
        collate_fn=BindingGymCollate(),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.workers) > 0,
    )
    print(
        f"BindingGYM rows={len(dataset)} assays={dataset.frame['assay_id'].nunique()} "
        f"batch_size={args.batch_size} workers={args.workers}"
    )
    print(f"structure_preprocess_seconds={preprocess_elapsed:.3f}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fold_predictions = {}
    total_start = time.perf_counter()
    for fold in folds:
        predictions, fold_elapsed = predict_fold(
            fold_models[fold],
            loader,
            device,
            fold,
            len(dataset),
        )
        fold_predictions[fold] = predictions
        print(f"fold{fold} inference_seconds={fold_elapsed:.3f}")
    total_elapsed = time.perf_counter() - total_start

    peak_gpu_mem_mb = None
    if device.type == "cuda":
        peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated(device) / 2**20)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = build_output(
        dataset.frame,
        fold_predictions,
        preprocess_elapsed,
        total_elapsed,
        peak_gpu_mem_mb,
    )
    output_path = output_dir / "bindinggym_predictions.csv"
    predictions.to_csv(output_path, index=False)

    metadata = {
        "config_path": str(args.config_path) if args.config_path else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "fold_checkpoints": (
            {
                str(fold): str(run["checkpoint_path"])
                for fold, run in sorted(fold_runs.items())
            }
            if fold_runs is not None
            else None
        ),
        "input_dir": str(args.input_dir),
        "manifest": str(args.manifest),
        "device": str(device),
        "folds": folds,
        "n_rows": int(len(predictions)),
        "n_assays": int(predictions["assay_id"].nunique()),
        "preprocess_seconds": float(preprocess_elapsed),
        "elapsed_seconds": float(total_elapsed),
        "samples_per_second": float(len(predictions) / total_elapsed)
        if total_elapsed > 0
        else None,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
    }
    (output_dir / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output_path)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
