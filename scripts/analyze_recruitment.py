#!/usr/bin/env python3
"""Trace PhysCoop member recruitment on complex-level validation folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import SkempiDatasetManager  # noqa: E402
from ddg_predictor import DDGPredictor  # noqa: E402
from trainer import CrossValidation, recursive_to  # noqa: E402
from utils import set_seed  # noqa: E402


REGIONS = (
    "mutation-interface overlap",
    "mutation-proximal non-interface",
    "interface-only",
    "other",
)


def resolve_device(device_name=None):
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_config(path: str) -> argparse.Namespace:
    with open(path, "r") as handle:
        return argparse.Namespace(**json.load(handle))


def load_state(cv_mgr, path: str, fold: int) -> None:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "models" in state:
        cv_mgr.load_state_dict(state)
    elif isinstance(state, dict) and "state_dict" in state:
        cv_mgr.load_state_dict(state["state_dict"])
    elif isinstance(state, dict) and "model_state_dict" in state:
        cv_mgr.load_mpnn_state_dict(state)
    elif isinstance(state, dict) and state and all(
        torch.is_tensor(value) for value in state.values()
    ):
        cv_mgr.models[fold].load_state_dict(state, strict=False)
    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")


def flatten(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def residue_regions(coords, chain_ids, mutation_mask, valid_mask):
    coords = coords.detach()
    chain_ids = chain_ids.detach()
    mutation_mask = mutation_mask.bool() & valid_mask.bool()
    valid = valid_mask.bool()

    pair_dist = torch.cdist(coords, coords)
    cross_chain = chain_ids.unsqueeze(1) != chain_ids.unsqueeze(0)
    interface = (
        (pair_dist <= 8.0)
        & cross_chain
        & valid.unsqueeze(1)
        & valid.unsqueeze(0)
    ).any(dim=1)

    if mutation_mask.any():
        mut_dist = pair_dist[:, mutation_mask].min(dim=1).values
        mutation_proximal = mut_dist <= 5.0
    else:
        mutation_proximal = torch.zeros_like(valid)

    labels = np.full(coords.shape[0], "other", dtype=object)
    labels[mutation_proximal.cpu().numpy() & interface.cpu().numpy()] = REGIONS[0]
    labels[mutation_proximal.cpu().numpy() & ~interface.cpu().numpy()] = REGIONS[1]
    labels[~mutation_proximal.cpu().numpy() & interface.cpu().numpy()] = REGIONS[2]
    labels[~valid.cpu().numpy()] = "invalid"
    return labels


def trace_layer(layer, batch_size_index: int = 0):
    trace = getattr(layer, "last_trace", None)
    if trace is None:
        raise RuntimeError("PhysCoop trace was not produced; enable trace_enabled on all layers.")
    idx = trace["selected_idx"][batch_size_index].detach().cpu()
    valid = trace["selected_valid"][batch_size_index].detach().cpu().bool()
    weights = trace["selected_weights"][batch_size_index].detach().cpu()
    base = trace["base_membership"][batch_size_index].detach().cpu()
    anchor_mask = trace["anchor_mask"][batch_size_index].detach().cpu().bool()
    selected_count = trace.get(
        "selected_count",
        valid.sum(dim=-1),
    )[batch_size_index].detach().cpu()
    effective_capacity = trace.get("effective_capacity")
    if effective_capacity is not None:
        effective_capacity = effective_capacity[batch_size_index].detach().cpu()
    return idx, valid, weights, base, anchor_mask, selected_count, effective_capacity


def layer_member_statistics(
    idx,
    selected_valid,
    weights,
    anchor_mask,
    selected_count,
    effective_capacity,
    valid_residue_mask,
):
    """Summarize both slot-level and unique-residue-level recruitment."""
    active_edges = anchor_mask.bool()
    valid_slots = selected_valid[active_edges]
    valid_weights = weights[active_edges]
    valid_counts = selected_count[active_edges].float()

    if valid_slots.numel() == 0:
        return {
            "active_hyperedge_count": 0,
            "candidate_slot_count": int(selected_valid.shape[-1]),
            "valid_member_slot_count": 0,
            "valid_member_slot_fraction": 0.0,
            "unique_selected_residue_count": 0,
            "unique_selected_residue_fraction": 0.0,
            "mean_selected_count": 0.0,
            "min_selected_count": 0.0,
            "max_selected_count": 0.0,
            "mean_effective_capacity": 0.0,
            "mean_soft_effective_size": 0.0,
        }

    selected_indices = idx[active_edges][valid_slots]
    unique_selected = torch.unique(selected_indices)
    valid_residue_count = int(valid_residue_mask.sum().item())
    candidate_slot_count = int(valid_slots.numel())
    valid_member_slot_count = int(valid_slots.sum().item())

    soft_sizes = []
    for edge_valid, edge_weights in zip(valid_slots, valid_weights):
        edge_weights = edge_weights[edge_valid]
        if edge_weights.numel() == 0:
            soft_sizes.append(0.0)
            continue
        normalized = edge_weights / edge_weights.sum().clamp_min(1e-8)
        soft_sizes.append(float(1.0 / normalized.square().sum().clamp_min(1e-8)))

    return {
        "active_hyperedge_count": int(active_edges.sum().item()),
        "candidate_slot_count": candidate_slot_count,
        "valid_member_slot_count": valid_member_slot_count,
        "valid_member_slot_fraction": float(
            valid_member_slot_count / max(candidate_slot_count, 1)
        ),
        "unique_selected_residue_count": int(unique_selected.numel()),
        "unique_selected_residue_fraction": float(
            unique_selected.numel() / max(valid_residue_count, 1)
        ),
        "mean_selected_count": float(valid_counts.mean().item()),
        "min_selected_count": float(valid_counts.min().item()),
        "max_selected_count": float(valid_counts.max().item()),
        "mean_effective_capacity": float(
            effective_capacity[active_edges].mean().item()
            if effective_capacity is not None
            else 0.0
        ),
        "mean_soft_effective_size": float(np.mean(soft_sizes)),
    }


def selected_residue_set(idx, selected_valid, anchor_mask):
    """Return the unique valid residues selected by active hyperedges."""
    active_idx = idx[anchor_mask.bool()]
    active_valid = selected_valid[anchor_mask.bool()]
    if active_idx.numel() == 0:
        return set()
    return set(active_idx[active_valid].tolist())


def jaccard_score(left, right):
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def scatter_mass(indices, weights, valid, length):
    mass = torch.zeros(length, dtype=weights.dtype)
    for edge_idx in range(indices.shape[0]):
        member_mask = valid[edge_idx]
        if member_mask.any():
            mass.scatter_add_(0, indices[edge_idx][member_mask], weights[edge_idx][member_mask])
    return mass


def analyze_fold(
    run_dir: Path,
    fold: int,
    device,
    max_items: int,
    output_dir: Path,
    workers: int = 0,
):
    config = load_config(str(run_dir / "train_config.json"))
    set_seed(config.seed)
    loader_manager = SkempiDatasetManager(
        config=config,
        num_cvfolds=config.num_cvfolds,
        num_workers=workers,
    )
    cv_mgr = CrossValidation(
        config=config,
        num_cvfolds=config.num_cvfolds,
        model_factory=DDGPredictor,
    ).to("cpu")
    checkpoint_path = run_dir / "checkpoint" / "ddg_model.ckpt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoint" / f"fold{fold}_best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "No supported checkpoint found. Expected either "
            f"{run_dir / 'checkpoint' / 'ddg_model.ckpt'} or "
            f"{run_dir / 'checkpoint' / f'fold{fold}_best_model.pt'}."
        )
    load_state(cv_mgr, str(checkpoint_path), fold=fold)
    model, _, _ = cv_mgr.get(fold)
    model.to(device)
    model.eval()

    layers = getattr(model.mpnn, "PhysicalGuidedAdaptiveHypergraphLayers", None)
    if layers is None:
        raise RuntimeError("The selected checkpoint does not contain PhysCoop layers.")
    for layer in layers:
        layer.trace_enabled = True

    output_dir.mkdir(parents=True, exist_ok=True)
    region_rows = []
    layer_rows = []
    transition_rows = []
    case_rows = []
    hyperedge_rows = []
    best_case = None

    for item_idx, batch in enumerate(tqdm(
        loader_manager.get_val_loader(fold),
        desc=f"fold{fold}",
        dynamic_ncols=True,
    )):
        if max_items > 0 and item_idx >= max_items:
            break
        batch = recursive_to(batch, device)
        with torch.inference_mode():
            model(batch)

        valid_mask = batch["mask"][0].bool()
        regions = residue_regions(
            batch["X"][0, :, 1, :],
            batch["chain_encoding_all"][0],
            batch["chain_M"][0],
            valid_mask,
        )
        valid_indices = np.flatnonzero(regions != "invalid")
        if len(valid_indices) == 0:
            continue

        complex_name = flatten(batch["complex"])
        mutstr = flatten(batch["mutstr"])
        selected_sets = {}
        for layer_index, layer in enumerate(layers):
            (
                idx,
                selected_valid,
                weights,
                base_membership,
                anchor_mask,
                selected_count,
                effective_capacity,
            ) = trace_layer(layer)
            if anchor_mask.numel() == 0 or not anchor_mask.any():
                continue
            mass = scatter_mass(idx, weights, selected_valid, len(regions)).numpy()
            initial = base_membership.any(dim=0).numpy()
            mass[regions == "invalid"] = -1.0
            member_stats = layer_member_statistics(
                idx=idx,
                selected_valid=selected_valid,
                weights=weights,
                anchor_mask=anchor_mask,
                selected_count=selected_count,
                effective_capacity=effective_capacity,
                valid_residue_mask=torch.from_numpy(regions != "invalid"),
            )
            selected_sets[layer_index + 1] = selected_residue_set(
                idx=idx,
                selected_valid=selected_valid,
                anchor_mask=anchor_mask,
            )

            n_top = max(1, int(np.ceil(0.05 * len(valid_indices))))
            ranked = np.argsort(-mass)
            top_indices = [int(i) for i in ranked if regions[i] != "invalid"][:n_top]
            top_regions = regions[top_indices]
            for region in REGIONS:
                region_rows.append(
                    {
                        "fold": fold,
                        "item": item_idx,
                        "layer": layer_index + 1,
                        "complex": complex_name,
                        "region": region,
                        "top_fraction": float(np.mean(top_regions == region)),
                        "candidate_fraction": float(np.mean(regions[valid_indices] == region)),
                    }
                )

            outside_initial = float(np.mean(~initial[top_indices]))
            mutation_interface = float(np.mean(top_regions == REGIONS[0]))
            layer_rows.append(
                {
                    "fold": fold,
                    "item": item_idx,
                    "layer": layer_index + 1,
                    "complex": complex_name,
                    "mutstr": mutstr,
                    "top_n": n_top,
                    "top_outside_initial_fraction": outside_initial,
                    "top_mutation_interface_fraction": mutation_interface,
                    "top_mean_mass": float(np.mean(mass[top_indices])),
                    "valid_residue_count": int((regions != "invalid").sum()),
                    **member_stats,
                }
            )

            initial_member_count = base_membership.gt(0).sum(dim=-1)
            for hyperedge_index in torch.nonzero(anchor_mask, as_tuple=False).flatten().tolist():
                member_mask = selected_valid[hyperedge_index]
                member_weights = weights[hyperedge_index][member_mask]
                if member_weights.numel() > 0:
                    normalized_weights = member_weights / member_weights.sum().clamp_min(1e-8)
                    soft_effective_size = float(
                        1.0 / normalized_weights.square().sum().clamp_min(1e-8)
                    )
                else:
                    soft_effective_size = 0.0
                hyperedge_rows.append(
                    {
                        "fold": fold,
                        "item": item_idx,
                        "layer": layer_index + 1,
                        "complex": complex_name,
                        "mutstr": mutstr,
                        "hyperedge_index": hyperedge_index,
                        "initial_member_count": int(initial_member_count[hyperedge_index]),
                        "dynamic_member_count": int(selected_count[hyperedge_index]),
                        "effective_capacity": (
                            float(effective_capacity[hyperedge_index])
                            if effective_capacity is not None
                            else np.nan
                        ),
                        "soft_effective_size": soft_effective_size,
                    }
                )

            case_score = outside_initial + mutation_interface
            if best_case is None or case_score > best_case["score"]:
                best_case = {
                    "score": case_score,
                    "fold": fold,
                    "item": item_idx,
                    "layer": layer_index + 1,
                    "complex": complex_name,
                    "mutstr": mutstr,
                    "top_indices": top_indices,
                }

            for residue_index in top_indices:
                case_rows.append(
                    {
                        "fold": fold,
                        "item": item_idx,
                        "layer": layer_index + 1,
                        "complex": complex_name,
                        "mutstr": mutstr,
                        "residue_index": residue_index,
                        "region": regions[residue_index],
                        "recruitment_mass": float(mass[residue_index]),
                        "initial_member": int(initial[residue_index]),
                    }
                )

        for left_layer, right_layer in ((1, 2), (2, 3), (1, 3)):
            left_set = selected_sets.get(left_layer, set())
            right_set = selected_sets.get(right_layer, set())
            transition_rows.append(
                {
                    "fold": fold,
                    "item": item_idx,
                    "complex": complex_name,
                    "mutstr": mutstr,
                    "left_layer": left_layer,
                    "right_layer": right_layer,
                    "left_count": len(left_set),
                    "right_count": len(right_set),
                    "intersection_count": len(left_set & right_set),
                    "union_count": len(left_set | right_set),
                    "jaccard": jaccard_score(left_set, right_set),
                    "changed_fraction": 1.0 - jaccard_score(left_set, right_set),
                }
            )

    pd.DataFrame(region_rows).to_csv(output_dir / f"fold{fold}_region_summary.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(output_dir / f"fold{fold}_layer_summary.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(
        output_dir / f"fold{fold}_layer_transition_summary.csv",
        index=False,
    )
    pd.DataFrame(case_rows).to_csv(
        output_dir / f"fold{fold}_top_recruited_residues.csv", index=False
    )
    hyperedge_columns = [
        "fold",
        "item",
        "layer",
        "complex",
        "mutstr",
        "hyperedge_index",
        "initial_member_count",
        "dynamic_member_count",
        "effective_capacity",
        "soft_effective_size",
    ]
    hyperedge_frame = pd.DataFrame(hyperedge_rows, columns=hyperedge_columns)
    hyperedge_frame.to_csv(
        output_dir / f"fold{fold}_hyperedge_sizes.csv",
        index=False,
    )
    if not hyperedge_frame.empty:
        size_summary = (
            hyperedge_frame.groupby("layer", as_index=False)
            .agg(
                num_hyperedges=("hyperedge_index", "size"),
                initial_mean=("initial_member_count", "mean"),
                initial_median=("initial_member_count", "median"),
                dynamic_mean=("dynamic_member_count", "mean"),
                dynamic_median=("dynamic_member_count", "median"),
                dynamic_min=("dynamic_member_count", "min"),
                dynamic_max=("dynamic_member_count", "max"),
                capacity_mean=("effective_capacity", "mean"),
                soft_effective_mean=("soft_effective_size", "mean"),
                soft_effective_median=("soft_effective_size", "median"),
            )
        )
        overall_summary = pd.DataFrame(
            {
                "layer": ["all"],
                "num_hyperedges": [len(hyperedge_frame)],
                "initial_mean": [hyperedge_frame["initial_member_count"].mean()],
                "initial_median": [hyperedge_frame["initial_member_count"].median()],
                "dynamic_mean": [hyperedge_frame["dynamic_member_count"].mean()],
                "dynamic_median": [hyperedge_frame["dynamic_member_count"].median()],
                "dynamic_min": [hyperedge_frame["dynamic_member_count"].min()],
                "dynamic_max": [hyperedge_frame["dynamic_member_count"].max()],
                "capacity_mean": [hyperedge_frame["effective_capacity"].mean()],
                "soft_effective_mean": [hyperedge_frame["soft_effective_size"].mean()],
                "soft_effective_median": [hyperedge_frame["soft_effective_size"].median()],
            }
        )
        size_summary = pd.concat([size_summary, overall_summary], ignore_index=True)
        size_distribution = (
            hyperedge_frame.groupby(
                ["layer", "dynamic_member_count"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "num_hyperedges"})
        )
    else:
        size_summary = pd.DataFrame(
            columns=[
                "layer",
                "num_hyperedges",
                "initial_mean",
                "initial_median",
                "dynamic_mean",
                "dynamic_median",
                "dynamic_min",
                "dynamic_max",
                "capacity_mean",
                "soft_effective_mean",
                "soft_effective_median",
            ]
        )
        size_distribution = pd.DataFrame(
            columns=["layer", "dynamic_member_count", "num_hyperedges"]
        )
    size_summary.to_csv(
        output_dir / f"fold{fold}_hyperedge_size_summary.csv",
        index=False,
    )
    size_distribution.to_csv(
        output_dir / f"fold{fold}_hyperedge_size_distribution.csv",
        index=False,
    )
    with open(output_dir / f"fold{fold}_representative_case.json", "w") as handle:
        json.dump(best_case or {}, handle, indent=2)
    return len(layer_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-items", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    count = analyze_fold(
        run_dir=run_dir,
        fold=args.fold,
        device=resolve_device(args.device),
        max_items=args.max_items,
        output_dir=output_dir,
        workers=args.workers,
    )
    print(f"Saved {count} layer summaries to {output_dir}")


if __name__ == "__main__":
    main()
