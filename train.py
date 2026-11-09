import argparse
import json
import os
import time
from types import SimpleNamespace

import pandas as pd
import torch
from tqdm.auto import tqdm

from dataset import SkempiDatasetManager
from ddg_predictor import DDGPredictor
from trainer import CrossValidation, recursive_to
from utils import eval_skempi_three_modes, set_seed


def resolve_folds(config):
    folds = getattr(config, "fold_indices", None)
    if folds is None:
        return list(range(config.num_cvfolds))
    return [int(fold) for fold in folds]


def make_run_dir(config, tag):
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    name = f"{timestamp}_{tag or config.ex_name}"
    path = os.path.join("results", name)
    os.makedirs(os.path.join(path, "checkpoint"), exist_ok=True)
    return path


def move_batch(batch, device):
    return recursive_to(batch, device)


def collect_predictions(model, loader, device):
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="validate", dynamic_ncols=True):
            batch = move_batch(batch, device)
            _, output, _ = model(batch)
            for complex_id, num_muts, true_value, prediction in zip(
                batch["complex"],
                batch["num_muts"],
                output["ddG_true"],
                output["ddG_pred"],
            ):
                rows.append(
                    {
                        "complex": complex_id,
                        "num_muts": num_muts,
                        "ddG": true_value.item(),
                        "ddG_pred": prediction.item(),
                        "method": "PhysCoop-DDG",
                    }
                )
    return pd.DataFrame(rows)


def metric_value(metrics, subset, column):
    row = metrics[metrics["mode"] == subset].iloc[0]
    return float(row[column])


def validate_fold(model, loader, device, results_dir, fold, iteration=None):
    predictions = collect_predictions(model, loader, device)
    suffix = f"_its{iteration}" if iteration is not None else ""
    predictions.to_csv(
        os.path.join(results_dir, f"fold{fold}{suffix}_results.csv"), index=False
    )
    return predictions, eval_skempi_three_modes(predictions)


def train_fold(model, optimizer, train_loader, val_loader, device, config, save_dir, fold):
    best_pearson = float("-inf")
    best_path = os.path.join(save_dir, "checkpoint", f"fold{fold}_best.pt")
    model.to(device)

    for iteration in range(config.max_iter):
        batch = move_batch(next(train_loader), device)
        if "3VR6_ABCDEF_GH" in batch["complex"]:
            continue

        model.train()
        loss, _, _ = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()

        if iteration % 100 == 0:
            print(
                f"fold={fold} iter={iteration} loss={loss.item():.6f}",
                flush=True,
            )

        should_validate = (
            iteration > 0
            and iteration % config.val_freq == 0
            and (
                config.max_iter < 50000
                or iteration >= getattr(config, "validation_start", 25000)
            )
        )
        if should_validate:
            _, metrics = validate_fold(
                model,
                val_loader,
                device,
                save_dir,
                fold,
                iteration=iteration,
            )
            pearson = metric_value(metrics, "all", "pearson")
            print(f"fold={fold} iter={iteration} all_pearson={pearson:.6f}", flush=True)
            if pearson > best_pearson:
                best_pearson = pearson
                torch.save(model.state_dict(), best_path)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"), strict=True)
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_path, best_pearson


def main():
    parser = argparse.ArgumentParser(description="Train PhysCoop-DDG on SKEMPI v2")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config_path, encoding="utf-8") as handle:
        config = SimpleNamespace(**json.load(handle))
    if getattr(config, "use_foldx_features", False):
        raise ValueError("This release does not use FoldX features")
    if getattr(config, "use_physquad", False):
        raise ValueError("This release excludes the legacy PhysQuad variant")

    set_seed(config.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    save_dir = make_run_dir(config, args.tag)
    config_path = os.path.join(save_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(vars(config), handle, indent=2)

    data_manager = SkempiDatasetManager(
        config=config,
        num_cvfolds=config.num_cvfolds,
        num_workers=getattr(config, "num_workers", 4),
    )
    cv_manager = CrossValidation(config, config.num_cvfolds, DDGPredictor)
    if getattr(config, "ckpt_path", None):
        checkpoint = torch.load(config.ckpt_path, map_location="cpu")
        cv_manager.load_mpnn_state_dict(checkpoint)
    if getattr(config, "sft_ckpt_path", None):
        checkpoint = torch.load(config.sft_ckpt_path, map_location="cpu")
        cv_manager.load_state_dict_inference(checkpoint)

    folds = resolve_folds(config)
    summary = []
    for fold in folds:
        model, optimizer, _ = cv_manager.get(fold)
        best_path, best_pearson = train_fold(
            model,
            optimizer,
            data_manager.get_train_loader(fold),
            data_manager.get_val_loader(fold),
            device,
            config,
            save_dir,
            fold,
        )
        predictions, metrics = validate_fold(
            model, data_manager.get_val_loader(fold), device, save_dir, fold
        )
        predictions.to_csv(os.path.join(save_dir, f"fold{fold}_final_predictions.csv"), index=False)
        for _, row in metrics.iterrows():
            summary.append({"fold": fold, **row.to_dict(), "best_pearson": best_pearson})
        print(f"fold={fold} checkpoint={best_path}", flush=True)

    pd.DataFrame(summary).to_csv(
        os.path.join(save_dir, "fold_metrics.csv"), index=False
    )
    torch.save(
        cv_manager.state_dict(),
        os.path.join(save_dir, "checkpoint", "ddg_model.ckpt"),
    )
    print(f"experiment_dir={save_dir}")


if __name__ == "__main__":
    main()
