#!/usr/bin/env python3
"""Evaluate structure-resolved BindingGYM predictions.

The script intentionally consumes a normalized prediction table instead of
assuming a particular raw BindingGYM archive layout. This keeps dataset
parsing separate from assay-level evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _finite_pair(frame, true_col, pred_col):
    values = frame[[true_col, pred_col]].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return values


def _correlation(values, method):
    if len(values) < 2:
        return np.nan
    if values.iloc[:, 0].nunique() < 2 or values.iloc[:, 1].nunique() < 2:
        return np.nan
    if method == "spearman":
        return float(spearmanr(values.iloc[:, 0], values.iloc[:, 1]).statistic)
    return float(pearsonr(values.iloc[:, 0], values.iloc[:, 1]).statistic)


def score_assays(frame, assay_col, true_col, pred_col, min_n):
    rows = []
    for assay_id, assay in frame.groupby(assay_col, dropna=False):
        values = _finite_pair(assay, true_col, pred_col)
        rows.append(
            {
                "assay_id": assay_id,
                "n": int(len(values)),
                "spearman": _correlation(values, "spearman")
                if len(values) >= min_n
                else np.nan,
                "pearson": _correlation(values, "pearson")
                if len(values) >= min_n
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean(values, rng, repeats):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])
    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _nullable_float(value):
    return float(value) if value is not None and np.isfinite(value) else None


def resource_summary(frame, assay_metrics, status_col, assay_col):
    if status_col in frame:
        successful = frame[frame[status_col].fillna("ok").astype(str).str.lower().eq("ok")]
    else:
        successful = frame

    summary = {
        "n_rows_input": int(len(frame)),
        "n_rows_successful": int(len(successful)),
        "n_assays_input": int(frame[assay_col].nunique(dropna=True)),
        "n_assays_evaluable": int(assay_metrics["spearman"].notna().sum()),
    }
    if "complex_id" in successful:
        summary["n_complexes_successful"] = int(successful["complex_id"].nunique(dropna=True))
    if "preprocess_seconds" in successful:
        seconds = pd.to_numeric(successful["preprocess_seconds"], errors="coerce").sum()
        summary["preprocess_seconds"] = float(seconds) if np.isfinite(seconds) else None
    if "inference_seconds" in successful:
        seconds = pd.to_numeric(successful["inference_seconds"], errors="coerce").sum()
        summary["inference_seconds"] = float(seconds) if np.isfinite(seconds) else None
        if np.isfinite(seconds) and seconds > 0:
            summary["samples_per_second"] = float(len(successful) / seconds)
            if "complex_id" in successful:
                n_complexes = successful["complex_id"].nunique(dropna=True)
                summary["complexes_per_second"] = float(n_complexes / seconds)
    if "peak_gpu_mem_mb" in successful:
        peak = pd.to_numeric(successful["peak_gpu_mem_mb"], errors="coerce").max()
        summary["peak_gpu_mem_mb"] = float(peak) if np.isfinite(peak) else None
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compute assay-level BindingGYM metrics from a normalized prediction CSV."
    )
    parser.add_argument("--predictions", required=True, help="Normalized prediction CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for metric outputs.")
    parser.add_argument("--assay-col", default="assay_id")
    parser.add_argument("--true-col", default="binding_score")
    parser.add_argument("--pred-col", default="pred_binding_score")
    parser.add_argument(
        "--negate-ddg",
        action="store_true",
        help="Create pred_binding_score as -pred_ddg when pred-col is absent.",
    )
    parser.add_argument("--status-col", default="status")
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_frame = pd.read_csv(args.predictions)
    frame = raw_frame.copy()
    required = {args.assay_col, args.true_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if args.pred_col not in frame.columns:
        if not args.negate_ddg or "pred_ddg" not in frame.columns:
            raise ValueError(
                f"Missing {args.pred_col!r}; provide it or use --negate-ddg with pred_ddg."
            )
        frame[args.pred_col] = -pd.to_numeric(frame["pred_ddg"], errors="coerce")

    if args.status_col in frame.columns:
        status = frame[args.status_col].fillna("ok").astype(str).str.lower()
        frame = frame[status.eq("ok")].copy()

    assay_metrics = score_assays(
        frame,
        assay_col=args.assay_col,
        true_col=args.true_col,
        pred_col=args.pred_col,
        min_n=args.min_n,
    )
    rng = np.random.default_rng(args.seed)
    macro_spearman, ci_low, ci_high = bootstrap_mean(
        assay_metrics["spearman"].dropna().to_numpy(),
        rng=rng,
        repeats=args.bootstrap_repeats,
    )
    macro_pearson, _, _ = bootstrap_mean(
        assay_metrics["pearson"].dropna().to_numpy(),
        rng=rng,
        repeats=args.bootstrap_repeats,
    )

    summary = resource_summary(
        raw_frame,
        assay_metrics,
        args.status_col,
        args.assay_col,
    )
    valid_spearman = assay_metrics["spearman"].dropna()
    summary.update(
        {
            "macro_spearman": _nullable_float(macro_spearman),
            "macro_spearman_ci95": [_nullable_float(ci_low), _nullable_float(ci_high)],
            "macro_pearson": _nullable_float(macro_pearson),
            "median_spearman": _nullable_float(valid_spearman.median()),
            "positive_spearman_fraction": _nullable_float(
                (valid_spearman > 0).mean() if len(valid_spearman) else np.nan
            ),
            "min_n": args.min_n,
            "true_col": args.true_col,
            "pred_col": args.pred_col,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assay_metrics.to_csv(output_dir / "assay_metrics.csv", index=False)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
