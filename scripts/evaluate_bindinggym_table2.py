#!/usr/bin/env python3
"""Reproduce the per-DMS metric convention used by H3-DDG Table 2.

The reference table reports unweighted means of per-DMS metrics for all
mutations, <3 mutations, and >=3 mutations. AUROC treats the top 10% of
experimental DMS scores within each assay as positives.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


REFERENCE_TABLE2 = (
    ("ProteinMPNN", "ALL", 0.0998, 0.2050, 0.5341, 3.4974),
    ("ProteinMPNN", "<3", 0.1137, 0.2439, 0.5404, 3.2328),
    ("ProteinMPNN", ">=3", 0.0734, 0.1614, 0.5930, 5.5921),
    ("BA-Cycle", "ALL", 0.1320, 0.1217, 0.5658, 1.2419),
    ("BA-Cycle", "<3", 0.1385, 0.1386, 0.5620, 1.0925),
    ("BA-Cycle", ">=3", 0.0830, 0.1955, 0.6190, 2.5822),
    ("Prompt-DDG", "ALL", 0.1880, 0.1818, 0.5198, 1.5216),
    ("Prompt-DDG", "<3", 0.2160, 0.2179, 0.5273, 1.3499),
    ("Prompt-DDG", ">=3", 0.1841, 0.2008, 0.6070, 3.5747),
    ("BA-DDG", "ALL", 0.2277, 0.2142, 0.5310, 1.1182),
    ("BA-DDG", "<3", 0.2553, 0.2471, 0.5395, 0.9716),
    ("BA-DDG", ">=3", 0.2037, 0.2259, 0.6307, 2.5191),
    ("H3-DDG", "ALL", 0.3057, 0.2725, 0.5703, 1.1294),
    ("H3-DDG", "<3", 0.3322, 0.3031, 0.5745, 1.0758),
    ("H3-DDG", ">=3", 0.2472, 0.2755, 0.6734, 2.4976),
)


def finite_values(frame, true_col, pred_col):
    values = frame[[true_col, pred_col]].apply(pd.to_numeric, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def correlation(values, method):
    if len(values) < 2:
        return np.nan
    true = values.iloc[:, 0].to_numpy()
    pred = values.iloc[:, 1].to_numpy()
    if np.unique(true).size < 2 or np.unique(pred).size < 2:
        return np.nan
    if method == "pearson":
        return float(pearsonr(true, pred).statistic)
    return float(spearmanr(true, pred).statistic)


def top_decile_auroc(values):
    true = values.iloc[:, 0].to_numpy()
    pred = values.iloc[:, 1].to_numpy()
    labels = true > np.percentile(true, 90)
    if np.unique(labels).size != 2:
        return np.nan
    return float(roc_auc_score(labels, pred))


def score_per_dms(frame, assay_col, true_col, pred_col, min_n):
    rows = []
    for assay_id, assay in frame.groupby(assay_col, dropna=False):
        values = finite_values(assay, true_col, pred_col)
        if len(values) < min_n:
            continue
        rows.append(
            {
                "assay_id": assay_id,
                "n_samples": int(len(values)),
                "pearson": correlation(values, "pearson"),
                "spearman": correlation(values, "spearman"),
                "auroc_top_decile": top_decile_auroc(values),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            (values.iloc[:, 0].to_numpy() - values.iloc[:, 1].to_numpy()) ** 2
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="PhysCoop-DDG (frozen SKEMPI)")
    parser.add_argument("--assay-col", default="assay_id")
    parser.add_argument("--true-col", default="binding_score")
    parser.add_argument("--pred-col", default="pred_binding_score")
    parser.add_argument("--status-col", default="status")
    parser.add_argument("--min-n", type=int, default=100)
    args = parser.parse_args()

    raw = pd.read_csv(args.predictions, low_memory=False)
    if args.status_col in raw:
        status = raw[args.status_col].fillna("ok").astype(str).str.lower()
        frame = raw[status.eq("ok")].copy()
    else:
        frame = raw.copy()

    required = {args.assay_col, args.true_col, args.pred_col, "num_muts"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    num_muts = pd.to_numeric(frame["num_muts"], errors="coerce")
    groups = {
        "ALL": frame,
        "<3": frame[num_muts.lt(3)],
        ">=3": frame[num_muts.ge(3)],
    }

    summaries = []
    per_dms = []
    for mutation_group, group in groups.items():
        assay_metrics = score_per_dms(
            group,
            assay_col=args.assay_col,
            true_col=args.true_col,
            pred_col=args.pred_col,
            min_n=args.min_n,
        )
        if assay_metrics.empty:
            raise ValueError(f"No evaluable DMS assays for mutation group {mutation_group}.")
        assay_metrics.insert(0, "mutation_group", mutation_group)
        per_dms.append(assay_metrics)
        summaries.append(
            {
                "method": args.method,
                "mutation_group": mutation_group,
                "n_samples": int(assay_metrics["n_samples"].sum()),
                "n_dms": int(len(assay_metrics)),
                "pearson": float(assay_metrics["pearson"].mean()),
                "spearman": float(assay_metrics["spearman"].mean()),
                "auroc_top_decile": float(assay_metrics["auroc_top_decile"].mean()),
                "rmse": float(assay_metrics["rmse"].mean()),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_dms_frame = pd.concat(per_dms, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    reference_frame = pd.DataFrame(
        REFERENCE_TABLE2,
        columns=["method", "mutation_group", "pearson", "spearman", "auroc_top_decile", "rmse"],
    )
    reference_frame.insert(2, "n_samples", np.nan)
    reference_frame.insert(3, "n_dms", np.nan)
    reference_frame["protocol"] = "H3-DDG Table 2 reference"
    summary_frame["protocol"] = "Frozen SKEMPI selected-checkpoint external evaluation"
    combined = pd.concat([reference_frame, summary_frame], ignore_index=True)

    per_dms_frame.to_csv(output_dir / "physcoop_per_dms_table2_metrics.csv", index=False)
    summary_frame.to_csv(output_dir / "physcoop_table2_metrics.csv", index=False)
    reference_frame.to_csv(output_dir / "reference_h3ddg_table2_extracted.csv", index=False)
    combined.to_csv(output_dir / "bindinggym_table2_comparison.csv", index=False)

    metadata = {
        "predictions": str(Path(args.predictions)),
        "method": args.method,
        "n_rows_input": int(len(raw)),
        "n_rows_successful": int(len(frame)),
        "min_n_per_dms": args.min_n,
        "mutation_groups": {
            "ALL": "all valid mutations",
            "<3": "one or two mutations",
            ">=3": "three or more mutations",
        },
        "per_dms_aggregation": "unweighted arithmetic mean",
        "auroc": "positive class is binding_score > within-DMS 90th percentile",
        "reference_source": "/Users/encore/Downloads/3112_Accurately_Predicting_Pro.pdf, Table 2",
        "reference_protocol_note": (
            "Reference Table 2 uses its published BindingGYM inter-assay protocol. "
            "The PhysCoop row is a frozen SKEMPI model external evaluation and is "
            "reported alongside, not as a directly matched protocol comparison."
        ),
    }
    with (output_dir / "table2_evaluation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(summary_frame.to_string(index=False))
    print(f"\nSaved Table 2-style metrics to {output_dir}")


if __name__ == "__main__":
    main()
