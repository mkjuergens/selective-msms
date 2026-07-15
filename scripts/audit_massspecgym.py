#!/usr/bin/env python3
"""Audit MassSpecGym folds, duplicate rows, candidate caps, and optional label equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ms_uq.data import candidate_fps_to_dense
from ms_uq.utils import resolve_candidate_paths


def normalize_inchikey(value) -> str:
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).split("-")[0]


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"metric": "rows", "value": int(len(df))}]
    for fold, count in df["fold"].value_counts().sort_index().items():
        rows.append({"metric": f"rows_{fold}", "value": int(count)})

    for key in ["smiles", "inchikey"]:
        overlaps = 0
        folds = sorted(df["fold"].dropna().unique())
        for i, left in enumerate(folds):
            left_values = set(df.loc[df["fold"] == left, key].dropna())
            for right in folds[i + 1:]:
                right_values = set(df.loc[df["fold"] == right, key].dropna())
                overlaps += len(left_values & right_values)
        rows.append({"metric": f"cross_fold_{key}_overlap", "value": int(overlaps)})

    exclude = [c for c in ["identifier"] if c in df.columns]
    dup_mask = df.duplicated(subset=[c for c in df.columns if c not in exclude], keep=False)
    rows.append({"metric": "duplicate_rows_excluding_identifier", "value": int(dup_mask.sum())})
    rows.append({"metric": "duplicate_groups_excluding_identifier", "value": int(df.loc[dup_mask].drop(columns=exclude).drop_duplicates().shape[0])})
    for fold, count in df.loc[dup_mask, "fold"].value_counts().sort_index().items():
        rows.append({"metric": f"duplicate_rows_{fold}", "value": int(count)})
    return pd.DataFrame(rows)


def summarize_candidates(candidate_json: Path) -> pd.DataFrame:
    data = json.loads(candidate_json.read_text())
    sizes = np.asarray([len(v.get("candidates", v)) if isinstance(v, dict) else len(v) for v in data.values()], dtype=np.int64)
    rows = [
        {"metric": "candidate_queries", "value": int(len(sizes))},
        {"metric": "candidate_total", "value": int(sizes.sum())},
        {"metric": "candidate_min", "value": int(sizes.min())},
        {"metric": "candidate_median", "value": float(np.median(sizes))},
        {"metric": "candidate_mean", "value": float(sizes.mean())},
        {"metric": "candidate_p90", "value": float(np.quantile(sizes, 0.90))},
        {"metric": "candidate_p95", "value": float(np.quantile(sizes, 0.95))},
        {"metric": "candidate_p99", "value": float(np.quantile(sizes, 0.99))},
        {"metric": "candidate_max", "value": int(sizes.max())},
        {"metric": "candidate_fraction_at_256", "value": float((sizes == 256).mean())},
        {"metric": "candidate_fraction_above_256", "value": float((sizes > 256).mean())},
    ]
    return pd.DataFrame(rows)


def audit_label_equivalence(df: pd.DataFrame, helper_dir: Path, candidate_setting: str, fold: str) -> pd.DataFrame:
    _, cand_fp_path, cand_inchi_path = resolve_candidate_paths(helper_dir, candidate_setting)
    fp = np.load(helper_dir / "fp_4096.npy", mmap_mode="r")
    inchis = np.load(helper_dir / "inchis.npy", mmap_mode="r")
    cand_fps = np.load(cand_fp_path)
    cand_inchis = np.load(cand_inchi_path)

    sub = df[df["fold"] == fold].copy()
    rows = []
    for smiles, group in sub.groupby("smiles", sort=False):
        idx = int(group.index[0])
        query_fp = fp[idx]
        query_inchi = normalize_inchikey(inchis[idx])
        dense_fps = candidate_fps_to_dense(
            cand_fps[smiles],
            n_candidates=len(cand_inchis[smiles]),
            fp_size=int(query_fp.shape[0]),
        )
        fp_labels = (dense_fps == query_fp.astype(bool)).all(axis=1)
        inchi_labels = np.asarray([normalize_inchikey(v) == query_inchi for v in cand_inchis[smiles]])
        rows.append({
            "smiles": smiles,
            "fold": fold,
            "n_rows": int(len(group)),
            "n_candidates": int(len(fp_labels)),
            "fp_positive": int(fp_labels.sum()),
            "inchi_positive": int(inchi_labels.sum()),
            "labels_equal": bool(np.array_equal(fp_labels, inchi_labels)),
            "n_label_differences": int(np.count_nonzero(fp_labels != inchi_labels)),
        })
    return pd.DataFrame(rows)


def write_markdown(out_path: Path, tables: Dict[str, pd.DataFrame]) -> None:
    lines: List[str] = ["# MassSpecGym Audit", ""]
    for name, table in tables.items():
        if name == "label_equivalence":
            summary = pd.DataFrame([
                {"metric": "unique_queries", "value": int(len(table))},
                {"metric": "labels_equal_queries", "value": int(table["labels_equal"].sum())},
                {"metric": "labels_mismatch_queries", "value": int((~table["labels_equal"]).sum())},
                {"metric": "affected_rows", "value": int(table.loc[~table["labels_equal"], "n_rows"].sum())},
                {"metric": "label_differences", "value": int(table["n_label_differences"].sum())},
            ])
            lines.extend([f"## {name}_summary", "", "```text", summary.to_string(index=False), "```", ""])
            preview = table.loc[~table["labels_equal"]].head(10)
            if not preview.empty:
                lines.extend([f"## {name}_mismatch_preview", "", "```text", preview.to_string(index=False), "```", ""])
            continue
        lines.extend([f"## {name}", "", "```text", table.to_string(index=False), "```", ""])
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_tsv", type=Path, required=True)
    ap.add_argument("--helper_dir", type=Path, required=True)
    ap.add_argument("--candidate_setting", default="formula")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--label_equivalence_fold", default="test")
    ap.add_argument("--skip_label_equivalence", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset_tsv, sep="\t")
    candidate_json, _, _ = resolve_candidate_paths(args.helper_dir, args.candidate_setting)

    tables: Dict[str, pd.DataFrame] = {
        "dataset": summarize_dataset(df),
        "candidates": summarize_candidates(candidate_json),
    }
    if not args.skip_label_equivalence:
        tables["label_equivalence"] = audit_label_equivalence(
            df, args.helper_dir, args.candidate_setting, args.label_equivalence_fold
        )

    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_markdown(args.out_dir / "audit_summary.md", tables)
    print(f"Saved audit outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
