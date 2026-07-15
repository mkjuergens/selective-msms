#!/usr/bin/env python3
"""Compile one canonical model/candidate score bundle into query-level revision features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ms_uq.evaluation.revision_features import target_ranks
from ms_uq.utils import resolve_candidate_paths
from scripts.run_meta_score_analysis import (
    build_features,
    canonical_views_for_metadata,
    load_scores,
    split_metadata,
)


def compile_query_scores(
    score_path: Path,
    fp_probs_path: Path,
    dataset_tsv: Path,
    helper_dir: Path,
    candidate_setting: str,
    split: str,
    out_path: Path,
    run_id: str,
    run_label: str,
    architecture: str,
    training_candidate_setting: str,
    evaluation_candidate_setting: str,
    pool_variant: str,
    query_mask_id: str,
    model_hash: str,
    candidate_pool_hash: str,
    temperature: float = 0.003,
    top_ks=(1, 5, 20),
    query_masks_path: Path | None = None,
    include_cardinality: bool = False,
    candidate_record_policy: str = "deduplicate",
    include_fingerprint_uncertainty: bool = False,
    candidate_tie_break: str = "candidate_id",
    aurc_convention: str = "discrete_prefix_mean",
    feature_convention: str = "canonical",
) -> pd.DataFrame:
    metadata = split_metadata(dataset_tsv, split)
    scores_flat, _, ptr, labels = load_scores(score_path)
    n_queries = ptr.numel() - 1
    metadata = metadata.iloc[:n_queries].copy()
    features, hits, _ = build_features(
        score_path, fp_probs_path, metadata, helper_dir, candidate_setting, top_ks,
        temperature, include_cardinality=include_cardinality,
        candidate_record_policy=candidate_record_policy,
        include_fingerprint_uncertainty=include_fingerprint_uncertainty,
        candidate_tie_break=candidate_tie_break,
        feature_convention=feature_convention,
    )
    _, candidate_fp_path, candidate_inchi_path = resolve_candidate_paths(helper_dir, candidate_setting)
    with np.load(candidate_fp_path) as candidate_fps, np.load(candidate_inchi_path) as candidate_inchis:
        ids_by_query, _ = canonical_views_for_metadata(
            candidate_inchis, candidate_fps, metadata, include_fps=False,
            record_policy=candidate_record_policy,
        )
    ranking_ids = ids_by_query if candidate_tie_break == "candidate_id" else None
    ranks = target_ranks(scores_flat, labels, ptr, candidate_ids=ranking_ids)
    bundle = torch.load(score_path, map_location="cpu")
    target_present = np.asarray(bundle.get("target_present", np.isfinite(ranks)), dtype=bool)
    raw_counts = np.asarray(bundle.get("raw_candidate_counts", np.diff(ptr.numpy())), dtype=np.int64)
    evaluation_counts = np.diff(ptr.numpy()).astype(np.int64)

    base = pd.DataFrame({
        "run_id": run_id,
        "split": split,
        "run_label": run_label,
        "architecture": architecture,
        "training_candidate_setting": training_candidate_setting,
        "evaluation_candidate_setting": evaluation_candidate_setting,
        "pool_variant": pool_variant,
        "model_hash": model_hash,
        "candidate_pool_hash": candidate_pool_hash,
        "query_mask_id": query_mask_id,
        "query_id": metadata["query_id"].astype(str),
        "molecule_group_id": metadata["molecule_group_id"].astype(str),
        "smiles": metadata["smiles"].astype(str),
        "target_present": target_present,
        "target_rank": ranks,
        "candidate_count_raw": raw_counts,
        "candidate_count": evaluation_counts,
        "candidate_record_policy": candidate_record_policy,
        "candidate_tie_break": candidate_tie_break,
        "T_train": 0.003,
        "T_eval": temperature,
        "aggregation": "score",
        "aurc_convention": aurc_convention,
        "feature_convention": feature_convention,
    })
    for name, values in features.items():
        base[name] = np.asarray(values, dtype=np.float64)

    if query_masks_path is not None:
        masks = pd.read_parquet(query_masks_path)
        mask_columns = ["query_id", query_mask_id]
        missing = [column for column in mask_columns if column not in masks]
        if missing:
            raise ValueError(f"Missing query-mask columns: {missing}")
        base = base.merge(masks[mask_columns], on="query_id", how="left", validate="one_to_one")
        if base[query_mask_id].isna().any():
            raise ValueError("Query mask does not cover every score row")
        base = base.loc[base[query_mask_id].astype(bool)].drop(columns=[query_mask_id])

    frames = []
    for k in top_ks:
        frame = base.copy()
        frame["K"] = int(k)
        frame["hit"] = (frame["target_rank"] <= k).astype(float)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if not (result.groupby("query_id")["hit"].apply(lambda v: list(v) == sorted(v)).all()):
        raise ValueError("Hit@1 <= Hit@5 <= Hit@20 assertion failed")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False, compression="zstd")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--fp_probs", type=Path, required=True)
    parser.add_argument("--dataset_tsv", type=Path, required=True)
    parser.add_argument("--helper_dir", type=Path, required=True)
    parser.add_argument("--candidate_setting", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--run_label", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--training_candidate_setting", required=True)
    parser.add_argument("--evaluation_candidate_setting", required=True)
    parser.add_argument("--pool_variant", required=True)
    parser.add_argument("--query_mask_id", required=True)
    parser.add_argument("--model_hash", required=True)
    parser.add_argument("--candidate_pool_hash", required=True)
    parser.add_argument("--temperature", type=float, default=0.003)
    parser.add_argument("--query_masks", type=Path)
    parser.add_argument("--include_cardinality", action="store_true")
    parser.add_argument("--candidate_record_policy", choices=["preserve", "deduplicate"], default="deduplicate")
    parser.add_argument("--include_fingerprint_uncertainty", action="store_true")
    parser.add_argument("--candidate_tie_break", choices=["source_order", "candidate_id"], default="candidate_id")
    parser.add_argument("--aurc_convention", choices=["discrete_prefix_mean", "manuscript_trapezoid_seed42"], default="discrete_prefix_mean")
    parser.add_argument("--feature_convention", choices=["canonical", "manuscript"], default="canonical")
    args = parser.parse_args()
    result = compile_query_scores(
        args.score, args.fp_probs, args.dataset_tsv, args.helper_dir, args.candidate_setting,
        args.split, args.out, args.run_id, args.run_label, args.architecture,
        args.training_candidate_setting, args.evaluation_candidate_setting,
        args.pool_variant, args.query_mask_id, args.model_hash, args.candidate_pool_hash,
        args.temperature, query_masks_path=args.query_masks,
        include_cardinality=args.include_cardinality,
        candidate_record_policy=args.candidate_record_policy,
        include_fingerprint_uncertainty=args.include_fingerprint_uncertainty,
        candidate_tie_break=args.candidate_tie_break,
        aurc_convention=args.aurc_convention,
        feature_convention=args.feature_convention,
    )
    print(f"Saved {len(result)} query/K rows to {args.out}")


if __name__ == "__main__":
    main()
