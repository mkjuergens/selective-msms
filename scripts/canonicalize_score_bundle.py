#!/usr/bin/env python3
"""Canonicalize a ragged candidate score bundle by TSV identity and deterministic deduplication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ms_uq.evaluation.revision_candidates import canonical_candidate_indices, normalize_inchikey
from ms_uq.utils import resolve_candidate_paths


def preserve_score_bundle(
    input_path: Path,
    output_path: Path,
    dataset_tsv: Path,
    helper_dir: Path,
    candidate_setting: str,
    split: str,
    label_mode: str,
    query_identity_source: str,
    max_queries: int | None = None,
) -> dict:
    """Attach provenance while preserving every candidate occurrence and source order."""
    raw = torch.load(input_path, map_location="cpu", mmap=True)
    if "labels_flat" not in raw:
        raise ValueError(f"{input_path} does not contain labels_flat")
    if max_queries is not None:
        n_available = raw["ptr"].numel() - 1
        n_keep = min(int(max_queries), n_available)
        flat_end = int(raw["ptr"][n_keep])
        raw = dict(raw)
        raw["scores_flat"] = raw["scores_flat"][:flat_end].clone()
        if "scores_stack_flat" in raw:
            raw["scores_stack_flat"] = raw["scores_stack_flat"][:, :flat_end].clone()
        raw["labels_flat"] = raw["labels_flat"][:flat_end].clone()
        raw["ptr"] = raw["ptr"][:n_keep + 1].clone()
    stack = raw.get("scores_stack_flat")
    if stack is None:
        stack = raw["scores_flat"].unsqueeze(0)
    ptr = raw["ptr"].long()
    labels = raw["labels_flat"].float()
    if int(ptr[-1]) != raw["scores_flat"].numel() or labels.numel() != int(ptr[-1]):
        raise ValueError("Score, label, and pointer lengths do not align")

    metadata = pd.read_csv(dataset_tsv, sep="\t")
    metadata = metadata.loc[metadata["fold"] == split].reset_index(drop=True)
    n_queries = ptr.numel() - 1
    if len(metadata) < n_queries:
        raise ValueError(f"Only {len(metadata)} metadata rows for {n_queries} score queries")
    metadata = metadata.iloc[:n_queries].copy()

    _, _, candidate_inchi_path = resolve_candidate_paths(helper_dir, candidate_setting)
    counts = np.diff(ptr.numpy()).astype(np.int64)
    expected_by_smiles = {}
    with np.load(candidate_inchi_path) as candidate_inchis:
        for query_index, smiles in enumerate(metadata["smiles"].astype(str)):
            expected = expected_by_smiles.get(smiles)
            if expected is None:
                expected = len(candidate_inchis[smiles])
                expected_by_smiles[smiles] = expected
            if counts[query_index] != expected:
                raise ValueError(
                    f"Candidate count mismatch for query {query_index} ({smiles}): "
                    f"scores={counts[query_index]}, helper={expected}"
                )

    target_present = np.asarray([
        bool(labels[int(ptr[i]):int(ptr[i + 1])].any()) for i in range(n_queries)
    ], dtype=bool)
    result = dict(raw)
    result.update({
        "ptr": ptr,
        "candidate_identity": "candidate records in helper order",
        "candidate_record_policy": "preserve",
        "candidate_deduplication": "none",
        "candidate_tie_break": "source_order",
        "label_mode": label_mode,
        "query_identity_source": query_identity_source,
        "query_ids": metadata["identifier"].astype(str).tolist(),
        "query_smiles": metadata["smiles"].astype(str).tolist(),
        "molecule_group_ids": metadata["inchikey"].map(normalize_inchikey).tolist(),
        "target_present": torch.as_tensor(target_present, dtype=torch.bool),
        "raw_candidate_counts": torch.as_tensor(counts, dtype=torch.long),
        "record_candidate_counts": torch.as_tensor(counts, dtype=torch.long),
        "source_score_path": str(input_path.resolve()),
        "candidate_setting": candidate_setting,
        "split": split,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    summary = {
        "n_queries": n_queries,
        "n_members": int(stack.shape[0]),
        "n_raw_scores": int(raw["scores_flat"].numel()),
        "n_record_scores": int(raw["scores_flat"].numel()),
        "n_target_absent": int((~target_present).sum()),
        "candidate_record_policy": "preserve",
        "label_mode": label_mode,
        "query_identity_source": query_identity_source,
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "candidate_setting": candidate_setting,
        "split": split,
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def canonicalize_score_bundle(
    input_path: Path,
    output_path: Path,
    dataset_tsv: Path,
    helper_dir: Path,
    candidate_setting: str,
    split: str,
    fp_size: int = 4096,
) -> dict:
    raw = torch.load(input_path, map_location="cpu")
    stack = raw.get("scores_stack_flat")
    if stack is None:
        stack = raw["scores_flat"].unsqueeze(0)
    stack = stack.double()
    ptr = raw["ptr"].long()
    metadata = pd.read_csv(dataset_tsv, sep="\t")
    metadata = metadata.loc[metadata["fold"] == split].reset_index(drop=True)
    n_queries = ptr.numel() - 1
    if len(metadata) < n_queries:
        raise ValueError(f"Only {len(metadata)} metadata rows for {n_queries} score queries")
    metadata = metadata.iloc[:n_queries].copy()

    _, candidate_fp_path, candidate_inchi_path = resolve_candidate_paths(helper_dir, candidate_setting)
    score_parts, label_parts = [], []
    canonical_ptr = [0]
    target_present, raw_counts, deduplicated_counts = [], [], []
    index_cache = {}
    with np.load(candidate_fp_path) as candidate_fps, np.load(candidate_inchi_path) as candidate_inchis:
        for query_index, row in enumerate(metadata.itertuples(index=False)):
            start, end = int(ptr[query_index]), int(ptr[query_index + 1])
            smiles = str(row.smiles)
            if smiles not in index_cache:
                candidate_ids, indices = canonical_candidate_indices(
                    candidate_inchis[smiles], candidate_fps[smiles], fp_size=fp_size
                )
                index_cache[smiles] = (candidate_ids, torch.as_tensor(indices, dtype=torch.long))
            candidate_ids, indices = index_cache[smiles]
            if end - start != len(candidate_inchis[smiles]):
                raise ValueError(
                    f"Candidate count mismatch for query {query_index} ({smiles}): "
                    f"scores={end-start}, helper={len(candidate_inchis[smiles])}"
                )
            local = stack[:, start:end].index_select(1, indices)
            target_id = normalize_inchikey(row.inchikey)
            labels = torch.as_tensor(candidate_ids == target_id, dtype=torch.float64)
            score_parts.append(local)
            label_parts.append(labels)
            canonical_ptr.append(canonical_ptr[-1] + len(candidate_ids))
            target_present.append(bool(labels.any()))
            raw_counts.append(end - start)
            deduplicated_counts.append(len(candidate_ids))

    canonical_stack = torch.cat(score_parts, dim=1) if score_parts else torch.empty((stack.shape[0], 0), dtype=torch.float64)
    aggregate = canonical_stack.mean(dim=0)
    if not torch.isfinite(canonical_stack).all():
        raise ValueError("Canonical member scores contain non-finite values")
    if aggregate.numel() and (float(aggregate.min()) < -1e-10 or float(aggregate.max()) > 1.0 + 1e-10):
        raise ValueError("Canonical cosine similarities fall outside [0, 1]")
    result = {
        "scores_flat": aggregate,
        "scores_stack_flat": canonical_stack,
        "ptr": torch.as_tensor(canonical_ptr, dtype=torch.long),
        "labels_flat": torch.cat(label_parts) if label_parts else torch.empty(0, dtype=torch.float64),
        "aggregation": "score",
        "metric_dtype": "float64",
        "candidate_identity": "TSV InChIKey connectivity block",
        "candidate_deduplication": "minimum SHA-256 fingerprint representative; candidate-ID order",
        "query_ids": metadata["identifier"].astype(str).tolist(),
        "query_smiles": metadata["smiles"].astype(str).tolist(),
        "molecule_group_ids": metadata["inchikey"].map(normalize_inchikey).tolist(),
        "target_present": np.asarray(target_present, dtype=bool),
        "raw_candidate_counts": np.asarray(raw_counts, dtype=np.int64),
        "deduplicated_candidate_counts": np.asarray(deduplicated_counts, dtype=np.int64),
        "source_score_path": str(input_path.resolve()),
        "candidate_setting": candidate_setting,
        "split": split,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    summary = {
        "n_queries": n_queries,
        "n_members": int(canonical_stack.shape[0]),
        "n_raw_scores": int(stack.shape[1]),
        "n_canonical_scores": int(canonical_stack.shape[1]),
        "n_target_absent": int((~np.asarray(target_present)).sum()),
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "candidate_setting": candidate_setting,
        "split": split,
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset_tsv", type=Path, required=True)
    parser.add_argument("--helper_dir", type=Path, required=True)
    parser.add_argument("--candidate_setting", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--record_policy", choices=["preserve", "deduplicate"], default="deduplicate")
    parser.add_argument("--label_mode", choices=["fingerprint", "inchikey", "inchikey_fallback"], default="fingerprint")
    parser.add_argument("--query_identity_source", choices=["precomputed", "tsv"], default="precomputed")
    parser.add_argument("--max_queries", type=int)
    args = parser.parse_args()
    if args.record_policy == "preserve":
        summary = preserve_score_bundle(
            args.input, args.output, args.dataset_tsv, args.helper_dir,
            args.candidate_setting, args.split, args.label_mode, args.query_identity_source,
            args.max_queries,
        )
    else:
        summary = canonicalize_score_bundle(
            args.input, args.output, args.dataset_tsv, args.helper_dir,
            args.candidate_setting, args.split,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
