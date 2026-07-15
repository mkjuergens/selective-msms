"""Deterministic candidate-set construction, auditing, and summaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ms_uq.data import candidate_fps_to_dense


CANDIDATE_CAP = 256


def summarize_candidate_sizes(setting: str, sizes: np.ndarray, cap: int = CANDIDATE_CAP) -> dict:
    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.size == 0:
        raise ValueError("Cannot summarize an empty candidate-size array")
    return {
        "setting": setting,
        "n_queries": int(sizes.size),
        "min": int(sizes.min()),
        "q25": float(np.quantile(sizes, 0.25)),
        "median": float(np.median(sizes)),
        "mean": float(sizes.mean()),
        "q75": float(np.quantile(sizes, 0.75)),
        "max": int(sizes.max()),
        f"n_equal_{cap}": int(np.count_nonzero(sizes == cap)),
        f"fraction_equal_{cap}": float(np.mean(sizes == cap)),
        f"n_above_{cap}": int(np.count_nonzero(sizes > cap)),
        f"fraction_above_{cap}": float(np.mean(sizes > cap)),
    }


def normalize_inchikey(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    return str(value).split("-")[0]


def truncation_key(candidate_id: str, seed: int) -> Tuple[bytes, str]:
    candidate_id = normalize_inchikey(candidate_id)
    digest = hashlib.sha256(f"{seed}|{candidate_id}".encode("utf-8")).digest()
    return digest, candidate_id


def candidate_fingerprint_rows(raw_fps: np.ndarray, n_candidates: int, fp_size: int = 4096) -> np.ndarray:
    """Return one dense or packed row per candidate without changing representation."""
    arr = np.asarray(raw_fps)
    if arr.ndim == 2:
        if arr.shape[0] != n_candidates:
            raise ValueError("Candidate fingerprint and identity counts differ")
        return arr
    if arr.ndim == 1 and arr.dtype == np.uint8:
        bytes_per_fp = (fp_size + 7) // 8
        expected = n_candidates * bytes_per_fp
        if arr.size < expected:
            raise ValueError("Packed candidate fingerprints are shorter than expected")
        return arr[:expected].reshape(n_candidates, bytes_per_fp)
    raise ValueError(f"Unsupported candidate fingerprint layout: {arr.shape} {arr.dtype}")


def canonical_candidate_indices(
    candidate_ids: Sequence, raw_fps: np.ndarray, fp_size: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose an order-independent fingerprint representative and sort identities."""
    normalized = np.asarray([normalize_inchikey(value) for value in candidate_ids], dtype=object)
    rows = candidate_fingerprint_rows(raw_fps, len(normalized), fp_size=fp_size)
    representative: Dict[str, Tuple[bytes, int]] = {}
    for index, (candidate_id, fp_row) in enumerate(zip(normalized, rows)):
        fp_hash = hashlib.sha256(np.ascontiguousarray(fp_row).tobytes()).digest()
        current = representative.get(str(candidate_id))
        if current is None or fp_hash < current[0]:
            representative[str(candidate_id)] = (fp_hash, index)
    ordered_ids = np.asarray(sorted(representative), dtype=object)
    indices = np.asarray([representative[str(candidate_id)][1] for candidate_id in ordered_ids], dtype=np.int64)
    return ordered_ids, indices


def canonical_candidate_view(
    candidate_ids: Sequence, raw_fps: np.ndarray, fp_size: int = 4096
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered_ids, indices = canonical_candidate_indices(candidate_ids, raw_fps, fp_size=fp_size)
    rows = candidate_fingerprint_rows(raw_fps, len(candidate_ids), fp_size=fp_size)
    selected = rows[indices]
    if np.asarray(raw_fps).ndim == 1:
        selected = selected.reshape(-1)
    return ordered_ids, selected, indices


@dataclass(frozen=True)
class CandidateSelection:
    candidate_ids: Tuple[str, ...]
    source_indices: Tuple[int, ...]
    inserted_target_ids: Tuple[str, ...]
    duplicate_occurrences: int
    target_ids: Tuple[str, ...]
    natural_ids: Tuple[str, ...]


def select_candidate_indices(
    candidate_ids: Sequence,
    target_ids: Iterable[str],
    cap: int = 256,
    seed: int = 42,
    representative_indices: Mapping[str, int] | None = None,
) -> CandidateSelection:
    """Deduplicate by identity and create protected and natural deterministic caps."""
    normalized = [normalize_inchikey(value) for value in candidate_ids]
    first_index: Dict[str, int] = {}
    if representative_indices is None:
        for index, candidate_id in enumerate(normalized):
            first_index.setdefault(candidate_id, index)
    else:
        first_index.update({normalize_inchikey(key): int(value) for key, value in representative_indices.items()})
    unique_ids = list(first_index)
    target_ids = tuple(sorted({normalize_inchikey(value) for value in target_ids if str(value)}))
    present_targets = [candidate_id for candidate_id in target_ids if candidate_id in first_index]
    inserted_targets = [candidate_id for candidate_id in target_ids if candidate_id not in first_index]
    if len(target_ids) > cap:
        raise ValueError(f"Query has {len(target_ids)} target identities but cap={cap}")

    negatives = [candidate_id for candidate_id in unique_ids if candidate_id not in target_ids]
    negatives.sort(key=lambda value: truncation_key(value, seed))
    protected_ids = present_targets + inserted_targets + negatives[: max(cap - len(target_ids), 0)]
    protected_ids = protected_ids[:cap]
    natural_ids = sorted(unique_ids, key=lambda value: truncation_key(value, seed))[:cap]
    source_indices = tuple(first_index.get(candidate_id, -1) for candidate_id in protected_ids)
    return CandidateSelection(
        candidate_ids=tuple(protected_ids),
        source_indices=source_indices,
        inserted_target_ids=tuple(inserted_targets),
        duplicate_occurrences=len(normalized) - len(unique_ids),
        target_ids=target_ids,
        natural_ids=tuple(natural_ids),
    )


def _select_fingerprints(
    raw_fps: np.ndarray,
    n_candidates: int,
    source_indices: Sequence[int],
    inserted_fps: Mapping[str, np.ndarray],
    candidate_ids: Sequence[str],
    fp_size: int,
) -> np.ndarray:
    arr = np.asarray(raw_fps)
    dense_output = arr.ndim == 2
    if dense_output:
        rows = arr.astype(bool, copy=False)
    elif arr.ndim == 1 and arr.dtype == np.uint8:
        bytes_per_fp = (fp_size + 7) // 8
        rows = arr[: n_candidates * bytes_per_fp].reshape(n_candidates, bytes_per_fp)
    else:
        raise ValueError(f"Unsupported candidate fingerprint layout: {arr.shape} {arr.dtype}")

    selected = []
    for candidate_id, source_index in zip(candidate_ids, source_indices):
        if source_index >= 0:
            selected.append(rows[source_index])
        else:
            query_fp = np.asarray(inserted_fps[candidate_id], dtype=np.uint8)
            selected.append(query_fp.astype(bool) if dense_output else np.packbits(query_fp, bitorder="big"))
    stacked = np.stack(selected) if selected else np.empty((0, rows.shape[1]), dtype=rows.dtype)
    return stacked if dense_output else stacked.reshape(-1)


def _natural_fingerprints(
    raw_fps: np.ndarray,
    candidate_ids: Sequence,
    natural_ids: Sequence[str],
    fp_size: int,
) -> np.ndarray:
    canonical_ids, canonical_indices = canonical_candidate_indices(candidate_ids, raw_fps, fp_size=fp_size)
    representative = dict(zip(canonical_ids.tolist(), canonical_indices.tolist()))
    indices = [representative[candidate_id] for candidate_id in natural_ids]
    return _select_fingerprints(raw_fps, len(candidate_ids), indices, {}, natural_ids, fp_size)


@dataclass(frozen=True)
class RecordCandidateSelection:
    source_indices: Tuple[int, ...]
    target_indices: Tuple[int, ...]
    n_exact_duplicate_occurrences: int
    n_connectivity_duplicate_occurrences: int


def record_truncation_key(record: str, source_index: int, seed: int) -> Tuple[bytes, int]:
    """Hash a candidate occurrence, not just its molecular identity."""
    payload = f"{seed}|{source_index}|{record}".encode("utf-8")
    return hashlib.sha256(payload).digest(), int(source_index)


def select_record_candidate_indices(
    candidate_records: Sequence,
    candidate_ids: Sequence,
    target_ids: Iterable[str],
    cap: int = 256,
    seed: int = 42,
) -> RecordCandidateSelection:
    """Select a deterministic cap while retaining duplicate occurrences and all targets."""
    records = [str(value) for value in candidate_records]
    normalized = [normalize_inchikey(value) for value in candidate_ids]
    if len(records) != len(normalized):
        raise ValueError("Candidate record and identity counts differ")
    targets = {normalize_inchikey(value) for value in target_ids if str(value)}
    target_indices = [index for index, value in enumerate(normalized) if value in targets]
    if not target_indices:
        raise ValueError("The uncapped source does not contain the target candidate")
    if len(target_indices) > cap:
        raise ValueError(f"Query has {len(target_indices)} target occurrences but cap={cap}")

    if len(records) <= cap:
        selected = list(range(len(records)))
    else:
        target_set = set(target_indices)
        negatives = [index for index in range(len(records)) if index not in target_set]
        negatives.sort(key=lambda index: record_truncation_key(records[index], index, seed))
        selected = sorted(target_indices + negatives[: cap - len(target_indices)])
    return RecordCandidateSelection(
        source_indices=tuple(selected),
        target_indices=tuple(target_indices),
        n_exact_duplicate_occurrences=len(records) - len(set(records)),
        n_connectivity_duplicate_occurrences=len(normalized) - len(set(normalized)),
    )


def build_record_preserving_formula_cap(
    dataset_tsv: Path,
    source_json_path: Path,
    source_fp_path: Path,
    source_inchi_path: Path,
    query_inchi_path: Path,
    output_dir: Path,
    fold: str = "test",
    cap: int = 256,
    seed: int = 42,
    fp_size: int = 4096,
    max_queries: int | None = None,
    write_manifest: bool = False,
) -> pd.DataFrame:
    """Cap PubChem candidate occurrences without identity deduplication or target insertion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(dataset_tsv, sep="\t", dtype=str)
    query_inchis = np.load(query_inchi_path, allow_pickle=False)
    if len(query_inchis) != len(metadata):
        raise ValueError("Precomputed query InChIKey count does not match the dataset TSV")
    metadata["query_identity"] = [normalize_inchikey(value) for value in query_inchis]
    fold_rows = metadata.loc[metadata["fold"] == fold].copy()
    identity_counts = fold_rows.groupby("smiles", sort=False)["query_identity"].nunique()
    if (identity_counts > 1).any():
        raise ValueError("One query SMILES maps to multiple precomputed target identities")
    targets = fold_rows.groupby("smiles", sort=False)["query_identity"].first()

    with source_json_path.open() as handle:
        source_records = json.load(handle)
    queries = [query for query in targets.index if query in source_records]
    if len(queries) != len(targets):
        missing = len(targets) - len(queries)
        raise ValueError(f"Record source misses {missing} query keys for fold={fold}")
    if max_queries is not None:
        queries = queries[: int(max_queries)]

    capped_fps: Dict[str, np.ndarray] = {}
    capped_inchis: Dict[str, np.ndarray] = {}
    capped_json: Dict[str, List[str]] = {}
    summary_rows = []
    manifest_rows = []
    with np.load(source_fp_path) as source_fps, np.load(source_inchi_path) as source_inchis:
        for query_index, query in enumerate(queries):
            records = list(source_records[query])
            raw_ids = source_inchis[query]
            if len(records) != len(raw_ids):
                raise ValueError(
                    f"Candidate JSON/InChI count mismatch for {query}: {len(records)} != {len(raw_ids)}"
                )
            selection = select_record_candidate_indices(
                records, raw_ids, [targets.loc[query]], cap=cap, seed=seed
            )
            indices = np.asarray(selection.source_indices, dtype=np.int64)
            selected_ids = np.asarray(raw_ids)[indices]
            selected_records = [records[index] for index in indices]
            capped_fps[query] = _select_fingerprints(
                source_fps[query], len(raw_ids), indices, {}, selected_records, fp_size,
            )
            capped_inchis[query] = selected_ids
            capped_json[query] = selected_records
            normalized_ids = [normalize_inchikey(value) for value in raw_ids]
            target_id = targets.loc[query]
            summary_rows.append({
                "query_smiles": query,
                "n_uncapped_records": len(records),
                "n_capped_records": len(indices),
                "n_exact_duplicate_occurrences": selection.n_exact_duplicate_occurrences,
                "n_connectivity_duplicate_occurrences": selection.n_connectivity_duplicate_occurrences,
                "n_target_occurrences_uncapped": len(selection.target_indices),
                "n_target_occurrences_capped": int(sum(normalized_ids[index] == target_id for index in indices)),
                "target_inserted": False,
                "source_order_preserved": True,
            })
            if write_manifest:
                selected_set = set(selection.source_indices)
                target_set = set(selection.target_indices)
                manifest_rows.extend({
                    "query_smiles": query,
                    "source_index": index,
                    "candidate_record": records[index],
                    "candidate_id": normalized_ids[index],
                    "is_target": index in target_set,
                    "included_in_cap": index in selected_set,
                } for index in range(len(records)))
            if (query_index + 1) % 100 == 0:
                print(f"Prepared record-preserving caps for {query_index + 1}/{len(queries)} query keys")

    prefix = output_dir / "MassSpecGym_retrieval_candidates_formula_pubchem_record_capped256"
    np.savez_compressed(prefix.with_name(prefix.name + "_fps.npz"), **capped_fps)
    np.savez_compressed(prefix.with_name(prefix.name + "_inchi.npz"), **capped_inchis)
    prefix.with_suffix(".json").write_text(json.dumps(capped_json, sort_keys=True))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "candidate_record_control_summary.csv", index=False)
    if write_manifest:
        pd.DataFrame(manifest_rows).to_parquet(
            output_dir / "candidate_record_manifest.parquet", index=False, compression="zstd"
        )
    return summary


def build_formula_candidate_controls(
    dataset_tsv: Path,
    source_fp_path: Path,
    source_inchi_path: Path,
    query_fp_path: Path,
    output_dir: Path,
    fold: str = "test",
    cap: int = 256,
    seed: int = 42,
    fp_size: int = 4096,
    max_queries: int | None = None,
    write_manifest: bool = True,
    write_natural: bool = True,
) -> pd.DataFrame:
    """Build a protected cap and, optionally, the natural-cap sensitivity helper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(dataset_tsv, sep="\t", dtype=str)
    fold_rows = metadata.loc[metadata["fold"] == fold].copy()
    query_targets = (
        fold_rows.assign(target_id=fold_rows["inchikey"].map(normalize_inchikey))
        .groupby("smiles", sort=False)["target_id"]
        .agg(lambda values: tuple(sorted(set(values))))
    )
    query_first_indices = fold_rows.reset_index().groupby("smiles", sort=False)["index"].first()
    query_fps = np.load(query_fp_path, mmap_mode="r")

    protected_fps: Dict[str, np.ndarray] = {}
    protected_inchis: Dict[str, np.ndarray] = {}
    protected_json: Dict[str, List[str]] = {}
    natural_fps: Dict[str, np.ndarray] = {}
    natural_inchis: Dict[str, np.ndarray] = {}
    natural_json: Dict[str, List[str]] = {}
    summary_rows = []
    manifest_writer = None
    manifest_path = output_dir / "candidate_manifest.parquet"

    with np.load(source_fp_path) as source_fps, np.load(source_inchi_path) as source_inchis:
        queries = [query for query in query_targets.index if query in source_inchis.files]
        if max_queries is not None:
            queries = queries[: int(max_queries)]
        for query_index, query in enumerate(queries):
            raw_ids = source_inchis[query]
            raw_fps = source_fps[query]
            canonical_ids, canonical_indices = canonical_candidate_indices(raw_ids, raw_fps, fp_size=fp_size)
            representatives = dict(zip(canonical_ids.tolist(), canonical_indices.tolist()))
            targets = query_targets.loc[query]
            selection = select_candidate_indices(
                raw_ids, targets, cap=cap, seed=seed, representative_indices=representatives
            )
            row_index = int(query_first_indices.loc[query])
            inserted_fps = {target: query_fps[row_index] for target in selection.inserted_target_ids}
            protected_fps[query] = _select_fingerprints(
                source_fps[query], len(raw_ids), selection.source_indices,
                inserted_fps, selection.candidate_ids, fp_size,
            )
            protected_inchis[query] = np.asarray(selection.candidate_ids, dtype="S27")
            protected_json[query] = list(selection.candidate_ids)
            if write_natural:
                natural_fps[query] = _natural_fingerprints(
                    source_fps[query], raw_ids, selection.natural_ids, fp_size,
                )
                natural_inchis[query] = np.asarray(selection.natural_ids, dtype="S27")
                natural_json[query] = list(selection.natural_ids)

            normalized_raw = [normalize_inchikey(value) for value in raw_ids]
            protected_set = set(selection.candidate_ids)
            natural_set = set(selection.natural_ids)
            target_set = set(selection.target_ids)
            summary_rows.append({
                "query_smiles": query,
                "n_raw": len(raw_ids),
                "n_deduplicated": len(set(normalized_raw)),
                "n_protected_capped": len(selection.candidate_ids),
                "n_natural_capped": len(selection.natural_ids),
                "duplicate_occurrences": selection.duplicate_occurrences,
                "n_target_identities": len(target_set),
                "n_targets_inserted": len(selection.inserted_target_ids),
                "n_targets_absent_natural_cap": len(target_set - natural_set),
            })
            if write_manifest:
                seen = set()
                rows = []
                for raw_index, candidate_id in enumerate(normalized_raw):
                    duplicate = candidate_id in seen
                    seen.add(candidate_id)
                    rows.append({
                        "candidate_setting": "formula_pubchem_uncapped",
                        "pool_variant": "local_uncapped_source",
                        "split": fold,
                        "query_smiles": query,
                        "candidate_id": candidate_id,
                        "raw_index": raw_index,
                        "is_duplicate_occurrence": duplicate,
                        "is_target": candidate_id in target_set,
                        "was_in_raw_database_result": True,
                        "was_inserted_for_closed_world": False,
                        "would_be_removed_by_unprotected_cap": candidate_id in target_set and candidate_id not in natural_set,
                        "included_protected_cap": candidate_id in protected_set and not duplicate,
                        "included_natural_cap": candidate_id in natural_set and not duplicate,
                        "n_raw": len(raw_ids),
                        "n_after_deduplication": len(set(normalized_raw)),
                    })
                for candidate_id in selection.inserted_target_ids:
                    rows.append({
                        "candidate_setting": "formula_pubchem_uncapped",
                        "pool_variant": "local_uncapped_source",
                        "split": fold,
                        "query_smiles": query,
                        "candidate_id": candidate_id,
                        "raw_index": -1,
                        "is_duplicate_occurrence": False,
                        "is_target": True,
                        "was_in_raw_database_result": False,
                        "was_inserted_for_closed_world": True,
                        "would_be_removed_by_unprotected_cap": True,
                        "included_protected_cap": True,
                        "included_natural_cap": False,
                        "n_raw": len(raw_ids),
                        "n_after_deduplication": len(set(normalized_raw)),
                    })
                table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
                if manifest_writer is None:
                    manifest_writer = pq.ParquetWriter(manifest_path, table.schema, compression="zstd")
                manifest_writer.write_table(table)
            if (query_index + 1) % 100 == 0:
                print(f"Prepared candidate controls for {query_index + 1}/{len(queries)} query keys")

    prefixes = {
        "formula_pubchem_capped256": (protected_fps, protected_inchis, protected_json),
    }
    if write_natural:
        prefixes["formula_pubchem_natural_capped256"] = (
            natural_fps, natural_inchis, natural_json
        )
    for setting, (fps, inchis, candidate_json) in prefixes.items():
        prefix = output_dir / f"MassSpecGym_retrieval_candidates_{setting}"
        np.savez_compressed(prefix.with_name(prefix.name + "_fps.npz"), **fps)
        np.savez_compressed(prefix.with_name(prefix.name + "_inchi.npz"), **inchis)
        prefix.with_suffix(".json").write_text(json.dumps(candidate_json, sort_keys=True))

    if manifest_writer is not None:
        manifest_writer.close()
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "candidate_control_summary.csv", index=False)
    return summary
