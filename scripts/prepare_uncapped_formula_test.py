#!/usr/bin/env python3
"""Prepare packed, test-only helpers for a large uncapped formula candidate JSON."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, inchi, rdMolDescriptors
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")

_CANDIDATE_MAP: Dict[str, Sequence[str]] = {}
_QUERY_META: Dict[str, dict] = {}
_FP_SIZE = 4096
_VALIDATE_FORMULA = False


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def inchikey_2d_from_mol(mol) -> str:
    return inchi.MolToInchiKey(mol).split("-")[0]


def inchikey_2d(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    return inchikey_2d_from_mol(mol) if mol is not None else None


def resolve_query_candidates(
    source: Mapping[str, Sequence[str]],
    metadata: pd.DataFrame,
) -> tuple[Dict[str, Sequence[str]], Dict[str, dict], Counter]:
    """Map source JSON keys to the original TSV SMILES representations."""
    canonical_to_source = {}
    inchikey_to_source = {}
    for source_key in tqdm(source, desc="Indexing source query keys"):
        canonical = canonical_smiles(source_key)
        key_2d = inchikey_2d(source_key)
        if canonical is not None:
            canonical_to_source.setdefault(canonical, source_key)
        if key_2d is not None:
            inchikey_to_source.setdefault(key_2d, source_key)

    candidate_map: Dict[str, Sequence[str]] = {}
    query_meta: Dict[str, dict] = {}
    methods: Counter = Counter()
    missing = []

    for row in metadata.itertuples(index=False):
        query = str(row.smiles)
        expected_key = str(row.inchikey).split("-")[0]
        if query in source:
            source_key = query
            method = "exact"
        else:
            canonical = canonical_smiles(query)
            source_key = canonical_to_source.get(canonical)
            method = "canonical"
            if source_key is None:
                source_key = inchikey_to_source.get(expected_key)
                method = "inchikey"

        if source_key is None:
            missing.append(query)
            continue
        if query in candidate_map:
            continue

        candidates = source[source_key]
        if not candidates:
            raise ValueError(f"Candidate pool is empty for query {query}")
        candidate_map[query] = candidates
        query_meta[query] = {
            "source_key": source_key,
            "inchikey_2d": expected_key,
            "smiles_inchikey_2d": inchikey_2d(query),
            "formula": str(row.formula),
            "match_method": method,
        }
        methods[method] += 1

    if missing:
        examples = ", ".join(missing[:5])
        raise ValueError(f"Could not match {len(missing)} query SMILES; examples: {examples}")
    return candidate_map, query_meta, methods


def _init_worker(candidate_map, query_meta, fp_size: int, validate_formula: bool) -> None:
    global _CANDIDATE_MAP, _QUERY_META, _FP_SIZE, _VALIDATE_FORMULA
    _CANDIDATE_MAP = candidate_map
    _QUERY_META = query_meta
    _FP_SIZE = int(fp_size)
    _VALIDATE_FORMULA = bool(validate_formula)


def _shard_path(shard_dir: Path, index: int) -> Path:
    return shard_dir / f"query_{index:05d}.npz"


def _build_query_shard(task: tuple[int, str, str]) -> tuple[int, int, int, int]:
    index, query, shard_path_str = task
    shard_path = Path(shard_path_str)
    if shard_path.exists():
        with np.load(shard_path, allow_pickle=False) as shard:
            stored_query = str(shard["query"].item())
            n_candidates = int(shard["n_candidates"].item())
            n_labels = int(shard["n_labels"].item())
            used_fallback = int(shard["used_label_fallback"].item()) if "used_label_fallback" in shard else 0
        if stored_query != query:
            raise ValueError(f"Shard/query mismatch in {shard_path}: {stored_query} != {query}")
        return index, n_candidates, n_labels, used_fallback

    candidates = _CANDIDATE_MAP[query]
    meta = _QUERY_META[query]
    n_candidates = len(candidates)
    n_bytes = (_FP_SIZE + 7) // 8
    packed = np.empty((n_candidates, n_bytes), dtype=np.uint8)
    keys = np.empty(n_candidates, dtype="S14")
    bit_array = np.empty(_FP_SIZE, dtype=np.uint8)
    formula_mismatches = 0

    for candidate_index, candidate in enumerate(candidates):
        mol = Chem.MolFromSmiles(str(candidate))
        if mol is None:
            raise ValueError(f"Could not parse candidate {candidate!r} for query {query!r}")
        bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=_FP_SIZE)
        DataStructs.ConvertToNumpyArray(bitvect, bit_array)
        packed[candidate_index] = np.packbits(bit_array, bitorder="big")
        keys[candidate_index] = inchikey_2d_from_mol(mol).encode("ascii")
        if _VALIDATE_FORMULA and rdMolDescriptors.CalcMolFormula(mol) != meta["formula"]:
            formula_mismatches += 1

    if formula_mismatches:
        raise ValueError(
            f"Found {formula_mismatches}/{n_candidates} formula mismatches for query {query!r}"
        )

    expected_key = meta["inchikey_2d"].encode("ascii")
    n_labels = int(np.count_nonzero(keys == expected_key))
    used_fallback = 0
    if n_labels == 0:
        smiles_key = str(meta["smiles_inchikey_2d"]).encode("ascii")
        n_labels = int(np.count_nonzero(keys == smiles_key))
        used_fallback = 1
    if n_labels == 0:
        raise ValueError(f"No candidate matches the TSV or SMILES-derived InChIKey for {query!r}")

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez(
        tmp_path,
        query=np.asarray(query),
        fps=packed.reshape(-1),
        inchis=keys,
        n_candidates=np.asarray(n_candidates, dtype=np.int64),
        n_labels=np.asarray(n_labels, dtype=np.int64),
        used_label_fallback=np.asarray(used_fallback, dtype=np.int64),
    )
    os.replace(tmp_path, shard_path)
    return index, n_candidates, n_labels, used_fallback


def build_shards(
    candidate_map: Dict[str, Sequence[str]],
    query_meta: Dict[str, dict],
    shard_dir: Path,
    fp_size: int,
    workers: int,
    validate_formula: bool = False,
) -> list[tuple[int, int, int, int]]:
    queries = list(candidate_map)
    tasks = [(i, query, str(_shard_path(shard_dir, i))) for i, query in enumerate(queries)]
    initargs = (candidate_map, query_meta, fp_size, validate_formula)

    if workers <= 1:
        _init_worker(*initargs)
        iterator = map(_build_query_shard, tasks)
        return list(tqdm(iterator, total=len(tasks), desc="Building candidate shards"))

    context = mp.get_context("fork")
    with context.Pool(workers, initializer=_init_worker, initargs=initargs) as pool:
        iterator = pool.imap_unordered(_build_query_shard, tasks, chunksize=1)
        return list(tqdm(iterator, total=len(tasks), desc="Building candidate shards"))


def consolidate_shards(
    queries: Iterable[str],
    shard_dir: Path,
    fp_path: Path,
    inchi_path: Path,
) -> None:
    queries = list(queries)
    fps = {}
    for index, query in enumerate(tqdm(queries, desc="Loading fingerprint shards")):
        with np.load(_shard_path(shard_dir, index), allow_pickle=False) as shard:
            if str(shard["query"].item()) != query:
                raise ValueError(f"Shard order mismatch at index {index}")
            fps[query] = shard["fps"].copy()
    np.savez(fp_path, **fps)
    del fps
    gc.collect()

    inchis = {}
    for index, query in enumerate(tqdm(queries, desc="Loading InChIKey shards")):
        with np.load(_shard_path(shard_dir, index), allow_pickle=False) as shard:
            inchis[query] = shard["inchis"].copy()
    np.savez(inchi_path, **inchis)


def write_json_atomic(path: Path, value) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(value, handle)
    os.replace(tmp_path, path)


def source_digest(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_json", type=Path, required=True)
    parser.add_argument("--dataset_tsv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fold", default="test")
    parser.add_argument("--fp_size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--shard_dir", type=Path, default=None)
    parser.add_argument("--validate_formula", action="store_true")
    parser.add_argument("--hash_source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare_only", action="store_true", help="Write the test JSON/manifest but do not fingerprint.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "MassSpecGym_retrieval_candidates_formula_uncapped"
    candidate_json = Path(f"{prefix}.json")
    fp_path = Path(f"{prefix}_fps.npz")
    inchi_path = Path(f"{prefix}_inchi.npz")
    manifest_path = Path(f"{prefix}_manifest.json")
    shard_dir = args.shard_dir or Path(f"{prefix}_shards")

    final_paths = [candidate_json, fp_path, inchi_path, manifest_path]
    if not args.overwrite and any(path.exists() for path in final_paths[1:3]):
        existing = ", ".join(str(path) for path in final_paths[1:3] if path.exists())
        raise FileExistsError(f"Refusing to overwrite existing helper files: {existing}")

    metadata = pd.read_csv(
        args.dataset_tsv,
        sep="\t",
        usecols=["smiles", "inchikey", "formula", "fold"],
        dtype=str,
    )
    metadata = metadata.loc[metadata["fold"] == args.fold].drop_duplicates("smiles")
    print(f"Loading {args.source_json} ...")
    with args.source_json.open() as handle:
        source = json.load(handle)
    candidate_map, query_meta, methods = resolve_query_candidates(source, metadata)
    del source
    gc.collect()

    sizes = np.asarray([len(candidates) for candidates in candidate_map.values()], dtype=np.int64)
    manifest = {
        "source_json": str(args.source_json.resolve()),
        "source_size_bytes": args.source_json.stat().st_size,
        "source_sha256": source_digest(args.source_json) if args.hash_source else None,
        "dataset_tsv": str(args.dataset_tsv.resolve()),
        "fold": args.fold,
        "fp_size": args.fp_size,
        "packed_bitorder": "big",
        "query_match_methods": dict(methods),
        "n_queries": int(len(candidate_map)),
        "n_candidate_occurrences": int(sizes.sum()),
        "candidate_size_min": int(sizes.min()),
        "candidate_size_median": float(np.median(sizes)),
        "candidate_size_mean": float(sizes.mean()),
        "candidate_size_max": int(sizes.max()),
        "n_above_256": int(np.count_nonzero(sizes > 256)),
        "test_only": args.fold == "test",
    }

    if args.overwrite or not candidate_json.exists():
        print(f"Writing {candidate_json} ...")
        write_json_atomic(candidate_json, candidate_map)
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))

    if args.prepare_only:
        return

    shard_dir.mkdir(parents=True, exist_ok=True)
    results = build_shards(
        candidate_map,
        query_meta,
        shard_dir,
        fp_size=args.fp_size,
        workers=args.workers,
        validate_formula=args.validate_formula,
    )
    manifest["n_positive_labels"] = int(sum(n_labels for _, _, n_labels, _ in results))
    manifest["n_queries_using_label_fallback"] = int(sum(fallback for _, _, _, fallback in results))
    manifest["shard_dir"] = str(shard_dir.resolve())
    write_json_atomic(manifest_path, manifest)

    print("Consolidating shards ...")
    consolidate_shards(candidate_map, shard_dir, fp_path, inchi_path)
    print(f"Saved {fp_path}")
    print(f"Saved {inchi_path}")
    print(f"Kept resumable shards in {shard_dir}")


if __name__ == "__main__":
    main()
