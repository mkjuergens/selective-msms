#!/usr/bin/env python3
"""Build a MassSpecGym candidate JSON shell from candidate helper NPZ files.

The retrieval dataset requires a candidate JSON to enumerate query keys before
the cleaned pipeline replaces candidates by the precomputed fingerprint/InChI
NPZ helpers. When only NPZ helpers are available, this script creates a
compatibility JSON with the same query order and candidate counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def normalize_candidate_map(raw) -> Dict[str, List[str]]:
    if not isinstance(raw, dict):
        raise TypeError("Candidate JSON must map query SMILES to candidate lists.")
    out: Dict[str, List[str]] = {}
    for query, candidates in raw.items():
        if isinstance(candidates, dict):
            candidates = candidates.get("candidates", candidates.get("smiles", []))
        out[str(query)] = [str(candidate) for candidate in candidates]
    return out


def load_candidate_counts(inchi_npz: Path) -> Dict[str, int]:
    data = np.load(inchi_npz, allow_pickle=False)
    return {str(key): int(len(data[key])) for key in data.files}


def build_compat_candidate_json(
    inchi_npz: Path,
    out_path: Path,
    source_json: Path | None = None,
    placeholder_prefix: str = "candidate",
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    counts = load_candidate_counts(inchi_npz)
    if source_json is not None:
        candidate_map = normalize_candidate_map(json.loads(source_json.read_text()))
        missing = sorted(set(counts) - set(candidate_map))
        extra = sorted(set(candidate_map) - set(counts))
        if missing or extra:
            raise ValueError(
                "Source JSON keys do not match the InChI NPZ keys "
                f"(missing={len(missing)}, extra={len(extra)})."
            )
        length_mismatches = [
            (query, len(candidate_map[query]), counts[query])
            for query in counts
            if len(candidate_map[query]) != counts[query]
        ]
        if length_mismatches:
            preview = ", ".join(
                f"{query}: json={json_len}, npz={npz_len}"
                for query, json_len, npz_len in length_mismatches[:5]
            )
            raise ValueError(f"Source JSON candidate counts do not match the NPZ: {preview}")
    else:
        candidate_map = {
            query: [f"{placeholder_prefix}_{i:06d}" for i in range(n_candidates)]
            for query, n_candidates in counts.items()
        }

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing candidate JSON: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(candidate_map))
    return candidate_map


def default_inchi_path(helper_dir: Path, candidate_setting: str) -> Path:
    return helper_dir / f"MassSpecGym_retrieval_candidates_{candidate_setting}_inchi.npz"


def default_out_path(helper_dir: Path, candidate_setting: str) -> Path:
    return helper_dir / f"MassSpecGym_retrieval_candidates_{candidate_setting}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--helper_dir", type=Path, required=True)
    ap.add_argument("--candidate_setting", default="mass")
    ap.add_argument("--inchi_npz", type=Path, default=None)
    ap.add_argument("--source_json", type=Path, default=None)
    ap.add_argument("--out_path", type=Path, default=None)
    ap.add_argument("--placeholder_prefix", default="candidate")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    inchi_npz = args.inchi_npz or default_inchi_path(args.helper_dir, args.candidate_setting)
    out_path = args.out_path or default_out_path(args.helper_dir, args.candidate_setting)
    candidate_map = build_compat_candidate_json(
        inchi_npz=inchi_npz,
        out_path=out_path,
        source_json=args.source_json,
        placeholder_prefix=args.placeholder_prefix,
        overwrite=args.overwrite,
    )
    sizes = np.asarray([len(candidates) for candidates in candidate_map.values()], dtype=np.int64)
    print(f"Saved {out_path}")
    print(
        "queries={queries} candidates={candidates} median={median:.1f} max={max_}".format(
            queries=len(sizes),
            candidates=int(sizes.sum()),
            median=float(np.median(sizes)) if len(sizes) else 0.0,
            max_=int(sizes.max()) if len(sizes) else 0,
        )
    )


if __name__ == "__main__":
    main()
