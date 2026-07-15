#!/usr/bin/env python3
"""Build candidate fingerprint/InChI helper NPZ files from a MassSpecGym candidate JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, inchi
from tqdm import tqdm


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return mol


def fingerprint_from_smiles(smiles: str, fp_size: int = 4096) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
    arr = np.zeros((fp_size,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr.astype(bool)


def inchikey_from_smiles(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    return inchi.MolToInchiKey(mol).split("-")[0]


def normalize_candidate_map(raw) -> Dict[str, Iterable[str]]:
    if not isinstance(raw, dict):
        raise TypeError("Candidate JSON must be an object mapping query SMILES to candidate SMILES lists.")
    out = {}
    for query, candidates in raw.items():
        if isinstance(candidates, dict):
            candidates = candidates.get("candidates", candidates.get("smiles", []))
        out[str(query)] = [str(c) for c in candidates]
    return out


def build_candidate_arrays(candidate_map: Dict[str, Iterable[str]], fp_size: int = 4096) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    fps, inchis = {}, {}
    for query, candidates in tqdm(candidate_map.items(), desc="Building helper arrays"):
        cand_list = list(candidates)
        if cand_list:
            fps[query] = np.stack([fingerprint_from_smiles(s, fp_size=fp_size) for s in cand_list], axis=0)
            inchis[query] = np.asarray([inchikey_from_smiles(s) for s in cand_list])
        else:
            fps[query] = np.zeros((0, fp_size), dtype=bool)
            inchis[query] = np.asarray([], dtype=str)
    return fps, inchis


def output_paths(candidate_json: Path, out_prefix: Path | None = None) -> Tuple[Path, Path]:
    prefix = out_prefix if out_prefix is not None else candidate_json.with_suffix("")
    return Path(f"{prefix}_fps.npz"), Path(f"{prefix}_inchi.npz")


def save_npz(path: Path, data: Dict[str, np.ndarray], compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compressed else np.savez
    save(path, **data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate_json", type=Path, required=True)
    ap.add_argument("--out_prefix", type=Path, default=None)
    ap.add_argument("--fp_size", type=int, default=4096)
    ap.add_argument("--compressed", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    fp_path, inchi_path = output_paths(args.candidate_json, args.out_prefix)
    if not args.overwrite and (fp_path.exists() or inchi_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing outputs: {fp_path}, {inchi_path}")

    raw = json.loads(args.candidate_json.read_text())
    candidate_map = normalize_candidate_map(raw)
    fps, inchis = build_candidate_arrays(candidate_map, fp_size=args.fp_size)
    save_npz(fp_path, fps, compressed=args.compressed)
    save_npz(inchi_path, inchis, compressed=args.compressed)
    print(f"Saved {fp_path}")
    print(f"Saved {inchi_path}")


if __name__ == "__main__":
    main()
