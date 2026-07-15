#!/usr/bin/env python3
"""Run the complete, resumable canonical revision evaluation without downloading or retraining."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ms_uq.evaluation.revision_candidates import build_record_preserving_formula_cap, normalize_inchikey
from ms_uq.evaluation.revision_features import peak_count
from ms_uq.evaluation.revision_reporting import SGR_SINGLE_MEASURES, merge_meta_predictions, run_sgr_stability


STAGES = [
    "preflight", "candidates", "scores", "metrics", "temperature", "meta", "sgr",
    "bootstrap", "figures", "tables", "report", "validate",
]
STAGE_DEPENDENCIES = {
    "candidates": ("preflight",),
    "scores": ("candidates",),
    "metrics": ("scores",),
    "temperature": ("scores",),
    "meta": ("scores",),
    "sgr": ("metrics", "meta"),
    "bootstrap": ("metrics", "meta"),
    "figures": ("bootstrap", "sgr"),
    "tables": ("bootstrap",),
    "report": ("figures", "tables", "temperature"),
    "validate": ("report", "tables", "sgr", "bootstrap"),
}
EVALUATION_TEMPERATURE = 0.003
TOP_KS = [1, 5, 20]
TARGET_RISKS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def expand_stage_dependencies(requested: Sequence[str]) -> List[str]:
    """Return requested stages and all prerequisites in canonical execution order."""
    if list(requested) == ["all"]:
        return list(STAGES)

    selected = set()

    def add(stage: str) -> None:
        for dependency in STAGE_DEPENDENCIES.get(stage, ()):
            add(dependency)
        selected.add(stage)

    for stage in requested:
        add(stage)
    return [stage for stage in STAGES if stage in selected]


@dataclass(frozen=True)
class ModelSource:
    run_label: str
    architecture: str
    training_candidates: str
    test_pred_dir: Path
    val_pred_dir: Optional[Path]
    checkpoint_files: tuple[Path, ...]
    prediction_samples: int = 5
    archived_test_score: Optional[Path] = None


@dataclass(frozen=True)
class ScoreCell:
    name: str
    model: str
    split: str
    helper_setting: str
    evaluation_setting: str
    pool_variant: str
    query_mask_id: str


class RevisionRunner:
    def __init__(self, args):
        self.args = args
        self.repo = Path(__file__).resolve().parents[1]
        self.data = args.data_dir.resolve()
        self.out = args.out_dir.resolve()
        self.python = Path(sys.executable).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.logs = self.out / "logs"
        self.logs.mkdir(exist_ok=True)
        self.helper_dir = self.out / "candidates" / "helpers"
        self.helper_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "stage_state.json"
        self.manifest_path = self.out / "run_manifest.json"
        self.hash_cache_path = self.out / "hash_cache.json"
        self.state = self._read_json(self.state_path, {})
        self.manifest = self._read_json(self.manifest_path, {})
        self.hash_cache = self._read_json(self.hash_cache_path, {})
        self.force_stages = set(args.force_stage)
        self.selected_stages = expand_stage_dependencies(args.stages)
        self.code_state = self._code_state()
        self.models = self._model_sources()
        self.cells = self._score_cells(args.analysis_scope)
        self.input_hashes: Dict[str, str] = {}

    @staticmethod
    def _read_json(path: Path, default):
        return json.loads(path.read_text()) if path.exists() else default

    @staticmethod
    def _atomic_json(path: Path, value) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
        temporary.replace(path)

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.repo, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout

    def _code_state(self) -> dict:
        sha = self._git("rev-parse", "HEAD").strip()
        status = self._git("status", "--porcelain=v1")
        diff = self._git("diff", "--binary")
        source_hash = hashlib.sha256()
        source_files = sorted(
            path for path in self.repo.rglob("*")
            if path.is_file() and ".git" not in path.parts and path.suffix in {".py", ".yml", ".yaml", ".toml", ".md"}
        )
        for path in source_files:
            source_hash.update(str(path.relative_to(self.repo)).encode())
            source_hash.update(hashlib.sha256(path.read_bytes()).digest())
        return {
            "commit_sha": sha,
            "worktree_clean": not bool(status.strip()),
            "git_status": status.splitlines(),
            "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "complete_source_sha256": source_hash.hexdigest(),
        }

    @staticmethod
    def _manifest_checkpoint(manifest: Path, metric: str = "reranker") -> Path:
        value = json.loads(manifest.read_text()).get(metric)
        if not value:
            raise ValueError(f"{manifest} does not contain checkpoint metric {metric}")
        checkpoint = Path(value)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        return checkpoint

    def _ensemble_checkpoints(self, root: Path) -> tuple[Path, ...]:
        manifests = sorted(root.glob("members/member_*/best_ckpts.json"))
        return tuple(self._manifest_checkpoint(path) for path in manifests)

    def _model_sources(self) -> Dict[str, ModelSource]:
        formula_mlp_root = self.data / "logs/ensemble_20251222-1218"
        mass_mlp_root = self.data / "logs/ensemble_20260709-1541_mass_bienc_T0003_retry"
        transformer_single = self.repo / "outputs/revision_runs/single_20260706-1138_transformer_single_formula/single/model/best_ckpts.json"
        transformer_ensemble = self.repo / "outputs/revision_runs/ensemble_20260707-0736_transformer_formula_seeds43_46_gpu2_4"
        transformer_checkpoints = (self._manifest_checkpoint(transformer_single),) + self._ensemble_checkpoints(transformer_ensemble)
        mc_dropout_root = self.data / "logs/mc_dropout_20260217-1126"
        laplace_pred_dir = self.data / "logs/laplace_bce/predictions/bienc"
        models = {
            "mlp_formula": ModelSource(
                "mlp_formula", "mlp", "formula_official_capped",
                formula_mlp_root / "predictions", self.repo / "outputs/revision_meta/mlp_val/pred",
                self._ensemble_checkpoints(formula_mlp_root),
            ),
            "transformer_formula": ModelSource(
                "transformer_formula", "transformer", "formula_official_capped",
                self.repo / "outputs/revision_analysis/transformer_ensemble_formula/pred",
                self.repo / "outputs/revision_meta/transformer_val/pred", transformer_checkpoints,
            ),
            "mlp_mass": ModelSource(
                "mlp_mass", "mlp", "mass_existing_capped256",
                self.repo / "outputs/predictions/mass_mlp_bienc", None,
                self._ensemble_checkpoints(mass_mlp_root),
            ),
            "mlp_mc_dropout": ModelSource(
                "mlp_mc_dropout", "mlp", "formula_official_capped",
                mc_dropout_root / "predictions", None,
                (self._manifest_checkpoint(mc_dropout_root / "single/model/best_ckpts.json", "cossim"),),
                prediction_samples=50,
                archived_test_score=self.data / "figures/eval_v6/mc_dropout/bienc/scores_ranker_score.pt",
            ),
            "mlp_laplace": ModelSource(
                "mlp_laplace", "mlp", "formula_official_capped",
                laplace_pred_dir, None,
                (
                    self.data / "logs/ensemble_20251222-1218/members/member_004/ckpts/cossim/cossim-01-6068.ckpt",
                    laplace_pred_dir / "laplace_state.pt",
                ),
                prediction_samples=50,
                archived_test_score=self.data / "figures/eval_v6/laplace/bienc/scores_ranker_score.pt",
            ),
        }
        for model in models.values():
            if not model.checkpoint_files:
                raise ValueError(f"{model.run_label}: no checkpoint provenance found")
            for path in model.checkpoint_files:
                if not path.exists():
                    raise FileNotFoundError(path)
            if model.archived_test_score is not None and not model.archived_test_score.exists():
                raise FileNotFoundError(model.archived_test_score)
        return models

    @staticmethod
    def _score_cells(analysis_scope: str) -> List[ScoreCell]:
        cells = [
            ScoreCell("mlp_formula__val__formula_official", "mlp_formula", "val", "formula", "formula_official_capped", "official_as_distributed_records", "official_formula_val"),
            ScoreCell("transformer_formula__val__formula_official", "transformer_formula", "val", "formula", "formula_official_capped", "official_as_distributed_records", "official_formula_val"),
            ScoreCell("mlp_formula__test__formula_official", "mlp_formula", "test", "formula", "formula_official_capped", "official_as_distributed_records", "official_formula_test"),
            ScoreCell("mlp_formula__test__formula_capped256", "mlp_formula", "test", "formula_pubchem_record_capped256", "formula_pubchem_capped256", "deterministic_record_cap_target_protected", "formula_paired_test"),
            ScoreCell("mlp_formula__test__formula_uncapped", "mlp_formula", "test", "formula_uncapped", "formula_pubchem_uncapped", "local_uncapped_records", "formula_paired_test"),
            ScoreCell("mlp_formula__test__mass", "mlp_formula", "test", "mass", "mass_existing_capped256", "opaque_existing_record_pool", "mass_paired_test"),
            ScoreCell("transformer_formula__test__formula_official", "transformer_formula", "test", "formula", "formula_official_capped", "official_as_distributed_records", "official_formula_test"),
            ScoreCell("transformer_formula__test__formula_capped256", "transformer_formula", "test", "formula_pubchem_record_capped256", "formula_pubchem_capped256", "deterministic_record_cap_target_protected", "formula_paired_test"),
            ScoreCell("transformer_formula__test__formula_uncapped", "transformer_formula", "test", "formula_uncapped", "formula_pubchem_uncapped", "local_uncapped_records", "formula_paired_test"),
            ScoreCell("mlp_mass__test__mass", "mlp_mass", "test", "mass", "mass_existing_capped256", "opaque_existing_record_pool", "mass_paired_test"),
            ScoreCell("mlp_mc_dropout__test__formula_official", "mlp_mc_dropout", "test", "formula", "formula_official_capped", "official_as_distributed_records", "official_formula_test"),
            ScoreCell("mlp_laplace__test__formula_official", "mlp_laplace", "test", "formula", "formula_official_capped", "official_as_distributed_records", "official_formula_test"),
        ]
        if analysis_scope == "extended":
            cells.append(ScoreCell(
                "mlp_mass__test__formula_official", "mlp_mass", "test", "formula",
                "formula_official_capped", "official_as_distributed_records", "official_formula_test",
            ))
        return cells

    def sha256(self, path: Path) -> str:
        path = path.resolve()
        stat = path.stat()
        if self.args.quick_hashes and stat.st_size > 100 * 1024 * 1024:
            return f"quick-smoke:{stat.st_size}:{stat.st_mtime_ns}"
        key = str(path)
        cached = self.hash_cache.get(key)
        fingerprint = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if cached and cached.get("fingerprint") == fingerprint:
            return cached["sha256"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self.hash_cache[key] = {"fingerprint": fingerprint, "sha256": value}
        self._atomic_json(self.hash_cache_path, self.hash_cache)
        return value

    def stage_signature(self, stage: str) -> str:
        payload = {
            "stage": stage,
            "code": self.code_state,
            "config": self.resolved_config(),
            "inputs": self.input_hashes,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def stage_complete(self, stage: str, outputs: Sequence[Path]) -> bool:
        if stage in self.force_stages or not self.args.resume:
            return False
        record = self.state.get(stage, {})
        return record.get("status") == "complete" and record.get("signature") == self.stage_signature(stage) and all(path.exists() for path in outputs)

    def mark_stage(self, stage: str, outputs: Sequence[Path]) -> None:
        self.state[stage] = {
            "status": "complete", "signature": self.stage_signature(stage),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": [str(path) for path in outputs],
        }
        self._atomic_json(self.state_path, self.state)

    def run_command(self, stage: str, name: str, command: Sequence[str]) -> None:
        command = [str(value) for value in command]
        log_path = self.logs / f"{stage}__{name}.log"
        started = datetime.now(timezone.utc).isoformat()
        print(f"[{stage}] {name}")
        print("  " + " ".join(command))
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in [str(self.repo), env.get("PYTHONPATH", "")] if value
        )
        with log_path.open("a") as log:
            log.write(f"\n[{started}] {' '.join(command)}\n")
            process = subprocess.run(
                command,
                cwd=self.repo,
                env=env,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        record = {"stage": stage, "name": name, "command": command, "log": str(log_path), "started_utc": started, "returncode": process.returncode}
        self.manifest.setdefault("commands", []).append(record)
        self._atomic_json(self.manifest_path, self.manifest)
        if process.returncode != 0:
            raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")

    def sgr_seeds(self) -> List[int]:
        return [42] if self.args.sgr_repeats == 1 else list(range(self.args.sgr_repeats))


    def resolved_config(self) -> dict:
        return {
            "data_dir": str(self.data), "out_dir": str(self.out), "device": self.args.device,
            "temperature": EVALUATION_TEMPERATURE, "top_ks": TOP_KS, "candidate_seed": 42,
            "candidate_record_policy": "preserve", "candidate_tie_break": "source_order",
            "score_dtype": "float32", "aurc_convention": "manuscript_trapezoid_seed42",
            "feature_convention": "manuscript",
            "bootstrap_seed": 42, "bootstrap_replicates": self.args.bootstrap_replicates,
            "sgr_seeds": self.sgr_seeds(), "sgr_delta": 0.001,
            "max_queries": self.args.max_queries,
            "quick_hashes": self.args.quick_hashes,
            "requested_stages": self.args.stages,
            "execution_stages": self.selected_stages,
            "analysis_scope": self.args.analysis_scope,
            "write_candidate_manifest": self.args.write_candidate_manifest,
            "meta_ablation": self.args.meta_ablation,
            "keep_raw_scores": self.args.keep_raw_scores,
            "models": {
                name: {**asdict(model), "test_pred_dir": str(model.test_pred_dir),
                       "val_pred_dir": str(model.val_pred_dir) if model.val_pred_dir else None,
                       "checkpoint_files": [str(path) for path in model.checkpoint_files]}
                for name, model in self.models.items()
            },
        }

    def initialize_manifest(self) -> None:
        versions = {}
        for package in ["torch", "numpy", "pandas", "scipy", "scikit-learn", "rdkit", "massspecgym", "pyarrow"]:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not-installed"
        self.manifest.update({
            "run_id": self.out.name,
            "utc_start": self.manifest.get("utc_start", datetime.now(timezone.utc).isoformat()),
            "command_line": sys.argv,
            "resolved_config": self.resolved_config(),
            "code_state": self.code_state,
            "versions": versions,
            "python": sys.version,
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "seeds": {"candidate_truncation": 42, "cross_validation": 42, "bootstrap": 42, "sgr": self.sgr_seeds()},
            "identity": {
                "official_formula_labels": "distributed fingerprint-equality labels",
                "new_candidate_labels": "precomputed query InChIKey connectivity block",
                "meta_cv_groups": "TSV InChIKey connectivity block",
            },
            "fingerprint": {"type": "Morgan bit vector", "radius": 2, "dimensions": 4096, "count": False, "chirality": "existing helper setting; unavailable in source manifest"},
            "training_temperature": EVALUATION_TEMPERATURE,
            "primary_evaluation_temperature": EVALUATION_TEMPERATURE,
            "aurc_convention": "manuscript_trapezoid_seed42",
            "candidate_record_policy": "preserve",
            "candidate_tie_break": "source_order",
            "feature_convention": "manuscript",
            "deviations": [
                "No data were downloaded and no ensemble was retrained.",
                "The local uncapped PubChem helper has no recoverable snapshot identifier; its content hash is authoritative.",
                "The existing mass pool is already capped at 256 records; its pre-cap pool, tolerance, adduct conversion, and database provenance are unavailable.",
                "The mass MLP used this same record-preserving mass helper during training, so no retraining is required.",
                "The mass MLP used dropout 0.25 whereas the formula MLP used dropout 0.30.",
                "Archived 50-draw MC-dropout and Laplace candidate-score bundles are reused in place after exact pointer, label, aggregation, and official-candidate alignment checks.",
                "All temperature-dependent selectors from those archived scores are recomputed at T_eval=0.003; old T_eval=1.0 metrics are not reused.",
                "SGR uses the submitted-paper spectrum-level random split; repeated molecule spectra preclude an independence-based guarantee.",
            ],
        })
        self._atomic_json(self.out / "resolved_config.json", self.resolved_config())
        self._atomic_json(self.manifest_path, self.manifest)

    def link_helpers(self) -> None:
        names = [
            "fp_4096.npy", "inchis.npy",
            "MassSpecGym_retrieval_candidates_formula.json", "MassSpecGym_retrieval_candidates_formula_fps.npz", "MassSpecGym_retrieval_candidates_formula_inchi.npz",
            "MassSpecGym_retrieval_candidates_formula_uncapped.json", "MassSpecGym_retrieval_candidates_formula_uncapped_fps.npz", "MassSpecGym_retrieval_candidates_formula_uncapped_inchi.npz",
            "MassSpecGym_retrieval_candidates_mass.json", "MassSpecGym_retrieval_candidates_mass_fps.npz", "MassSpecGym_retrieval_candidates_mass_inchi.npz",
        ]
        for name in names:
            source, target = self.data / name, self.helper_dir / name
            if not source.exists():
                raise FileNotFoundError(source)
            if target.exists() or target.is_symlink():
                if target.resolve() != source.resolve():
                    raise ValueError(f"Helper link points to unexpected source: {target}")
            else:
                target.symlink_to(source)

    def _presence_map(
        self, setting: str, rows: pd.DataFrame,
        target_column: str = "precomputed_molecule_group_id",
    ) -> Dict[tuple, bool]:
        path = self.data / f"MassSpecGym_retrieval_candidates_{setting}_inchi.npz"
        result = {}
        with np.load(path) as candidates:
            for smiles, group in rows.groupby("smiles", sort=False):
                identities = {normalize_inchikey(value) for value in candidates[smiles]} if smiles in candidates.files else set()
                for target in group[target_column].unique():
                    result[(smiles, target)] = target in identities
        return result

    def preflight(self) -> None:
        outputs = [self.out / "query_manifest.parquet", self.out / "query_masks.parquet", self.out / "input_hashes.csv"]
        if self.stage_complete("preflight", outputs):
            print("[preflight] using validated cached outputs")
            hashes = pd.read_csv(self.out / "input_hashes.csv")
            self.input_hashes = dict(zip(hashes.path, hashes.sha256))
            return
        self.link_helpers()
        dataset_path = self.data / "MassSpecGym.tsv"
        metadata = pd.read_csv(dataset_path, sep="\t")
        metadata["query_id"] = metadata["identifier"].astype(str)
        metadata["molecule_group_id"] = metadata["inchikey"].map(normalize_inchikey)
        expected = {"train": (194119, 22746), "val": (19429, 3185), "test": (17556, 2998)}
        for split, (n_spectra, n_molecules) in expected.items():
            subset = metadata[metadata.fold == split]
            observed = (len(subset), subset.molecule_group_id.nunique())
            if observed != (n_spectra, n_molecules):
                raise ValueError(f"Official {split} counts {observed} do not match expected {(n_spectra, n_molecules)}")
            if subset.query_id.duplicated().any():
                raise ValueError(f"query_id is not unique in {split}")
        group_sets = {split: set(metadata.loc[metadata.fold == split, "molecule_group_id"]) for split in expected}
        if any(group_sets[a] & group_sets[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
            raise ValueError("Molecule groups overlap official folds")

        precomputed = np.load(self.data / "inchis.npy")
        precomputed_ids = np.asarray([normalize_inchikey(value) for value in precomputed], dtype=object)
        metadata["precomputed_molecule_group_id"] = precomputed_ids
        metadata["identity_mismatch"] = metadata.molecule_group_id.to_numpy() != precomputed_ids
        test = metadata[metadata.fold == "test"].copy()
        formula_presence = self._presence_map("formula", test)
        uncapped_presence = self._presence_map("formula_uncapped", test)
        mass_presence = self._presence_map("mass", test)
        key = list(zip(test.smiles, test.precomputed_molecule_group_id))
        test["official_formula_test"] = [formula_presence[value] for value in key]
        test["formula_reconstructed_test"] = [value in uncapped_presence for value in key]
        test["formula_paired_test"] = [uncapped_presence.get(value, False) for value in key]
        test["mass_supported_test"] = [mass_presence.get(value, False) for value in key]
        test["mass_paired_test"] = test.mass_supported_test & test.official_formula_test
        test["mask_exclusion_reason"] = np.where(
            ~test.formula_reconstructed_test, "missing local uncapped formula pool",
            np.where(~test.mass_supported_test, "precomputed target absent from opaque mass helper", ""),
        )
        masks = test[[
            "query_id", "official_formula_test", "formula_reconstructed_test", "formula_paired_test",
            "mass_supported_test", "mass_paired_test", "mask_exclusion_reason",
        ]]
        query_columns = [
            "query_id", "identifier", "fold", "smiles", "inchikey", "molecule_group_id",
            "precomputed_molecule_group_id", "identity_mismatch", "formula", "precursor_formula",
            "parent_mass", "precursor_mz", "adduct", "instrument_type", "collision_energy",
        ]
        query_manifest = metadata[query_columns].copy()
        query_manifest["n_peaks"] = metadata["mzs"].map(peak_count).astype(np.int32)
        for column in masks.columns[1:]:
            query_manifest[column] = query_manifest.query_id.map(dict(zip(masks.query_id, masks[column])))
        query_manifest.to_parquet(self.out / "query_manifest.parquet", index=False, compression="zstd")
        masks.to_parquet(self.out / "query_masks.parquet", index=False, compression="zstd")
        pd.DataFrame([{
            "n_rows": len(metadata), "n_identity_mismatches": int(metadata.identity_mismatch.sum()),
            "n_test_unique_smiles": int(test.smiles.nunique()), "n_test_molecules": int(test.molecule_group_id.nunique()),
            "n_mass_target_absent_spectra": int((~test.mass_supported_test).sum()),
            "n_mass_target_absent_molecules": int(test.loc[~test.mass_supported_test, "molecule_group_id"].nunique()),
        }]).to_csv(self.out / "dataset_audit.csv", index=False)

        prediction_specs = []
        for model in self.models.values():
            prediction_specs.append((model.test_pred_dir / "fp_probs.pt", 17556, model.prediction_samples))
            if model.val_pred_dir is not None:
                prediction_specs.append((model.val_pred_dir / "fp_probs.pt", 19429, model.prediction_samples))
        for path, expected_rows, expected_samples in prediction_specs:
            data = torch.load(path, map_location="cpu", mmap=True)
            stack = data["stack"] if isinstance(data, dict) else data
            if tuple(stack.shape) != (expected_rows, expected_samples, 4096):
                raise ValueError(f"Unexpected frozen prediction shape {tuple(stack.shape)} in {path}")
            for start in range(0, expected_rows, 256):
                if not torch.isfinite(stack[start:start + 256]).all():
                    raise ValueError(f"Non-finite frozen predictions in {path}")
            del data, stack

        input_paths = [
            dataset_path, self.data / "fp_4096.npy", self.data / "inchis.npy",
            self.data / "massspecgym_118m_mira.json",
        ]
        for setting in ["formula", "formula_uncapped", "mass"]:
            input_paths.extend(self.data / f"MassSpecGym_retrieval_candidates_{setting}{suffix}" for suffix in [".json", "_fps.npz", "_inchi.npz"])
        input_paths.extend(path for path, _, _ in prediction_specs)
        input_paths.extend(model.test_pred_dir / "ranker.pt" for model in self.models.values())
        input_paths.extend(path for model in self.models.values() for path in model.checkpoint_files)
        input_paths.extend(
            model.archived_test_score for model in self.models.values()
            if model.archived_test_score is not None
        )
        rows = []
        for path in dict.fromkeys(map(Path, input_paths)):
            digest = self.sha256(path)
            self.input_hashes[str(path.resolve())] = digest
            rows.append({"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest})
        pd.DataFrame(rows).to_csv(self.out / "input_hashes.csv", index=False)
        checkpoint_rows = []
        for model in self.models.values():
            for member, checkpoint in enumerate(model.checkpoint_files):
                checkpoint_rows.append({"run_label": model.run_label, "member": member, "checkpoint": str(checkpoint), "sha256": self.input_hashes[str(checkpoint.resolve())]})
        pd.DataFrame(checkpoint_rows).to_csv(self.out / "checkpoint_manifest.csv", index=False)
        self.manifest["input_hash_digest"] = hashlib.sha256(json.dumps(self.input_hashes, sort_keys=True).encode()).hexdigest()
        self._atomic_json(self.manifest_path, self.manifest)
        self.mark_stage("preflight", outputs)

    def candidates(self) -> None:
        outputs = [
            self.helper_dir / "MassSpecGym_retrieval_candidates_formula_pubchem_record_capped256_fps.npz",
            self.helper_dir / "candidate_record_control_summary.csv",
        ]
        if self.stage_complete("candidates", outputs):
            print("[candidates] using cached record-preserving deterministic cap")
            return
        self.link_helpers()
        build_record_preserving_formula_cap(
            self.data / "MassSpecGym.tsv",
            self.data / "MassSpecGym_retrieval_candidates_formula_uncapped.json",
            self.data / "MassSpecGym_retrieval_candidates_formula_uncapped_fps.npz",
            self.data / "MassSpecGym_retrieval_candidates_formula_uncapped_inchi.npz",
            self.data / "inchis.npy",
            self.helper_dir,
            cap=256, seed=42, max_queries=self.args.max_queries,
            write_manifest=self.args.write_candidate_manifest and self.args.max_queries is None,
        )
        self.mark_stage("candidates", outputs)

    def model_hash(self, model: ModelSource) -> str:
        digests = [self.input_hashes.get(str(path.resolve())) or self.sha256(path) for path in model.checkpoint_files]
        return hashlib.sha256("".join(digests).encode()).hexdigest()

    def candidate_hash(self, setting: str) -> str:
        paths = [
            self.helper_dir / f"MassSpecGym_retrieval_candidates_{setting}_fps.npz",
            self.helper_dir / f"MassSpecGym_retrieval_candidates_{setting}_inchi.npz",
        ]
        return hashlib.sha256("".join(self.sha256(path) for path in paths).encode()).hexdigest()

    def score_dir(self, cell: ScoreCell) -> Path:
        return self.out / "scores" / cell.name

    def cell_helper_dir(self, cell: ScoreCell) -> Path:
        directory = self.score_dir(cell) / "helpers"
        directory.mkdir(parents=True, exist_ok=True)
        for name in ["fp_4096.npy", "inchis.npy"]:
            source, target = self.helper_dir / name, directory / name
            if not target.exists():
                target.symlink_to(source)
        for suffix in ["_fps.npz", "_inchi.npz"]:
            name = f"MassSpecGym_retrieval_candidates_{cell.helper_setting}{suffix}"
            source, target = self.helper_dir / name, directory / name
            if not target.exists():
                target.symlink_to(source)
        json_name = f"MassSpecGym_retrieval_candidates_{cell.helper_setting}.json"
        json_source, json_path = self.helper_dir / json_name, directory / json_name
        if not json_path.exists():
            json_path.symlink_to(json_source)
        return directory

    def evaluation_score(self, cell: ScoreCell) -> Path:
        return self.score_dir(cell) / "record_scores.pt"

    def _archived_score_compatible(self, path: Path, model: ModelSource) -> bool:
        if not path.exists():
            return False
        try:
            bundle = torch.load(path, map_location="cpu", mmap=True)
            stack = bundle.get("scores_stack_flat")
            ptr = bundle["ptr"].long()
            labels = bundle["labels_flat"]
            expected_queries = min(self.args.max_queries or 17556, 17556)
            if stack is None or tuple(stack.shape) != (model.prediction_samples, int(ptr[-1])):
                return False
            if ptr.numel() - 1 != expected_queries or labels.numel() != int(ptr[-1]):
                return False
            if bundle["scores_flat"].dtype != torch.float32 or stack.dtype != torch.float32:
                return False
            if not torch.allclose(bundle["scores_flat"], stack.mean(dim=0), atol=1e-6, rtol=1e-6):
                return False
            reference = self.evaluation_score(
                self._cell("mlp_formula", "test", "formula_official_capped")
            )
            if reference.exists() and reference.resolve() != path.resolve():
                canonical = torch.load(reference, map_location="cpu", mmap=True)
                if not torch.equal(ptr, canonical["ptr"].long()):
                    return False
                if not torch.equal(labels.float(), canonical["labels_flat"].float()):
                    return False
            return True
        except (KeyError, RuntimeError, ValueError):
            return False

    def _prepare_archived_score(self, cell: ScoreCell, model: ModelSource) -> None:
        source = model.archived_test_score
        if source is None:
            raise ValueError(f"{cell.name} has no archived score source")
        output = self.evaluation_score(cell)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.args.max_queries is None:
            if output.exists() or output.is_symlink():
                if output.is_symlink() and output.resolve() == source.resolve():
                    pass
                elif self._archived_score_compatible(output, model):
                    return
                else:
                    output.unlink()
            if not output.exists():
                output.symlink_to(source)
        else:
            if not self._archived_score_compatible(output, model) or "scores" in self.force_stages:
                if output.is_symlink():
                    output.unlink()
                self.run_command("scores", cell.name + "__archived_prefix", [
                    self.python, self.repo / "scripts/canonicalize_score_bundle.py",
                    "--input", source, "--output", output,
                    "--dataset_tsv", self.data / "MassSpecGym.tsv",
                    "--helper_dir", self.helper_dir, "--candidate_setting", "formula",
                    "--split", "test", "--record_policy", "preserve",
                    "--label_mode", "fingerprint", "--query_identity_source", "precomputed",
                    "--max_queries", str(self.args.max_queries),
                ])
        if not self._archived_score_compatible(output, model):
            raise RuntimeError(f"Archived score bundle failed canonical alignment: {output}")

    @staticmethod
    def _raw_score_compatible(path: Path, cell: ScoreCell) -> bool:
        if not path.exists():
            return False
        try:
            bundle = torch.load(path, map_location="cpu")
            expected_label = "fingerprint" if cell.helper_setting == "formula" else "inchikey"
            return bool(
                bundle["scores_flat"].dtype == torch.float32
                and bundle.get("score_dtype") == "float32"
                and bundle.get("candidate_setting") == cell.helper_setting
                and bundle.get("split") == cell.split
                and bundle.get("label_mode") == expected_label
                and bundle.get("query_identity_source") == "precomputed"
                and bundle.get("candidate_record_policy") == "preserve"
                and bundle.get("candidate_tie_break") == "source_order"
                and torch.isfinite(bundle["scores_flat"]).all()
            )
        except (KeyError, RuntimeError, ValueError):
            return False

    def scores(self) -> None:
        outputs = [self.evaluation_score(cell) for cell in self.cells]
        if self.stage_complete("scores", outputs):
            print("[scores] using cached record-preserving score bundles")
            return
        for cell in self.cells:
            directory = self.score_dir(cell)
            raw_dir = directory / "raw"
            raw_score = raw_dir / "scores_ranker_score.pt"
            evaluation_bundle = self.evaluation_score(cell)
            model = self.models[cell.model]
            if model.archived_test_score is not None:
                self._prepare_archived_score(cell, model)
                continue
            evaluation_compatible = self._raw_score_compatible(evaluation_bundle, cell)
            if evaluation_compatible and "scores" not in self.force_stages:
                if not self.args.keep_raw_scores:
                    shutil.rmtree(raw_dir, ignore_errors=True)
                continue
            cell_helpers = self.cell_helper_dir(cell)
            pred_dir = model.val_pred_dir if cell.split == "val" else model.test_pred_dir
            if pred_dir is None:
                raise ValueError(f"{cell.name} has no prediction source")
            raw_compatible = self._raw_score_compatible(raw_score, cell)
            if not raw_compatible or "scores" in self.force_stages:
                command = [
                    self.python, self.repo / "scripts/prepare_split_scores.py",
                    "--dataset_tsv", self.data / "MassSpecGym.tsv", "--helper_dir", cell_helpers,
                    "--pred_dir", pred_dir, "--out_dir", raw_dir, "--split", cell.split,
                    "--architecture", model.architecture, "--candidate_setting", cell.helper_setting,
                    "--label_mode", "fingerprint" if cell.helper_setting == "formula" else "inchikey",
                    "--query_identity_source", "precomputed",
                    "--missing_target_policy", "error", "--lazy_candidate_helpers",
                    "--aggregation", "score", "--temperature", str(EVALUATION_TEMPERATURE),
                    "--score_dtype", "float32", "--device", self.args.device,
                    "--batch_size", str(self.args.batch_size), "--num_workers", str(self.args.num_workers),
                ]
                if self.args.max_queries is not None:
                    command.extend(["--max_queries", str(self.args.max_queries)])
                if raw_score.exists():
                    command.append("--overwrite")
                self.run_command("scores", cell.name + "__raw", command)
            if not evaluation_compatible or "scores" in self.force_stages:
                label_mode = "fingerprint" if cell.helper_setting == "formula" else "inchikey"
                self.run_command("scores", cell.name + "__records", [
                    self.python, self.repo / "scripts/canonicalize_score_bundle.py",
                    "--input", raw_score, "--output", evaluation_bundle,
                    "--dataset_tsv", self.data / "MassSpecGym.tsv", "--helper_dir", cell_helpers,
                    "--candidate_setting", cell.helper_setting, "--split", cell.split,
                    "--record_policy", "preserve", "--label_mode", label_mode,
                    "--query_identity_source", "precomputed",
                ])
            if not self._raw_score_compatible(evaluation_bundle, cell):
                raise RuntimeError(f"Invalid record-preserving score bundle: {evaluation_bundle}")
            if not self.args.keep_raw_scores:
                shutil.rmtree(raw_dir)
        self.mark_stage("scores", outputs)

    def query_score_path(self, cell: ScoreCell) -> Path:
        return self.out / "query_scores" / f"{cell.name}.parquet"

    def metrics(self) -> None:
        outputs = [self.query_score_path(cell) for cell in self.cells]
        if self.stage_complete("metrics", outputs):
            print("[metrics] using cached record-level query-score rows")
            return
        for cell in self.cells:
            output = self.query_score_path(cell)
            if output.exists() and "metrics" not in self.force_stages:
                continue
            model = self.models[cell.model]
            cell_helpers = self.cell_helper_dir(cell)
            pred_dir = model.val_pred_dir if cell.split == "val" else model.test_pred_dir
            command = [
                self.python, self.repo / "scripts/compile_revision_scores.py",
                "--score", self.evaluation_score(cell), "--fp_probs", pred_dir / "fp_probs.pt",
                "--dataset_tsv", self.data / "MassSpecGym.tsv", "--helper_dir", cell_helpers,
                "--candidate_setting", cell.helper_setting, "--split", cell.split, "--out", output,
                "--run_id", self.out.name, "--run_label", cell.model,
                "--architecture", model.architecture,
                "--training_candidate_setting", model.training_candidates,
                "--evaluation_candidate_setting", cell.evaluation_setting,
                "--pool_variant", cell.pool_variant, "--query_mask_id", cell.query_mask_id,
                "--model_hash", self.model_hash(model), "--candidate_pool_hash", self.candidate_hash(cell.helper_setting),
                "--temperature", str(EVALUATION_TEMPERATURE),
                "--candidate_record_policy", "preserve",
                "--candidate_tie_break", "source_order",
                "--aurc_convention", "manuscript_trapezoid_seed42",
                "--feature_convention", "manuscript",
            ]
            if cell.evaluation_setting == "formula_official_capped":
                command.append("--include_fingerprint_uncertainty")
            if cell.split == "test":
                command.extend(["--query_masks", self.out / "query_masks.parquet"])
            if self.args.analysis_scope == "extended":
                command.append("--include_cardinality")
            self.run_command("metrics", cell.name, command)
        self.mark_stage("metrics", outputs)

    def _cell(self, model: str, split: str, evaluation: str) -> ScoreCell:
        matches = [cell for cell in self.cells if cell.model == model and cell.split == split and cell.evaluation_setting == evaluation]
        if len(matches) != 1:
            raise ValueError(f"Expected one cell for {(model, split, evaluation)}, found {len(matches)}")
        return matches[0]

    def temperature(self) -> None:
        out_dir = self.out / "figures" / "temperature"
        output = out_dir / "temperature_sensitivity_rel_aurc.csv"
        if self.stage_complete("temperature", [output, out_dir / "temperature_sensitivity_rel_aurc.pdf"]):
            print("[temperature] using cached results")
            return
        mlp = self.score_dir(self._cell("mlp_formula", "test", "formula_official_capped"))
        transformer = self.score_dir(self._cell("transformer_formula", "test", "formula_official_capped"))
        self.run_command("temperature", "formula_ensembles", [
            self.python, self.repo / "scripts/run_temperature_sensitivity.py",
            "--model", f"mlp={mlp}", "--model", f"transformer={transformer}",
            "--score_filename", "record_scores.pt", "--dataset_tsv", self.data / "MassSpecGym.tsv",
            "--helper_dir", self.helper_dir, "--candidate_setting", "formula", "--out_dir", out_dir,
            "--candidate_record_policy", "preserve", "--candidate_tie_break", "source_order",
            "--aurc_convention", "manuscript_trapezoid_seed42",
            "--feature_convention", "manuscript",
            "--rankwise_temp", str(EVALUATION_TEMPERATURE), "--temperatures", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0",
            *(["--max_queries", str(self.args.max_queries)] if self.args.max_queries is not None else []),
        ])
        self.mark_stage("temperature", [output, out_dir / "temperature_sensitivity_rel_aurc.pdf"])

    def meta_variants(self) -> List[str]:
        return ["full", "no_gap_at_k"] if self.args.meta_ablation else ["full"]


    def meta(self) -> None:
        fold_map = self.out / "meta" / "meta_cv_fold_assignments.csv"
        outputs = []
        for model_name in ["mlp_formula", "transformer_formula"]:
            for variant in self.meta_variants():
                outputs.append(self.out / "meta" / model_name / variant / "meta_predictions.parquet")
        if self.stage_complete("meta", outputs):
            print("[meta] using cached validation-only logistic models")
            return
        for model_name in ["mlp_formula", "transformer_formula"]:
            model = self.models[model_name]
            val_cell = self._cell(model_name, "val", "formula_official_capped")
            test_cell = self._cell(model_name, "test", "formula_official_capped")
            for variant in self.meta_variants():
                out_dir = self.out / "meta" / model_name / variant
                prediction_output = out_dir / "meta_predictions.parquet"
                if prediction_output.exists() and "meta" not in self.force_stages:
                    continue
                command = [
                    self.python, self.repo / "scripts/run_meta_score_analysis.py",
                    "--model_label", model_name, "--dataset_tsv", self.data / "MassSpecGym.tsv",
                    "--helper_dir", self.helper_dir, "--candidate_setting", "formula",
                    "--val_score", self.evaluation_score(val_cell), "--val_fp_probs", model.val_pred_dir / "fp_probs.pt",
                    "--test_score", self.evaluation_score(test_cell), "--test_fp_probs", model.test_pred_dir / "fp_probs.pt",
                    "--out_dir", out_dir, "--temperature", str(EVALUATION_TEMPERATURE), "--meta_model", "logistic",
                    "--candidate_record_policy", "preserve", "--candidate_tie_break", "source_order",
                    "--aurc_convention", "manuscript_trapezoid_seed42",
                    "--feature_convention", "manuscript",
                    "--cv_fold_assignments", fold_map, "--delta", "0.001",
                    "--target_risks", *map(str, TARGET_RISKS),
                ]
                if self.args.max_queries is not None:
                    command.extend(["--max_queries", str(self.args.max_queries)])
                if variant == "no_gap_at_k":
                    command.append("--exclude_score_gap_at_k")
                self.run_command("meta", model_name + "__" + variant, command)
        self.mark_stage("meta", outputs)

    def meta_specs(self):
        specs = [
            ("mlp_formula", "meta_full", self.out / "meta/mlp_formula/full/meta_predictions.parquet"),
            ("transformer_formula", "meta_full", self.out / "meta/transformer_formula/full/meta_predictions.parquet"),
        ]
        if self.args.meta_ablation:
            specs.extend([
                ("mlp_formula", "meta_no_gap_at_k", self.out / "meta/mlp_formula/no_gap_at_k/meta_predictions.parquet"),
                ("transformer_formula", "meta_no_gap_at_k", self.out / "meta/transformer_formula/no_gap_at_k/meta_predictions.parquet"),
            ])
        return specs

    def combined_query_scores(self) -> pd.DataFrame:
        scores = pd.concat([pd.read_parquet(self.query_score_path(cell)) for cell in self.cells], ignore_index=True)
        return merge_meta_predictions(scores, self.meta_specs())

    def sgr(self) -> None:
        out_dir = self.out / "sgr"
        outputs = [out_dir / name for name in ["sgr_score_selection.csv", "sgr_partitions.csv", "sgr_thresholds.csv", "sgr_evaluation.csv"]]
        if self.stage_complete("sgr", outputs):
            print("[sgr] using cached risk-control analysis")
            return
        scores = self.combined_query_scores()
        out_dir.mkdir(parents=True, exist_ok=True)
        run_sgr_stability(scores, out_dir, seeds=self.sgr_seeds(), target_risks=TARGET_RISKS, delta=0.001)
        thresholds = pd.read_csv(out_dir / "sgr_thresholds.csv")
        evaluation = pd.read_csv(out_dir / "sgr_evaluation.csv")
        seed42 = thresholds[thresholds.seed == 42].merge(
            evaluation[evaluation.seed == 42],
            on=["run_label", "K", "measure", "seed", "target_risk", "delta", "n_cal", "n_eval", "feasible"],
            validate="one_to_one",
        )
        seed42["category"] = "retrieval"
        seed42["loss"] = seed42.K.map(lambda k: f"hit@{k}")
        seed42.to_csv(out_dir / "sgr_results_seed42.csv", index=False)
        self.mark_stage("sgr", outputs)

    def bootstrap(self) -> None:
        out_dir = self.out / "results"
        outputs = [out_dir / "metrics_tidy.csv", out_dir / "query_scores.parquet"]
        if self.args.bootstrap_replicates > 0:
            outputs.append(out_dir / "bootstrap_replicates.parquet")
        if self.stage_complete("bootstrap", outputs):
            print("[bootstrap] using cached clustered intervals")
            return
        command = [self.python, self.repo / "scripts/aggregate_revision_results.py"]
        for cell in self.cells:
            command.extend(["--query_score", self.query_score_path(cell)])
        for run_label, output_name, path in self.meta_specs():
            command.extend(["--meta", f"{run_label},{output_name}={path}"])
        command.extend(["--out_dir", out_dir, "--bootstrap_replicates", str(self.args.bootstrap_replicates), "--bootstrap_seed", "42"])
        self.run_command("bootstrap", "canonical_results", command)
        self.mark_stage("bootstrap", outputs)

    def figures(self) -> None:
        outputs = [
            self.out / "figures/meta/mlp_formula_rc_aurc_paired_retrieval_score_meta.pdf",
            self.out / "figures/meta/transformer_formula_rc_aurc_paired_retrieval_score_meta.pdf",
            self.out / "figures/candidates/candidate_distribution_histograms.pdf",
            self.out / "figures/candidates/candidate_size_stratification.pdf",
            self.out / "figures/meta/mlp_formula_rc_aurc_paired_retrieval_score.pdf",
            self.out / "figures/meta/transformer_formula_rc_aurc_paired_retrieval_score.pdf",
            self.out / "figures/meta/mlp_mc_dropout_rc_aurc_paired_retrieval_score.pdf",
            self.out / "figures/meta/mlp_laplace_rc_aurc_paired_retrieval_score.pdf",
            self.out / "figures/sgr_coverage_mlp_formula_seed42.pdf",
            self.out / "figures/sgr_coverage_mlp_formula_seed42.png",
            self.out / "figures/sgr_coverage_transformer_formula_seed42.pdf",
            self.out / "figures/sgr_coverage_transformer_formula_seed42.png",
        ]
        if self.stage_complete("figures", outputs):
            print("[figures] using cached paper figures")
            return
        self.run_command("figures", "meta_joint", [
            self.python, self.repo / "scripts/plot_meta_joint_results.py",
            "--query_scores", self.out / "results/query_scores.parquet",
            "--out_dir", self.out / "figures/meta",
        ])
        self.run_command("figures", "candidate_histograms", [
            self.python, self.repo / "scripts/build_candidate_pool_appendix.py",
            "--dataset_tsv", self.data / "MassSpecGym.tsv", "--helper_dir", self.helper_dir,
            "--out_dir", self.out / "figures/candidates", "--distributions_only",
            *(["--max_queries", str(self.args.max_queries)] if self.args.max_queries is not None else []),
        ])
        self.run_command("figures", "candidate_size_stratification", [
            self.python, self.repo / "scripts/plot_candidate_size_stratification.py",
            "--query_scores", self.out / "results/query_scores.parquet",
            "--out_path", self.out / "figures/candidates/candidate_size_stratification.pdf",
            "--run_label", "mlp_formula",
        ])
        for run_label in ["mlp_formula", "transformer_formula"]:
            self.run_command("figures", "sgr_seed42__" + run_label, [
                self.python, self.repo / "scripts/plot_sgr_analysis.py",
                "--sgr_csv", self.out / "sgr/sgr_results_seed42.csv",
                "--out_path", self.out / f"figures/sgr_coverage_{run_label}_seed42.pdf",
                "--run_label", run_label,
            ])
        self.mark_stage("figures", outputs)

    def tables(self) -> None:
        outputs = [
            self.out / "results/table_supervised_combination.csv",
            self.out / "results/table_uq_methods.csv",
            self.out / "results/table_candidate_settings.csv",
            self.out / "results/old_to_canonical_crosswalk.csv",
        ]
        if self.stage_complete("tables", outputs):
            print("[tables] using cached table exports")
            return
        canonical = pd.read_csv(self.out / "results/metrics_tidy.csv")
        crosswalk_rows = []
        old_specs = [
            ("mlp_formula", "formula_official_capped", self.repo / "outputs/revision_candidate_comparison/formula_capped/mlp/rel_aurc_retrieval_score.csv"),
            ("transformer_formula", "formula_official_capped", self.repo / "outputs/revision_candidate_comparison/formula_capped/transformer/rel_aurc_retrieval_score.csv"),
            ("mlp_formula", "formula_pubchem_uncapped", self.repo / "outputs/revision_uncapped/eval/mlp/rel_aurc_retrieval_score.csv"),
            ("transformer_formula", "formula_pubchem_uncapped", self.repo / "outputs/revision_uncapped/eval/transformer/rel_aurc_retrieval_score.csv"),
            ("mlp_mass", "mass_existing_capped256", self.repo / "outputs/mass_mlp_bienc/eval/ensemble/bienc_mass/rel_aurc_retrieval_score.csv"),
        ]
        for run_label, setting, path in old_specs:
            if not path.exists():
                continue
            old = pd.read_csv(path, index_col=0)
            for measure in old.index:
                for column in old.columns:
                    if not column.startswith("hit@"):
                        continue
                    k = int(column.split("@")[1])
                    match = canonical[(canonical.run_label == run_label) & (canonical.evaluation_candidate_setting == setting) & (canonical.K == k) & (canonical.measure == measure) & (canonical.metric == "rel_aurc")]
                    crosswalk_rows.append({
                        "run_label": run_label, "evaluation_candidate_setting": setting, "K": k,
                        "measure": measure, "old_source": str(path), "old_value": float(old.loc[measure, column]),
                        "canonical_value": float(match.value.iloc[0]) if len(match) == 1 else np.nan,
                    })
        pd.DataFrame(crosswalk_rows).to_csv(outputs[2], index=False)
        self.mark_stage("tables", outputs)

    def report(self) -> None:
        output = self.out / "report/index.html"
        if self.stage_complete("report", [output]):
            print("[report] using cached static report")
            return
        self.run_command("report", "html", [
            self.python, self.repo / "scripts/build_revision_share_report.py",
            "--source_dir", self.out, "--out_dir", self.out / "report",
        ])
        self.mark_stage("report", [output])

    def validate(self) -> None:
        output = self.out / "validation_report.json"
        if self.stage_complete("validate", [output]):
            print("[validate] using cached acceptance report")
            return
        checks = []
        def add(name, passed, observed):
            checks.append({"name": name, "passed": bool(passed), "observed": observed})
        result_report = json.loads((self.out / "results/validation_report.json").read_text())
        add("result validation passed", result_report["passed"], result_report)
        required = [
            self.out / "run_manifest.json", self.out / "input_hashes.csv", self.out / "query_manifest.parquet",
            self.out / "candidates/helpers/candidate_record_control_summary.csv",
            self.out / "results/query_scores.parquet", self.out / "results/metrics_tidy.csv",
            self.out / "results/table_supervised_combination.csv", self.out / "results/table_uq_methods.csv",
            self.out / "results/table_candidate_settings.csv", self.out / "sgr/sgr_evaluation.csv",
            self.out / "figures/candidates/candidate_size_stratification.pdf",
            self.out / "figures/sgr_coverage_mlp_formula_seed42.pdf",
            self.out / "figures/sgr_coverage_transformer_formula_seed42.pdf",
            self.out / "report/index.html",
        ]
        missing = [str(path) for path in required if not path.exists()]
        add("all required artifacts exist", not missing, missing)
        scores = pd.read_parquet(self.out / "results/query_scores.parquet")
        official = scores[(scores.split == "test") & (scores.evaluation_candidate_setting == "formula_official_capped")]
        official_counts = {
            f"{run_label}|K={int(k)}": int(count)
            for (run_label, k), count in official.groupby(["run_label", "K"]).size().items()
        }
        add(
            "official formula test has 17,556 spectra per K/model",
            all(len(group) == min(self.args.max_queries or 17556, 17556) for _, group in official.groupby(["run_label", "K"])),
            official_counts,
        )
        add(
            "all evaluated candidate pools preserve record occurrences",
            set(scores["candidate_record_policy"].astype(str)) == {"preserve"},
            sorted(scores["candidate_record_policy"].astype(str).unique().tolist()),
        )
        add(
            "all canonical result rows use T_eval=0.003",
            np.allclose(scores["T_eval"].to_numpy(dtype=float), EVALUATION_TEMPERATURE),
            sorted(scores["T_eval"].drop_duplicates().astype(float).tolist()),
        )
        expected_uq_models = {"mlp_formula", "transformer_formula", "mlp_mc_dropout", "mlp_laplace"}
        observed_uq_models = set(official["run_label"].astype(str))
        add("all official-candidate UQ models are present", expected_uq_models <= observed_uq_models, sorted(observed_uq_models))
        sgr = pd.read_csv(self.out / "sgr/sgr_evaluation.csv")
        sgr_sets = sgr.groupby(["run_label", "K"])["measure"].agg(lambda values: set(values))
        add(
            "SGR includes every manuscript single-score curve",
            all(set(SGR_SINGLE_MEASURES) <= measures for measures in sgr_sets),
            {f"{model}|K={int(k)}": sorted(measures) for (model, k), measures in sgr_sets.items()},
        )
        official_k1 = official.loc[official["K"] == 1]
        record_totals = official_k1.groupby("run_label")["candidate_count"].sum().astype(int).to_dict()
        add(
            "official formula test preserves all 2,909,859 distributed candidate occurrences",
            bool(record_totals) and (
                all(value == 2909859 for value in record_totals.values())
                if self.args.max_queries is None
                else len(set(record_totals.values())) == 1
            ),
            record_totals,
        )
        cap_summary = pd.read_csv(self.out / "candidates/helpers/candidate_record_control_summary.csv")
        cap_ok = bool(
            (cap_summary["n_capped_records"] <= 256).all()
            and (cap_summary["n_target_occurrences_capped"] >= 1).all()
            and (~cap_summary["target_inserted"].astype(bool)).all()
            and cap_summary["source_order_preserved"].astype(bool).all()
        )
        add("record-level cap is at most 256, target-retaining, and insertion-free", cap_ok, {
            "n_queries": len(cap_summary),
            "max_capped_records": int(cap_summary["n_capped_records"].max()),
            "min_capped_target_occurrences": int(cap_summary["n_target_occurrences_capped"].min()),
        })
        if self.args.max_queries is None:
            expected_hits = {
                "mlp_formula": {1: 0.1312, 5: 0.2729, 20: 0.4757},
                "transformer_formula": {1: 0.1744, 5: 0.3564, 20: 0.5702},
            }
            observed_hits = {
                run_label: {
                    int(k): float(group["hit"].mean())
                    for k, group in model_rows.groupby("K")
                }
                for run_label, model_rows in official.groupby("run_label")
                if run_label in expected_hits
            }
            parity = all(
                abs(observed_hits[model][k] - expected) <= 2e-4
                for model, values in expected_hits.items()
                for k, expected in values.items()
            )
            add("official Hit@K reproduces manuscript values", parity, observed_hits)
        manifest_hash = self.sha256(self.out / "results/metrics_tidy.csv")
        add("record-level metrics hash recorded", True, manifest_hash)
        report = {"passed": all(row["passed"] for row in checks), "checks": checks, "metrics_sha256": manifest_hash}
        self._atomic_json(output, report)
        if not report["passed"]:
            raise RuntimeError("Canonical rerun failed acceptance validation")
        self.mark_stage("validate", [output])

    def execute(self) -> None:
        self.initialize_manifest()
        if self.args.stages != ["all"] and self.selected_stages != self.args.stages:
            print(f"[runner] expanded stages with prerequisites: {','.join(self.selected_stages)}")
        methods = {
            "preflight": self.preflight, "candidates": self.candidates, "scores": self.scores,
            "metrics": self.metrics, "temperature": self.temperature, "meta": self.meta,
            "sgr": self.sgr, "bootstrap": self.bootstrap, "figures": self.figures,
            "tables": self.tables, "report": self.report, "validate": self.validate,
        }
        for stage in STAGES:
            if stage not in self.selected_stages:
                continue
            started = time.time()
            methods[stage]()
            print(f"[{stage}] complete in {(time.time() - started) / 60:.1f} min")
        print(f"Canonical revision rerun available at {self.out}")


def parse_stages(value: str) -> List[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if values == ["all"]:
        return values
    unknown = sorted(set(values) - set(STAGES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown stages: {unknown}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/home/mira/data/msuq"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/revision_rerun_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stages", type=parse_stages, default=["all"])
    parser.add_argument("--force-stage", action="append", default=[], choices=STAGES)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--analysis-scope", choices=["core", "extended"], default="core")
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--sgr-repeats", type=int, default=1,
                        help="1 uses paper-parity seed 42; values >1 use seeds 0..N-1")
    parser.add_argument("--meta-ablation", action="store_true",
                        help="Also fit the optional meta model without score_gap_at_K")
    parser.add_argument("--write-candidate-manifest", action="store_true",
                        help="Write the optional multi-million-row candidate provenance table")
    parser.add_argument("--quick-hashes", action="store_true", help="Smoke-only: avoid hashing files larger than 100 MiB")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--keep-raw-scores", action="store_true",
                        help="Retain temporary pre-provenance score bundles (roughly doubles score storage)")
    args = parser.parse_args()
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates must be non-negative")
    if args.sgr_repeats < 1:
        parser.error("--sgr-repeats must be positive")
    RevisionRunner(args).execute()


if __name__ == "__main__":
    main()
