"""Canonical paper-result layout and deterministic release packaging."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml


MODEL_IDS = {
    "mlp_formula": "ensemble_mlp_formula",
    "transformer_formula": "ensemble_transformer_formula",
    "mlp_mass": "ensemble_mlp_mass",
    "mlp_mc_dropout": "mc_dropout_mlp_formula",
    "mlp_laplace": "laplace_mlp_formula",
}

SETTING_IDS = {
    "formula_official_capped": "formula_official",
    "formula_pubchem_capped256": "formula_paired_capped",
    "formula_pubchem_uncapped": "formula_uncapped",
    "mass_existing_capped256": "mass_capped",
}

EVALUATIONS = (
    ("mlp_formula__val__formula_official", "formula_official", "ensemble_mlp_formula", "validation"),
    ("mlp_formula__test__formula_official", "formula_official", "ensemble_mlp_formula", "test"),
    ("transformer_formula__val__formula_official", "formula_official", "ensemble_transformer_formula", "validation"),
    ("transformer_formula__test__formula_official", "formula_official", "ensemble_transformer_formula", "test"),
    ("mlp_mc_dropout__test__formula_official", "formula_official", "mc_dropout_mlp_formula", "test"),
    ("mlp_laplace__test__formula_official", "formula_official", "laplace_mlp_formula", "test"),
    ("mlp_formula__test__formula_capped256", "formula_paired_capped", "ensemble_mlp_formula", "test"),
    ("transformer_formula__test__formula_capped256", "formula_paired_capped", "ensemble_transformer_formula", "test"),
    ("mlp_formula__test__formula_uncapped", "formula_uncapped", "ensemble_mlp_formula", "test"),
    ("transformer_formula__test__formula_uncapped", "formula_uncapped", "ensemble_transformer_formula", "test"),
    ("mlp_formula__test__mass", "mass_capped", "ensemble_mlp_formula", "test"),
    ("mlp_mass__test__mass", "mass_capped", "ensemble_mlp_mass", "test"),
)

PREDICTION_ROLES = {
    "647fe7e978bb541e5966ca4b7920076fc598ca00b07ebf0d66935a6b2f926e85": ("ensemble_mlp_formula", "test"),
    "f0655e27546989bca1b4b063540f1bcf9ea77703606463ab83a8a2506e645b0c": ("ensemble_mlp_formula", "validation"),
    "416130df96aead6b321448eca94f213b8490a7c53d0000e54060506ad5034e58": ("ensemble_transformer_formula", "test"),
    "d3116526cb0244fe6dd2c85c5d03cbf531363f3a7e77762c9425d8e779f497ac": ("ensemble_transformer_formula", "validation"),
    "b593f95d8105f9946320d02544007d625825d59240f6323af5434f40b5b04a9c": ("ensemble_mlp_mass", "test"),
    "5928328f6a8ef49776adf6d99591ca07e60b89146500d87f458a48d3c89ddb9d": ("mc_dropout_mlp_formula", "test"),
    "95b3da4826efb6cb97e395a06c61067ef85371c2b7cdac2be517cdd51103dd62": ("laplace_mlp_formula", "test"),
}

RANKER_HASH = "e1ed5484a17470a497acfaac15abdc585730ac0336021c1cf572ecbcb67addad"
ARCHIVED_SCORE_HASHES = {
    "e53b3b23cc1322ea49f38299ab3ca3ed4e41d9acf5f27a0e9c04df7567d538d0",
    "fd7ddcd511d40d9de9c32edb7dccb487259475becd323a46521a397714eee6cb",
}
EXPECTED_METRICS_SHA256 = "2fd72890fae6573757c8754fbde023493a2f20c94d83f17a979a04617ca83d40"
EXPECTED_HITS = {
    "mlp_formula": {1: 0.13118022328548645, 5: 0.27289815447710186, 20: 0.47573479152426523},
    "transformer_formula": {1: 0.174356345408977, 5: 0.3564023695602643, 20: 0.5701754385964912},
}


@dataclass(frozen=True)
class ReleaseMember:
    archive: str
    archive_path: str
    role: str
    model: str
    split: str
    size_bytes: int
    sha256: str


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, target: Path, *, hardlink: bool = True) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if hardlink:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path, *, hardlink: bool = True) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            _copy_file(path, target / path.relative_to(source), hardlink=hardlink)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sanitize_record_metadata(source: Path, score_target: Path, *, cell: str, setting: str, model: str, split: str) -> dict:
    if source.is_file():
        raw = json.loads(source.read_text())
    else:
        bundle = torch.load(score_target, map_location="cpu", mmap=True)
        ptr = bundle["ptr"].long()
        stack = bundle.get("scores_stack_flat")
        raw = {
            "candidate_record_policy": "preserve",
            "candidate_setting": "formula",
            "label_mode": "fingerprint",
            "n_members": int(stack.shape[0]) if stack is not None else 1,
            "n_queries": int(ptr.numel() - 1),
            "n_raw_scores": int(ptr[-1]),
            "n_record_scores": int(ptr[-1]),
            "n_target_absent": 0,
            "query_identity_source": "precomputed",
        }
    keep = {
        key: raw[key]
        for key in [
            "candidate_record_policy", "candidate_setting", "label_mode", "n_members",
            "n_queries", "n_raw_scores", "n_record_scores", "n_target_absent",
            "query_identity_source",
        ]
        if key in raw
    }
    keep.update({
        "evaluation_id": cell,
        "candidate_setting_id": setting,
        "model_id": model,
        "split": split,
        "score_bundle": "scores.pt",
        "score_bundle_sha256": sha256(score_target),
        "T_eval": 0.003,
        "candidate_records_deduplicated": False,
    })
    return keep


def _filtered_metrics(metrics: pd.DataFrame, model_id: str, setting_id: str, split: str) -> pd.DataFrame:
    run_label = next(key for key, value in MODEL_IDS.items() if value == model_id)
    setting = next(key for key, value in SETTING_IDS.items() if value == setting_id)
    raw_split = "val" if split == "validation" else split
    return metrics[
        (metrics["run_label"] == run_label)
        & (metrics["evaluation_candidate_setting"] == setting)
        & (metrics["split"] == raw_split)
    ].copy()


def _sanitized_inputs(source_run: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_run / "input_hashes.csv")
    frame["filename"] = frame["path"].map(lambda value: Path(value).name)
    frame["role"] = "external_data"
    frame.loc[frame.sha256.isin(PREDICTION_ROLES), "role"] = "prediction"
    frame.loc[frame.sha256 == RANKER_HASH, "role"] = "shared_prediction_metadata"
    frame.loc[frame.sha256.isin(ARCHIVED_SCORE_HASHES), "role"] = "archived_score_bundle"
    checkpoints = set(pd.read_csv(source_run / "checkpoint_manifest.csv")["sha256"])
    frame.loc[frame.sha256.isin(checkpoints), "role"] = "checkpoint"
    return frame[["filename", "role", "size_bytes", "sha256"]].drop_duplicates()


def validate_paper_results(results_dir: Path, *, strict_hash: bool = True) -> dict:
    required = [
        results_dir / "numerical/query_scores.parquet",
        results_dir / "numerical/metrics.csv",
        results_dir / "provenance/input_files.tsv",
        results_dir / "provenance/evaluation_matrix.tsv",
    ]
    checks = []

    def add(name: str, passed: bool, observed: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "observed": observed})

    missing = [str(path) for path in required if not path.is_file()]
    add("required files exist", not missing, missing)
    if missing:
        return {"passed": False, "checks": checks}

    metric_hash = sha256(results_dir / "numerical/metrics.csv")
    add("metrics hash matches frozen run", (not strict_hash) or metric_hash == EXPECTED_METRICS_SHA256, metric_hash)
    scores = pd.read_parquet(results_dir / "numerical/query_scores.parquet")
    official = scores[
        (scores["split"] == "test")
        & (scores["evaluation_candidate_setting"] == "formula_official_capped")
    ]
    counts = official.groupby(["run_label", "K"]).size().to_dict()
    add("official test contains 17,556 spectra per model and K", bool(counts) and all(value == 17556 for value in counts.values()), {str(k): int(v) for k, v in counts.items()})
    selectors = [
        "s1", "score_gap", "log_n_candidates", "n_candidates", "confidence",
        "retrieval_total", "retrieval_aleatoric", "retrieval_epistemic",
        "normalized_entropy", "rank_var_1", "rank_var_5", "rank_var_20",
        "bitwise_epistemic", "bitwise_aleatoric", "bitwise_total",
        "score_gap_at_1", "score_gap_at_5", "score_gap_at_20",
    ]
    finite = all(np.isfinite(scores[column].dropna().to_numpy(float)).all() for column in selectors if column in scores)
    add("selectors are finite", finite, selectors)
    add("all rows use T_eval=0.003", bool(np.allclose(scores["T_eval"].to_numpy(float), 0.003)), sorted(scores.T_eval.unique().tolist()))
    add("candidate records are preserved", set(scores.candidate_record_policy.astype(str)) == {"preserve"}, sorted(scores.candidate_record_policy.unique().tolist()))
    observed_hits = {}
    for model, expected in EXPECTED_HITS.items():
        rows = official[official.run_label == model]
        observed_hits[model] = {int(k): float(group.hit.mean()) for k, group in rows.groupby("K")}
    parity = all(abs(observed_hits[model][k] - value) < 1e-12 for model, values in EXPECTED_HITS.items() for k, value in values.items())
    add("official Hit@K reproduces manuscript values", parity, observed_hits)
    evaluation_dirs = [path.parent for path in results_dir.glob("evaluations/*/*/*/metadata.json")]
    add("all 12 evaluation bundles exist", len(evaluation_dirs) == len(EVALUATIONS), len(evaluation_dirs))
    bundle_ok = all((path / "scores.pt").is_file() and (path / "metrics.csv").is_file() for path in evaluation_dirs)
    add("each evaluation has scores, metrics, and metadata", bundle_ok, [str(path) for path in evaluation_dirs])
    report = {"passed": all(row["passed"] for row in checks), "metrics_sha256": metric_hash, "checks": checks}
    return report


def build_paper_results(source_run: Path, results_dir: Path, *, hardlink: bool = True, force: bool = False) -> dict:
    """Create the minimal canonical result tree without recomputing inference."""
    source_run = source_run.resolve()
    if not (source_run / "results/query_scores.parquet").is_file():
        raise FileNotFoundError(f"Not a completed paper run: {source_run}")
    if results_dir.exists():
        if not force:
            raise FileExistsError(f"{results_dir} exists; pass force=True to replace it")
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    metrics = pd.read_csv(source_run / "results/metrics_tidy.csv")
    _copy_file(source_run / "results/query_scores.parquet", results_dir / "numerical/query_scores.parquet", hardlink=hardlink)
    _copy_file(source_run / "results/metrics_tidy.csv", results_dir / "numerical/metrics.csv", hardlink=hardlink)

    matrix_rows = []
    for cell, setting, model, split in EVALUATIONS:
        source = source_run / "scores" / cell
        target = results_dir / "evaluations" / setting / model / split
        _copy_file(source / "record_scores.pt", target / "scores.pt", hardlink=hardlink)
        filtered = _filtered_metrics(metrics, model, setting, split)
        target.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(target / "metrics.csv", index=False)
        metadata = _sanitize_record_metadata(source / "record_scores.json", target / "scores.pt", cell=cell, setting=setting, model=model, split=split)
        _write_json(target / "metadata.json", metadata)
        matrix_rows.append({
            "candidate_setting": setting,
            "model": model,
            "split": split,
            "n_queries": metadata["n_queries"],
            "n_candidate_records": metadata["n_record_scores"],
            "score_sha256": metadata["score_bundle_sha256"],
        })

    analysis_sources = {
        "candidate_sets": source_run / "figures/candidates",
        "temperature": source_run / "figures/temperature",
        "sgr": source_run / "sgr",
        "meta_models": source_run / "meta",
    }
    for name, source in analysis_sources.items():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() not in {".pdf", ".png"}:
                _copy_file(path, results_dir / "analyses" / name / path.relative_to(source), hardlink=hardlink)
    for path in sorted((source_run / "figures").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_run / "figures")
        if path.suffix.lower() in {".pdf", ".png"}:
            _copy_file(path, results_dir / "figures" / relative, hardlink=hardlink)
        elif relative.parts[0] == "meta":
            _copy_file(path, results_dir / "analyses/meta_models/plots" / Path(*relative.parts[1:]), hardlink=hardlink)
        elif relative.parts[0] == "correlation":
            _copy_file(path, results_dir / "analyses/correlation" / Path(*relative.parts[1:]), hardlink=hardlink)
    for path in sorted((source_run / "results").glob("table_*")):
        _copy_file(path, results_dir / "tables" / path.name, hardlink=hardlink)
    for name in ["paired_differences.csv", "candidate_distribution_summary.csv"]:
        path = source_run / "results" / name
        if path.is_file():
            _copy_file(path, results_dir / "tables" / name, hardlink=hardlink)
    provenance = results_dir / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix_rows).to_csv(provenance / "evaluation_matrix.tsv", sep="\t", index=False)
    _sanitized_inputs(source_run).to_csv(provenance / "input_files.tsv", sep="\t", index=False)
    for name in ["dataset_audit.csv", "query_manifest.parquet", "query_masks.parquet"]:
        path = source_run / name
        if path.is_file():
            _copy_file(path, provenance / name, hardlink=hardlink)
    frozen = json.loads((source_run / "run_manifest.json").read_text())
    frozen_code = frozen.get("code_state", {})
    public_manifest = {
        "evaluation_code_state": {
            key: frozen_code.get(key)
            for key in ["commit_sha", "complete_source_sha256", "tracked_diff_sha256", "worktree_clean"]
        },
        "versions": frozen.get("versions"),
        "python": frozen.get("python"),
        "platform": frozen.get("platform"),
        "cuda": frozen.get("cuda"),
        "seeds": frozen.get("seeds"),
        "training_temperature": 0.003,
        "primary_evaluation_temperature": 0.003,
        "candidate_record_policy": "preserve",
        "candidate_tie_break": "source_order",
        "deviations": frozen.get("deviations", []),
    }
    _write_json(provenance / "run_manifest.json", public_manifest)
    report = validate_paper_results(results_dir)
    _write_json(provenance / "validation_report.json", report)
    if not report["passed"]:
        raise RuntimeError("Canonical paper-result validation failed")
    return report


def load_paper_config(repo: Path) -> dict:
    return yaml.safe_load((repo / "config/paper.yml").read_text())


def write_external_data(source_run: Path, output: Path, config: Mapping[str, object]) -> None:
    inputs = _sanitized_inputs(source_run)
    inputs = inputs[inputs.role == "external_data"].copy()
    sources = config.get("external_data", {})
    rows = []
    for row in inputs.itertuples(index=False):
        spec = sources.get(row.filename, {}) if isinstance(sources, Mapping) else {}
        rows.append({
            "filename": row.filename,
            "size_bytes": int(row.size_bytes),
            "sha256": row.sha256,
            "source_url": spec.get("url", "unknown"),
            "preparation": spec.get("preparation", "Use the existing data-preparation scripts."),
            "provenance_note": spec.get("note", ""),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)


def discover_release_sources(source_run: Path) -> tuple[list[dict], list[dict], Path]:
    inputs = pd.read_csv(source_run / "input_hashes.csv")
    predictions = []
    for row in inputs.itertuples(index=False):
        role = PREDICTION_ROLES.get(row.sha256)
        if role:
            path = Path(row.path)
            if not path.is_file() or sha256(path) != row.sha256:
                raise RuntimeError(f"Prediction source failed hash verification: {path}")
            predictions.append({"model": role[0], "split": role[1], "path": path, "sha256": row.sha256})
    if len(predictions) != 7:
        raise RuntimeError(f"Expected seven prediction tensors, found {len(predictions)}")
    ranker_rows = inputs[inputs.sha256 == RANKER_HASH]
    if ranker_rows.empty:
        raise RuntimeError("Shared ranker metadata was not recorded")
    ranker = Path(ranker_rows.iloc[0].path)

    checkpoint_frame = pd.read_csv(source_run / "checkpoint_manifest.csv")
    checkpoints = []
    for row in checkpoint_frame.itertuples(index=False):
        path = Path(row.checkpoint)
        if not path.is_file() or sha256(path) != row.sha256:
            raise RuntimeError(f"Checkpoint source failed hash verification: {path}")
        checkpoints.append({
            "model": MODEL_IDS[row.run_label], "member": int(row.member),
            "path": path, "sha256": row.sha256,
        })
    if len(checkpoints) != 18:
        raise RuntimeError(f"Expected 18 checkpoint/state files, found {len(checkpoints)}")
    return predictions, checkpoints, ranker


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_STORED
    return info


def _write_zip(output: Path, members: Sequence[tuple[Path | bytes, str, str, str, str]]) -> list[ReleaseMember]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    rows = []
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for source, archive_path, role, model, split in sorted(members, key=lambda row: row[1]):
            if not archive_path.startswith("artifacts/") or PurePosixPath(archive_path).is_absolute() or ".." in PurePosixPath(archive_path).parts:
                raise ValueError(f"Unsafe archive path: {archive_path}")
            info = _zip_info(archive_path)
            if isinstance(source, bytes):
                archive.writestr(info, source)
                digest = hashlib.sha256(source).hexdigest()
                size = len(source)
            else:
                with source.resolve().open("rb") as handle, archive.open(info, "w", force_zip64=True) as target:
                    digest_obj = hashlib.sha256()
                    size = 0
                    for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                        target.write(chunk)
                        digest_obj.update(chunk)
                        size += len(chunk)
                digest = digest_obj.hexdigest()
            rows.append(ReleaseMember(output.name, archive_path, role, model, split, size, digest))
    temporary.replace(output)
    return rows


def _tracked_source_members(repo: Path, extra: Sequence[Path]) -> list[tuple[Path | bytes, str, str, str, str]]:
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True, stdout=subprocess.PIPE).stdout.split(b"\0")
    paths = [repo / value.decode() for value in tracked if value and (repo / value.decode()).is_file()]
    paths.extend(path for path in extra if path.is_file() and path not in paths)
    return [(path, f"artifacts/source/{path.relative_to(repo).as_posix()}", "source", "", "") for path in sorted(set(paths))]


def _release_readme(config: Mapping[str, object]) -> str:
    paper = config.get("paper", {})
    return f"""# Selective MS/MS Paper Artifacts

Artifacts for [{paper.get('title', 'Selective prediction for MS/MS retrieval')}]({paper.get('url', 'https://arxiv.org/abs/2603.10950')}).

## Files

- `source.zip`: repository snapshot, environment lock, and external-data manifest.
- `results.zip`: validated scores, numerical results, analyses, figures, tables, and provenance.
- `predictions.zip`: seven frozen fingerprint-prediction tensors; shared ranker metadata is stored once.
- `checkpoints.zip`: 18 checkpoint/state files for the five released model groups.
- `MANIFEST.tsv`: member-level sizes, roles, and SHA-256 hashes.
- `SHA256SUMS`: hashes for the six companion files in this deposit. A checksum file cannot include its own stable hash.

## Extraction

Extract the four archives in the root of a clone of the repository. Every archive writes only below `artifacts/`.

```bash
unzip source.zip
unzip results.zip
unzip predictions.zip
unzip checkpoints.zip
python scripts/run_paper_evaluation.py validate --artifacts artifacts
python scripts/run_paper_evaluation.py report --artifacts artifacts --output-dir outputs/paper_results_reproduced
```

`report` requires no MassSpecGym download. Full candidate rescoring additionally requires the files and hashes in `artifacts/source/EXTERNAL_DATA.tsv`.

## Result Coverage

Official formula candidates include the formula MLP and transformer ensembles, MC Dropout, and Laplace. Paired capped and uncapped formula candidates include the formula MLP and transformer ensembles. Mass candidates include the formula-trained and mass-trained MLP ensembles. The evaluation temperature is `0.003` throughout the primary analysis.
"""


def build_release(repo: Path, source_run: Path, results_dir: Path, release_dir: Path) -> dict:
    config = load_paper_config(repo)
    result_report = validate_paper_results(results_dir)
    if not result_report["passed"]:
        raise RuntimeError("Refusing to package invalid paper results")
    predictions, checkpoints, ranker = discover_release_sources(source_run)

    external = repo / "EXTERNAL_DATA.tsv"
    write_external_data(source_run, external, config)
    members: list[ReleaseMember] = []
    members += _write_zip(release_dir / "source.zip", _tracked_source_members(repo, [external]))
    result_members = [
        (path, f"artifacts/results/{path.relative_to(results_dir).as_posix()}", "result", "", "")
        for path in sorted(results_dir.rglob("*")) if path.is_file()
    ]
    members += _write_zip(release_dir / "results.zip", result_members)

    prediction_members: list[tuple[Path | bytes, str, str, str, str]] = []
    prediction_index = []
    for item in predictions:
        archive_path = f"artifacts/models/{item['model']}/predictions/{item['split']}/fp_probs.pt"
        prediction_members.append((item["path"], archive_path, "prediction", item["model"], item["split"]))
        prediction_index.append({"model": item["model"], "split": item["split"], "path": archive_path, "sha256": item["sha256"], "ranker": "artifacts/models/shared/ranker.pt"})
    prediction_members.append((ranker, "artifacts/models/shared/ranker.pt", "shared_prediction_metadata", "shared", ""))
    prediction_members.append((json.dumps(prediction_index, indent=2, sort_keys=True).encode() + b"\n", "artifacts/models/predictions.json", "metadata", "", ""))
    members += _write_zip(release_dir / "predictions.zip", prediction_members)

    checkpoint_members = []
    for item in checkpoints:
        suffix = item["path"].suffix
        filename = f"member_{item['member'] + 1:02d}{suffix}"
        if item["model"] == "laplace_mlp_formula":
            filename = "laplace_state.pt" if item["path"].name == "laplace_state.pt" else "base_model.ckpt"
        elif item["model"] == "mc_dropout_mlp_formula":
            filename = "model.ckpt"
        archive_path = f"artifacts/models/{item['model']}/checkpoints/{filename}"
        checkpoint_members.append((item["path"], archive_path, "checkpoint", item["model"], ""))
    members += _write_zip(release_dir / "checkpoints.zip", checkpoint_members)

    release_dir.mkdir(parents=True, exist_ok=True)
    readme = release_dir / "README.md"
    readme.write_text(_release_readme(config))
    with (release_dir / "MANIFEST.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ReleaseMember.__dataclass_fields__), delimiter="\t")
        writer.writeheader()
        writer.writerows([member.__dict__ for member in members])
    checksummed = ["README.md", "MANIFEST.tsv", "source.zip", "results.zip", "predictions.zip", "checkpoints.zip"]
    (release_dir / "SHA256SUMS").write_text("".join(f"{sha256(release_dir / name)}  {name}\n" for name in checksummed))
    return verify_release(release_dir)


def verify_release(release_dir: Path) -> dict:
    expected = {"README.md", "MANIFEST.tsv", "SHA256SUMS", "source.zip", "results.zip", "predictions.zip", "checkpoints.zip"}
    observed = {path.name for path in release_dir.iterdir() if path.is_file() and not path.name.startswith(".")}
    checks = [{"name": "exactly seven release files", "passed": observed == expected, "observed": sorted(observed)}]
    checksum_ok = True
    for line in (release_dir / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        checksum_ok &= (sha256(release_dir / name) == digest)
    checks.append({"name": "companion checksums match", "passed": checksum_ok, "observed": "SHA256SUMS"})
    bad_members = []
    zip_errors = []
    for name in ["source.zip", "results.zip", "predictions.zip", "checkpoints.zip"]:
        with zipfile.ZipFile(release_dir / name) as archive:
            error = archive.testzip()
            if error:
                zip_errors.append(f"{name}:{error}")
            for info in archive.infolist():
                parts = PurePosixPath(info.filename).parts
                is_link = ((info.external_attr >> 16) & 0o170000) == 0o120000
                if not parts or parts[0] != "artifacts" or ".." in parts or PurePosixPath(info.filename).is_absolute() or is_link:
                    bad_members.append(f"{name}:{info.filename}")
    checks.append({"name": "ZIP64 archives pass CRC checks", "passed": not zip_errors, "observed": zip_errors})
    checks.append({"name": "archive members are relative regular files under artifacts", "passed": not bad_members, "observed": bad_members})
    manifest = pd.read_csv(release_dir / "MANIFEST.tsv", sep="\t")
    checks.append({"name": "manifest contains seven predictions and 18 checkpoints", "passed": int((manifest.role == "prediction").sum()) == 7 and int((manifest.role == "checkpoint").sum()) == 18, "observed": {"predictions": int((manifest.role == "prediction").sum()), "checkpoints": int((manifest.role == "checkpoint").sum())}})
    forbidden = [path for path in manifest.archive_path if Path(path).name.startswith("MassSpecGym") or Path(path).name == "massspecgym_118m_mira.json"]
    checks.append({"name": "no MassSpecGym data payloads are included", "passed": not forbidden, "observed": forbidden})
    report = {"passed": all(row["passed"] for row in checks), "checks": checks}
    _write_json(release_dir.parent / f"{release_dir.name}_verification.json", report)
    if not report["passed"]:
        raise RuntimeError("Release verification failed")
    return report


def build_inventory(repo: Path, output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(repo).as_posix()
        if rel.startswith("outputs/paper_results/"):
            category, reason = "canonical", "Included in results.zip"
        elif rel.startswith("outputs/"):
            category, reason = "redundant", "Superseded paper run or intermediate output"
        elif rel.startswith(("artifacts/", "releases/")):
            category, reason = "release", "Generated release material"
        elif "__pycache__" in path.parts or rel.endswith((".pyc", ".DS_Store")) or ".egg-info/" in rel:
            category, reason = "temporary", "Regenerable cache or metadata"
        elif rel.startswith(("scripts/", "ms_uq/", "config/", "tests/")) or rel in {"README.md", "LICENSE", "pyproject.toml", ".gitignore", "EXTERNAL_DATA.tsv", "environment.lock.yml"}:
            category, reason = "canonical", "Source release"
        else:
            category, reason = "review", "Manual classification required"
        rows.append({"path": rel, "size_bytes": path.stat().st_size, "category": category, "reason": reason})
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, sep="\t", index=False)
    return frame


def build_cleanup_manifest(repo: Path, output: Path) -> pd.DataFrame:
    protected = {"outputs/paper_results"}
    rows = []
    output_root = repo / "outputs"
    for path in sorted(output_root.iterdir()) if output_root.exists() else []:
        rel = path.relative_to(repo).as_posix()
        if rel in protected:
            continue
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else path.stat().st_size
        rows.append({"path": rel, "size_bytes": size, "reason": "Superseded intermediate or historical output", "release_replacement": "artifacts/results"})
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, sep="\t", index=False)
    return frame
