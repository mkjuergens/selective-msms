"""Canonical paper-result layout and deterministic release packaging."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


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


def load_paper_config(repo: Path) -> dict:
    return yaml.safe_load((repo / "config/paper.yml").read_text())


def zenodo_doi(config: Mapping[str, object]) -> str:
    paper = config.get("paper", {})
    doi = str(paper.get("zenodo_doi", "")).strip() if isinstance(paper, Mapping) else ""
    if not doi.startswith("10.5281/zenodo.") or not doi.removeprefix("10.5281/zenodo.").isdigit():
        raise ValueError("config/paper.yml must contain a reserved Zenodo DOI")
    return doi


def _git_source_commit(repo: Path, *, require_clean: bool = True) -> str:
    if require_clean:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, check=True,
            stdout=subprocess.PIPE, text=True,
        ).stdout.strip()
        if status:
            raise RuntimeError("Commit intentional source changes before finalizing the release")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout.strip()


def _indexed_file(data_dir: Path, relative_path: str, size: int, digest: str) -> Path:
    path = (data_dir / relative_path).resolve()
    if not path.is_relative_to(data_dir.resolve()):
        raise ValueError(f"Indexed path escapes data directory: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != size or sha256(path) != digest:
        raise RuntimeError(f"Indexed file failed verification: {path}")
    return path


def discover_checkpoint_sources(data_dir: Path) -> list[dict]:
    """Load and verify the released checkpoint/state files from data/models."""
    data_dir = data_dir.resolve()
    model_dir = data_dir / "models"

    checkpoint_frame = pd.read_csv(model_dir / "checkpoints.tsv", sep="\t")
    checkpoints = []
    for row in checkpoint_frame.itertuples(index=False):
        checkpoints.append({
            "model": row.model,
            "member": int(row.member),
            "path": _indexed_file(data_dir, row.path, int(row.size_bytes), row.sha256),
            "sha256": row.sha256,
        })
    if len(checkpoints) != 18:
        raise RuntimeError(f"Expected 18 checkpoint/state files, found {len(checkpoints)}")
    return checkpoints


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
            root = PurePosixPath(archive_path).parts[0] if PurePosixPath(archive_path).parts else ""
            if root not in {"data", "source"} or PurePosixPath(archive_path).is_absolute() or ".." in PurePosixPath(archive_path).parts:
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
    return [(path, f"source/{path.relative_to(repo).as_posix()}", "source", "", "") for path in sorted(set(paths))]


def _source_archive_members(repo: Path, extra: Sequence[Path], source_commit: str) -> list[tuple[Path | bytes, str, str, str, str]]:
    members = _tracked_source_members(repo, extra)
    members.append((
        f"{source_commit}\n".encode(),
        "source/SOURCE_COMMIT",
        "source_metadata", "", "",
    ))
    return members


def _release_readme(config: Mapping[str, object], source_commit: str) -> str:
    paper = config.get("paper", {})
    doi = zenodo_doi(config)
    return f"""# Selective MS/MS Paper Data

Data and models for [{paper.get('title', 'Selective prediction for MS/MS retrieval')}]({paper.get('url', 'https://arxiv.org/abs/2603.10950')}).

Zenodo DOI: [{doi}](https://doi.org/{doi})

Source commit: `{source_commit}`

## Pick What You Need

- `results.zip`: exact scores, tables, figures, and the browsable report.
- `checkpoints.zip`: 18 model checkpoint/state files for new inference.
- `source.zip`: the matching code, environment, and data notes.
- `MANIFEST.tsv` and `SHA256SUMS`: archive contents and checksums.

## Fastest Route

Extract `results.zip` in the repository root, then run:

```bash
python scripts/evaluate.py validate --data data
python scripts/evaluate.py report --data data --output outputs/report
```

That is enough to reproduce the released figures and tables. No MassSpecGym download or GPU is needed.

## Rerun Predictions

Extract `checkpoints.zip` and prepare the MassSpecGym v1 files described in the repository README:

```bash
python scripts/evaluate.py predict \
  --data data \
  --massspecgym-data /path/to/massspecgym-data \
  --device cuda:0
```

The official spectrum and candidate files come from [MassSpecGym](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/tree/main/data); fingerprint and InChIKey helpers can be generated with [`ms-mole`](https://github.com/gdewael/ms-mole#reproduction-steps). The uncapped and paper-specific mass extensions must match `EXTERNAL_DATA.tsv` in `source.zip`.

Predictions use about 18 GB locally. MC Dropout and Laplace samples may vary slightly across systems; the exact paper scores remain in `results.zip`.

## Included Results

The release covers the MLP and transformer ensembles, MC Dropout, Laplace, formula candidates, paired capped and uncapped formula candidates, mass candidates, temperature analysis, meta-models, and selective risk control. Primary results use `T_eval=0.003`.
"""


def _write_manifest(path: Path, members: Sequence[ReleaseMember]) -> None:
    archive_order = {name: index for index, name in enumerate([
        "source.zip", "results.zip", "checkpoints.zip",
    ])}
    ordered = sorted(members, key=lambda row: (archive_order.get(row.archive, 99), row.archive_path))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ReleaseMember.__dataclass_fields__), delimiter="\t")
        writer.writeheader()
        writer.writerows([member.__dict__ for member in ordered])


def _write_release_metadata(
    release_dir: Path,
    config: Mapping[str, object],
    source_commit: str,
    members: Sequence[ReleaseMember],
) -> None:
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "README.md").write_text(_release_readme(config, source_commit))
    _write_manifest(release_dir / "MANIFEST.tsv", members)
    checksummed = ["README.md", "MANIFEST.tsv", "source.zip", "results.zip", "checkpoints.zip"]
    (release_dir / "SHA256SUMS").write_text("".join(
        f"{sha256(release_dir / name)}  {name}\n" for name in checksummed
    ))


def build_release(repo: Path, data_dir: Path, release_dir: Path) -> dict:
    data_dir = data_dir.resolve()
    results_dir = data_dir / "results"
    config = load_paper_config(repo)
    result_report = validate_paper_results(results_dir)
    if not result_report["passed"]:
        raise RuntimeError("Refusing to package invalid paper results")
    checkpoints = discover_checkpoint_sources(data_dir)

    external = repo / "EXTERNAL_DATA.tsv"
    if not external.is_file():
        raise FileNotFoundError(external)
    source_commit = _git_source_commit(repo)
    members: list[ReleaseMember] = []
    members += _write_zip(
        release_dir / "source.zip",
        _source_archive_members(repo, [external], source_commit),
    )
    result_members = [
        (path, f"data/results/{path.relative_to(results_dir).as_posix()}", "result", "", "")
        for path in sorted(results_dir.rglob("*")) if path.is_file()
    ]
    members += _write_zip(release_dir / "results.zip", result_members)

    checkpoint_members = [
        (
            item["path"],
            f"data/{item['path'].relative_to(data_dir).as_posix()}",
            "checkpoint",
            item["model"],
            "",
        )
        for item in checkpoints
    ]
    checkpoint_members.append((
        data_dir / "models/checkpoints.tsv",
        "data/models/checkpoints.tsv",
        "metadata",
        "",
        "",
    ))
    members += _write_zip(release_dir / "checkpoints.zip", checkpoint_members)

    stale_predictions = release_dir / "predictions.zip"
    if stale_predictions.exists():
        stale_predictions.unlink()
    _write_release_metadata(release_dir, config, source_commit, members)
    return verify_release(release_dir)


def verify_release(release_dir: Path) -> dict:
    expected = {"README.md", "MANIFEST.tsv", "SHA256SUMS", "source.zip", "results.zip", "checkpoints.zip"}
    observed = {path.name for path in release_dir.iterdir() if path.is_file() and not path.name.startswith(".")}
    checks = [{"name": "exactly six release files", "passed": observed == expected, "observed": sorted(observed)}]
    checksum_ok = True
    for line in (release_dir / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        checksum_ok &= (sha256(release_dir / name) == digest)
    checks.append({"name": "companion checksums match", "passed": checksum_ok, "observed": "SHA256SUMS"})
    bad_members = []
    zip_errors = []
    for name in ["source.zip", "results.zip", "checkpoints.zip"]:
        with zipfile.ZipFile(release_dir / name) as archive:
            error = archive.testzip()
            if error:
                zip_errors.append(f"{name}:{error}")
            expected_root = "source" if name == "source.zip" else "data"
            for info in archive.infolist():
                parts = PurePosixPath(info.filename).parts
                is_link = ((info.external_attr >> 16) & 0o170000) == 0o120000
                if not parts or parts[0] != expected_root or ".." in parts or PurePosixPath(info.filename).is_absolute() or is_link:
                    bad_members.append(f"{name}:{info.filename}")
    checks.append({"name": "ZIP64 archives pass CRC checks", "passed": not zip_errors, "observed": zip_errors})
    checks.append({"name": "archive members use the source/ and data/ roots", "passed": not bad_members, "observed": bad_members})
    with zipfile.ZipFile(release_dir / "source.zip") as source_archive:
        source_names = set(source_archive.namelist())
        commit_name = "source/SOURCE_COMMIT"
        config_name = "source/config/paper.yml"
        source_commit = source_archive.read(commit_name).decode().strip() if commit_name in source_names else ""
        source_config = yaml.safe_load(source_archive.read(config_name)) if config_name in source_names else {}
    source_doi = zenodo_doi(source_config) if source_config else ""
    readme_text = (release_dir / "README.md").read_text()
    source_metadata_ok = (
        len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit)
        and bool(source_doi)
        and source_doi in readme_text
        and source_commit in readme_text
    )
    checks.append({
        "name": "source commit and Zenodo DOI are recorded consistently",
        "passed": source_metadata_ok,
        "observed": {"source_commit": source_commit, "zenodo_doi": source_doi},
    })
    manifest = pd.read_csv(release_dir / "MANIFEST.tsv", sep="\t")
    n_predictions = int((manifest.role == "prediction").sum())
    n_checkpoints = int((manifest.role == "checkpoint").sum())
    checks.append({"name": "manifest contains no predictions and 18 checkpoints", "passed": n_predictions == 0 and n_checkpoints == 18, "observed": {"predictions": n_predictions, "checkpoints": n_checkpoints}})
    forbidden = [path for path in manifest.archive_path if Path(path).name.startswith("MassSpecGym") or Path(path).name == "massspecgym_118m_mira.json"]
    checks.append({"name": "no MassSpecGym data payloads are included", "passed": not forbidden, "observed": forbidden})
    report = {"passed": all(row["passed"] for row in checks), "checks": checks}
    _write_json(release_dir.parent / f"{release_dir.name}_verification.json", report)
    if not report["passed"]:
        raise RuntimeError("Release verification failed")
    return report
