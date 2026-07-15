# Selective Prediction for Molecular Retrieval from MS/MS

Code and frozen artifacts for selective molecular-structure retrieval from tandem mass spectra. The project evaluates when fingerprint-based retrieval predictions should be accepted or deferred using risk-coverage curves and selective risk control.

Paper: [arXiv:2603.10950](https://arxiv.org/abs/2603.10950)

Artifacts: [10.5281/zenodo.19108280](https://doi.org/10.5281/zenodo.19108280)

## Repository Layout

```text
config/paper.yml                  Frozen paper settings and logical artifact IDs
ms_uq/                            Training, retrieval, uncertainty, and reporting code
scripts/run_paper_evaluation.py   Full evaluation, report-only reproduction, validation
scripts/prepare_release.py        Canonical result and release packaging
scripts/train.py                  Single-model training
scripts/train_ensemble.py         Ensemble launcher
scripts/run_evaluation.py         General evaluation entrypoint
scripts/run_sgr_evaluation.py     General selective-risk-control entrypoint
tests/                            Numerical and pipeline regression tests
```

Generated data, results, model artifacts, and release archives are deliberately excluded from Git.

## Installation

The released `source.zip` contains `environment.lock.yml`, which records the environment used for the paper artifacts.

```bash
conda env create -f environment.lock.yml
conda activate selective_msms
pip install -e ".[dev]"
pytest -q
```

CUDA is recommended for candidate rescoring and prediction generation. Report-only reproduction and validation run on CPU and do not require MassSpecGym files.

## Released Artifacts

The Zenodo deposit contains exactly seven files:

```text
README.md
MANIFEST.tsv
SHA256SUMS
source.zip
results.zip
predictions.zip
checkpoints.zip
```

Extract the archives from the root of this repository. They write only below `artifacts/`.

```bash
unzip source.zip
unzip results.zip
unzip predictions.zip
unzip checkpoints.zip
```

The model artifact groups are:

- `ensemble_mlp_formula`
- `ensemble_transformer_formula`
- `ensemble_mlp_mass`
- `mc_dropout_mlp_formula`
- `laplace_mlp_formula`

The result matrix covers official formula candidates, paired capped and uncapped formula candidates, and the available capped mass candidates. Candidate records are preserved without deduplication in the primary analysis, and `T_eval=0.003` is used throughout.

## Report-Only Reproduction

Recreate the numerical/figure bundle and static HTML index from `results.zip` alone:

```bash
python scripts/run_paper_evaluation.py report \
  --artifacts artifacts \
  --output-dir outputs/paper_results_reproduced

python scripts/run_paper_evaluation.py validate \
  --artifacts artifacts
```

Open `outputs/paper_results_reproduced/report/index.html` in a browser. This path requires no MassSpecGym data and performs no model inference.

## Full Paper Evaluation

Full candidate rescoring requires the released predictions/checkpoints plus the external files listed in `artifacts/source/EXTERNAL_DATA.tsv`. Put those external files in one local directory, verify their SHA-256 hashes, and run:

```bash
python scripts/run_paper_evaluation.py full \
  --data-dir /path/to/massspecgym-data \
  --artifacts artifacts \
  --out-dir outputs/paper_run \
  --device cuda:0
```

The full command is resumable. Use `--stages` for a subset and `--force-stage NAME` to invalidate one stage. It recomputes candidate scores and uncertainty features from frozen fingerprint predictions; it does not retrain the models or download data.

## Training

Single-model and ensemble training remain available independently of the frozen paper artifacts:

```bash
python scripts/train_ensemble.py \
  /path/to/MassSpecGym.tsv \
  /path/to/helpers \
  outputs/training \
  --method ensemble \
  --n_members 5 \
  --architecture mlp \
  --candidate_setting formula \
  --rankwise_loss bienc \
  --rankwise_temp 0.003 \
  --devices "[0,1,2,3,4]" \
  --max_parallel 5
```

Use `--candidate_setting mass --label_mode inchikey` with the matching mass helper files for mass-candidate training. See `--help` on each entrypoint for all runtime and architecture options.

## Preparing a Release

Maintainers can inventory, build, finalize, and verify the canonical release with:

```bash
python scripts/prepare_release.py plan

python scripts/prepare_release.py build \
  --source-run /path/to/completed-paper-run

python scripts/prepare_release.py finalize

python scripts/prepare_release.py verify
```

`prepare_release.py` does not download data or retrain models. It verifies all source hashes, materializes `outputs/paper_results`, creates deterministic ZIP64 archives, and rejects external MassSpecGym payloads in the deposit.
After committing source or DOI metadata, `finalize` rebuilds only the small `source.zip`, `README.md`, `MANIFEST.tsv`, and `SHA256SUMS`; it preserves the large scientific payload archives.

## License

See [LICENSE](LICENSE).
