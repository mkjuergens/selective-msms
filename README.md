# Selective-MSMS

Code and data for **"When Should We Trust the Annotation? Selective Prediction for Molecular Structure Retrieval from Mass Spectra"**.

We study when molecular-structure retrieval predictions from tandem mass spectra (MS/MS) should be accepted or deferred. The repository evaluates fingerprint-, retrieval-, and representation-level confidence scores using risk-coverage curves and selective risk control on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark.

- **Paper:** [arXiv:2603.10950](https://arxiv.org/abs/2603.10950)
- **Data and models:** [10.5281/zenodo.19108280](https://doi.org/10.5281/zenodo.19108280)

## Overview

<p align="center">
  <img src="docs/figures/figure_1.png" alt="Overview of the selective molecular-retrieval framework" width="700"/>
</p>

## Installation

The exact environment used for the paper is recorded in `environment.lock.yml`.

```bash
conda env create -f environment.lock.yml
conda activate selective_msms
pip install -e ".[dev]"
pytest -q
```

CUDA is needed only for prediction generation and candidate rescoring. Reproducing the released tables, figures, and report runs on CPU.

## Released Data

The Zenodo record separates the large files so that only the required parts need to be downloaded:

| File | Contents | Needed for |
|---|---|---|
| `results.zip` | Scores, metrics, tables, and figures | Report reproduction |
| `predictions.zip` | Seven frozen fingerprint-prediction tensors | Full rescoring |
| `checkpoints.zip` | The 18 model checkpoint/state files | Model provenance and new inference |
| `source.zip` | Exact source snapshot | Archival reference |

Extract the data archives in the repository root. They create `data/results/` and `data/models/`.

```bash
unzip results.zip
unzip predictions.zip
unzip checkpoints.zip
```

Large MassSpecGym files are not included. Their exact filenames, sources, sizes, hashes, and preparation notes are listed in [`EXTERNAL_DATA.tsv`](EXTERNAL_DATA.tsv).

## Reproduce Results

`results.zip` is sufficient to recreate the browsable report without downloading MassSpecGym:

```bash
python scripts/evaluate.py report --data data --output outputs/report
python scripts/evaluate.py validate --data data
```

Open `outputs/report/index.html` to browse the reproduced figures and tables.

For full candidate rescoring and analysis, extract the predictions and checkpoints and provide the external MassSpecGym directory:

```bash
python scripts/evaluate.py full \
  --data data \
  --massspecgym-data /path/to/massspecgym-data \
  --output outputs/evaluation \
  --device cuda:0
```

The full evaluation is resumable and does not retrain the models or download data. It uses score-level ensemble aggregation, cosine similarity, preserved candidate records, and `T_eval=0.003`.

## Training

Train the formula-candidate MLP ensemble used in the main experiments with:

```bash
python scripts/train_ensemble.py \
  /path/to/MassSpecGym.tsv \
  /path/to/candidate-files \
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

Use `--candidate_setting mass --label_mode inchikey` with the corresponding mass-candidate files. `scripts/train.py` trains one model; `scripts/train_ensemble.py` launches independent ensemble members.

## Repository Layout

```text
config/paper.yml                       Paper settings
ms_uq/                                 Models, inference, uncertainty, and evaluation
ms_uq/paper/                           Internal paper analyses and figures
scripts/evaluate.py                    Reproduce or rerun the paper evaluation
scripts/train.py                       Train one model
scripts/train_ensemble.py              Train an ensemble
scripts/prepare_uncapped_candidates.py Build the local uncapped formula helpers
tests/                                 Regression tests
```

Generated data, models, and outputs are excluded from Git.

## Acknowledgements

The model architecture, dataset pipeline, and ranking-loss implementation build on [ms-mole](https://github.com/gdewael/ms-mole) and its accompanying paper, ["Small molecule retrieval from tandem mass spectrometry: what are we optimizing for?"](https://arxiv.org/abs/2602.16507), by De Waele et al.

This work uses the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark by Bushuiev et al.

## Citation

```bibtex
@article{jurgens2026should,
  title   = {When Should We Trust the Annotation? Selective Prediction for Molecular Structure Retrieval from Mass Spectra},
  author  = {J{\"u}rgens, Mira and De Waele, Gaetan and Rakhshaninejad, Morteza and Waegeman, Willem},
  journal = {arXiv preprint arXiv:2603.10950},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
