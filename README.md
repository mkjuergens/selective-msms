# Selective-MSMS
Code for **"When Should We Trust the Annotation? Selective Prediction for Molecular Structure Retrieval from Mass Spectra"**.

[Paper](https://arxiv.org/abs/2603.10950)

We introduce a selective prediction framework for molecular structure retrieval from tandem mass spectra (MS/MS), enabling models to abstain from predictions when uncertainty is too high.

All experiments are conducted on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark.

## Overview
<p align="center">
  <img src="docs/figures/figure_1.png" alt="Overview of the methodology" width="700"/>
</p>

<p align="center">
  <em></em>
</p>

## Installation

```bash
conda create -n selective-msms python=3.11
conda activate selective-msms

# Install MassSpecGym library
pip install massspecgym

# Install this package
git clone https://github.com/BioML-UGent/selective-msms.git
pip install -e ./selective-msms/
```
## Data Preparation

This codebase uses a custom dataset (`RetrievalDataset_PrecompFPandInchi` from [ms-mole](https://github.com/gdewael/ms-mole)) that requires precomputed Morgan fingerprints and InChI keys alongside the MassSpecGym data. The following files are expected in a `helper_dir/` directory:

| File | Description |
|------|-------------|
| `MassSpecGym.tsv` | MassSpecGym dataset (auto-downloaded by the library) |
| `fp_4096.npy` | Precomputed 4096-bit Morgan fingerprints for all dataset molecules |
| `inchis.npy` | InChI keys for all dataset molecules |
| `MassSpecGym_retrieval_candidates_formula.json` | Retrieval candidate lists (grouped by molecular formula) |
| `MassSpecGym_retrieval_candidates_formula_fps.npz` | Precomputed fingerprints for all candidates |
| `MassSpecGym_retrieval_candidates_formula_inchi.npz` | InChI keys for all candidates |
| `MassSpecGym_retrieval_candidates_formula_uncapped.*` | Optional uncapped formula-candidate helpers for revision analyses |
| `ground_truth_bits_labels_test.pt` | Ground-truth fingerprints and labels for the test set |

The candidate lists and dataset TSV are provided by MassSpecGym. Fingerprint files must be precomputed from the SMILES strings using RDKit Morgan fingerprints (radius 2, 4096 bits), following the same procedure as [ms-mole](https://github.com/gdewael/ms-mole).


## Repository Structure

```
selective-msms/
├── ms_uq/                          
│   ├── core/                       
│   ├── models/                     
│   ├── inference/                  
│   ├── unc_measures/               # contains all uncertainty measures and decompositions
│   ├── evaluation/                 # contains functions for evaluating selective prediction performance
│   ├── utils/                      
│   ├── data.py                     
│   └── loss.py                     
├── scripts/                        
│   ├── train.py                    
│   ├── train_ensemble.py           # wrapper to train a second-order model: ensemble, mc_dropout, single
│   ├── make_predictions.py         # generate predictions for the test set
│   ├── run_evaluation.py           # evaluation script producing visual and analytical results
│   ├── run_sgr_evaluation.py       # evaluation script for risk control analysis
│   └── plot_sgr_analysis.py        
├── config/                         
│   ├── eval.yml                    # template for risk-coverage analysis
│   ├── sgr.yml                     # template for selective-risk-control analysis
│   ├── eval_paper_mlp_formula.yml  # paper MLP/formula reproduction config
│   └── sgr_paper_mlp_formula.yml   # paper SGR reproduction config
└── tests/
```


## Pretrained Artifacts

We provide precomputed predictions and archived paper results for the models used
in the paper, so that the main evaluation tables and figures can be inspected or
reproduced without retraining the models.

**Download**: [Zenodo DOI TODO](https://zenodo.org/TODO)

<!-- Alternative: [HuggingFace Hub](https://huggingface.co/datasets/BioML-UGent/selective-msms-predictions) -->

The first artifact contains model predictions and generated paper outputs only.
It does not bundle the MassSpecGym data, helper files, or checkpoints. Prepare
those files separately as described in [Data Preparation](#data-preparation) if
you want to rerun the evaluation scripts.

After downloading, extract the archive into the repository root. The expected
layout is:

```text
MANIFEST.tsv
checksums.sha256
outputs/
├── predictions/
│   ├── ensemble_ranking/       # Deep Ensemble (S=5), ranking loss
│   │   ├── fp_probs.pt         # (N, 5, 4096) bitwise probabilities
│   │   └── ranker.pt           # learned biencoder scorer
│   ├── ensemble_focal/         # Deep Ensemble (S=5), focal loss
│   │   └── fp_probs.pt
│   ├── mcdo_ranking/           # MC Dropout (S=50), ranking loss
│   │   ├── fp_probs.pt
│   │   └── ranker.pt
│   ├── mcdo_focal/             # MC Dropout (S=50), focal loss
│   │   └── fp_probs.pt
│   ├── laplace_ranking/        # Laplace (S=50), ranking loss
│   │   ├── fp_probs.pt
│   │   └── ranker.pt
│   └── laplace_focal/          # Laplace (S=50), focal loss
│       └── fp_probs.pt
└── paper_results/              # archived CSVs, score tensors, and figures
```

Each `fp_probs.pt` is a dict `{"stack": tensor(N, S, 4096), "meta": {...}}` containing
per-sample bitwise fingerprint probabilities from the test set. Ranking-loss
models additionally include a `ranker.pt` learned biencoder scorer; without it,
scoring falls back to cosine similarity and the ranking-loss numbers will differ.

The files in `outputs/paper_results/` are the generated evaluation artifacts
from the paper runs. They can be inspected directly. To regenerate them from
`outputs/predictions/`, the shared data files listed in
[Data Preparation](#data-preparation), including `ground_truth_bits_labels_test.pt`,
must be available in `helper_dir`.

### Evaluate Without Training

After downloading and extracting the artifact:

```bash
# 1. Prepare MassSpecGym data and helper files separately.

# 2. Update dataset_tsv/helper_dir/gt_path in the config if needed.

# 3. Run risk-coverage/AURC evaluation from the precomputed predictions.
python scripts/run_evaluation.py --config config/eval_paper_mlp_formula.yml --group ensemble

# 4. Run SGR risk-control evaluation from the precomputed predictions.
python scripts/run_sgr_evaluation.py --config config/sgr_paper_mlp_formula.yml --group ensemble
```

The evaluation scripts detect `fp_probs.pt` in each `pred_dir` and skip
prediction generation, proceeding directly to candidate scoring and uncertainty
analysis.

## Reproducing Paper Results

The main retrieval baseline in the paper is the MLP Deep Ensemble trained with
the ranking/BiEnc loss and evaluated with formula-filtered candidates. It uses
score-level aggregation and the learned `ranker.pt`.

| Model | Candidate setting | Aggregation | Hit@1 | Hit@5 | Hit@20 |
|---|---|---|---:|---:|---:|
| Deep Ensemble (Ranking/BiEnc) | formula | score | 13.12 | 27.29 | 47.57 |
| Deep Ensemble (Focal) | formula | probability | 11.43 | 23.80 | 42.36 |

For the ranking/BiEnc ensemble, compare new revision experiments primarily
against:

```text
outputs/paper_results/eval_v6/ensemble/bienc/hit_rates_aggregate.csv
outputs/paper_results/eval_v6/ensemble/bienc/rel_aurc_retrieval_score.csv
```

If these archived result files are not present, regenerate them from the
precomputed predictions with `config/eval_paper_mlp_formula.yml`.



### Full pipeline (training from scratch)


### 1. Train a model (single, Deep Ensemble, MC Dropout)
To train a single model or an ensemble model using the architecture and the ranking loss function, run the following command. Needs to contain paths to massspecgym data.
```bash
python scripts/train_ensemble.py \
  /path/to/MassSpecGym.tsv \
  /path/to/helper_dir \
  outputs/logs \
  --method ensemble \
  --n_members 5 \
  --architecture mlp \
  --candidate_setting formula \
  --rankwise_loss bienc \
  --rankwise_temp 0.003 \
  --lr 0.0001 \
  --layer_dim 1024 \
  --bin_width 0.1 \
  --devices "[0,1]"
```

<!-- ### 2. Generate predictions

```bash
python scripts/make_predictions.py \
    --ens_dir <path>/logs/ensemble/ \
    --dataset_tsv <path>/MassSpecGym.tsv \
    --helper_dir <path>/helper/ \
    --device cuda:0
```

For Laplace approximation predictions:

```bash
python scripts/make_predictions.py \
    --mode laplace_bce \
    --ckpt <path>/best.ckpt \
    --dataset_tsv <path>/MassSpecGym.tsv \
    --helper_dir <path>/helper/ \
    --device cuda:0
``` -->

### 2. Evaluate (predictions + risk-coverage analysis)

The evaluation script handles prediction generation, candidate scoring, uncertainty computation, and plot generation in a single pipeline:

```bash
python scripts/run_evaluation.py --config config/eval.yml --group ensemble
```

This produces rejection curves, AURC bar charts, relAURC tables, and correlation heatmaps.

### 3. Risk-controlled evaluation (SGR)

```bash
python scripts/run_sgr_evaluation.py --config config/sgr.yml --group ensemble
```


This computes coverage at target risk levels with the SGR algorithm and generates calibration results.



### Revision Analyses

The revision experiments use the official MassSpecGym validation fold for learned
meta-scores and reserve the test fold for final relAURC/SGR reporting.

Run the reviewer-critical frozen-model revision matrix with one resumable command:

```bash
python scripts/run_revision_rerun.py \
  --data-dir /data/home/mira/data/msuq \
  --out-dir outputs/revision_rerun_v1 \
  --device cuda:0
```

The default `core` scope covers official formula MLP/transformer results, paired
capped-versus-uncapped formula evaluation for both architectures, formula-versus-
mass evaluation for the MLP, mass-trained-versus-formula-trained MLP comparison on
the same mass pool, temperature sensitivity, validation-only logistic meta-scores,
and paper-parity SGR with seed 42. It omits bootstrap resampling and secondary
sensitivity analyses so the complete run remains practical.

The runner does not download data or retrain an ensemble. It validates and reuses
existing fingerprint predictions, rebuilds candidate scores and derived metrics,
and records commands, hashes, package versions, seeds, and resolved paths in
`run_manifest.json`. Completed stages are reused only when their code/config/input
signature still matches. Use `--stages temperature,meta`, `--force-stage meta`, or
`--no-resume` for controlled reruns.

The previous maximal scope remains available explicitly:

```bash
python scripts/run_revision_rerun.py \
  --data-dir /data/home/mira/data/msuq \
  --out-dir outputs/revision_rerun_extended \
  --device cuda:0 \
  --analysis-scope extended \
  --bootstrap-replicates 2000 \
  --sgr-repeats 100 \
  --meta-ablation \
  --write-candidate-manifest
```

A plumbing-only smoke run is available with
`--max-queries 64 --bootstrap-replicates 5 --quick-hashes`; its numerical results
are not suitable for the manuscript.

```bash
# Dataset/candidate audit and capped fingerprint-vs-InChIKey label check
python scripts/audit_massspecgym.py \
  --dataset_tsv /path/to/MassSpecGym.tsv \
  --helper_dir /path/to/helper_dir \
  --candidate_setting formula \
  --out_dir outputs/revision_audit

# Temperature sensitivity from cached MLP and transformer score files
python scripts/run_temperature_sensitivity.py \
  --model mlp=/path/to/mlp/eval_dir \
  --model transformer=/path/to/transformer/eval_dir \
  --gt_path /path/to/ground_truth_bits_labels_test.pt \
  --out_dir outputs/revision_temperature

# Prepare validation/test bundles for a trained ensemble
python scripts/prepare_split_scores.py \
  --split val \
  --dataset_tsv /path/to/MassSpecGym.tsv \
  --helper_dir /path/to/helper_dir \
  --pred_dir outputs/revision_meta/model_val/pred \
  --out_dir outputs/revision_meta/model_val \
  --architecture mlp \
  --candidate_setting formula \
  --label_mode fingerprint \
  --ckpts /comma/separated/member/checkpoints \
  --device cuda:0

# Train logistic meta-scores on validation and evaluate frozen scores on test
python scripts/run_meta_score_analysis.py \
  --model_label mlp \
  --dataset_tsv /path/to/MassSpecGym.tsv \
  --helper_dir /path/to/helper_dir \
  --val_score outputs/revision_meta/model_val/scores_ranker_score.pt \
  --val_fp_probs outputs/revision_meta/model_val/pred/fp_probs.pt \
  --test_score /path/to/test/scores_ranker_score.pt \
  --test_fp_probs /path/to/test/pred/fp_probs.pt \
  --out_dir outputs/revision_meta/model
```

## Uncapped Formula Evaluation

The uncapped PubChem JSON is too large for the dense generic helper builder.
Prepare packed helpers for the official test fold with the resumable builder:

```bash
python scripts/prepare_uncapped_formula_test.py \
  --source_json /data/home/mira/data/msuq/massspecgym_118m_mira.json \
  --dataset_tsv /data/home/mira/data/msuq/MassSpecGym.tsv \
  --output_dir /data/home/mira/data/msuq \
  --workers 16
```

The script rewrites canonical query keys to the original TSV SMILES, preserves
candidate order, stores packed 4096-bit fingerprints, and labels candidates by
2D InChIKey. It intentionally builds test-only helpers; do not use
`candidate_setting: formula_uncapped` for training with these files.

After helper generation, evaluate both existing formula-trained ensembles and
then run fixed-split SGR:

```bash
python scripts/run_evaluation.py \
  --config config/eval_uncapped_formula.yml \
  --group ensemble

python scripts/run_sgr_evaluation.py \
  --config config/sgr_uncapped_formula.yml \
  --group ensemble
```

Both configs use `T_eval=0.003`, fresh score directories, and
`label_mode: inchikey_fallback`. The fingerprint fallback handles one audited
TSV/InChIKey inconsistency in the test fold.

## Scoring Functions

The framework compares the following scoring functions for selective prediction:

| Scoring function | Level | Order | Description |
|---|---|---|---|
| Confidence (max prob) | Retrieval | 1st | Maximum softmax probability over candidates |
| Score gap | Retrieval | 1st | Difference between top-1 and top-2 aggregated scores |
| Margin | Retrieval | 1st | Difference between top-1 and top-2 probabilities |
| Retrieval entropy (A/E/T) | Retrieval | 2nd | Entropy decomposition over candidate distributions |
| Normalized entropy | Retrieval | 2nd | Candidate entropy divided by `log |C|` |
| Rank variance | Retrieval | 2nd | Variance of candidate ranks across posterior samples |
| Bitwise entropy (A/E/T) | Fingerprint | 2nd | Entropy decomposition over predicted fingerprint bits |
| k-NN distance | Input | 1st | Deep k-nearest-neighbor distance |
| Mahalanobis distance | Input | 1st | Mahalanobis distance in encoder space |


## Device Usage

Training is GPU-oriented. By default, `scripts/train_ensemble.py` trains a
5-member ensemble sequentially on one CUDA device (`--devices "[0]"`). To train
ensemble members in parallel, pass multiple GPU indices, e.g.
`--devices "[0,1,2,3]"`; the launcher assigns one ensemble member process to
each selected GPU and runs at most `len(--devices)` members concurrently unless
`--max_parallel` is set.

This is not multi-GPU training of a single model. Each ensemble member is trained
as an independent process on one selected GPU.

Evaluation has two stages. Generating `fp_probs.pt` from checkpoints and
ranker-based candidate scoring can use CUDA via the YAML `device` field, e.g.
`device: cuda:0`. Once prediction and score files are saved, the downstream
risk-coverage analysis, AURC/relAURC tables, SGR summaries, and visualisations
can be run on CPU by setting `device: cpu`.

## Smoke Test

The following commands run a minimal end-to-end check of the training and
evaluation pipeline. This is intended only to verify that the code runs; the
resulting metrics are not meaningful.

```bash
# Train a tiny 2-member ensemble for one epoch and one batch
python scripts/train_ensemble.py \
  /path/to/MassSpecGym.tsv \
  /path/to/helper_dir \
  outputs/smoke_train \
  --method ensemble \
  --n_members 2 \
  --max_parallel 1 \
  --devices "[0]" \
  --accelerator gpu \
  --precision 32-true \
  --batch_size 4 \
  --bin_width 1.0 \
  --layer_dim 64 \
  --n_layers 2 \
  --bitwise_loss bce \
  --rankwise_loss bienc \
  --rankwise_temp 0.003 \
  --max_epochs 1 \
  --limit_train_batches 1 \
  --limit_val_batches 1 \
  --num_sanity_val_steps 0 \
  --skip_test=True \
  --tag smoke

```

## Acknowledgements

The model architecture and training code are adapted from [ms-mole](https://github.com/gdewael/ms-mole) by De Waele et al.


This work builds on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark by Bushuiev et al.

## Citation

```bibtex
@article{jurgens2026should,
  title={When should we trust the annotation? Selective prediction for molecular structure retrieval from mass spectra},
  author={J{\"u}rgens, Mira and De Waele, Gaetan and Rakhshaninejad, Morteza and Waegeman, Willem},
  journal={arXiv preprint arXiv:2603.10950},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.