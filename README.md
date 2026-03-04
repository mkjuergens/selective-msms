# Selective-MSMS
Code for **"When Should We Trust the Annotation? Selective Prediction for Molecular Structure Retrieval from Mass Spectra"**.

We introduce a selective prediction framework for molecular structure retrieval from tandem mass spectra (MS/MS), enabling models to abstain from predictions when uncertainty is too high. The framework evaluates uncertainty quantification strategies at two levels of granularity — fingerprint-level and retrieval-level — and applies distribution-free risk control to obtain subsets of annotations with provable error guarantees.

All experiments are conducted on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark.

> **Paper**: [arXiv link TODO] · **Data**: [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym)

## Installation

```bash
conda create -n selective-msms python=3.11
conda activate selective-msms

# Install MassSpecGym (dependency)
pip install massspecgym

# Install this package
git clone https://github.com/BioML-UGent/selective-msms.git
pip install -e ./selective-msms/
```

## Repository Structure

```
selective-msms/
├── ms_uq/                          # Python package
│   ├── core/                       # Entropy, similarity functions
│   ├── models/                     # MLP architecture, Laplace approximation
│   ├── inference/                  # Ensemble/MC Dropout sampling, retrieval scoring
│   ├── unc_measures/               # Uncertainty measures (bitwise, retrieval, distance)
│   ├── evaluation/                 # Metrics, rejection curves, SGR, visualisation
│   ├── utils/                      # Data loading, checkpoint utilities
│   ├── data.py                     # Dataset with precomputed fingerprints
│   └── loss.py                     # Fingerprint prediction losses
├── scripts/                        # Entry-point scripts
│   ├── train.py                    # Train a single model
│   ├── train_ensemble.py           # Train a deep ensemble
│   ├── make_predictions.py         # Generate predictions (ensemble/MC dropout/Laplace)
│   ├── run_evaluation.py           # Compute RC curves, AURC, correlation analysis
│   ├── run_sgr_evaluation.py       # Risk-controlled evaluation (SGR)
│   └── plot_sgr_analysis.py        # SGR coverage figure
├── config/                         # YAML configuration files
│   ├── eval.yml                    # Main evaluation config
│   └── sgr.yml                     # SGR risk control config
└── tests/
```

## Reproducing Paper Results

The pipeline has four stages. Each stage reads from the output of the previous one.

### 1. Train ensemble

```bash
python scripts/train_ensemble.py \
    --members 5 \
    --dataset_tsv <path>/MassSpecGym.tsv \
    --helper_dir <path>/helper/ \
    --output_dir <path>/logs/ensemble/
```

### 2. Generate predictions

```bash
python scripts/make_predictions.py \
    --mode ensemble \
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
```

### 3. Evaluate (risk-coverage analysis)

```bash
python scripts/run_evaluation.py --config config/eval.yml
```

This produces rejection curves, AURC bar charts, and correlation heatmaps.

### 4. Risk-controlled evaluation (SGR)

```bash
python scripts/run_sgr_evaluation.py --config config/sgr.yml
```

This computes coverage at target risk levels with the SGR algorithm and generates calibration results.

## Scoring Functions

The framework evaluates scoring functions from three families:

| Family | Measures | Level |
|--------|----------|-------|
| First-order retrieval | Confidence (κ_conf), Score gap (κ_gap) | Retrieval |
| Second-order uncertainty | Aleatoric (κ_ret^al), Epistemic (κ_ret^ep), Rank variance (κ_rank) | Retrieval |
| Bitwise uncertainty | Bitwise aleatoric, epistemic, total | Fingerprint |
| Distance-based | k-NN distance, Mahalanobis distance | Embedding |

## Acknowledgements

The model architecture and training code are adapted from [ms-mole](https://github.com/gdewael/ms-mole) by De Waele et al. We thank the authors for making their code publicly available.

This work builds on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark by Bushuiev et al.

## Citation

```bibtex
@article{TODO,
  title={When Should We Trust the Annotation? Selective Prediction for Molecular Structure Retrieval from Mass Spectra},
  author={TODO},
  journal={Journal of Cheminformatics},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.