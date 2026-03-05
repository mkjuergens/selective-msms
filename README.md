# Selective-MSMS
Code for **"When Should We Trust the Annotation? Selective Prediction for Molecular Structure Retrieval from Mass Spectra"**.

We introduce a selective prediction framework for molecular structure retrieval from tandem mass spectra (MS/MS), enabling models to abstain from predictions when uncertainty is too high.

All experiments are conducted on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark.

<!-- > **Paper**: [arXiv link TODO] · **Data**: [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) -->

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
├── ms_uq/                          
│   ├── core/                       
│   ├── models/                     
│   ├── inference/                  
│   ├── unc_measures/               
│   ├── evaluation/                 
│   ├── utils/                      
│   ├── data.py                     
│   └── loss.py                     
├── scripts/                        
│   ├── train.py                    # Train a single model
│   ├── train_ensemble.py           # Train a deep ensemble or mc dropout model
│   ├── make_predictions.py         # Generate predictions 
│   ├── run_evaluation.py           # Compute RC curves, AURC, correlation analysis (either with prediciotns or from scratch)
│   ├── run_sgr_evaluation.py       # Risk-controlled evaluation (SGR)
│   └── plot_sgr_analysis.py        # SGR coverage figure
├── config/                         
│   └── sgr.yml                     
└── tests/
```

## Reproducing Paper Results

Currently still under construction!

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


## Acknowledgements

The model architecture and training code are adapted from [ms-mole](https://github.com/gdewael/ms-mole) by De Waele et al.


This work builds on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark by Bushuiev et al.

## Citation

<!-- ```bibtex
@article{TODO,
  title={When Should We Trust the Annotation? Selective Prediction for Molecular Structure Retrieval from Mass Spectra},
  author={TODO},
  journal={Journal of Cheminformatics},
  year={2025}
}
``` -->

## License

MIT License. See [LICENSE](LICENSE) for details.