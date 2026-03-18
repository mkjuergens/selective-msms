# Selective-MSMS
Code for **"When Should We Trust the Annotation? Selective Prediction for Molecular Structure Retrieval from Mass Spectra"**.

We introduce a selective prediction framework for molecular structure retrieval from tandem mass spectra (MS/MS), enabling models to abstain from predictions when uncertainty is too high.

All experiments are conducted on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark.

<!-- > **Paper**: [arXiv link TODO] · **Data**: [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) -->

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
│   └── sgr.yml                     
└── tests/
```

## Reproducing Paper Results

Currently still under construction!

### 1. Train a model (single, Deep Ensemble, MC Dropout)
To train a single model or an ensemble model using the architecture and the ranking loss function, run the following command. Needs to contain paths to massspecgym data.
```bash
python scripts/train_ensemble.py \
    --<path>/MassSpecGym.tsv \ # path to massspecgym tsv 
    --<path>/helper/ \ # directory with helper files
    --<path>/logs \ # directory where logs should be saved
    --method ensemble \
    --n_members 5 \
    --rankwise_loss bienc \
    --rankwise_temp 0.003 \
    --lr 0.0001 \
    --layer_dim 1024 \
    --bin_width 0.1 \
    --devices "[1,2]" \
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