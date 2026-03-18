from __future__ import annotations
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import numpy as np
import pandas as pd

from ms_uq.utils import load_predictions, load_ground_truth, make_test_loader, is_confidence_score
from ms_uq.inference.retrieve import scores_from_loader
from ms_uq.inference import load_ranker
from ms_uq.evaluation import hit_at_k_ragged, compute_fingerprint_losses
from ms_uq.evaluation.selective_risk import fit_sgr
from ms_uq.evaluation.visualisation import plot_sgr_coverage_combined, plot_sgr_risk_calibration
from ms_uq.unc_measures.eval_measures import compute_uncertainties, RETRIEVAL_MEASURES, FINGERPRINT_MEASURES


@dataclass
class SGRConfig:
    """SGR evaluation configuration."""
    pred_dir: str = ""
    gt_path: str = ""
    dataset_tsv: str = ""
    helper_dir: str = ""
    delta: float = 0.005

    # Ranker support (for ranking-loss models)
    ranker_path: str = ""  # path to ranker.pt; if empty, looks in pred_dir

    retrieval_losses: List[str] = field(default_factory=lambda: ["hit@1", "hit@5", "hit@20"])
    retrieval_target_risks: List[float] = field(default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    retrieval_measures: Optional[List[str]] = None

    fingerprint_losses: List[str] = field(default_factory=lambda: ["tanimoto", "cosine", "hamming"])
    fingerprint_target_risks: Dict[str, List[float]] = field(default_factory=lambda: {
        "tanimoto": [0.60, 0.70, 0.80, 0.90],
        "cosine": [0.60, 0.70, 0.80, 0.90],
        "hamming": [0.30, 0.35, 0.40, 0.45],
    })
    fingerprint_measures: Optional[List[str]] = None

    metric: str = "cosine"
    aggregation: str = "score"
    temperature: float = 1.0
    device: str = "cuda:0"
    batch_size: int = 256
    num_workers: int = 2
    bin_width: float = 0.1
    overwrite: bool = False


def _find_ranker(pred_dir: Path, config: SGRConfig) -> Optional[torch.nn.Module]:
    """Load ranker from explicit path or pred_dir/ranker.pt."""
    ranker_candidates = []
    if config.ranker_path:
        ranker_candidates.append(Path(config.ranker_path))
    ranker_candidates.append(pred_dir / "ranker.pt")

    for rp in ranker_candidates:
        if rp.exists():
            ranker = load_ranker(rp, device=config.device)
            if ranker is not None:
                print(f"  [ranker] Loaded from {rp}")
                return ranker
    return None


def load_data(pred_dir: Path, out_dir: Path, config: SGRConfig, loader=None):
    """Load predictions and compute retrieval scores.

    Score file naming follows run_evaluation.py convention:
      scores_{ranker|metric}_{aggregation}.pt
    This ensures that if run_evaluation.py already computed scores (including
    with a ranker for ranking-loss models), they are reused.
    """
    Pbits, _, _, _ = load_predictions(
        pred_dir, metric=config.metric, aggregation=config.aggregation, require_scores=False
    )
    if Pbits is None:
        raise FileNotFoundError(f"No fp_probs.pt in {pred_dir}")

    # Try loading ranker (needed for ranking-loss models)
    ranker = _find_ranker(pred_dir, config)
    score_prefix = "ranker" if ranker else config.metric
    scores_file = out_dir / f"scores_{score_prefix}_{config.aggregation}.pt"

    # Also check if run_evaluation.py cached scores in a different location
    if not scores_file.exists() and not config.overwrite:
        # Check pred_dir and common parent directories
        for alt_dir in [pred_dir, pred_dir.parent]:
            alt = alt_dir / scores_file.name
            if alt.exists():
                scores_file = alt
                break

    if scores_file.exists() and not config.overwrite:
        print(f"  Loading cached scores: {scores_file}")
        data = torch.load(scores_file, map_location="cpu")
    else:
        print(f"  Computing scores (ranker={'yes' if ranker else 'no'})...")
        if loader is None:
            loader = make_test_loader(config.dataset_tsv, config.helper_dir, config.bin_width,
                                      config.batch_size, config.num_workers)
        data = scores_from_loader(Pbits, loader, metric=config.metric, aggregation=config.aggregation,
                                  temperature=config.temperature, return_labels=True,
                                  return_per_sample=True, ranker=ranker, device=config.device)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, scores_file)

    return Pbits, data["scores_flat"], data.get("scores_stack_flat"), data["ptr"]


def compute_losses(Pbits, scores_flat, ptr, labels_flat, y_bits, config: SGRConfig) -> Dict[str, np.ndarray]:
    """Compute all losses."""
    losses = {}

    for loss in config.retrieval_losses:
        if loss.startswith("hit@"):
            k = int(loss.split("@")[1])
            hits = hit_at_k_ragged(scores_flat, labels_flat, ptr, k=k)
            losses[loss] = (1 - hits).numpy()

    if y_bits is not None:
        fp_pred = Pbits.mean(dim=1) if Pbits.dim() == 3 else Pbits
        fp_losses = compute_fingerprint_losses(fp_pred, y_bits)
        for name in config.fingerprint_losses:
            key = f"{name}_loss" if not name.endswith("_loss") else name
            if key in fp_losses:
                losses[name] = fp_losses[key].numpy()

    return losses


def compute_all_uncertainties(Pbits, scores_stack, scores_flat, ptr, config: SGRConfig):
    """Compute uncertainty measures."""
    ret_measures = config.retrieval_measures or list(RETRIEVAL_MEASURES)
    for loss in config.retrieval_losses:
        if loss.startswith("hit@"):
            k = int(loss.split("@")[1])
            if f"rank_var_{k}" not in ret_measures:
                ret_measures.append(f"rank_var_{k}")

    fp_measures = config.fingerprint_measures or list(FINGERPRINT_MEASURES)

    return compute_uncertainties(
        Pbits=Pbits, scores_stack=scores_stack, scores_agg=scores_flat, ptr=ptr,
        fingerprint_measures=fp_measures, retrieval_measures=ret_measures,
        temperature=config.temperature, negate_confidence=False,
    ), ret_measures, fp_measures


def run_sgr_for_losses(losses: Dict, uncertainties: Dict, measures: List[str],
                       target_risks_map: Dict, delta: float, binary_loss: bool = True) -> Dict:
    """
    Run SGR for a set of losses. Returns results dict compatible with plotting functions.

    Parameters
    ----------
    losses : dict
        Mapping loss_name -> loss array.
    uncertainties : dict
        Mapping measure_name -> uncertainty scores.
    measures : list
        Which measures to evaluate.
    target_risks_map : dict
        Mapping loss_name -> list of target risk values.
    delta : float
        Confidence parameter for SGR.
    binary_loss : bool
        If True, use binomial bounds (for hit@k). If False, use Hoeffding (for fingerprint).

    Returns
    -------
    dict
        Results in format expected by plotting functions.
    """
    results = {}

    for loss_name, loss_vals in losses.items():
        target_risks = target_risks_map.get(loss_name, target_risks_map.get("default", [0.1, 0.2, 0.3]))
        results[loss_name] = {"sgr": {}, "aurcs": {}, "base_error": float(loss_vals.mean()), "target_risks": target_risks}

        for measure in measures:
            if measure not in uncertainties:
                continue

            unc = uncertainties[measure]
            is_conf = is_confidence_score(measure)
            conf = unc if is_conf else -unc

            # AURC
            idx = np.argsort(-conf)
            cumsum = np.cumsum(loss_vals[idx])
            counts = np.arange(1, len(loss_vals) + 1)
            risks, coverages = cumsum / counts, counts / len(loss_vals)
            results[loss_name]["aurcs"][measure] = float(np.trapezoid(risks, coverages) if hasattr(np, 'trapezoid') else np.trapz(risks, coverages))

            # SGR
            results[loss_name]["sgr"][measure] = {}
            for r_star in target_risks:
                results[loss_name]["sgr"][measure][r_star] = fit_sgr(
                    conf, loss_vals, r_star, delta, higher_is_confident=True, binary_loss=binary_loss
                )

    return results


def run_sgr_evaluation(pred_dir: Path, out_dir: Path, config: SGRConfig,
                       gt_path: Optional[Path] = None, loader=None, label: str = "") -> Dict:
    """Run complete SGR evaluation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nSGR Evaluation: {label or pred_dir.name}\n{'='*60}")

    # Load data
    Pbits, scores_flat, scores_stack, ptr = load_data(Path(pred_dir), out_dir, config, loader)
    y_bits, labels_flat = load_ground_truth(gt_path or Path(config.gt_path))

    # Compute
    all_losses = compute_losses(Pbits, scores_flat, ptr, labels_flat, y_bits, config)
    uncertainties, ret_measures, fp_measures = compute_all_uncertainties(Pbits, scores_stack, scores_flat, ptr, config)

    retrieval_losses = {k: v for k, v in all_losses.items() if k.startswith("hit@")}
    fingerprint_losses = {k: v for k, v in all_losses.items() if k in config.fingerprint_losses}

    print(f"  Samples: {len(labels_flat)}")
    print(f"  Retrieval: {list(retrieval_losses.keys())} | Fingerprint: {list(fingerprint_losses.keys())}")

    # Run SGR
    retrieval_risks = {loss: config.retrieval_target_risks for loss in retrieval_losses}
    retrieval_results = run_sgr_for_losses(retrieval_losses, uncertainties, ret_measures, retrieval_risks, config.delta, binary_loss=True)
    fingerprint_results = run_sgr_for_losses(fingerprint_losses, uncertainties, fp_measures, config.fingerprint_target_risks, config.delta, binary_loss=False)

    # Print summary
    for name, results in [("Retrieval", retrieval_results), ("Fingerprint", fingerprint_results)]:
        if results:
            print(f"\n  {name}:")
            for loss, data in results.items():
                top = sorted(data["aurcs"].items(), key=lambda x: x[1])[:3]
                print(f"    {loss} (base={data['base_error']:.1%}): best={top[0][0]} AURC={top[0][1]:.4f}")

    # Plot
    if retrieval_results:
        plot_sgr_coverage_combined(retrieval_results, "", out_dir / "sgr_retrieval_coverage.pdf", sharey=True)
        plot_sgr_coverage_combined(retrieval_results, "", out_dir / "sgr_retrieval_coverage.png", sharey=True)
        plot_sgr_risk_calibration(retrieval_results, "", out_dir / "sgr_retrieval_calibration.png", sharey=True)
        plot_sgr_risk_calibration(retrieval_results, "", out_dir / "sgr_retrieval_calibration.pdf", sharey=True)

    if fingerprint_results:
        plot_sgr_coverage_combined(fingerprint_results, "", out_dir / "sgr_fingerprint_coverage.pdf", sharey=False)
        plot_sgr_coverage_combined(fingerprint_results, "", out_dir / "sgr_fingerprint_coverage.png", sharey=False)
        plot_sgr_risk_calibration(fingerprint_results, "", out_dir / "sgr_fingerprint_calibration.png", sharey=False)

    # Save CSV
    rows = []
    for cat, res in [("retrieval", retrieval_results), ("fingerprint", fingerprint_results)]:
        for loss, data in res.items():
            for measure, sgr_dict in data["sgr"].items():
                for r_star, r in sgr_dict.items():
                    rows.append({"category": cat, "loss": loss, "measure": measure, "target_risk": r_star,
                                 "coverage": r.coverage, "empirical_risk": r.empirical_risk, "feasible": r.feasible,
                                 "aurc": data["aurcs"].get(measure, np.nan)})
    pd.DataFrame(rows).to_csv(out_dir / "sgr_results.csv", index=False)

    print(f"\nSaved to {out_dir}")
    return {"retrieval": retrieval_results, "fingerprint": fingerprint_results}


def run_from_config(config_path: Path, group: str):
    """Run from YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    common, sgr_cfg = cfg["common"], cfg.get("sgr", {})
    group_cfg = cfg["model_groups"][group]
    base_out = Path(common["base_out_dir"]) / group_cfg.get("out_subdir", group)

    loader = make_test_loader(common["dataset_tsv"], common["helper_dir"],
                              common.get("bin_width", 0.1), common.get("batch_size", 256), common.get("num_workers", 2))

    for model_name, model_cfg in group_cfg["models"].items():
        config = SGRConfig(
            pred_dir=model_cfg["pred_dir"],
            gt_path=common.get("gt_path", ""),
            dataset_tsv=common["dataset_tsv"],
            helper_dir=common["helper_dir"],
            delta=sgr_cfg.get("delta", 0.005),
            ranker_path=model_cfg.get("ranker_path", ""),
            retrieval_losses=sgr_cfg.get("retrieval_losses", ["hit@1", "hit@5", "hit@20"]),
            retrieval_target_risks=sgr_cfg.get("retrieval_target_risks", [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]),
            retrieval_measures=sgr_cfg.get("retrieval_measures"),
            fingerprint_losses=sgr_cfg.get("fingerprint_losses", ["tanimoto", "cosine", "hamming"]),
            fingerprint_target_risks=sgr_cfg.get("fingerprint_target_risks", {}),
            fingerprint_measures=sgr_cfg.get("fingerprint_measures"),
            metric=common.get("metric", "cosine"),
            aggregation=model_cfg.get("aggregation", common.get("aggregation", "score")),
            temperature=common.get("temperature", 1.0),
            device=common.get("device", "cuda:0"),
            batch_size=common.get("batch_size", 256),
            num_workers=common.get("num_workers", 2),
            overwrite=model_cfg.get("overwrite", False),
        )

        run_sgr_evaluation(Path(model_cfg["pred_dir"]), base_out / model_name / "sgr", config,
                          gt_path=Path(common["gt_path"]) if common.get("gt_path") else None,
                          loader=loader, label=model_cfg.get("label", model_name))


def main():
    ap = argparse.ArgumentParser(description="SGR evaluation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--group", required=True)
    args = ap.parse_args()
    run_from_config(Path(args.config), args.group)
    print("\n✓ Done\n")


if __name__ == "__main__":
    main()
