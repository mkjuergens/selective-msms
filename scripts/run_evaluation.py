from __future__ import annotations
import argparse
import gc
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ms_uq.utils import make_test_loader, load_predictions, load_ground_truth, discover_ensemble_ckpts, make_train_val_test_loaders
from ms_uq.inference.retrieve import scores_from_loader
from ms_uq.inference import Predictor, MCDropoutSampler, EnsembleSampler, head_probs_fn, load_ranker, save_ranker_from_model
from ms_uq.models.fingerprint_mlp import FingerprintPredicter
from ms_uq.evaluation import (
    hit_at_k_ragged, 
    compute_aurc_table,
    compute_fingerprint_losses,
    plot_aurc_bars, 
    plot_risk_coverage_curves, 
    plot_rc_and_aurc_paired,
    plot_member_vs_agg,
    plot_correlation_heatmap
)
from ms_uq.unc_measures.eval_measures import (
    compute_uncertainties, FINGERPRINT_MEASURES, RETRIEVAL_MEASURES, DISTANCE_MEASURES,
    fit_distance_model, get_embeddings_from_pbits,
)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    dataset_tsv: str = ""
    helper_dir: str = ""
    gt_path: str = ""
    
    mode: str = "ensemble"
    ckpt: str = ""
    ckpts: str = ""
    ens_dir: str = ""
    ens_metric: str = "focal"
    passes: int = 50
    
    # Laplace
    laplace_samples: int = 50
    laplace_tau_w: float = 1.0
    laplace_tau_b: float = 1.0
    laplace_tune_prior: bool = True
    laplace_tune_method: str = "marglik"
    laplace_max_batches: Optional[int] = 200
    laplace_diagnostics: bool = True
    
    metric: str = "cosine"
    aggregations: List[str] = field(default_factory=lambda: ["score", "max_score_topk"])
    topk_k: int = 80
    topk_temp: float = 0.1
    temperature: float = 1.0
    top_k_hits: List[int] = field(default_factory=lambda: [1, 5, 20])
    weighted_method: str = "entropy"
    
    fingerprint_measures: Optional[List[str]] = None
    retrieval_measures: Optional[List[str]] = None
    
    distance_measures: Optional[List[str]] = None
    distance_n_neighbors: int = 10
    distance_metric: str = "cosine"
    distance_normalize: bool = True
    distance_covariance: str = "shrinkage"
    distance_knn_aggregation: str = "kth"
    distance_embedding_type: str = "fingerprint"
    
    device: str = "cuda:0"
    batch_size: int = 256
    num_workers: int = 2
    bin_width: float = 0.1
    overwrite: bool = False


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_predictions(out_dir: Path, config: EvalConfig, loader: Optional[DataLoader] = None,
                         train_loader: Optional[DataLoader] = None, 
                         val_loader: Optional[DataLoader] = None) -> Tuple[Path, Optional[Path]]:
    """Generate fingerprint predictions from checkpoints."""
    try:
        from massspecgym.models.base import Stage
        torch.serialization.add_safe_globals([Stage])
    except ImportError:
        pass
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_path = out_dir / "fp_probs.pt"
    ranker_path = out_dir / "ranker.pt"
    
    if fp_path.exists() and not config.overwrite:
        print(f"  Using cached {fp_path}")
        return fp_path, ranker_path if ranker_path.exists() else None
    
    if loader is None:
        loader = make_test_loader(config.dataset_tsv, config.helper_dir, config.bin_width,
                                   config.batch_size, config.num_workers)
    
    mode = config.mode.lower()
    if mode == "ensemble" and config.ckpt and not config.ckpts and not config.ens_dir:
        mode = "mcdo" if config.passes > 1 else "single"
    
    # Laplace: delegate entirely to laplace_bce module
    if mode == "laplace":
        from ms_uq.models.laplace_bce import generate_laplace_predictions, LaplaceConfig
        lp_cfg = LaplaceConfig(
            tau_w=config.laplace_tau_w,
            tau_b=config.laplace_tau_b,
            n_samples=config.laplace_samples,
            tune_prior=config.laplace_tune_prior,
            tune_method=config.laplace_tune_method,
            max_batches=config.laplace_max_batches,
            diagnostics=config.laplace_diagnostics,
        )
        return generate_laplace_predictions(
            out_dir=out_dir,
            ckpt=config.ckpt,
            test_loader=loader,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.device,
            overwrite=config.overwrite,
            cfg=lp_cfg,
            make_loaders_fn=lambda: make_train_val_test_loaders(
                config.dataset_tsv, config.helper_dir, config.bin_width,
                config.batch_size, config.num_workers
            ),
            save_ranker_fn=save_ranker_from_model,
        )
    
    ckpt_for_ranker = None
    if mode == "mcdo":
        print(f"  Mode: MC Dropout ({config.passes} passes)")
        sampler = MCDropoutSampler(Path(config.ckpt), FingerprintPredicter, passes=config.passes, device=config.device)
        ckpt_for_ranker = config.ckpt
    elif mode == "ensemble":
        if config.ckpts:
            ckpt_list = [Path(p.strip()) for p in config.ckpts.split(",") if p.strip()]
        elif config.ens_dir and config.ens_metric:
            ckpt_list = discover_ensemble_ckpts(config.ens_dir, config.ens_metric, prefer="best")
        else:
            raise ValueError("Ensemble requires --ckpts or (--ens_dir and --ens_metric)")
        print(f"  Mode: Ensemble ({len(ckpt_list)} members)")
        sampler = EnsembleSampler(ckpt_list, FingerprintPredicter, mc_dropout_eval=False, device=config.device)
        ckpt_for_ranker = ckpt_list[0]
    elif mode == "single":
        print(f"  Mode: Single model")
        sampler = EnsembleSampler([Path(config.ckpt)], FingerprintPredicter, mc_dropout_eval=False, device=config.device)
        ckpt_for_ranker = config.ckpt
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if ckpt_for_ranker:
        try:
            model = FingerprintPredicter.load_from_checkpoint(ckpt_for_ranker, map_location=config.device)
            save_ranker_from_model(model, ranker_path)
            del model; _cleanup()
        except Exception:
            pass
    
    predictor = Predictor(sampler, head_probs_fn("loss.fp_pred_head", torch.sigmoid))
    predictor.predict_stack(loader, fp_path, save_every=100, overwrite=config.overwrite)
    del sampler, predictor; _cleanup()
    
    print(f"  Saved: {fp_path}")
    return fp_path, ranker_path if ranker_path.exists() else None


def _fit_distance_model_for_eval(pred_dir: Path, config: EvalConfig,
                                  test_loader, train_loader=None):
    """Fit distance model on training embeddings for distance-based uncertainty."""
    from ms_uq.unc_measures.distance_unc import DistanceUncertainty, extract_embeddings_from_loader
    
    pred_dir = Path(pred_dir)
    emb_type = config.distance_embedding_type
    suffix = f"_{emb_type}" if emb_type != "encoder" else ""
    distance_model_path = pred_dir / f"distance_model{suffix}.pt"
    train_emb_path = pred_dir / f"train_embeddings{suffix}.pt"
    
    if distance_model_path.exists() and not config.overwrite:
        print(f"  Loading cached distance model from {distance_model_path}")
        return DistanceUncertainty.load(distance_model_path)
    
    ckpt_path = _get_checkpoint_path(config)
    if ckpt_path is None:
        print("  Warning: No checkpoint available for distance model, skipping")
        return None
    
    if train_emb_path.exists() and not config.overwrite:
        print(f"  Loading cached training embeddings from {train_emb_path}")
        train_embeddings = torch.load(train_emb_path, map_location="cpu")
    else:
        print(f"  Extracting training {emb_type} embeddings...")
        if train_loader is None:
            train_loader, _, _ = make_train_val_test_loaders(
                config.dataset_tsv, config.helper_dir, config.bin_width,
                config.batch_size, config.num_workers
            )
        train_embeddings = extract_embeddings_from_loader(
            ckpt_path, train_loader, device=config.device, 
            show_progress=True, embedding_type=emb_type
        )
        torch.save(train_embeddings, train_emb_path)
    
    print("  Fitting distance model...")
    distance_model = fit_distance_model(
        train_embeddings,
        n_neighbors=config.distance_n_neighbors,
        metric=config.distance_metric,
        normalize=config.distance_normalize,
        covariance=config.distance_covariance,
        knn_aggregation=config.distance_knn_aggregation,
    )
    distance_model.save(distance_model_path)
    return distance_model


def _get_checkpoint_path(config: EvalConfig) -> Optional[Path]:
    """Get a single checkpoint path from config."""
    if config.ckpt:
        return Path(config.ckpt)
    if config.ckpts:
        first = config.ckpts.split(",")[0].strip()
        return Path(first) if first else None
    if config.ens_dir and config.ens_metric:
        ckpts = discover_ensemble_ckpts(config.ens_dir, config.ens_metric, prefer="best")
        return ckpts[0] if ckpts else None
    return None


def _extract_test_embeddings(pred_dir: Path, config: EvalConfig, test_loader) -> Optional[torch.Tensor]:
    """Extract test embeddings for distance-based uncertainty."""
    from ms_uq.unc_measures.distance_unc import extract_embeddings_from_loader
    
    emb_type = config.distance_embedding_type
    suffix = f"_{emb_type}" if emb_type != "encoder" else ""
    test_emb_path = Path(pred_dir) / f"test_embeddings{suffix}.pt"
    
    if test_emb_path.exists() and not config.overwrite:
        return torch.load(test_emb_path, map_location="cpu")
    
    ckpt_path = _get_checkpoint_path(config)
    if ckpt_path is None:
        return None
    
    print(f"  Extracting test {emb_type} embeddings...")
    test_embeddings = extract_embeddings_from_loader(
        ckpt_path, test_loader, device=config.device, 
        show_progress=True, embedding_type=emb_type
    )
    torch.save(test_embeddings, test_emb_path)
    return test_embeddings


def compute_retrieval_metrics(scores_flat: torch.Tensor, labels_flat: torch.Tensor,
                               ptr: torch.Tensor, top_k_list: List[int]) -> Tuple[Dict, Dict]:
    """Compute hit@k metrics and corresponding losses."""
    hits, losses = {}, {}
    for k in top_k_list:
        h = hit_at_k_ragged(scores_flat, labels_flat, ptr, k=k)
        hits[f"hit@{k}"] = h.numpy()
        losses[f"hit@{k}"] = (1 - h).numpy()
    return hits, losses


def evaluate_aggregation(Pbits: torch.Tensor, scores_path: Path, y_bits: Optional[torch.Tensor],
                          labels_flat: torch.Tensor, config: EvalConfig,
                          distance_model=None, test_embeddings=None) -> Dict:
    """Evaluate a single aggregation method."""
    data = torch.load(scores_path, map_location="cpu")
    scores_flat, ptr = data["scores_flat"], data["ptr"]
    scores_stack = data.get("scores_stack_flat")
    
    hits, retrieval_losses = compute_retrieval_metrics(scores_flat, labels_flat, ptr, config.top_k_hits)
    
    fp_losses = {}
    if y_bits is not None:
        fp_losses_tensors = compute_fingerprint_losses(
            Pbits.mean(dim=1) if Pbits.dim() == 3 else Pbits, y_bits,
            binarize=True,                   # new: binarize for evaluation
        )
        fp_losses = {k: v.numpy() for k, v in fp_losses_tensors.items() if "loss" in k}
    
    ret_measures = list(config.retrieval_measures or RETRIEVAL_MEASURES)
    for k in config.top_k_hits:
        rk = f"rank_var_{k}"
        if rk not in ret_measures:
            ret_measures.append(rk)
    
    uncertainties = compute_uncertainties(
        Pbits=Pbits, scores_stack=scores_stack, scores_agg=scores_flat, ptr=ptr,
        fingerprint_measures=config.fingerprint_measures, retrieval_measures=ret_measures,
        distance_measures=config.distance_measures, distance_model=distance_model,
        test_embeddings=test_embeddings, temperature=config.temperature, negate_confidence=False,
    )
    
    retrieval_aurc = compute_aurc_table(
        uncertainties, hit_rates=hits, include_oracle=True, include_random=True
    )
    
    fp_aurc = None
    if fp_losses:
        fp_aurc = compute_aurc_table(
            uncertainties, losses=fp_losses, include_oracle=True, include_random=True
        )
    
    return {
        "hits": hits, "retrieval_losses": retrieval_losses, "fp_losses": fp_losses,
        "uncertainties": uncertainties, "retrieval_aurc": retrieval_aurc, "fp_aurc": fp_aurc,
    }


def evaluate_members(scores_stack: torch.Tensor, labels_flat: torch.Tensor,
                      ptr: torch.Tensor, top_k_hits: List[int]) -> Dict[str, np.ndarray]:
    """Evaluate individual ensemble members.
    
    Returns
    -------
    dict
        Per-instance hit arrays. Keys are 'hit@{k}', values are (S, N) arrays
        where S is the number of members and N is the number of test instances.
        Row s corresponds to member/sample s in the same order as scores_stack.
    """
    if scores_stack is None or scores_stack.dim() != 2:
        return {}
    S = scores_stack.shape[0]
    member_hits = {f"hit@{k}": [] for k in top_k_hits}
    for s in range(S):
        for k in top_k_hits:
            h = hit_at_k_ragged(scores_stack[s], labels_flat, ptr, k=k)
            member_hits[f"hit@{k}"].append(h.numpy())
    # Stack to (S, N) arrays
    return {k: np.stack(v, axis=0) for k, v in member_hits.items()}


def _get_checkpoint_names(config: EvalConfig) -> List[str]:
    """Extract ordered checkpoint names from config for provenance tracking."""
    if config.ckpts:
        return [Path(p.strip()).name for p in config.ckpts.split(",") if p.strip()]
    if config.ens_dir and config.ens_metric:
        ckpts = discover_ensemble_ckpts(config.ens_dir, config.ens_metric, prefer="best")
        return [p.name for p in ckpts]
    if config.ckpt:
        return [Path(config.ckpt).name]
    return []


def _save_hit_rates(agg_results: Dict, member_hits: Dict, config: EvalConfig, out_dir: Path):
    """Save aggregate and per-member hit rates to disk.
    
    Saves:
      - hit_rates_aggregate.csv: one row per aggregation method, columns are Hit@K
      - hit_rates_members.npz: per-instance arrays (S, N) keyed by 'hit@{k}',
        plus 'checkpoint_names' for provenance
      - hit_rates_members_summary.csv: mean ± std per member with checkpoint names
    """
    # --- Aggregate hit rates ---
    rows = []
    for agg, results in agg_results.items():
        row = {"aggregation": agg}
        for key, arr in results["hits"].items():
            row[key] = arr.mean()
        rows.append(row)
    df_agg = pd.DataFrame(rows).set_index("aggregation")
    df_agg.to_csv(out_dir / "hit_rates_aggregate.csv", float_format="%.6f")
    
    if not member_hits:
        return
    
    # --- Per-instance member hit rates (S, N) arrays ---
    ckpt_names = _get_checkpoint_names(config)
    save_dict = dict(member_hits)  # copy: keys are 'hit@{k}', values are (S, N) arrays
    if ckpt_names:
        save_dict["checkpoint_names"] = np.array(ckpt_names, dtype=object)
    np.savez(out_dir / "hit_rates_members.npz", **save_dict)
    
    # --- Summary table: one row per member ---
    S = next(iter(member_hits.values())).shape[0]
    rows = []
    for s in range(S):
        row = {"member": s}
        if s < len(ckpt_names):
            row["checkpoint"] = ckpt_names[s]
        for key, arr in member_hits.items():
            row[key] = arr[s].mean()
        rows.append(row)
    df_members = pd.DataFrame(rows)
    
    # Append mean ± std row
    summary_row = {"member": "mean±std"}
    if ckpt_names:
        summary_row["checkpoint"] = ""
    for key in member_hits:
        means = np.array([r[key] for r in rows])
        summary_row[key] = f"{means.mean():.6f}±{means.std():.6f}"
    df_members = pd.concat([df_members, pd.DataFrame([summary_row])], ignore_index=True)
    df_members.to_csv(out_dir / "hit_rates_members_summary.csv", index=False)
    print(f"  Saved hit rates to {out_dir}")


def _compute_rel_aurc(aurc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute relative AURC normalised between oracle (0) and random (1).
    
    relAURC(κ) = (AURC(κ) - AURC_oracle) / (AURC_random - AURC_oracle)
    
    Rows 'oracle' and 'random' are used as references and excluded from output.
    """
    if "oracle" not in aurc_df.index or "random" not in aurc_df.index:
        return pd.DataFrame()
    
    oracle = aurc_df.loc["oracle"]
    random = aurc_df.loc["random"]
    denom = random - oracle
    # Avoid division by zero
    denom = denom.replace(0, np.nan)
    
    scoring_rows = [idx for idx in aurc_df.index if idx not in ("oracle", "random")]
    rel = (aurc_df.loc[scoring_rows] - oracle) / denom
    return rel


def _save_uncertainties(uncertainties: Dict[str, np.ndarray], out_dir: Path, agg: str):
    """Cache uncertainty values to disk for later reuse (e.g. SGR evaluation)."""
    unc_path = out_dir / f"uncertainties_{agg}.npz"
    np.savez(unc_path, **uncertainties)


def run_evaluation(pred_dir: Path, out_dir: Path, config: EvalConfig,
                   gt_path: Optional[Path] = None, loader=None, train_loader=None, 
                   val_loader=None, label: str = ""):
    """Run full evaluation pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}\nEvaluating: {label or pred_dir.name}\n{'='*60}")
    
    if loader is None:
        loader = make_test_loader(config.dataset_tsv, config.helper_dir, config.bin_width,
                                   config.batch_size, config.num_workers)
    
    fp_path = pred_dir / "fp_probs.pt"
    if not fp_path.exists() or config.overwrite:
        print("\n[predict] Generating predictions...")
        fp_path, ranker_path = generate_predictions(
            pred_dir, config, loader=loader, 
            train_loader=train_loader, val_loader=val_loader
        )
    else:
        ranker_path = pred_dir / "ranker.pt"
    
    ranker = None
    if ranker_path and Path(ranker_path).exists():
        ranker = load_ranker(ranker_path, device=config.device)
        if ranker:
            print(f"[ranker] Loaded from {ranker_path}")
    
    print("\n[load] Loading data...")
    Pbits, _, _, _ = load_predictions(pred_dir, metric=config.metric, aggregation="score", require_scores=False)
    if Pbits is None:
        raise FileNotFoundError(f"No fp_probs.pt in {pred_dir}")
    y_bits, labels_flat = load_ground_truth(gt_path or Path(config.gt_path))
    
    # Distance-based uncertainty
    distance_model, test_embeddings = None, None
    if config.distance_measures:
        print("\n[distance] Setting up distance-based uncertainty...")
        distance_model = _fit_distance_model_for_eval(pred_dir, config, loader, train_loader)
        if distance_model is not None:
            test_embeddings = _extract_test_embeddings(pred_dir, config, loader)
    
    # Compute scores per aggregation
    print("\n[scores] Computing scores...")
    scores_paths = {}
    for agg in config.aggregations:
        fname = f"scores_{'ranker' if ranker else config.metric}_{agg}.pt"
        scores_file = out_dir / fname
        if scores_file.exists() and not config.overwrite:
            print(f"  {agg}: cached")
        else:
            print(f"  {agg}: computing...")
            result = scores_from_loader(
                Pbits, loader, metric=config.metric, aggregation=agg,
                temperature=config.temperature, topk_k=config.topk_k, topk_temp=config.topk_temp,
                weighted_method=config.weighted_method, return_labels=True, return_per_sample=True,
                ranker=ranker, device=config.device,
            )
            torch.save(result, scores_file)
        scores_paths[agg] = scores_file
    
    # Evaluate
    print("\n[eval] Evaluating aggregations...")
    agg_results = {}
    for agg, path in scores_paths.items():
        print(f"  {agg}...")
        agg_results[agg] = evaluate_aggregation(
            Pbits, path, y_bits, labels_flat, config, 
            distance_model=distance_model, test_embeddings=test_embeddings
        )
    
    print("\n[eval] Evaluating members...")
    member_hits = {}
    if "score" in scores_paths:
        data = torch.load(scores_paths["score"], map_location="cpu")
        if data.get("scores_stack_flat") is not None:
            member_hits = evaluate_members(data["scores_stack_flat"], labels_flat, data["ptr"], config.top_k_hits)
    
    print("\n[plot] Generating plots...")
    _generate_plots(agg_results, member_hits, config, out_dir, label)
    
    print("\n[save] Saving hit rates and uncertainties...")
    _save_hit_rates(agg_results, member_hits, config, out_dir)
    for agg, results in agg_results.items():
        _save_uncertainties(results["uncertainties"], out_dir, agg)
    
    _print_summary(agg_results, member_hits, config, out_dir)
    
    return agg_results, member_hits


def _generate_plots(agg_results: Dict, member_hits: Dict, config: EvalConfig, out_dir: Path, label: str):
    """Generate all evaluation plots."""
    for agg, results in agg_results.items():
        available = list(results["uncertainties"].keys())
        ret_plot = list(config.retrieval_measures or available)
        if config.distance_measures:
            ret_plot = [m for m in set(ret_plot) | set(config.distance_measures) if m in available]
        fp_plot = list(config.fingerprint_measures or available)
        
        if not results["retrieval_aurc"].empty:
            plot_aurc_bars(results["retrieval_aurc"], [f"hit@{k}" for k in config.top_k_hits],
                          out_dir / f"aurc_retrieval_{agg}.pdf", measures=ret_plot)
        
        if results["fp_aurc"] is not None and not results["fp_aurc"].empty:
            plot_aurc_bars(results["fp_aurc"], ["cosine_loss", "tanimoto_loss", "hamming_loss"],
                          out_dir / f"aurc_fingerprint_{agg}.png", measures=fp_plot)
        
        plot_risk_coverage_curves(results["retrieval_losses"], results["uncertainties"],
                                   out_dir / f"risk_coverage_retrieval_{agg}.pdf",
                                   loss_cols=[f"hit@{k}" for k in config.top_k_hits], measures=ret_plot)
        
        # Combined paired figure (risk-coverage + AURC bars aligned)
        if not results["retrieval_aurc"].empty:
            plot_rc_and_aurc_paired(
                results["retrieval_losses"], results["uncertainties"],
                results["retrieval_aurc"],
                out_dir / f"rc_aurc_paired_retrieval_{agg}.pdf",
                loss_cols=[f"hit@{k}" for k in config.top_k_hits],
                measures=ret_plot,
            )
        
        if results["fp_losses"]:
            plot_risk_coverage_curves(results["fp_losses"], results["uncertainties"],
                                       out_dir / f"risk_coverage_fingerprint_{agg}.png",
                                       loss_cols=["cosine_loss", "tanimoto_loss", "hamming_loss"], measures=fp_plot)
            if results["fp_aurc"] is not None and not results["fp_aurc"].empty:
                plot_rc_and_aurc_paired(
                    results["fp_losses"], results["uncertainties"],
                    results["fp_aurc"],
                    out_dir / f"rc_aurc_paired_fingerprint_{agg}.pdf",
                    loss_cols=["cosine_loss", "tanimoto_loss", "hamming_loss"],
                    measures=fp_plot,
                )
        
        results["retrieval_aurc"].to_csv(out_dir / f"aurc_retrieval_{agg}.csv")
        rel_aurc = _compute_rel_aurc(results["retrieval_aurc"])
        if not rel_aurc.empty:
            rel_aurc.to_csv(out_dir / f"rel_aurc_retrieval_{agg}.csv", float_format="%.6f")
        if results["fp_aurc"] is not None:
            results["fp_aurc"].to_csv(out_dir / f"aurc_fingerprint_{agg}.csv")
            rel_fp = _compute_rel_aurc(results["fp_aurc"])
            if not rel_fp.empty:
                rel_fp.to_csv(out_dir / f"rel_aurc_fingerprint_{agg}.csv", float_format="%.6f")

        # plot heatmap of uncertainty correlations
        unc = results["uncertainties"]
        if len(unc) >= 2:
            corr_df = pd.DataFrame(unc).corr(method="spearman")
            plot_correlation_heatmap(
                corr_df,
                out_dir / f"correlation_heatmap_{agg}.pdf",
            )
    
    if member_hits:
        # Convert (S, N) per-instance arrays to (S,) per-member means for plotting
        member_means = {k: v.mean(axis=1) for k, v in member_hits.items()}
        agg_hits = {agg: {k: r["hits"][k].mean() for k in r["hits"]} for agg, r in agg_results.items()}
        plot_member_vs_agg(member_means, agg_hits, out_dir / "member_vs_aggregate.pdf")


def _print_summary(agg_results: Dict, member_hits: Dict, config: EvalConfig, out_dir: Path):
    """Print evaluation summary."""
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for agg, results in agg_results.items():
        print(f"\n{agg}:")
        for k in config.top_k_hits:
            key = f"hit@{k}"
            if key in results["hits"]:
                print(f"  Hit@{k}: {results['hits'][key].mean():.4f}")
        # Print relAURC if available
        rel_aurc = _compute_rel_aurc(results["retrieval_aurc"])
        if not rel_aurc.empty and "hit@1" in rel_aurc.columns:
            best_idx = rel_aurc["hit@1"].idxmin()
            print(f"  Best κ (relAURC Hit@1): {best_idx} = {rel_aurc.loc[best_idx, 'hit@1']:.4f}")
    if member_hits:
        print(f"\nMembers (S={next(iter(member_hits.values())).shape[0]}):")
        ckpt_names = _get_checkpoint_names(config)
        for k in config.top_k_hits:
            key = f"hit@{k}"
            if key in member_hits:
                per_member_means = member_hits[key].mean(axis=1)  # (S,)
                print(f"  Hit@{k}: {per_member_means.mean():.4f} ± {per_member_means.std():.4f}")
        if ckpt_names:
            for s, name in enumerate(ckpt_names):
                vals = "  ".join(f"Hit@{k}: {member_hits[f'hit@{k}'][s].mean():.4f}" for k in config.top_k_hits)
                print(f"    [{s}] {name}  {vals}")
    print(f"\nResults saved to {out_dir}")


def _config_from_dicts(common: dict, model_cfg: dict, group_cfg: dict) -> EvalConfig:
    """Build EvalConfig by layering: common defaults < group overrides < model overrides.
    
    Only keys that are valid EvalConfig fields are passed through.
    """
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(EvalConfig)}
    merged = {}
    for d in [common, group_cfg, model_cfg]:
        merged.update({k: v for k, v in d.items() if k in valid_fields and v is not None})
    # Remove sub-dicts that aren't EvalConfig fields (e.g. 'models', 'out_subdir')
    return EvalConfig(**{k: v for k, v in merged.items() if k in valid_fields})


def run_from_config(config_path: Path, group: str):
    """Run evaluation from YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    common = cfg["common"]
    if group not in cfg["model_groups"]:
        raise ValueError(f"Group '{group}' not found. Available: {list(cfg['model_groups'].keys())}")
    
    group_cfg = cfg["model_groups"][group]
    base_out = Path(common["base_out_dir"]) / group_cfg.get("out_subdir", group)
    
    loader = make_test_loader(common["dataset_tsv"], common["helper_dir"],
                               common.get("bin_width", 0.1),
                               common.get("batch_size", 256),
                               common.get("num_workers", 2))
    
    for model_name, model_cfg in group_cfg["models"].items():
        if model_cfg is None:
            model_cfg = {}
        
        model_config = _config_from_dicts(common, model_cfg, group_cfg)
        
        run_evaluation(
            Path(model_cfg["pred_dir"]), base_out / model_name, model_config,
            gt_path=Path(common["gt_path"]) if common.get("gt_path") else None,
            loader=loader, label=model_cfg.get("label", model_name),
        )


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluation pipeline for molecular retrieval with UQ.")
    ap.add_argument("--config", help="YAML config file")
    ap.add_argument("--group", help="Model group to evaluate")
    ap.add_argument("--pred_dir", help="Prediction directory")
    ap.add_argument("--out_dir", help="Output directory")
    ap.add_argument("--gt_path", help="Ground truth path")
    ap.add_argument("--dataset_tsv", help="Dataset TSV path")
    ap.add_argument("--helper_dir", help="Helper directory")
    ap.add_argument("--label", default="")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "tanimoto"])
    ap.add_argument("--aggregations", default="score,max_score_topk")
    ap.add_argument("--topk_k", type=int, default=80)
    ap.add_argument("--topk_temp", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k_hits", default="1,5,20")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--bin_width", type=float, default=0.1)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.config:
        if not args.group:
            raise ValueError("--group required with --config")
        run_from_config(Path(args.config), args.group)
    else:
        required = [args.pred_dir, args.out_dir, args.gt_path, args.dataset_tsv, args.helper_dir]
        if not all(required):
            raise ValueError("Direct mode requires: --pred_dir, --out_dir, --gt_path, --dataset_tsv, --helper_dir")
        config = EvalConfig(
            dataset_tsv=args.dataset_tsv, helper_dir=args.helper_dir, gt_path=args.gt_path,
            metric=args.metric,
            aggregations=[a.strip() for a in args.aggregations.split(",")],
            topk_k=args.topk_k, topk_temp=args.topk_temp, temperature=args.temperature,
            top_k_hits=[int(k) for k in args.top_k_hits.split(",")],
            batch_size=args.batch_size, num_workers=args.num_workers, bin_width=args.bin_width,
        )
        run_evaluation(Path(args.pred_dir), Path(args.out_dir), config,
                       gt_path=Path(args.gt_path), label=args.label)
    print("\n✓ Evaluation complete\n")


if __name__ == "__main__":
    main()