"""
Evaluation metrics for retrieval and uncertainty.

SINGLE SOURCE for:
- Hit@k computation
- Fingerprint losses (Tanimoto, cosine, Hamming)
- Score statistics
- AURC tables
- Correlation matrices
"""

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Dict, List, Optional
from scipy.stats import spearmanr

from ms_uq.core.entropy import entropy_normalized
from ms_uq.core.similarity import (
    cosine_pairwise,
    tanimoto_pairwise,
    hamming_distance,
)
from ms_uq.evaluation.rejection_curve import rejection_curve, aurc_from_curve, compute_aurc_with_baselines



def compute_fingerprint_losses(
    fp_pred: Tensor,
    fp_true: Tensor,
    threshold: float = 0.5,
    binarize: bool = False,          
) -> Dict[str, Tensor]:
    fp_pred = fp_pred.float()
    fp_true = fp_true.float()
    """
    Compute fingerprint-level losses between predictions and ground truth.
    
    Parameters
    ----------
    fp_pred : (N, K)
        Predicted fingerprints (probabilities in [0, 1]).
    fp_true : (N, K)
        Ground truth fingerprints (binary or probabilities).
    threshold : float
        Threshold for binarizing predictions (for Hamming loss).
    binarize : bool
        Whether to binarize predictions and ground truth before computing losses.
    
    Returns
    -------
    dict with losses (all shape (N,), lower = better):
        - tanimoto_loss: 1 - Tanimoto similarity
        - cosine_loss: 1 - cosine similarity
        - hamming_loss: fraction of differing bits
    """
    if binarize:                      # ← new: 2 lines
        fp_pred = (fp_pred > threshold).float()
    fp_pred = fp_pred.float()
    fp_true = fp_true.float()
    
    # Similarities → losses
    tanimoto_sim = tanimoto_pairwise(fp_pred, fp_true)
    cosine_sim = cosine_pairwise(fp_pred, fp_true)
    hamming = hamming_distance(fp_pred, fp_true, threshold=threshold)
    
    return {
        "tanimoto_loss": 1.0 - tanimoto_sim,
        "cosine_loss": 1.0 - cosine_sim,
        "hamming_loss": hamming,
        # Also return similarities for convenience
        "tanimoto_sim": tanimoto_sim,
        "cosine_sim": cosine_sim,
    }


def compute_all_losses(
    Pbits: Tensor,
    y_bits: Tensor,
    scores_agg: Tensor,
    labels_flat: Tensor,
    ptr: Tensor,
    top_k_hits: List[int] = [1, 5, 20],
    threshold: float = 0.5,
    binarize: bool = False,          # ← new
) -> Dict[str, np.ndarray]:
    """
    Compute all losses: hit@k and fingerprint losses.
    
    Parameters
    ----------
    Pbits : (N, S, K) or (N, K)
        Predicted fingerprint probabilities.
    y_bits : (N, K)
        Ground truth fingerprints.
    scores_agg : (M,)
        Aggregated retrieval scores.
    labels_flat : (M,)
        Candidate labels.
    ptr : (N+1,)
        Ragged pointers.
    top_k_hits : list
        k values for hit@k computation.
    threshold : float
        Binarization threshold for Hamming.
    
    Returns
    -------
    dict with all losses as numpy arrays (N,)
    """
    losses = {}
    
    # Hit@k losses (1 - hit rate)
    for k in top_k_hits:
        hits = hit_at_k_ragged(scores_agg, labels_flat, ptr, k=k)
        losses[f"hit@{k}"] = (1.0 - hits).numpy()
    
    # Fingerprint losses
    fp_pred = Pbits.mean(dim=1) if Pbits.dim() == 3 else Pbits
    fp_losses = compute_fingerprint_losses(fp_pred, y_bits, threshold=threshold, binarize=binarize)
    
    for name, val in fp_losses.items():
        if "loss" in name:  
            losses[name] = val.numpy()
    
    return losses


def hit_at_k_ragged(
    scores_flat: Tensor,
    labels_flat: Tensor,
    ptr: Tensor,
    k: int = 1,
    tie_break: str = "first"
) -> Tensor:
    """
    Hit@k for ragged candidate lists.
    
    Parameters
    ----------
    scores_flat : (total_candidates,) flattened scores
    labels_flat : (total_candidates,) binary labels (1 = correct)
    ptr : (N+1,) pointers into flat arrays
    k : number of top candidates to consider
    tie_break : "random" or "first"
    
    Returns
    -------
    (N,) binary hit indicators
    """
    N = ptr.numel() - 1
    hits = torch.zeros(N, dtype=torch.float32, device=scores_flat.device)
    
    for i in range(N):
        start, end = ptr[i].item(), ptr[i + 1].item()
        s = scores_flat[start:end]
        lab = labels_flat[start:end]
        
        if tie_break == "random":
            generator = torch.Generator(device=s.device).manual_seed(i)
            s = s + torch.rand(s.shape, generator=generator, device=s.device, dtype=s.dtype) * 1e-12
        elif tie_break != "first":
            raise ValueError("tie_break must be one of: first, random")

        n_cand = end - start
        k_actual = min(k, n_cand)
        topk_idx = torch.argsort(s, descending=True, stable=True)[:k_actual]
        hits[i] = lab[topk_idx].any().float()
    
    return hits


def compute_score_statistics(
    scores_flat: Tensor,
    ptr: Tensor,
    k: int = 5,
    temperature: float = 0.003
) -> Dict[str, Tensor]:
    """
    Compute score-based statistics per query.
    
    Returns
    -------
    dict with keys:
        - margin12: score difference between rank 1 and 2
        - entropy_rel: normalized entropy of softmax distribution
        - top1_prob: probability of top-1 candidate
        - topk_prob: cumulative probability of top-k candidates
    """
    N = ptr.numel() - 1
    device = scores_flat.device
    
    margin12 = torch.zeros(N, device=device)
    entropy_rel = torch.zeros(N, device=device)
    top1_prob = torch.zeros(N, device=device)
    topk_prob = torch.zeros(N, device=device)
    
    for i in range(N):
        start, end = ptr[i].item(), ptr[i + 1].item()
        s = scores_flat[start:end]
        n_cand = end - start
        
        # Softmax probabilities
        probs = torch.softmax(s / temperature, dim=0)
        
        # Sort descending
        sorted_scores, _ = s.sort(descending=True)
        sorted_probs, _ = probs.sort(descending=True)
        
        # Margin
        if n_cand >= 2:
            margin12[i] = sorted_scores[0] - sorted_scores[1]
        else:
            margin12[i] = float('inf')
        
        # Entropy (normalized)
        entropy_rel[i] = entropy_normalized(probs.unsqueeze(0), dim=-1).squeeze()
        
        # Top-k probabilities
        top1_prob[i] = sorted_probs[0]
        k_actual = min(k, n_cand)
        topk_prob[i] = sorted_probs[:k_actual].sum()
    
    return {
        "margin12": margin12,
        "entropy_rel": entropy_rel,
        "top1_prob": top1_prob,
        "topk_prob": topk_prob
    }

def compute_correlations(
    metrics: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Spearman correlation matrix: metrics vs targets."""
    metric_names = list(metrics.keys())
    target_names = list(targets.keys())
    
    corr_matrix = np.zeros((len(metric_names), len(target_names)))
    
    for i, m in enumerate(metric_names):
        for j, t in enumerate(target_names):
            rho, _ = spearmanr(metrics[m], targets[t])
            corr_matrix[i, j] = rho
    
    return pd.DataFrame(corr_matrix, index=metric_names, columns=target_names)


def compute_aurc_general(
    metrics: Dict[str, np.ndarray],
    losses: Dict[str, np.ndarray],
    include_oracle: bool = True,
    include_random: bool = True,
    confidence_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute AURC for all metrics vs all loss types (general version).
    
    Works with any loss type: hit@k (as hits, not losses), hamming_loss, 
    tanimoto_loss, cosine_loss, or any continuous loss.
    
    Parameters
    ----------
    metrics : dict
        Uncertainty/confidence metrics (name -> (N,) values).
    losses : dict
        Loss or performance values (name -> (N,) values):
        - hit@k: binary (1 = correct, 0 = wrong) - will be converted to loss
        - *_loss: continuous loss (higher = worse) - used directly
        - *_sim: continuous similarity (higher = better) - converted to loss
    include_oracle : bool
        Include oracle (perfect rejection) baseline.
    include_random : bool
        Include random baseline.
    confidence_metrics : list, optional
        Metrics where higher = more confident. Auto-detected if None.
    
    Returns
    -------
    pd.DataFrame
        AURC values (metrics as rows, losses as columns).
    """
    from ms_uq.utils import is_confidence_score
    
    if confidence_metrics is None:
        confidence_metrics = [m for m in metrics if is_confidence_score(m)]
    
    results = {}
    
    for loss_name, loss_vals in losses.items():
        results[loss_name] = {}
        
        # Convert to proper loss format
        if loss_name.startswith("hit"):
            # hit@k: 1 = correct → loss = 1 - hit
            loss_array = 1.0 - loss_vals.astype(np.float32)
        elif "_sim" in loss_name or "similarity" in loss_name.lower():
            # Similarity: higher = better → loss = 1 - sim
            loss_array = 1.0 - loss_vals.astype(np.float32)
        else:
            # Already a loss
            loss_array = loss_vals.astype(np.float32)
        
        loss_tensor = torch.tensor(loss_array, dtype=torch.float32)
        
        # Oracle: sort by loss (reject high loss first)
        if include_oracle:
            rej_pct, kept_loss = rejection_curve(loss_tensor, loss_tensor)
            results[loss_name]["oracle"] = aurc_from_curve(rej_pct, kept_loss)
        
        # Analytic expected random AURC equals the mean loss.
        if include_random:
            results[loss_name]["random"] = float(np.mean(loss_array, dtype=np.float64))
        
        # All metrics
        for metric_name, metric_vals in metrics.items():
            unc = metric_vals.copy()
            # Negate confidence scores so higher = more uncertain
            if metric_name in confidence_metrics:
                unc = -unc
            
            rej_pct, kept_loss = rejection_curve(
                loss_tensor,
                torch.tensor(unc, dtype=torch.float32)
            )
            results[loss_name][metric_name] = aurc_from_curve(rej_pct, kept_loss)
    
    return pd.DataFrame(results)


# Update compute_aurc_table to use the general version
def compute_aurc_table(
    metrics: Dict[str, np.ndarray],
    hit_rates: Optional[Dict[str, np.ndarray]] = None,
    losses: Optional[Dict[str, np.ndarray]] = None,
    include_oracle: bool = False,
    include_random: bool = False,
) -> pd.DataFrame:
    """
    Compute AURC for all metrics vs all hit rates or losses.
    
    Parameters
    ----------
    metrics : dict
        Uncertainty/confidence metrics.
    hit_rates : dict, optional
        Hit@k binary arrays (1 = correct). Use this OR losses.
    losses : dict, optional
        Loss values (name -> values). Use this OR hit_rates.
    include_oracle : bool
        Include oracle (perfect rejection) baseline.
    include_random : bool
        Include random baseline.
    
    Returns
    -------
    pd.DataFrame
        AURC values (metrics as index, hit rates/losses as columns).
    """
    if hit_rates is not None and losses is not None:
        # Merge both
        all_losses = {**hit_rates, **losses}
    elif hit_rates is not None:
        all_losses = hit_rates
    elif losses is not None:
        all_losses = losses
    else:
        raise ValueError("Must provide either hit_rates or losses")
    
    return compute_aurc_general(
        metrics, all_losses,
        include_oracle=include_oracle,
        include_random=include_random
    )


def compute_aurc_all_losses(
    metrics: Dict[str, np.ndarray],
    losses: Dict[str, np.ndarray],
    include_oracle: bool = True,
    include_random: bool = True,
) -> pd.DataFrame:
    """
    Compute AURC for all metrics vs all loss types.
    
    This is a more general version that works with any loss (not just hit@k).
    
    Parameters
    ----------
    metrics : dict
        Uncertainty/confidence metrics (name -> (N,) values).
    losses : dict
        Loss values (name -> (N,) values). Can include:
        - hit@k losses (1 - hit rate)
        - tanimoto_loss
        - cosine_loss
        - hamming_loss
    include_oracle : bool
        Include oracle (perfect rejection) baseline.
    include_random : bool
        Include random baseline.
    
    Returns
    -------
    pd.DataFrame
        AURC values (metrics as rows, losses as columns).
    """
    from ms_uq.utils import is_confidence_score
    
    results = {}
    
    for loss_name, loss_vals in losses.items():
        results[loss_name] = {}
        loss_tensor = torch.tensor(loss_vals, dtype=torch.float32)
        
        # Oracle: sort by loss (reject high loss first)
        if include_oracle:
            rej_pct, kept_loss = rejection_curve(loss_tensor, loss_tensor)
            results[loss_name]["oracle"] = aurc_from_curve(rej_pct, kept_loss)
        
        # Analytic expected random AURC equals the mean loss.
        if include_random:
            results[loss_name]["random"] = float(np.mean(loss_vals, dtype=np.float64))
        
        # All metrics
        for metric_name, metric_vals in metrics.items():
            unc = metric_vals.copy()
            # Negate confidence scores so higher = more uncertain
            if is_confidence_score(metric_name):
                unc = -unc
            
            rej_pct, kept_loss = rejection_curve(
                loss_tensor,
                torch.tensor(unc, dtype=torch.float32)
            )
            results[loss_name][metric_name] = aurc_from_curve(rej_pct, kept_loss)
    
    return pd.DataFrame(results)


def evaluate_uncertainty_vs_losses(
    Pbits: Tensor,
    y_bits: Tensor,
    scores_agg: Tensor,
    labels_flat: Tensor,
    ptr: Tensor,
    uncertainty_metrics: Dict[str, np.ndarray],
    top_k_hits: List[int] = [1, 5, 20],
    include_oracle: bool = True,
    include_random: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive evaluation of uncertainty metrics vs multiple loss types.
    
    Parameters
    ----------
    Pbits : (N, S, K) or (N, K)
        Predicted fingerprint probabilities.
    y_bits : (N, K)
        Ground truth fingerprints.
    scores_agg : (M,)
        Aggregated retrieval scores.
    labels_flat : (M,)
        Candidate labels.
    ptr : (N+1,)
        Ragged pointers.
    uncertainty_metrics : dict
        Uncertainty metrics to evaluate (name -> values).
    top_k_hits : list
        k values for hit@k.
    include_oracle : bool
        Include oracle baseline.
    include_random : bool
        Include random baseline.
    
    Returns
    -------
    dict with:
        - 'aurc': DataFrame with AURC for all metrics vs all losses
        - 'correlations': DataFrame with Spearman correlations
        - 'losses': dict of computed losses
        - 'loss_means': dict of mean loss values
    """
    # Compute all losses
    losses = compute_all_losses(
        Pbits, y_bits, scores_agg, labels_flat, ptr,
        top_k_hits=top_k_hits
    )
    
    # AURC table
    aurc_df = compute_aurc_all_losses(
        uncertainty_metrics, losses,
        include_oracle=include_oracle,
        include_random=include_random
    )
    
    # Correlations: uncertainty vs losses (higher loss = worse, so expect positive correlation)
    corr_df = compute_correlations(uncertainty_metrics, losses)
    
    # Mean losses
    loss_means = {name: vals.mean() for name, vals in losses.items()}
    
    return {
        "aurc": aurc_df,
        "correlations": corr_df,
        "losses": losses,
        "loss_means": loss_means,
    }