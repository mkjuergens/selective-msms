"""
Uncertainty measures for evaluation.

Provides a unified API for computing uncertainty measures with separate
configuration for fingerprint-level, retrieval-level, and distance-based evaluation.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
from torch import Tensor

from ms_uq.unc_measures.bitwise_unc import BitwiseUncertainty, compute_sparse_aware_epistemic
from ms_uq.unc_measures.retrieval_unc import RetrievalUncertainty
from ms_uq.unc_measures.distance_unc import DistanceUncertainty


FINGERPRINT_MEASURES = {
    "bitwise_epistemic",
    "bitwise_aleatoric", 
    "bitwise_total",
    "bitwise_epistemic_sparse",
    "bitwise_epistemic_active",
}

RETRIEVAL_MEASURES = {
    "retrieval_epistemic",
    "retrieval_aleatoric",
    "retrieval_total",
    "normalized_entropy",
    "rank_var_1",
    "rank_var_5",
    "rank_var_20",
    "confidence",
    "margin",
    "score_gap",
    "n_candidates",
    "ambiguity_ratio",
}

DISTANCE_MEASURES = {
    "knn_distance",
    "mahalanobis",
    "relative_mahalanobis",
    "centroid_distance",
}


def compute_fingerprint_uncertainties(
    Pbits: Tensor,
    measures: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Compute fingerprint-level uncertainty measures."""
    if Pbits.dim() == 2:
        Pbits = Pbits.unsqueeze(1)
    
    N, S, K = Pbits.shape
    results = {}
    
    if S <= 1:
        return results
    
    if measures is None:
        measures = list(FINGERPRINT_MEASURES)
    measures = [m for m in measures if m in FINGERPRINT_MEASURES]
    
    if any(m in measures for m in ["bitwise_epistemic", "bitwise_aleatoric", "bitwise_total"]):
        bitwise = BitwiseUncertainty(aggregate="sum", kind="entropy", weighting="none")
        if batch_size is not None and N > batch_size:
            parts = [
                bitwise.forward(Pbits[start:start + batch_size].float())
                for start in range(0, N, batch_size)
            ]
            unc = {
                name: torch.cat([part[name] for part in parts], dim=0)
                for name in parts[0]
            }
        else:
            unc = bitwise.forward(Pbits.float())
        
        if "bitwise_epistemic" in measures:
            results["bitwise_epistemic"] = unc["epistemic"].cpu().numpy()
        if "bitwise_aleatoric" in measures:
            results["bitwise_aleatoric"] = unc["aleatoric"].cpu().numpy()
        if "bitwise_total" in measures:
            results["bitwise_total"] = unc["total"].cpu().numpy()
    
    if "bitwise_epistemic_sparse" in measures:
        results["bitwise_epistemic_sparse"] = compute_sparse_aware_epistemic(
            Pbits, method="logit"
        ).cpu().numpy()
    
    if "bitwise_epistemic_active" in measures:
        results["bitwise_epistemic_active"] = compute_sparse_aware_epistemic(
            Pbits, method="active_bit"
        ).cpu().numpy()
    
    return results


def compute_retrieval_uncertainties(
    scores_stack: Tensor,
    ptr: Tensor,
    scores_agg: Optional[Tensor] = None,
    measures: Optional[List[str]] = None,
    temperature: float = 0.003,
    negate_confidence: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute retrieval-level uncertainty measures."""
    results = {}
    
    if scores_stack is None:
        return results
    if scores_stack.dim() == 1:
        scores_stack = scores_stack.unsqueeze(0)
    
    if measures is None:
        measures = list(RETRIEVAL_MEASURES)
    measures = [m for m in measures if m in RETRIEVAL_MEASURES]
    
    rank_k_values = []
    for m in measures:
        if m.startswith("rank_var_"):
            try:
                rank_k_values.append(int(m.split("_")[-1]))
            except ValueError:
                pass
    
    retrieval = RetrievalUncertainty(
        temperature=temperature,
        normalize_entropy=False,
        top_k_list=rank_k_values or [1, 5, 20],
    )
    unc = retrieval.forward(scores_stack, ptr, scores_agg)
    
    if "retrieval_epistemic" in measures:
        results["retrieval_epistemic"] = unc["entropy_epistemic"].cpu().numpy()
    if "retrieval_aleatoric" in measures:
        results["retrieval_aleatoric"] = unc["entropy_aleatoric"].cpu().numpy()
    if "retrieval_total" in measures:
        results["retrieval_total"] = unc["entropy_total"].cpu().numpy()
    if "normalized_entropy" in measures:
        results["normalized_entropy"] = unc["normalized_entropy"].cpu().numpy()
    
    for k in rank_k_values:
        key = f"rank_var_{k}"
        if key in measures and key in unc:
            results[key] = unc[key].cpu().numpy()
    
    sign = -1.0 if negate_confidence else 1.0
    if "confidence" in measures:
        results["confidence"] = (sign * unc["confidence_top1"]).cpu().numpy()
    if "margin" in measures:
        results["margin"] = (sign * unc["margin"]).cpu().numpy()
    if "score_gap" in measures:
        results["score_gap"] = (sign * unc["score_gap"]).cpu().numpy()
    
    # Candidate set metrics (higher = more uncertain/difficult)
    if "n_candidates" in measures:
        results["n_candidates"] = unc["n_candidates"].cpu().numpy()
    if "ambiguity_ratio" in measures:
        results["ambiguity_ratio"] = unc["ambiguity_ratio"].cpu().numpy()
    
    return results


def compute_distance_uncertainties(
    test_embeddings: Tensor,
    distance_model: DistanceUncertainty,
    measures: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute distance-based uncertainty measures.
    
    Parameters
    ----------
    test_embeddings : (N, D) tensor
        Embeddings for test samples (e.g., mean fingerprint predictions).
    distance_model : DistanceUncertainty
        Fitted distance uncertainty model.
    measures : list of str, optional
        Which measures to compute. If None, computes all.
    
    Returns
    -------
    dict mapping measure_name -> (N,) array
    """
    results = {}
    
    if distance_model is None or test_embeddings is None:
        return results
    
    if measures is None:
        measures = list(DISTANCE_MEASURES)
    measures = [m for m in measures if m in DISTANCE_MEASURES]
    
    if not measures:
        return results
    
    # Compute all distance measures
    unc = distance_model.forward(test_embeddings)
    
    for measure in measures:
        if measure in unc:
            results[measure] = unc[measure].cpu().numpy()
    
    return results


def fit_distance_model(
    train_embeddings: Tensor,
    n_neighbors: int = 10,
    metric: str = "cosine",
    normalize: bool = True,
    covariance: str = "shrinkage",
    knn_aggregation: str = "kth",
) -> DistanceUncertainty:
    """
    Fit a distance uncertainty model on training embeddings.
    
    Parameters
    ----------
    train_embeddings : (N_train, D) tensor
    n_neighbors : int
        Number of neighbors for k-NN.
    metric : str
        "cosine" or "euclidean".
    normalize : bool
        L2-normalize embeddings.
    covariance : str
        "shrinkage", "diagonal", or "full".
    knn_aggregation : str
        "kth" (k-th neighbor) or "mean".
    
    Returns
    -------
    Fitted DistanceUncertainty model.
    """
    model = DistanceUncertainty(
        n_neighbors=n_neighbors,
        metric=metric,
        normalize=normalize,
        covariance=covariance,
        knn_aggregation=knn_aggregation,
    )
    model.fit(train_embeddings)
    return model


def get_embeddings_from_pbits(Pbits: Tensor) -> Tensor:
    """
    Extract embeddings from fingerprint predictions.
    
    For ensemble predictions (N, S, K), returns mean across samples.
    For single predictions (N, K), returns as-is.
    """
    if Pbits.dim() == 3:
        return Pbits.mean(dim=1)
    return Pbits


def compute_uncertainties(
    Pbits: Optional[Tensor] = None,
    scores_stack: Optional[Tensor] = None,
    scores_agg: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    fingerprint_measures: Optional[List[str]] = None,
    retrieval_measures: Optional[List[str]] = None,
    distance_measures: Optional[List[str]] = None,
    distance_model: Optional[DistanceUncertainty] = None,
    test_embeddings: Optional[Tensor] = None,
    temperature: float = 0.003,
    negate_confidence: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute all uncertainties with separate config for fingerprint, retrieval, and distance.
    
    Parameters
    ----------
    Pbits : (N, S, K) or (N, K) tensor, optional
        Fingerprint predictions.
    scores_stack : (S, M) tensor, optional
        Per-member candidate scores.
    scores_agg : (M,) tensor, optional
        Aggregated candidate scores.
    ptr : (N+1,) tensor, optional
        Ragged array pointers.
    fingerprint_measures : list of str, optional
        Which fingerprint measures to compute.
    retrieval_measures : list of str, optional
        Which retrieval measures to compute.
    distance_measures : list of str, optional
        Which distance measures to compute.
    distance_model : DistanceUncertainty, optional
        Fitted distance model. Required if distance_measures is not None/empty.
    test_embeddings : (N, D) tensor, optional
        Pre-computed test embeddings for distance measures. If None and Pbits
        is provided, will fall back to using mean fingerprints as embeddings.
    temperature : float
        Softmax temperature for retrieval measures.
    negate_confidence : bool
        If True, negate confidence measures so higher = more uncertain.
    
    Returns
    -------
    dict mapping measure_name -> (N,) array
    """
    results = {}
    
    if Pbits is not None:
        results.update(compute_fingerprint_uncertainties(Pbits, fingerprint_measures))
    
    if scores_stack is not None and ptr is not None:
        results.update(compute_retrieval_uncertainties(
            scores_stack, ptr, scores_agg, retrieval_measures,
            temperature, negate_confidence
        ))
    
    if distance_measures and distance_model is not None:
        # Use pre-computed embeddings if available, else fall back to fingerprints
        if test_embeddings is not None:
            emb = test_embeddings
        elif Pbits is not None:
            emb = get_embeddings_from_pbits(Pbits)
        else:
            emb = None
        
        if emb is not None:
            results.update(compute_distance_uncertainties(
                emb, distance_model, distance_measures
            ))
    
    return results
