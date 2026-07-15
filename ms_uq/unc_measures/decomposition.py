"""
Unified uncertainty decomposition primitives.

Provides both entropy-based and variance-based decompositions with
consistent API. Used by bitwise_unc.py and retrieval_unc.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import torch
from torch import Tensor

from ms_uq.core.entropy import binary_entropy, categorical_entropy

_EPS = 1e-12


@dataclass
class UncertaintyDecomposition:
    """
    Container for aleatoric/epistemic decomposition.
    
    Both entropy-based and variance-based measures are computed
    for maximum flexibility in downstream use.
    """
    # Entropy-based (information-theoretic)
    total_entropy: Tensor
    aleatoric_entropy: Tensor
    epistemic_entropy: Tensor
    
    # Variance-based (for propagation via delta method)
    total_variance: Tensor
    aleatoric_variance: Tensor
    epistemic_variance: Tensor
    
    # Mean prediction
    mean: Tensor
    
    def get(self, kind: Literal["entropy", "variance"] = "entropy") -> dict:
        """Get decomposition as dict with standard keys."""
        if kind == "entropy":
            return {
                "total": self.total_entropy,
                "aleatoric": self.aleatoric_entropy,
                "epistemic": self.epistemic_entropy,
            }
        else:
            return {
                "total": self.total_variance,
                "aleatoric": self.aleatoric_variance,
                "epistemic": self.epistemic_variance,
            }


def decompose_binary(
    P: Tensor,
    sample_dim: int = 1,
    reduce_dim: Optional[int] = None,
    reduce_method: Literal["mean", "sum"] = "mean",
) -> UncertaintyDecomposition:
    """
    Decompose uncertainty for binary (Bernoulli) predictions.
    
    Parameters
    ----------
    P : (..., S, ...) 
        Probabilities from S samples. sample_dim indicates which dim is samples.
    sample_dim : int
        Which dimension contains the samples. (e.g. from MC Dopout, ensemble)
    reduce_dim : int or None
        If set, reduce across this dimension (e.g., bits) after decomposition.
    reduce_method : str
        'mean' or 'sum' for reduction.
    
    Returns
    -------
    UncertaintyDecomposition with all measures.
    
    Notes
    -----
    Entropy-based (information-theoretic):
        Total = H[E[P]] = binary entropy of mean
        Aleatoric = E[H[P]] = expected binary entropy
        Epistemic = Total - Aleatoric = mutual information I(Y; θ)
    
    Variance-based (for delta method propagation):
        Total = Var[Y] = E[θ(1-θ)] + Var[θ]  (law of total variance)
        Aleatoric = E[θ(1-θ)] = expected Bernoulli variance
        Epistemic = Var[θ] = variance of predictions across samples
    """
    # Move sample dim to position 1 for consistent processing
    P = P.movedim(sample_dim, 1)  

    # mean probability across samples
    P_mean = P.mean(dim=1)  # (..., ...)
    
    # total uncertainty: H[E[P]]
    H_total = binary_entropy(P_mean)
    
    # aleatoric: E[H[P]]
    H_per_sample = binary_entropy(P) 
    H_aleatoric = H_per_sample.mean(dim=1)
    
    # Epistemic: I(Y; θ) = H[E[P]] - E[H[P]]
    H_epistemic = (H_total - H_aleatoric).clamp_min(0)
    
    # === Variance-based ===
    # Epistemic: Var[θ] across samples
    V_epistemic = P.var(dim=1, unbiased=P.shape[1] > 1)
    
    # Aleatoric: E[θ(1-θ)] - expected Bernoulli variance
    bernoulli_var = P * (1 - P)
    V_aleatoric = bernoulli_var.mean(dim=1)
    
    # Total: by law of total variance
    V_total = V_aleatoric + V_epistemic
    
    # Optional reduction across another dimension (e.g., bits)
    if reduce_dim is not None:
        # Adjust reduce_dim since we moved sample_dim
        if reduce_dim > sample_dim:
            reduce_dim -= 1
        
        reduce_fn = torch.mean if reduce_method == "mean" else torch.sum
        H_total = reduce_fn(H_total, dim=reduce_dim)
        H_aleatoric = reduce_fn(H_aleatoric, dim=reduce_dim)
        H_epistemic = reduce_fn(H_epistemic, dim=reduce_dim)
        V_total = reduce_fn(V_total, dim=reduce_dim)
        V_aleatoric = reduce_fn(V_aleatoric, dim=reduce_dim)
        V_epistemic = reduce_fn(V_epistemic, dim=reduce_dim)
        P_mean = reduce_fn(P_mean, dim=reduce_dim)
    
    return UncertaintyDecomposition(
        total_entropy=H_total,
        aleatoric_entropy=H_aleatoric,
        epistemic_entropy=H_epistemic,
        total_variance=V_total,
        aleatoric_variance=V_aleatoric,
        epistemic_variance=V_epistemic,
        mean=P_mean,
    )


def decompose_categorical(
    logits_or_probs: Tensor,
    sample_dim: int = 1,
    category_dim: int = -1,
    is_logits: bool = True,
    temperature: float = 0.003,
    normalize_entropy: bool = False,
) -> UncertaintyDecomposition:
    """
    Decompose uncertainty for categorical (softmax) predictions.
    
    Parameters
    ----------
    logits_or_probs : (..., S, ..., C)
        Logits or probabilities from S samples over C categories.
    sample_dim : int
        Which dimension contains samples.
    category_dim : int
        Which dimension contains categories (i.e. candidates).
    is_logits : bool
        If True, apply softmax; if False, assume already probabilities.
    temperature : float
        Softmax temperature (only used if is_logits=True).
    normalize_entropy : bool
        If True, divide entropy by log(C).
    
    Returns
    -------
    UncertaintyDecomposition
    
    Notes
    -----
    Entropy-based:
        Total = H[E[p]] = entropy of mean distribution
        Aleatoric = E[H[p]] = expected entropy
        Epistemic = Total - Aleatoric = I(Y; θ)
    
    Variance-based (summed over categories):
        Aleatoric = E[sum_c p_c(1-p_c)]
        Epistemic = sum_c Var[p_c]
        Total = Aleatoric + Epistemic
    """
    ndim = logits_or_probs.dim()
    
    # Normalize negative indices to positive for correct comparisons
    if sample_dim < 0:
        sample_dim = ndim + sample_dim
    if category_dim < 0:
        category_dim = ndim + category_dim
    
    # Convert to probabilities if needed
    if is_logits:
        logits = logits_or_probs.float() / max(temperature, _EPS)
        logits = logits - logits.max(dim=category_dim, keepdim=True).values
        probs = torch.softmax(logits, dim=category_dim)
    else:
        probs = logits_or_probs.float()
    
    # Get number of categories (i.e., classes)
    C = probs.shape[category_dim]
    
    # Mean across samples (NOTE: we average probabilities here, not samples)
    P_mean = probs.mean(dim=sample_dim)
    
    # === Entropy-based ===
    # After averaging over sample_dim, category_dim shifts if it was after sample_dim
    cat_dim_after_sample_mean = category_dim - 1 if category_dim > sample_dim else category_dim
    H_total = categorical_entropy(P_mean, dim=cat_dim_after_sample_mean)
    
    # Entropy per sample (over categories)
    H_per_sample = categorical_entropy(probs, dim=category_dim)
    # After entropy over category_dim, we average over sample_dim
    # sample_dim position doesn't change if category_dim > sample_dim
    sample_dim_after_cat_entropy = sample_dim if category_dim > sample_dim else sample_dim
    H_aleatoric = H_per_sample.mean(dim=sample_dim_after_cat_entropy)
    
    H_epistemic = (H_total - H_aleatoric).clamp_min(0)
    
    if normalize_entropy and C > 1:
        log_C = torch.log(torch.tensor(C, dtype=torch.float32, device=probs.device))
        H_total = H_total / log_C
        H_aleatoric = H_aleatoric / log_C
        H_epistemic = H_epistemic / log_C
    
    # === Variance-based ===
    # Aleatoric: E[p(1-p)] summed over categories, then averaged over samples
    bernoulli_var = (probs * (1 - probs)).sum(dim=category_dim)  # sum over categories
    # Now average over sample_dim (which hasn't shifted since category_dim > sample_dim or vice versa)
    sample_dim_after_cat_sum = sample_dim if category_dim > sample_dim else sample_dim
    V_aleatoric = bernoulli_var.mean(dim=sample_dim_after_cat_sum)
    
    # Epistemic: Var[p] across samples, then summed over categories
    V_per_category = probs.var(dim=sample_dim, unbiased=probs.shape[sample_dim] > 1)
    # After var over sample_dim, category_dim shifts if it was after sample_dim
    cat_dim_after_var = category_dim - 1 if category_dim > sample_dim else category_dim
    V_epistemic = V_per_category.sum(dim=cat_dim_after_var)
    
    V_total = V_aleatoric + V_epistemic
    
    return UncertaintyDecomposition(
        total_entropy=H_total,
        aleatoric_entropy=H_aleatoric,
        epistemic_entropy=H_epistemic,
        total_variance=V_total,
        aleatoric_variance=V_aleatoric,
        epistemic_variance=V_epistemic,
        mean=P_mean,
    )