"""
Entropy and information-theoretic functions.
"""

import torch
from torch import Tensor

EPS = 1e-12


def binary_entropy(p: Tensor) -> Tensor:
    """Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)."""
    p = p.clamp(0, 1)
    q = 1 - p
    return -(torch.special.xlogy(p, p) + torch.special.xlogy(q, q))


def categorical_entropy(probs: Tensor, dim: int = -1) -> Tensor:
    """Categorical entropy H(p) = -sum(p * log(p))."""
    probs = probs.clamp(min=EPS)
    return -(probs * probs.log()).sum(dim)


# Alias for backward compatibility with retrieval_unc.py
entropy_from_probs = categorical_entropy


def entropy_normalized(probs: Tensor, dim: int = -1) -> Tensor:
    """
    Normalized entropy in [0, 1].
    Returns entropy / log(K) where K is the number of categories.
    """
    K = probs.shape[dim]
    if K <= 1:
        return torch.zeros_like(probs.select(dim, 0))
    
    H = categorical_entropy(probs, dim=dim)
    max_entropy = torch.tensor(float(K), dtype=probs.dtype, device=probs.device).log()
    return H / max_entropy


def kl_divergence(p: Tensor, q: Tensor, dim: int = -1) -> Tensor:
    """KL divergence KL(p || q) = sum(p * log(p/q))."""
    p = p.clamp(min=EPS)
    q = q.clamp(min=EPS)
    return (p * (p.log() - q.log())).sum(dim)

def mutual_information(p_joint: Tensor, p_marginal1: Tensor, p_marginal2: Tensor) -> Tensor:
    """
    Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
    
    For uncertainty decomposition: I = H(E[p]) - E[H(p)]
    """
    return categorical_entropy(p_marginal1) + categorical_entropy(p_marginal2) - categorical_entropy(p_joint)