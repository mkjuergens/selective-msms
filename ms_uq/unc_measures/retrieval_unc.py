"""
Retrieval uncertainty decomposition for molecular candidate ranking.

This module provides uncertainty measures for retrieval/ranking tasks:
- Entropy-based decomposition (aleatoric/epistemic)
- Rank-based disagreement measures  
- Confidence metrics (margin, score_gap)
- Candidate set difficulty metrics
"""

from __future__ import annotations
from typing import Dict, Optional, List

import torch
import numpy as np
from torch import Tensor

from ms_uq.unc_measures.base import BaseUncertainty

_EPS = 1e-12


class RetrievalUncertainty(BaseUncertainty):
    """
    Unified retrieval uncertainty for ragged candidate lists.
    
    Computes four categories of uncertainty measures:
    
    1. **Entropy-based** (probability distribution uncertainty):
       - entropy_total: H[E[p]] - entropy of mean probabilities
       - entropy_aleatoric: E[H[p]] - expected entropy per sample
       - entropy_epistemic: total - aleatoric (mutual information)
       - normalized_entropy: H[E[p]] / log(n_candidates)
    
    2. **Rank-based** (disagreement in rankings):
       - rank_var_k: Var[rank of aggregated top-k candidates across samples]
       - topk_agreement: fraction of samples with same candidate in top-k
    
    3. **Confidence** (score/probability magnitude):
       - confidence_topk: sum of P(top-k candidates)
       - margin: P(top-1) - P(top-2)
       - score_gap: score(top-1) - score(top-2) in aggregated scores
    
    4. **Candidate set** (difficulty/ambiguity):
       - n_candidates: number of candidates (more = harder)
       - ambiguity_ratio: score(top-2) / score(top-1)
    
    Parameters
    ----------
    temperature : float
        Softmax temperature for converting scores to probabilities.
    normalize_entropy : bool
        If True, normalize entropy by log(n_candidates).
    top_k_list : list of int
        k values for rank variance and confidence computation.
    """
    
    def __init__(
        self,
        temperature: float = 0.003,
        normalize_entropy: bool = False,
        top_k_list: Optional[List[int]] = None,
    ):
        super().__init__()
        self.temperature = max(temperature, _EPS)
        self.normalize_entropy = normalize_entropy
        self.top_k_list = top_k_list or [1, 5, 20]
    
    def forward(
        self,
        scores_stack: Tensor,
        ptr: Tensor,
        scores_agg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute all retrieval uncertainty measures.
        
        Parameters
        ----------
        scores_stack : (S, M) tensor
            Per-sample scores. S = number of samples/members, M = total candidates.
        ptr : (N+1,) tensor
            Ragged array pointers. ptr[i]:ptr[i+1] gives candidates for query i.
        scores_agg : (M,) tensor, optional
            Aggregated scores. If None, uses mean of scores_stack.
        
        Returns
        -------
        dict with all uncertainty measures (each shape (N,)):
            Entropy-based:
                - entropy_total, entropy_aleatoric, entropy_epistemic
            Rank-based:
                - rank_var_1, rank_var_5, rank_var_20, ...
                - top1_agreement, top5_agreement, top20_agreement, ...
            Confidence:
                - confidence_top1, confidence_top5, confidence_top20, ...
                - margin, score_gap
            Candidate set:
                - n_candidates, ambiguity_ratio
        """
        # Handle 1D input (single model)
        if scores_stack.dim() == 1:
            scores_stack = scores_stack.unsqueeze(0)
        
        S, M = scores_stack.shape
        N = ptr.numel() - 1
        device = scores_stack.device
        
        # Default aggregation: mean
        if scores_agg is None:
            scores_agg = scores_stack.mean(dim=0)
        
        # Pre-allocate outputs
        results = {
            # Entropy-based
            "entropy_total": np.zeros(N, dtype=np.float32),
            "entropy_aleatoric": np.zeros(N, dtype=np.float32),
            "entropy_epistemic": np.zeros(N, dtype=np.float32),
            "normalized_entropy": np.zeros(N, dtype=np.float32),
            # Confidence
            "margin": np.zeros(N, dtype=np.float32),
            "score_gap": np.zeros(N, dtype=np.float32),
            # Candidate set
            "n_candidates": np.zeros(N, dtype=np.float32),
            "ambiguity_ratio": np.zeros(N, dtype=np.float32),
        }
        
        # Add rank-based and confidence for each k
        for k in self.top_k_list:
            results[f"rank_var_{k}"] = np.zeros(N, dtype=np.float32)
            results[f"top{k}_agreement"] = np.zeros(N, dtype=np.float32)
            results[f"confidence_top{k}"] = np.zeros(N, dtype=np.float32)
        
        # Process each query
        for i in range(N):
            start, end = int(ptr[i]), int(ptr[i + 1])
            n_cand = end - start
            
            # Store n_candidates
            results["n_candidates"][i] = float(n_cand)
            
            if n_cand == 0:
                continue
            
            if n_cand == 1:
                # Trivial case: only one candidate
                for k in self.top_k_list:
                    results[f"confidence_top{k}"][i] = 1.0
                    results[f"top{k}_agreement"][i] = 1.0
                results["margin"][i] = 1.0
                results["score_gap"][i] = 1.0
                results["ambiguity_ratio"][i] = 0.0
                results["normalized_entropy"][i] = 0.0
                continue
            
            # Extract scores for this query
            scores_i = scores_stack[:, start:end]  # (S, n_cand)
            agg_i = scores_agg[start:end]  # (n_cand,)
            
            # === 1. Entropy-based decomposition ===
            logits = scores_i.float() / self.temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)  # (S, n_cand)
            
            P_mean = probs.mean(dim=0)  # (n_cand,)
            
            # Total: H[E[p]]
            H_total = -(P_mean * torch.log(P_mean + _EPS)).sum()
            
            # Aleatoric: E[H[p]]
            H_per_sample = -(probs * torch.log(probs + _EPS)).sum(dim=-1)
            H_aleatoric = H_per_sample.mean()
            
            # Epistemic: I(Y; θ) = H[E[p]] - E[H[p]]
            H_epistemic = (H_total - H_aleatoric).clamp_min(0)
            H_normalized = H_total / np.log(float(n_cand)) if n_cand > 1 else H_total.new_tensor(0.0)
            
            if self.normalize_entropy and n_cand > 1:
                log_C = np.log(float(n_cand))
                H_total = H_total / log_C
                H_aleatoric = H_aleatoric / log_C
                H_epistemic = H_epistemic / log_C
            
            results["entropy_total"][i] = H_total.item()
            results["entropy_aleatoric"][i] = H_aleatoric.item()
            results["entropy_epistemic"][i] = H_epistemic.item()
            results["normalized_entropy"][i] = H_normalized.item()
            
            # === 2. Rank-based disagreement ===
            # Get rankings from each sample
            sample_ranks = scores_i.argsort(dim=-1, descending=True)  # (S, n_cand)
            agg_ranks = agg_i.argsort(descending=True)  # (n_cand,)
            
            for k in self.top_k_list:
                k_eff = min(k, n_cand)
                
                # Top-k candidates according to aggregated scores
                topk_agg = set(agg_ranks[:k_eff].tolist())
                
                # Rank variance: For each of the top-k aggregated candidates,
                # what is the variance of their rank across samples?
                rank_vars = []
                for cand_idx in agg_ranks[:k_eff].tolist():
                    # Find rank of this candidate in each sample
                    cand_ranks = (sample_ranks == cand_idx).float().argmax(dim=-1)  # (S,)
                    rank_vars.append(cand_ranks.float().var().item())
                results[f"rank_var_{k}"][i] = np.mean(rank_vars) if rank_vars else 0.0
                
                # Top-k agreement: fraction of samples whose top-k overlaps with aggregated top-k
                agreements = []
                for s in range(S):
                    topk_s = set(sample_ranks[s, :k_eff].tolist())
                    overlap = len(topk_agg & topk_s) / k_eff
                    agreements.append(overlap)
                results[f"top{k}_agreement"][i] = np.mean(agreements)
                
                # Confidence: sum of probabilities for top-k candidates
                topk_probs = P_mean[agg_ranks[:k_eff]].sum().item()
                results[f"confidence_top{k}"][i] = topk_probs
            
            # === 3. Margin and score gap ===
            sorted_probs = P_mean.sort(descending=True).values
            results["margin"][i] = (sorted_probs[0] - sorted_probs[1]).item()
            
            sorted_scores = agg_i.sort(descending=True).values
            results["score_gap"][i] = (sorted_scores[0] - sorted_scores[1]).item()
            
            # === 4. Candidate set metrics ===
            # Ambiguity ratio: top2/top1 (high = ambiguous)
            if sorted_scores[0].abs() > _EPS:
                results["ambiguity_ratio"][i] = (sorted_scores[1] / sorted_scores[0]).item()
            else:
                results["ambiguity_ratio"][i] = 0.0
        
        # Convert to tensors
        return {k: torch.from_numpy(v) for k, v in results.items()}
    
    def compute(self, scores_stack: Tensor, ptr: Tensor, scores_agg: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Alias for forward()."""
        return self.forward(scores_stack, ptr, scores_agg)


# Backward compatibility alias
RetrievalUncertaintyRagged = RetrievalUncertainty