"""
Selection with Guaranteed Risk (SGR) for Molecular Retrieval.

Based on: Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks", NeurIPS 2017
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import binom
from scipy.optimize import brentq


@dataclass
class SGRResult:
    """Result from SGR fitting."""
    threshold: float
    risk_bound: float
    empirical_risk: float
    coverage: float
    n_selected: int
    n_total: int
    target_risk: float
    delta: float
    feasible: bool = True
    eval_coverage: Optional[float] = None
    eval_empirical_risk: Optional[float] = None
    eval_n_selected: Optional[int] = None
    
    def __repr__(self) -> str:
        return (f"SGRResult(bound={self.risk_bound:.4f}, coverage={self.coverage:.4f}, "
                f"feasible={self.feasible})")


@dataclass
class SGRComparisonResult:
    """Result from comparing multiple uncertainty scores."""
    results: Dict[str, SGRResult]
    best_by_coverage: str
    target_risk: float
    
    def to_dataframe(self) -> pd.DataFrame:
        rows = [{
            "method": name,
            "threshold": r.threshold,
            "risk_bound": r.risk_bound,
            "empirical_risk": r.empirical_risk,
            "coverage": r.coverage,
            "n_selected": r.n_selected,
            "feasible": r.feasible
        } for name, r in self.results.items()]
        return pd.DataFrame(rows)


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_binomial_bound(empirical_risk: float, n_samples: int, delta: float) -> float:
    """
    Compute Clopper-Pearson upper bound on true risk (for BINARY losses only).
    
    Finds b such that P(X ≤ n_errors | Binomial(n, b)) = δ,
    giving a 1-δ upper confidence bound on the true risk.
    
    Parameters
    ----------
    empirical_risk : float
        Observed error rate.
    n_samples : int
        Number of samples.
    delta : float
        Significance level (e.g., 0.05 for 95% confidence).
    
    Returns
    -------
    bound : float
        Upper bound b* such that P(true_risk > b*) < δ.
    """
    if n_samples == 0:
        return 1.0
    
    # Use round for numerical stability (avoid floor(0.999999) = 0)
    n_errors = int(round(empirical_risk * n_samples))
    n_errors = min(n_errors, n_samples)  # Clamp to valid range
    
    if n_errors >= n_samples:
        return 1.0
    
    try:
        return float(brentq(lambda b: binom.cdf(n_errors, n_samples, b) - delta, 1e-10, 1 - 1e-10))
    except ValueError:
        return 1.0


def compute_hoeffding_bound(empirical_risk: float, n_samples: int, delta: float,
                            loss_range: float = 1.0) -> float:
    """
    Compute Hoeffding upper bound on true risk (for CONTINUOUS bounded losses).
    
    Uses Hoeffding's inequality: P(μ > μ̂ + ε) ≤ exp(-2nε²/R²)
    where R is the range of the loss.
    
    Parameters
    ----------
    empirical_risk : float
        Observed mean loss.
    n_samples : int
        Number of samples.
    delta : float
        Significance level.
    loss_range : float
        Range of possible loss values (max - min). Default 1.0 for [0,1] losses.
    
    Returns
    -------
    bound : float
        Upper bound such that P(true_risk > bound) < δ.
    """
    if n_samples == 0:
        return 1.0
    
    epsilon = loss_range * np.sqrt(np.log(1.0 / delta) / (2 * n_samples))
    return min(1.0, empirical_risk + epsilon)


class SelectiveGuaranteedRisk:
    """
    SGR classifier that finds threshold guaranteeing selective risk <= r*.
    
    Parameters
    ----------
    higher_is_confident : bool
        If True, higher scores mean more confident predictions.
    binary_loss : bool
        If True, use binomial bounds (for 0/1 losses like hit@k).
        If False, use Hoeffding bounds (for continuous losses like Tanimoto).
    """
    
    def __init__(self, higher_is_confident: bool = True, binary_loss: bool = True):
        self.higher_is_confident = higher_is_confident
        self.binary_loss = binary_loss
        self.threshold_: Optional[float] = None
        self.result_: Optional[SGRResult] = None
    
    def fit(
        self,
        confidence: Union[torch.Tensor, np.ndarray],
        losses: Union[torch.Tensor, np.ndarray],
        target_risk: float,
        delta: float = 0.001,
        verbose: bool = False
    ) -> SGRResult:
        """Find threshold guaranteeing selective risk <= target_risk."""
        confidence = _to_numpy(confidence)
        losses = _to_numpy(losses)
        
        if not self.higher_is_confident:
            confidence = -confidence
        
        m = len(confidence)
        k = int(np.ceil(np.log2(m)))
        
        sorted_idx = np.argsort(confidence)
        sorted_conf = confidence[sorted_idx]
        sorted_loss = losses[sorted_idx]
        
        z_min, z_max = 0, m - 1
        best_result = None
        
        for i in range(k):
            z = (z_min + z_max) // 2
            theta = sorted_conf[z]
            n_selected = m - z
            
            emp_risk = sorted_loss[z:].sum() / n_selected if n_selected > 0 else 1.0
            coverage = n_selected / m
            
            # Choose bound based on loss type
            if self.binary_loss:
                bound = compute_binomial_bound(emp_risk, n_selected, delta / k)
            else:
                bound = compute_hoeffding_bound(emp_risk, n_selected, delta / k)
            
            if verbose:
                print(f"  Iter {i+1}: z={z}, θ={theta:.4f}, risk={emp_risk:.4f}, "
                      f"cov={coverage:.3f}, bound={bound:.4f}")
            
            result = SGRResult(
                threshold=float(theta), risk_bound=float(bound),
                empirical_risk=float(emp_risk), coverage=float(coverage),
                n_selected=int(n_selected), n_total=int(m),
                target_risk=float(target_risk), delta=float(delta),
                feasible=bound < target_risk
            )
            
            if bound < target_risk:
                z_max = z
                best_result = result
            else:
                z_min = z + 1
        
        if best_result is None:
            warnings.warn(f"Could not achieve target risk r*={target_risk}")
            best_result = SGRResult(
                threshold=float('inf'), risk_bound=1.0,
                empirical_risk=float(losses.mean()), coverage=0.0,
                n_selected=0, n_total=int(m),
                target_risk=float(target_risk), delta=float(delta),
                feasible=False
            )
        
        self.threshold_ = best_result.threshold
        self.result_ = best_result
        return best_result
    
    def evaluate(
        self,
        confidence: Union[torch.Tensor, np.ndarray],
        losses: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[float, float]:
        """Evaluate on new data. Returns (risk, coverage)."""
        if self.threshold_ is None:
            raise RuntimeError("Must call fit() first")
        
        confidence = _to_numpy(confidence)
        losses = _to_numpy(losses)
        
        if not self.higher_is_confident:
            confidence = -confidence
        
        mask = confidence >= self.threshold_
        n_selected = mask.sum()
        
        if n_selected == 0:
            return 1.0, 0.0
        
        return float(losses[mask].sum() / n_selected), float(n_selected / len(losses))
    
    def select(self, confidence: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Return boolean mask of samples to predict on."""
        if self.threshold_ is None:
            raise RuntimeError("Must call fit() first")
        
        confidence = _to_numpy(confidence)
        if not self.higher_is_confident:
            confidence = -confidence
        return confidence >= self.threshold_


def fit_sgr(
    confidence: Union[torch.Tensor, np.ndarray],
    losses: Union[torch.Tensor, np.ndarray],
    target_risk: float,
    delta: float = 0.001,
    higher_is_confident: bool = True,
    binary_loss: bool = True,
    verbose: bool = False
) -> SGRResult:
    """
    Convenience function to fit SGR.
    
    Parameters
    ----------
    confidence : array-like
        Confidence scores (higher = more confident if higher_is_confident=True).
    losses : array-like
        Loss values per sample.
    target_risk : float
        Desired risk level r*.
    delta : float
        Confidence parameter (1-delta probability of guarantee).
    higher_is_confident : bool
        Whether higher confidence scores mean more confident.
    binary_loss : bool
        If True, use binomial bounds (for 0/1 losses like hit@k).
        If False, use Hoeffding bounds (for continuous losses like Tanimoto).
    verbose : bool
        Print iteration details.
    
    Returns
    -------
    SGRResult
        Result containing threshold, coverage, and risk bounds.
    """
    return SelectiveGuaranteedRisk(higher_is_confident, binary_loss).fit(
        confidence, losses, target_risk, delta, verbose
    )


def make_cal_eval_split(
    n_samples: int,
    cal_fraction: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create the deterministic SGR calibration/evaluation split used in the paper."""
    if n_samples <= 1:
        raise ValueError("SGR split requires at least two samples.")
    if not 0.0 < cal_fraction < 1.0:
        raise ValueError("cal_fraction must be strictly between 0 and 1.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_cal = int(n_samples * cal_fraction)
    n_cal = min(max(n_cal, 1), n_samples - 1)
    return perm[:n_cal], perm[n_cal:]


def attach_eval_result(
    sgr: SelectiveGuaranteedRisk,
    result: SGRResult,
    confidence: Union[torch.Tensor, np.ndarray],
    losses: Union[torch.Tensor, np.ndarray],
) -> SGRResult:
    """Evaluate a fitted SGR selector on held-out data and attach eval fields."""
    if result.feasible:
        eval_risk, eval_coverage = sgr.evaluate(confidence, losses)
        result.eval_empirical_risk = float(eval_risk)
        result.eval_coverage = float(eval_coverage)
        result.eval_n_selected = int(round(eval_coverage * len(losses)))
    else:
        result.eval_empirical_risk = np.nan
        result.eval_coverage = np.nan
        result.eval_n_selected = 0
    return result


def compare_uncertainty_scores(
    uncertainties: Dict[str, Union[torch.Tensor, np.ndarray]],
    losses: Union[torch.Tensor, np.ndarray],
    target_risk: float,
    delta: float = 0.001,
    higher_is_confident: Optional[Dict[str, bool]] = None
) -> SGRComparisonResult:
    """Compare multiple uncertainty scores for SGR."""
    from ms_uq.utils import is_confidence_score
    
    if higher_is_confident is None:
        higher_is_confident = {name: is_confidence_score(name) for name in uncertainties}
    
    results = {}
    best_name, best_coverage = None, -1.0
    
    for name, scores in uncertainties.items():
        result = SelectiveGuaranteedRisk(higher_is_confident.get(name, False)).fit(
            scores, losses, target_risk, delta
        )
        results[name] = result
        
        if result.feasible and result.coverage > best_coverage:
            best_coverage = result.coverage
            best_name = name
    
    return SGRComparisonResult(results, best_name or "none_feasible", target_risk)


def sgr_risk_coverage_table(
    confidence: Union[torch.Tensor, np.ndarray],
    losses: Union[torch.Tensor, np.ndarray],
    target_risks: List[float],
    delta: float = 0.001,
    higher_is_confident: bool = True
) -> pd.DataFrame:
    """Generate table of SGR results for multiple target risks."""
    rows = []
    for r_star in target_risks:
        result = fit_sgr(confidence, losses, r_star, delta, higher_is_confident)
        rows.append({
            "target_risk": r_star,
            "empirical_risk": result.empirical_risk,
            "coverage": result.coverage,
            "risk_bound": result.risk_bound,
            "n_selected": result.n_selected,
            "feasible": result.feasible
        })
    return pd.DataFrame(rows)