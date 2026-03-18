#!/usr/bin/env python3
"""
Smoke test for the ms_uq package.

Verifies:
  1. All subpackage imports resolve without error
  2. BitwiseUncertainty: output shapes, entropy decomposition invariants
  3. RetrievalUncertainty: output shapes, entropy decomposition invariants
  4. DistanceUncertainty: fit/forward cycle, output shapes
  5. compute_uncertainties unified API
  6. Rejection curves and AURC bounds
  7. Selective risk (SGR) basic contract
  8. All kept plotting functions are importable

Run:
    python tests/smoke_test.py
"""
from __future__ import annotations
import sys
import traceback

import torch
import numpy as np

PASS, FAIL = 0, 0


def check(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [OK] {name}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        FAIL += 1


def test_imports():
    """All core submodules import without error.
    
    Note: ms_uq.data and ms_uq.utils.helper_functions.create_dataset
    require massspecgym at runtime but not at import time.
    """
    import ms_uq.core
    import ms_uq.core.entropy
    import ms_uq.core.similarity
    import ms_uq.evaluation.metrics
    import ms_uq.evaluation.rejection_curve
    import ms_uq.evaluation.selective_risk
    import ms_uq.evaluation.visualisation
    import ms_uq.unc_measures
    import ms_uq.unc_measures.base
    import ms_uq.unc_measures.bitwise_unc
    import ms_uq.unc_measures.retrieval_unc
    import ms_uq.unc_measures.distance_unc
    import ms_uq.unc_measures.decomposition
    import ms_uq.unc_measures.eval_measures
    import ms_uq.utils
    import ms_uq.utils.helper_functions

def test_bitwise_entropy():
    """Entropy decomposition: total >= aleatoric, epistemic >= 0."""
    from ms_uq.unc_measures.bitwise_unc import BitwiseUncertainty
    torch.manual_seed(0)
    N, S, K = 16, 5, 128
    P = torch.rand(N, S, K)

    bw = BitwiseUncertainty(aggregate="sum", kind="entropy", weighting="none")
    out = bw(P)
    assert out["total"].shape == (N,), f"shape: {out['total'].shape}"
    assert out["aleatoric"].shape == (N,)
    assert out["epistemic"].shape == (N,)
    assert (out["epistemic"] >= -1e-7).all(), "epistemic entropy < 0"
    assert (out["total"] >= out["aleatoric"] - 1e-7).all(), "total < aleatoric"


def test_bitwise_variance():
    """Variance decomposition: total = aleatoric + epistemic."""
    from ms_uq.unc_measures.bitwise_unc import BitwiseUncertainty
    torch.manual_seed(0)
    N, S, K = 16, 5, 128
    P = torch.rand(N, S, K)

    bw = BitwiseUncertainty(aggregate="sum", kind="both", weighting="none")
    out = bw(P)
    assert out["total_var"].shape == (N,)
    # Law of total variance: Var[Y] = E[Var[Y|θ]] + Var[E[Y|θ]]
    residual = (out["total_var"] - out["aleatoric_var"] - out["epistemic_var"]).abs()
    assert (residual < 1e-5).all(), f"variance decomposition violated: max residual {residual.max():.6f}"


def test_bitwise_sparse_aware():
    """Sparse-aware epistemic methods run without error."""
    from ms_uq.unc_measures.bitwise_unc import compute_sparse_aware_epistemic
    torch.manual_seed(0)
    P = torch.rand(8, 5, 128)
    for method in ("logit", "active_bit", "relative"):
        out = compute_sparse_aware_epistemic(P, method=method)
        assert out.shape == (8,), f"{method}: shape {out.shape}"


# ═══════════════════════════════════════════════════════════════════════
# 3. RetrievalUncertainty (ragged)
# ═══════════════════════════════════════════════════════════════════════
def test_retrieval_uncertainty():
    """Ragged retrieval uncertainty: shapes, entropy invariants, rank variance."""
    from ms_uq.unc_measures.retrieval_unc import RetrievalUncertainty
    torch.manual_seed(0)
    N, S = 10, 5
    # Ragged candidate counts: 3..20 candidates per query
    n_cands = torch.randint(3, 21, (N,))
    M = int(n_cands.sum())
    ptr = torch.zeros(N + 1, dtype=torch.long)
    ptr[1:] = n_cands.cumsum(0)

    scores_stack = torch.randn(S, M)

    ru = RetrievalUncertainty(temperature=1.0, top_k_list=[1, 5])
    out = ru(scores_stack, ptr)

    # Shape checks
    for key in ("entropy_total", "entropy_aleatoric", "entropy_epistemic",
                "rank_var_1", "rank_var_5", "confidence_top1", "margin",
                "score_gap", "n_candidates"):
        assert out[key].shape == (N,), f"{key}: shape {out[key].shape}"

    # Entropy invariants
    assert (out["entropy_epistemic"] >= -1e-7).all(), "epistemic entropy < 0"
    assert (out["entropy_total"] >= out["entropy_aleatoric"] - 1e-7).all(), "total < aleatoric"

    # Rank variance is non-negative
    assert (out["rank_var_1"] >= -1e-7).all()
    assert (out["rank_var_5"] >= -1e-7).all()


def test_distance_uncertainty():
    """Fit on synthetic train embeddings, forward on test, check shapes."""
    from ms_uq.unc_measures.distance_unc import DistanceUncertainty
    torch.manual_seed(0)
    N_train, N_test, D = 100, 20, 64
    X_train = torch.randn(N_train, D)
    X_test = torch.randn(N_test, D)

    du = DistanceUncertainty(n_neighbors=5, metric="cosine", normalize=True,
                              covariance="shrinkage", knn_aggregation="kth")
    du.fit(X_train)
    out = du(X_test)

    for key in ("knn_distance", "mahalanobis", "centroid_distance"):
        assert out[key].shape == (N_test,), f"{key}: shape {out[key].shape}"
        assert torch.isfinite(out[key]).all(), f"{key}: non-finite values"
        assert (out[key] >= 0).all(), f"{key}: negative distance"


def test_compute_uncertainties():
    """Unified API returns expected keys for fingerprint + retrieval measures."""
    from ms_uq.unc_measures.eval_measures import compute_uncertainties
    torch.manual_seed(0)
    N, S, K = 10, 5, 128
    Pbits = torch.rand(N, S, K)

    # Build ragged scores
    n_cands = torch.randint(3, 15, (N,))
    M = int(n_cands.sum())
    ptr = torch.zeros(N + 1, dtype=torch.long)
    ptr[1:] = n_cands.cumsum(0)
    scores_stack = torch.randn(S, M)

    out = compute_uncertainties(
        Pbits=Pbits, scores_stack=scores_stack, ptr=ptr,
        fingerprint_measures=["bitwise_total", "bitwise_epistemic"],
        retrieval_measures=["confidence", "score_gap", "rank_var_1", "retrieval_total"],
        temperature=1.0, negate_confidence=False,
    )
    assert isinstance(out, dict)
    for key in ("bitwise_total", "bitwise_epistemic", "confidence",
                "score_gap", "rank_var_1", "retrieval_total"):
        assert key in out, f"missing key: {key}"
        assert out[key].shape == (N,), f"{key}: shape {out[key].shape}"


def test_rejection_curve_and_aurc():
    """Rejection curve: monotone coverage, AURC in [oracle, random]."""
    from ms_uq.evaluation.rejection_curve import (
        rejection_curve, aurc_from_curve, compute_oracle_aurc, compute_random_aurc,
    )
    torch.manual_seed(0)
    N = 200
    hits = (torch.rand(N) > 0.6).float()  # ~40% hit rate
    loss = 1.0 - hits
    # Good uncertainty: correlated with loss
    u_good = loss + 0.1 * torch.randn(N)
    # Random uncertainty
    u_rand = torch.rand(N)

    rej, kept = rejection_curve(loss, u_good)
    assert rej.shape == (N,) and kept.shape == (N,)
    assert rej[0].item() == 0.0, "first rejection% should be 0"

    aurc_good = aurc_from_curve(rej, kept)
    aurc_rand = aurc_from_curve(*rejection_curve(loss, u_rand))
    aurc_oracle = compute_oracle_aurc(hits.numpy())[0]
    aurc_random = compute_random_aurc(hits.numpy())[0]

    assert aurc_oracle <= aurc_good + 1e-4, f"oracle ({aurc_oracle:.4f}) should be <= good ({aurc_good:.4f})"
    assert aurc_good <= aurc_random + 0.05, f"good ({aurc_good:.4f}) should be <= random ({aurc_random:.4f})"


def test_sgr_basic():
    """SGR: fit on synthetic data, check output structure."""
    from ms_uq.evaluation.selective_risk import fit_sgr, SGRResult
    torch.manual_seed(42)
    N = 500
    loss = (torch.rand(N) > 0.7).float()  # ~30% error
    confidence = -loss + 0.2 * torch.randn(N)  # higher = more confident

    result = fit_sgr(
        confidence=confidence.numpy(), losses=loss.numpy(),
        target_risk=0.2, delta=0.05,
        higher_is_confident=True,
    )
    assert isinstance(result, SGRResult), f"expected SGRResult, got {type(result)}"
    assert hasattr(result, "threshold")
    assert hasattr(result, "coverage")


def test_plot_imports():
    """All 7 kept plot functions are importable."""
    from ms_uq.evaluation.visualisation import (
        plot_risk_coverage_curves,
        plot_aurc_bars,
        plot_rc_and_aurc_paired,
        plot_member_vs_agg,
        plot_correlation_heatmap,
        plot_sgr_coverage_combined,
        plot_sgr_risk_calibration,
    )


def test_binary_entropy():
    """H(0)≈0, H(1)≈0, H(0.5)=log(2)."""
    from ms_uq.core.entropy import binary_entropy
    # Boundary values: should be close to zero (exact 0 requires EPS handling)
    assert binary_entropy(torch.tensor(0.0)).item() < 1e-5, "H(0) should be ~0"
    assert binary_entropy(torch.tensor(1.0)).item() < 1e-5, "H(1) should be ~0"
    # Maximum entropy at p=0.5
    h_half = binary_entropy(torch.tensor(0.5)).item()
    assert abs(h_half - np.log(2)) < 1e-5, f"H(0.5) = {h_half}, expected {np.log(2)}"


def test_similarity_matrix():
    """Cosine self-similarity is identity-like on normalized vectors."""
    from ms_uq.core.similarity import similarity_matrix
    torch.manual_seed(0)
    X = torch.randn(5, 32)
    S = similarity_matrix(X, X, metric="cosine")
    assert S.shape == (5, 5)
    # Diagonal should be ~1.0
    diag = S.diag()
    assert (diag - 1.0).abs().max() < 1e-5, f"self-similarity not 1: {diag}"


def test_confidence_detection():
    """is_confidence_score correctly classifies known measure names."""
    from ms_uq.utils.helper_functions import is_confidence_score
    assert is_confidence_score("confidence") is True
    assert is_confidence_score("score_gap") is True
    assert is_confidence_score("margin") is True
    assert is_confidence_score("retrieval_epistemic") is False
    assert is_confidence_score("bitwise_total") is False
    assert is_confidence_score("rank_var_1") is False

if __name__ == "__main__":
    print("=" * 60)
    print("ms_uq smoke test")
    print("=" * 60)

    tests = [
        ("imports", test_imports),
        ("bitwise entropy decomposition", test_bitwise_entropy),
        ("bitwise variance decomposition", test_bitwise_variance),
        ("bitwise sparse-aware epistemic", test_bitwise_sparse_aware),
        ("retrieval uncertainty (ragged)", test_retrieval_uncertainty),
        ("distance uncertainty (fit/forward)", test_distance_uncertainty),
        ("compute_uncertainties unified API", test_compute_uncertainties),
        ("rejection curve + AURC bounds", test_rejection_curve_and_aurc),
        ("SGR basic contract", test_sgr_basic),
        ("plot function imports", test_plot_imports),
        ("binary entropy", test_binary_entropy),
        ("similarity matrix", test_similarity_matrix),
        ("confidence score detection", test_confidence_detection),
    ]

    for name, fn in tests:
        check(name, fn)

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    print(f"{'=' * 60}")
    sys.exit(1 if FAIL > 0 else 0)
