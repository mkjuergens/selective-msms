import argparse
from pathlib import Path
import torch

from ms_uq.unc_measures.bitwise_unc import BitwiseUncertainty
from ms_uq.unc_measures.retrieval_unc import RetrievalUncertainty


def synth_tests():
    torch.manual_seed(0)
    N, S, K = 8, 5, 16

    P = torch.rand(N, S, K)
    uq_b = BitwiseUncertainty(aggregate="sum", kind="both", weighting="none")(P)

    # Entropy keys
    for k in ("aleatoric", "epistemic", "total"):
        assert uq_b[k].shape == (N,), f"bitwise {k} wrong shape: {uq_b[k].shape}"
    # Variance keys
    for k in ("aleatoric_var", "epistemic_var", "total_var"):
        assert uq_b[k].shape == (N,), f"bitwise {k} wrong shape: {uq_b[k].shape}"

    # Invariants
    assert torch.all(uq_b["epistemic"] >= -1e-7), "epistemic entropy < 0"
    assert torch.all(uq_b["total"] >= uq_b["aleatoric"] - 1e-7), "total < aleatoric"

    # Ragged retrieval
    n_cands = torch.randint(3, 15, (N,))
    M = int(n_cands.sum())
    ptr = torch.zeros(N + 1, dtype=torch.long)
    ptr[1:] = n_cands.cumsum(0)
    scores = torch.randn(S, M)

    uq_r = RetrievalUncertainty(temperature=1.0, top_k_list=[1, 5])(scores, ptr)
    for k in ("entropy_aleatoric", "entropy_epistemic", "entropy_total",
              "rank_var_1", "rank_var_5"):
        assert uq_r[k].shape == (N,), f"retrieval {k} wrong shape: {uq_r[k].shape}"
    assert torch.all(uq_r["entropy_epistemic"] >= -1e-7)
    assert torch.all(uq_r["entropy_total"] >= uq_r["entropy_aleatoric"] - 1e-7)

    print("[OK] synthetic UQ invariants and shapes")


def file_tests(pred_dir: Path, metric: str, T: float | None):
    """Validate UQ computation on real prediction files."""
    P = torch.load(pred_dir / "fp_probs.pt", map_location="cpu")["stack"].float()
    uq_b = BitwiseUncertainty(aggregate="sum", kind="entropy", weighting="none")(P)
    print(f"[OK] bitwise UQ vectors: shape {uq_b['total'].shape}")

    scores_files = sorted(pred_dir.glob(f"scores_*{metric}*.pt"))
    if not scores_files:
        print(f"[SKIP] no scores file matching *{metric}* in {pred_dir}")
        return

    D = torch.load(scores_files[0], map_location="cpu")
    scores_stack = D.get("scores_stack_flat")
    ptr = D["ptr"].long()
    if scores_stack is None:
        print("[SKIP] no per-sample scores in file")
        return

    N = ptr.numel() - 1
    RU = RetrievalUncertainty(temperature=1.0 if T is None else float(T), top_k_list=[1])
    uq = RU(scores_stack, ptr)
    assert (uq["entropy_epistemic"] >= -1e-7).all()
    assert (uq["entropy_total"] >= uq["entropy_aleatoric"] - 1e-7).all()
    print(f"[OK] retrieval UQ over {N} spectra, metric={metric}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Quick UQ sanity checks.")
    ap.add_argument("--pred_dir", type=str, default=None,
                    help="Folder with fp_probs.pt and scores_*<metric>*.pt")
    ap.add_argument("--metric", type=str, default="cosine")
    ap.add_argument("--temp", type=float, default=None)
    args = ap.parse_args()

    synth_tests()
    if args.pred_dir:
        file_tests(Path(args.pred_dir), args.metric, args.temp)
