import argparse
from pathlib import Path
import torch

from ms_uq.unc_measures.bitwise_unc import BitwiseUncertainty
from ms_uq.unc_measures.retrieval_unc import RetrievalUncertainty

def synth_tests():
    torch.manual_seed(0)
    N, S, K, C = 8, 5, 16, 7

    P = torch.rand(N, S, K)
    uq_b = BitwiseUncertainty(aggregate="sum")(P)
    for k in ("aleatoric_var","epistemic_var","total_var","aleatoric_ent","epistemic_ent","total_ent"):
        assert uq_b[k].shape == (N,), f"bitwise {k} wrong shape: {uq_b[k].shape}"
    # invariants
    assert torch.all(uq_b["epistemic_ent"] >= -1e-7)
    assert torch.all(uq_b["total_ent"]    >= uq_b["aleatoric_ent"] - 1e-7)

    scores = torch.randn(N, S, C)
    uq_r = RetrievalUncertainty(k=1, temperature=1.0, mc_samples=256)(scores)
    for k in ("aleatoric_var","epistemic_var","total_var","aleatoric_ent","epistemic_ent","total_ent"):
        assert uq_r[k].shape == (N,), f"retrieval {k} wrong shape: {uq_r[k].shape}"
    assert torch.all(uq_r["epistemic_ent"] >= -1e-7)
    assert torch.all(uq_r["total_ent"]    >= uq_r["aleatoric_ent"] - 1e-7)

    print("[OK] synthetic UQ invariants and shapes")

def file_tests(pred_dir: Path, metric: str, T: float | None):
    # Load (N,S,K) bitwise probs
    P = torch.load(pred_dir/"fp_probs.pt", map_location="cpu")["stack"].float()
    uq_b = BitwiseUncertainty(reduce="sum")(P)
    print(f"[OK] bitwise UQ vectors: shape {uq_b['total_ent'].shape}")

    # Load ragged scores (S, ΣM), slice per spectrum, run retrieval-UQ rank-1
    D = torch.load(pred_dir/f"scores_ragged_{metric}.pt", map_location="cpu")
    S, SM = D["scores_stack_flat"].shape
    ptr   = D["ptr"].long()
    N = ptr.numel() - 1
    RU = RetrievalUncertainty(k=1, temperature=1.0 if T is None else float(T), mc_samples=256)

    AU, EU, TU = [], [], []
    for n in range(N):
        s, e = int(ptr[n]), int(ptr[n+1])
        logits_ns = D["scores_stack_flat"][:, s:e].unsqueeze(0)  # (1,S,Cn)
        uq = RU(logits_ns)
        AU.append(uq["aleatoric_ent"].squeeze(0))
        EU.append(uq["epistemic_ent"].squeeze(0))
        TU.append(uq["total_ent"].squeeze(0))
    AU, EU, TU = torch.stack(AU), torch.stack(EU), torch.stack(TU)
    assert (EU >= -1e-7).all() and (TU >= AU - 1e-7).all()
    print(f"[OK] retrieval UQ (rank-1) over {N} spectra, metric={metric}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Quick UQ sanity checks (synthetic + optional file-based).")
    ap.add_argument("--pred_dir", type=str, default=None, help="Folder with fp_probs.pt and scores_ragged_<metric>.pt")
    ap.add_argument("--metric",  type=str, default="cosine")
    ap.add_argument("--temp",    type=float, default=None, help="Temperature for retrieval-UQ (if None, use 1.0)")
    args = ap.parse_args()

    synth_tests()
    if args.pred_dir:
        file_tests(Path(args.pred_dir), args.metric, args.temp)
