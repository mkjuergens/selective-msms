from pathlib import Path
import pytest
import torch
import argparse
import sys
from tqdm import tqdm

from ms_uq.inference.predictor import (
    Predictor, MCDropoutSampler, EnsembleSampler, head_probs_fn
)
from ms_uq.inference.retrieve import (
    scores_ragged_from_loader, ragged_softmax
)
from ms_uq.models.fingerprint_mlp import FingerprintPredicter
from ms_uq.data import RetrievalDataset_PrecompFPandInchi
from massspecgym.data.data_module import MassSpecDataModule
from massspecgym.data.transforms import SpecBinner, MolFingerprinter


def pytest_addoption(parser):
    g = parser.getgroup("msuq")
    g.addoption("--mode", choices=["mcdo", "ensemble"], required=True)
    g.addoption("--ckpt", default=None, help="Single checkpoint for mcdo.")
    g.addoption("--ckpts", default=None, help="Comma-separated ckpts for ensemble.")
    g.addoption("--passes", type=int, default=10, help="MC-Dropout samples (default 10 for quick test).")

    g.addoption("--dataset_tsv", required=True)
    g.addoption("--helper_dir", required=True)
    g.addoption("--out_dir", required=True)

    g.addoption("--device", default="cuda:0")
    g.addoption("--batch_size", type=int, default=128)
    g.addoption("--num_workers", type=int, default=2)
    g.addoption("--bin_width", type=float, default=0.1)

    g.addoption("--metric", choices=["cosine", "tanimoto"], default="cosine")
    g.addoption("--temperature", type=float, default=None,
                help="If set, also compute ragged candidate probabilities at this T.")
    g.addoption("--n_bits", type=int, default=None, help="Expected K (e.g. 4096).")


@pytest.fixture(scope="session")
def opts(request):
    o = request.config.getoption
    return {
        "mode": o("--mode"),
        "ckpt": o("--ckpt"),
        "ckpts": o("--ckpts"),
        "passes": int(o("--passes")),
        "dataset_tsv": o("--dataset_tsv"),
        "helper_dir": o("--helper_dir"),
        "out_dir": str(Path(o("--out_dir")).expanduser().resolve()),
        "device": o("--device"),
        "batch_size": int(o("--batch_size")),
        "num_workers": int(o("--num_workers")),
        "bin_width": float(o("--bin_width")),
        "metric": o("--metric"),
        "temperature": o("--temperature"),
        "n_bits": o("--n_bits"),
    }


def _make_test_loader(dataset_tsv: str, helper_dir: str, bw: float, bs: int, nw: int):
    ds = RetrievalDataset_PrecompFPandInchi(
        spec_transform=SpecBinner(max_mz=1005, bin_width=bw, to_rel_intensities=True),
        mol_transform=MolFingerprinter(fp_size=4096),
        pth=dataset_tsv,
        fp_pth=Path(helper_dir) / "fp_4096.npy",
        inchi_pth=Path(helper_dir) / "inchis.npy",
        candidates_pth=Path(helper_dir) / "MassSpecGym_retrieval_candidates_formula.json",
        candidates_fp_pth=Path(helper_dir) / "MassSpecGym_retrieval_candidates_formula_fps.npz",
        candidates_inchi_pth=Path(helper_dir) / "MassSpecGym_retrieval_candidates_formula_inchi.npz",
    )
    dm = MassSpecDataModule(dataset=ds, batch_size=bs, num_workers=nw)
    dm.prepare_data()
    dm.setup(stage="test")
    return dm.test_dataloader()  # sequential sampler (required for ragged alignment)


def check_bits(P: torch.Tensor, n_bits: int | None):
    assert P.ndim == 3, f"bitwise probs must be (N,S,K), got {tuple(P.shape)}"
    N, S, K = P.shape
    assert N > 0 and S > 0 and K > 0
    if n_bits is not None:
        assert K == n_bits, f"expected K={n_bits}, got {K}"
    assert torch.isfinite(P).all(), "NaN/Inf in bitwise probabilities"
    assert (P >= 0).all() and (P <= 1).all(), "bitwise probabilities not in [0,1]"
    return N, S, K

def check_scores_ragged(D: dict, N: int, S: int):
    Pflat = D["scores_stack_flat"]
    ptr   = D["ptr"].long()
    ids   = D["cand_ids"].long()
    assert Pflat.ndim == 2 and ptr.ndim == 1 and ids.ndim == 1
    S2, SM = Pflat.shape
    assert S2 == S, f"sample count mismatch scores S={S2} vs bits S={S}"
    assert ptr.numel() == N + 1 and ptr[0].item() == 0 and ptr[-1].item() == SM
    assert ids.numel() == SM
    assert torch.isfinite(Pflat).all(), "NaN/Inf in ragged scores"
    return S2, SM, ptr, ids


def check_probs_ragged(Pflat: torch.Tensor, ptr: torch.Tensor):
    assert torch.isfinite(Pflat).all()
    assert (Pflat >= 0).all() and (Pflat <= 1 + 1e-6).all()
    max_err = 0.0
    N = ptr.numel() - 1
    for n in range(N):
        s, e = int(ptr[n]), int(ptr[n+1])
        max_err = max(max_err, float((Pflat[:, s:e].sum(1) - 1.0).abs().max()))
    assert max_err < 1e-4, f"ragged candidate prob sums not ~1 (max |sum-1|={max_err})"



def parse_args():
    ap = argparse.ArgumentParser("MC-Dropout / Ensemble → (N,S,K) + ragged candidate scores (+probs) + checks")
    ap.add_argument("--mode", choices=["mcdo", "ensemble"], required=True)
    ap.add_argument("--ckpt",  default=None, help="Single checkpoint for mcdo.")
    ap.add_argument("--ckpts", default=None, help="Comma-separated checkpoints for ensemble.")
    ap.add_argument("--passes", type=int, default=50, help="MC-Dropout samples (S).")
    ap.add_argument("--dataset_tsv", required=True)
    ap.add_argument("--helper_dir",  required=True)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--bin_width", type=float, default=0.1)
    ap.add_argument("--metric", choices=["cosine","tanimoto"], default="cosine")
    ap.add_argument("--temperature", type=float, default=None,
                    help="If set, also compute ragged candidate probabilities at this T.")
    ap.add_argument("--n_bits", type=int, default=None, help="Expected K (e.g., 4096).")
    ap.add_argument("--skip_predict", action="store_true",
                    help="Skip predicting; only check existing files in --out_dir.")
    return ap.parse_args()


def main():
    a = parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_path = out_dir / "fp_probs.pt"
    scores_path = out_dir / f"scores_ragged_{a.metric}.pt"

    # dataloader
    dl = _make_test_loader(a.dataset_tsv, a.helper_dir, a.bin_width, a.batch_size, a.num_workers)

    # (1) predict (N,S,K)
    if not a.skip_predict and not fp_path.exists():
        if a.mode == "mcdo":
            if not a.ckpt:
                print("ERROR: --ckpt required for --mode mcdo", file=sys.stderr); sys.exit(2)
            sampler = MCDropoutSampler(Path(a.ckpt), FingerprintPredicter,
                                       passes=a.passes, device=a.device)
        else:
            if not a.ckpts:
                print("ERROR: --ckpts required for --mode ensemble", file=sys.stderr); sys.exit(2)
            ckpts = [Path(p.strip()) for p in a.ckpts.split(",") if p.strip()]
            if not ckpts:
                print("ERROR: no valid checkpoints parsed from --ckpts", file=sys.stderr); sys.exit(2)
            sampler = EnsembleSampler(ckpts, FingerprintPredicter, mc_dropout_eval=False, device=a.device)

        predictor = Predictor(sampler, head_probs_fn("loss.fp_pred_head", torch.sigmoid))
        # wrap dl with tqdm so you see per-batch progress
        for s_idx, (_name, model) in enumerate(tqdm(sampler, total=len(sampler), desc="samples")):
            rows = []
            for batch in tqdm(dl, desc=f"predict {s_idx}", leave=False):
                with torch.no_grad():
                    rows.append(predictor.predict_fn(model, batch).cpu())
        predictor.predict_stack(dl, fp_path)
    assert fp_path.exists(), f"missing {fp_path}"

    # check fp_probs
    Pbits = torch.load(fp_path, map_location="cpu")["stack"].float()
    N, S, K = check_bits(Pbits, a.n_bits)
    print(f"[ok] fp_probs: (N,S,K)=({N},{S},{K})")

    # (3) ragged candidate scores
    if not scores_path.exists():
        print("[info] computing ragged candidate scores…")
        out = scores_ragged_from_loader(fp_path,
                                        tqdm(dl, desc="scoring", leave=False),
                                        metric=a.metric, outfile=scores_path)
        if isinstance(out, dict):
            torch.save(out, scores_path)
    Dscores = torch.load(scores_path, map_location="cpu")
    _, SM, ptr, ids = check_scores_ragged(Dscores, N, S)
    print(f"[ok] scores_ragged_{a.metric}: (S,ΣM)=({S},{SM}), ptr len={ptr.numel()}, ids len={ids.numel()}")

    # (4) ragged candidate probabilities at T
    if a.temperature is not None:
        print(f"[info] computing ragged candidate probabilities at T={a.temperature}…")
        Pflat = ragged_softmax(Dscores["scores_stack_flat"], ptr, float(a.temperature))
        check_probs_ragged(Pflat, ptr)
        probs_path = out_dir / f"cand_probs_ragged_{a.metric}.pt"
        torch.save({"probs_stack_flat": Pflat.half(), "ptr": ptr, "cand_ids": ids,
                    "temperature": float(a.temperature)}, probs_path)
        print(f"[ok] cand_probs_ragged_{a.metric}: saved with T={a.temperature}")

    print("[done] all checks passed.")




if __name__ == "__main__":
    main()
