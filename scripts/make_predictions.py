from __future__ import annotations
from pathlib import Path
import argparse
import sys
import gc
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ms_uq.utils import (
    make_test_loader,
    make_train_val_test_loaders,
    discover_ensemble_ckpts,
)
from ms_uq.inference import (
    Predictor,
    MCDropoutSampler,
    EnsembleSampler,
    head_probs_fn,
    save_ranker_from_model,
)
from ms_uq.inference.retrieve import scores_from_loader
from ms_uq.models.registry import get_model_class
from ms_uq.models.laplace_bce import LaplaceConfig, generate_laplace_predictions

try:
    from torch.serialization import add_safe_globals
    from massspecgym.models.base import Stage
    add_safe_globals([Stage])
except Exception:
    pass


def _cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def save_fp_probs(
    mode: str,
    ckpt: Optional[str],
    ckpts_csv: Optional[str],
    ens_dir: Optional[str],
    ens_metric: Optional[str],
    passes: int,
    device: str,
    dl: DataLoader,
    out_fp_path: Path,
    overwrite: bool,
    save_ranker: bool = True,
    model_cls=None,
) -> Optional[Path]:
    """
    Save fingerprint probabilities and optionally ranker weights.
    
    Parameters
    ----------
    mode : str
        'mcdo', 'ensemble', or 'single'
    ckpt : str, optional
        Single checkpoint path (for mcdo/single)
    ckpts_csv : str, optional
        Comma-separated checkpoint paths
    ens_dir : str, optional
        Ensemble directory
    ens_metric : str, optional
        Metric for ensemble checkpoint discovery
    passes : int
        MC dropout passes
    device : str
        Device to use
    dl : DataLoader
        Test dataloader
    out_fp_path : Path
        Output path for fingerprint probabilities
    overwrite : bool
        Overwrite existing files
    save_ranker : bool
        If True, also save ranker weights if model has one
    
    Returns
    -------
    ranker_path : Path or None
        Path to saved ranker, or None if no ranker
    """
    ranker_path = None
    if model_cls is None:
        model_cls = get_model_class("mlp")
    
    if out_fp_path.exists() and not overwrite:
        print(f"[predict] using cached {out_fp_path}")
        # Check if ranker already exists
        potential_ranker = out_fp_path.parent / "ranker.pt"
        if potential_ranker.exists():
            return potential_ranker
        return None

    if mode == "mcdo":
        if not ckpt:
            sys.exit("ERROR: --ckpt required for --mode mcdo")
        sampler = MCDropoutSampler(Path(ckpt), model_cls, passes=passes, device=device)
        # For mcdo, load model once to check for ranker
        if save_ranker:
            model = model_cls.load_from_checkpoint(ckpt, map_location=device)
            ranker_out = out_fp_path.parent / "ranker.pt"
            if save_ranker_from_model(model, ranker_out):
                ranker_path = ranker_out
            del model
            _cleanup()
            
    elif mode == "ensemble":
        if ckpts_csv:
            ckpt_list = [Path(p.strip()) for p in ckpts_csv.split(",") if p.strip()]
        elif ens_dir and ens_metric:
            ckpt_list = discover_ensemble_ckpts(ens_dir, ens_metric, prefer="best")
        else:
            sys.exit("ERROR: ensemble requires --ens_dir and --ens_metric (or --ckpts).")
        if not ckpt_list:
            sys.exit("ERROR: no ensemble checkpoints found.")
        print(f"[ensemble] {len(ckpt_list)} members discovered.")
        sampler = EnsembleSampler(ckpt_list, model_cls, mc_dropout_eval=False, device=device)
        # For ensemble, use first member's ranker (should be same architecture)
        if save_ranker and ckpt_list:
            model = model_cls.load_from_checkpoint(ckpt_list[0], map_location=device)
            ranker_out = out_fp_path.parent / "ranker.pt"
            if save_ranker_from_model(model, ranker_out):
                ranker_path = ranker_out
            del model
            _cleanup()
            
    else:  # single
        if not ckpt:
            sys.exit("ERROR: --ckpt required for --mode single")
        sampler = EnsembleSampler([Path(ckpt)], model_cls, mc_dropout_eval=False, device=device)
        if save_ranker:
            model = model_cls.load_from_checkpoint(ckpt, map_location=device)
            ranker_out = out_fp_path.parent / "ranker.pt"
            if save_ranker_from_model(model, ranker_out):
                ranker_path = ranker_out
            del model
            _cleanup()

    predictor = Predictor(sampler, head_probs_fn("loss.fp_pred_head", torch.sigmoid))
    predictor.predict_stack(dl, out_fp_path, save_every=100, overwrite=overwrite)

    del sampler, predictor
    _cleanup()
    
    return ranker_path


def save_fp_probs_laplace_bce(
    ckpt_path: str, device: str,
    train_dl: DataLoader, val_dl: Optional[DataLoader], test_dl: DataLoader,
    out_fp_path: Path, n_samples: int = 50,
    tau_w: float = 1.0, tau_b: float = 1.0,
    max_train_batches: int = 200, out_chunk: int = 512,
    prior_opt: str = "gridsearch",
    overwrite: bool = False,
    model_cls=None,
) -> None:
    """Save fingerprint probabilities using the current Laplace-BCE helper."""
    if out_fp_path.exists() and not overwrite:
        print(f"[laplace_bce] using cached {out_fp_path}")
        return

    if model_cls is None:
        model_cls = get_model_class("mlp")

    tune_method = "marglik" if prior_opt == "marglik" else "val_bce"
    cfg = LaplaceConfig(
        tau_w=tau_w,
        tau_b=tau_b,
        n_samples=n_samples,
        tune_prior=True,
        tune_method=tune_method,
        max_batches=max_train_batches,
        out_chunk=out_chunk,
    )

    generate_laplace_predictions(
        out_dir=out_fp_path.parent,
        ckpt=ckpt_path,
        test_loader=test_dl,
        train_loader=train_dl,
        val_loader=val_dl,
        device=device,
        overwrite=overwrite,
        cfg=cfg,
        save_ranker_fn=save_ranker_from_model,
        model_cls=model_cls,
    )

def save_scores(
    fp_path: Path,
    out_dir: Path,
    dl: DataLoader,
    metric: str = "cosine",
    aggregation: str = "score",
    temperature: float = 1.0,
    overwrite: bool = False,
    ranker_path: Optional[Path] = None,
    device: str = "cpu",
) -> Path:
    """
    Compute and save retrieval scores with specified aggregation.
    
    Parameters
    ----------
    fp_path : Path
        Path to fingerprint probabilities file.
    out_dir : Path
        Output directory.
    dl : DataLoader
        Test dataloader.
    metric : str
        Similarity metric: 'cosine', 'tanimoto', or 'iou'.
        Ignored if ranker_path is provided.
    aggregation : str
        'score': Average similarity scores across samples
        'fingerprint': Average fingerprints, then compute similarity
        'probability': Average softmax probabilities (uses temperature)
    temperature : float
        Softmax temperature for 'probability' aggregation.
        Higher = softer distribution, lower = sharper.
    overwrite : bool
        Overwrite existing files.
    ranker_path : Path, optional
        Path to saved ranker weights. If provided, uses learned
        similarity instead of metric-based similarity.
    device : str
        Device for ranker computation.
    
    Returns
    -------
    Path to saved scores file.
    """
    from ms_uq.inference import load_ranker
    
    # Check for ranker
    ranker = None
    use_ranker = False
    
    # Auto-detect ranker in same directory as fp_probs
    if ranker_path is None:
        potential_ranker = fp_path.parent / "ranker.pt"
        if potential_ranker.exists():
            ranker_path = potential_ranker
    
    if ranker_path is not None and ranker_path.exists():
        ranker = load_ranker(ranker_path, device=device)
        if ranker is not None:
            use_ranker = True
            print(f"[scores] Using learned ranker from {ranker_path}")
    
    # Build output filename
    if use_ranker:
        base_name = f"scores_ragged_ranker_{aggregation}"
    else:
        base_name = f"scores_ragged_{metric}_{aggregation}"
    
    if aggregation == "probability":
        out_path = out_dir / f"{base_name}_T{temperature}.pt"
    else:
        out_path = out_dir / f"{base_name}.pt"
    
    if out_path.exists() and not overwrite:
        print(f"[scores] using cached {out_path}")
        return out_path
    
    # Load fingerprint probabilities
    data = torch.load(fp_path, map_location="cpu")
    fp_probs = (data["stack"] if isinstance(data, dict) else data).float()
    
    score_method = "ranker" if use_ranker else metric
    print(f"[scores] computing {score_method} scores with '{aggregation}' aggregation" + 
          (f" (T={temperature})" if aggregation == "probability" else "") + "...")
    
    result = scores_from_loader(
        fp_probs=fp_probs,
        loader=dl,
        metric=metric,
        aggregation=aggregation,
        temperature=temperature,
        return_labels=True,
        return_per_sample=True,
        show_progress=True,
        ranker=ranker,
        device=device,
    )
    
    # Store temperature in result for reference
    result["temperature"] = temperature
    if use_ranker:
        result["ranker_path"] = str(ranker_path)
    
    torch.save(result, out_path)
    print(f"[scores] saved {out_path}")
    
    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ... existing arguments ...
    ap.add_argument("--mode", choices=["mcdo", "ensemble", "single", "laplace_bce"], required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--ckpts")
    ap.add_argument("--ens_dir")
    ap.add_argument("--ens_metric")
    ap.add_argument("--passes", type=int, default=50)

    # Laplace BCE options
    ap.add_argument("--laplace_samples", type=int, default=50)
    ap.add_argument("--la_prior_opt", choices=["marglik", "gridsearch", "CV"], default="gridsearch")
    ap.add_argument("--la_tau_w", type=float, default=1.0)
    ap.add_argument("--la_tau_b", type=float, default=1.0)
    ap.add_argument("--la_max_train_batches", type=int, default=200)
    ap.add_argument("--la_out_chunk", type=int, default=512)

    # Data
    ap.add_argument("--dataset_tsv", required=True)
    ap.add_argument("--helper_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--architecture", choices=["mlp", "transformer"], default="mlp")
    ap.add_argument("--candidate_setting", choices=["formula", "mass"], default="formula")

    # Runtime
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--bin_width", type=float, default=0.1)
    ap.add_argument("--max_mz", type=float, default=1005.0)
    ap.add_argument("--n_peaks", type=int, default=128)
    ap.add_argument("--prec_mz_intensity", type=float, default=1.1)
    ap.add_argument("--pin_memory", action="store_true")

    # Scoring
    ap.add_argument("--metric", choices=["cosine", "tanimoto", "iou"], default="cosine")
    ap.add_argument("--aggregation", choices=["score", "fingerprint", "probability"], default="score",
                    help="Ensemble aggregation: 'score' (avg scores), 'fingerprint' (avg FPs), "
                         "'probability' (avg softmax probs, uses --temperature)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for 'probability' aggregation. "
                         "Lower = sharper (more confident), higher = softer.")
    ap.add_argument("--overwrite", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_path = out_dir / "fp_probs.pt"

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Aggregation: {args.aggregation}" + 
          (f" (T={args.temperature})" if args.aggregation == "probability" else ""))
    print(f"Metric: {args.metric}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    # ... existing loader and fp_probs code ...

    model_cls = get_model_class(args.architecture)
    loader_kwargs = dict(
        architecture=args.architecture,
        candidate_setting=args.candidate_setting,
        max_mz=args.max_mz,
        n_peaks=args.n_peaks,
        prec_mz_intensity=args.prec_mz_intensity,
    )

    # Create loaders and compute fingerprint probabilities
    if args.mode == "laplace_bce":
        train_dl, val_dl, test_dl = make_train_val_test_loaders(
            args.dataset_tsv, args.helper_dir, args.bin_width,
            args.batch_size, args.num_workers, args.pin_memory,
            **loader_kwargs
        )
        save_fp_probs_laplace_bce(
            args.ckpt, args.device, train_dl, val_dl, test_dl, fp_path,
            n_samples=args.laplace_samples, tau_w=args.la_tau_w,
            tau_b=args.la_tau_b, max_train_batches=args.la_max_train_batches,
            out_chunk=args.la_out_chunk, prior_opt=args.la_prior_opt,
            overwrite=args.overwrite,
            model_cls=model_cls
        )
        dl = test_dl
    else:
        dl = make_test_loader(
            args.dataset_tsv, args.helper_dir, args.bin_width,
            args.batch_size, args.num_workers, args.pin_memory,
            **loader_kwargs
        )
        save_fp_probs(
            args.mode, args.ckpt, args.ckpts, args.ens_dir, args.ens_metric,
            args.passes, args.device, dl, fp_path, args.overwrite,
            model_cls=model_cls
        )

    # Verify fp_probs
    P = torch.load(fp_path, map_location="cpu")
    Pstack = P["stack"] if isinstance(P, dict) else P
    print(f"[ok] fp_probs: shape {tuple(Pstack.shape)}")
    del P, Pstack
    gc.collect()

    # Compute scores with aggregation
    scores_path = save_scores(
        fp_path, out_dir, dl,
        metric=args.metric,
        aggregation=args.aggregation,
        temperature=args.temperature,
        overwrite=args.overwrite,
    )

    D = torch.load(scores_path, map_location="cpu")
    print(f"[ok] scores: aggregated shape {tuple(D['scores_flat'].shape)}")
    if "scores_stack_flat" in D:
        print(f"     per-sample shape {tuple(D['scores_stack_flat'].shape)}")
    if "temperature" in D:
        print(f"     temperature: {D['temperature']}")

    print(f"\n[done] Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
