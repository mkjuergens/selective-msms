from __future__ import annotations
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ms_uq.inference import (
    EnsembleSampler,
    MCDropoutSampler,
    Predictor,
    head_probs_fn,
    save_ranker_from_model,
)
from ms_uq.models.registry import get_model_class
from ms_uq.utils import (
    discover_ensemble_ckpts,
    make_test_loader,
    make_train_val_test_loaders,
)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    dataset_tsv: str = ""
    helper_dir: str = ""
    gt_path: str = ""
    architecture: str = "mlp"
    candidate_setting: str = "formula"
    max_mz: float = 1005.0
    n_peaks: int = 128
    prec_mz_intensity: float = 1.1
    label_mode: str = "fingerprint"
    query_identity_source: str = "precomputed"
    missing_target_policy: str = "error"
    lazy_candidate_helpers: bool = False
    
    mode: str = "ensemble"
    ckpt: str = ""
    ckpts: str = ""
    ens_dir: str = ""
    ens_metric: str = "focal"
    passes: int = 50
    
    # Laplace
    laplace_samples: int = 50
    laplace_tau_w: float = 1.0
    laplace_tau_b: float = 1.0
    laplace_tune_prior: bool = True
    laplace_tune_method: str = "marglik"
    laplace_max_batches: Optional[int] = 200
    laplace_diagnostics: bool = True
    
    temperature: float = 0.003

    device: str = "cuda:0"
    batch_size: int = 256
    num_workers: int = 2
    bin_width: float = 0.1
    test_subset_size: Optional[int] = None
    overwrite: bool = False


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _loader_kwargs(config: EvalConfig) -> Dict:
    return {
        "architecture": config.architecture,
        "candidate_setting": config.candidate_setting,
        "max_mz": config.max_mz,
        "n_peaks": config.n_peaks,
        "prec_mz_intensity": config.prec_mz_intensity,
        "label_mode": config.label_mode,
        "query_identity_source": config.query_identity_source,
        "missing_target_policy": config.missing_target_policy,
        "lazy_candidate_helpers": config.lazy_candidate_helpers,
    }


def _temperature_tag(temperature: float) -> str:
    return f"T{float(temperature):g}".replace("-", "m").replace(".", "p")


def score_cache_name(prefix: str, aggregation: str, temperature: float) -> str:
    if aggregation == "probability":
        return f"scores_{prefix}_{aggregation}_{_temperature_tag(temperature)}.pt"
    return f"scores_{prefix}_{aggregation}.pt"


def generate_predictions(out_dir: Path, config: EvalConfig, loader: Optional[DataLoader] = None,
                         train_loader: Optional[DataLoader] = None, 
                         val_loader: Optional[DataLoader] = None) -> Tuple[Path, Optional[Path]]:
    """Generate fingerprint predictions from checkpoints."""
    try:
        from massspecgym.models.base import Stage
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if add_safe_globals is not None:
            add_safe_globals([Stage])
    except ImportError:
        pass
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_path = out_dir / "fp_probs.pt"
    ranker_path = out_dir / "ranker.pt"
    
    if fp_path.exists() and not config.overwrite:
        print(f"  Using cached {fp_path}")
        return fp_path, ranker_path if ranker_path.exists() else None
    
    if loader is None:
        loader = make_test_loader(
            config.dataset_tsv, config.helper_dir, config.bin_width,
            config.batch_size, config.num_workers, subset_size=config.test_subset_size,
            **_loader_kwargs(config),
        )
    
    mode = config.mode.lower()
    model_cls = get_model_class(config.architecture)
    if mode == "ensemble" and config.ckpt and not config.ckpts and not config.ens_dir:
        mode = "mcdo" if config.passes > 1 else "single"
    
    # Laplace: delegate entirely to laplace_bce module
    if mode == "laplace":
        from ms_uq.models.laplace_bce import generate_laplace_predictions, LaplaceConfig
        lp_cfg = LaplaceConfig(
            tau_w=config.laplace_tau_w,
            tau_b=config.laplace_tau_b,
            n_samples=config.laplace_samples,
            tune_prior=config.laplace_tune_prior,
            tune_method=config.laplace_tune_method,
            max_batches=config.laplace_max_batches,
            diagnostics=config.laplace_diagnostics,
        )
        return generate_laplace_predictions(
            out_dir=out_dir,
            ckpt=config.ckpt,
            test_loader=loader,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.device,
            overwrite=config.overwrite,
            cfg=lp_cfg,
            make_loaders_fn=lambda: make_train_val_test_loaders(
                config.dataset_tsv, config.helper_dir, config.bin_width,
                config.batch_size, config.num_workers,
                **_loader_kwargs(config),
            ),
            model_cls=model_cls,
            save_ranker_fn=save_ranker_from_model,
        )
    
    ckpt_for_ranker = None
    if mode == "mcdo":
        print(f"  Mode: MC Dropout ({config.passes} passes)")
        sampler = MCDropoutSampler(Path(config.ckpt), model_cls, passes=config.passes, device=config.device)
        ckpt_for_ranker = config.ckpt
    elif mode == "ensemble":
        if config.ckpts:
            ckpt_list = [Path(p.strip()) for p in config.ckpts.split(",") if p.strip()]
        elif config.ens_dir and config.ens_metric:
            ckpt_list = discover_ensemble_ckpts(config.ens_dir, config.ens_metric, prefer="best")
        else:
            raise ValueError("Ensemble requires --ckpts or (--ens_dir and --ens_metric)")
        print(f"  Mode: Ensemble ({len(ckpt_list)} members)")
        sampler = EnsembleSampler(ckpt_list, model_cls, mc_dropout_eval=False, device=config.device)
        ckpt_for_ranker = ckpt_list[0]
    elif mode == "single":
        print(f"  Mode: Single model")
        sampler = EnsembleSampler([Path(config.ckpt)], model_cls, mc_dropout_eval=False, device=config.device)
        ckpt_for_ranker = config.ckpt
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if ckpt_for_ranker:
        try:
            model = model_cls.load_from_checkpoint(ckpt_for_ranker, map_location=config.device)
            save_ranker_from_model(model, ranker_path)
            del model; _cleanup()
        except Exception:
            pass
    
    predictor = Predictor(sampler, head_probs_fn("loss.fp_pred_head", torch.sigmoid))
    predictor.predict_stack(loader, fp_path, save_every=100, overwrite=config.overwrite)
    del sampler, predictor; _cleanup()
    
    print(f"  Saved: {fp_path}")
    return fp_path, ranker_path if ranker_path.exists() else None
