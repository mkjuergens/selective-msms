from __future__ import annotations
import gc
import json
import shutil
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
    laplace_state: str = ""
    
    temperature: float = 0.003

    device: str = "cuda:0"
    batch_size: int = 256
    num_workers: int = 2
    bin_width: float = 0.1
    test_subset_size: Optional[int] = None
    overwrite: bool = False
    seed: int = 42


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
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
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
            prediction_seed=config.seed,
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
            state_path=Path(config.laplace_state) if config.laplace_state else None,
        )
    
    ckpt_for_ranker = None
    if mode == "mcdo":
        print(f"  Mode: MC Dropout ({config.passes} passes)")
        sampler = MCDropoutSampler(Path(config.ckpt), model_cls, passes=config.passes, device=config.device, seed=config.seed)
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

@dataclass(frozen=True)
class PaperPredictionSpec:
    model: str
    dataset_split: str
    output_split: str
    architecture: str
    candidate_setting: str
    mode: str
    samples: int


PAPER_PREDICTION_SPECS = (
    PaperPredictionSpec("ensemble_mlp_formula", "val", "validation", "mlp", "formula", "ensemble", 5),
    PaperPredictionSpec("ensemble_mlp_formula", "test", "test", "mlp", "formula", "ensemble", 5),
    PaperPredictionSpec("ensemble_transformer_formula", "val", "validation", "transformer", "formula", "ensemble", 5),
    PaperPredictionSpec("ensemble_transformer_formula", "test", "test", "transformer", "formula", "ensemble", 5),
    PaperPredictionSpec("ensemble_mlp_mass", "test", "test", "mlp", "mass", "ensemble", 5),
    PaperPredictionSpec("mc_dropout_mlp_formula", "test", "test", "mlp", "formula", "mcdo", 50),
    PaperPredictionSpec("laplace_mlp_formula", "test", "test", "mlp", "formula", "laplace", 50),
)


def paper_prediction_path(data_dir: Path, spec: PaperPredictionSpec) -> Path:
    return data_dir / "models" / spec.model / "predictions" / spec.output_split / "fp_probs.pt"


def missing_paper_predictions(data_dir: Path) -> list[PaperPredictionSpec]:
    """Return paper prediction tensors that have not been generated locally."""
    return [spec for spec in PAPER_PREDICTION_SPECS if not paper_prediction_path(data_dir, spec).is_file()]

def regenerate_paper_predictions(
    data_dir: Path,
    massspecgym_data: Path,
    *,
    device: str = "cuda:0",
    batch_size: int = 256,
    num_workers: int = 2,
    seed: int = 42,
    overwrite: bool = False,
) -> list[dict]:
    """Generate the seven paper prediction tensors from released checkpoints."""
    from ms_uq.paper.release import discover_checkpoint_sources

    data_dir = data_dir.resolve()
    massspecgym_data = massspecgym_data.resolve()
    indexed = discover_checkpoint_sources(data_dir)
    by_model: dict[str, list[dict]] = {}
    for item in indexed:
        by_model.setdefault(item["model"], []).append(item)
    for items in by_model.values():
        items.sort(key=lambda item: item["member"])

    loaders: dict[tuple[str, str], dict[str, DataLoader]] = {}
    rows = []
    shared_ranker = data_dir / "models/shared/ranker.pt"

    for spec in PAPER_PREDICTION_SPECS:
        output = paper_prediction_path(data_dir, spec)
        if output.is_file() and not overwrite:
            rows.append({
                "model": spec.model,
                "split": spec.output_split,
                "path": output.relative_to(data_dir).as_posix(),
                "size_bytes": output.stat().st_size,
                "samples": spec.samples,
                "seed": seed,
                "stochastic": spec.mode in {"mcdo", "laplace"},
            })
            continue
        model_files = [item["path"] for item in by_model[spec.model]]
        laplace_state = ""
        if spec.mode == "laplace":
            checkpoints = [path for path in model_files if path.suffix == ".ckpt"]
            states = [path for path in model_files if path.name == "laplace_state.pt"]
            if len(checkpoints) != 1 or len(states) != 1:
                raise RuntimeError("Laplace release must contain one base checkpoint and one state file")
            ckpt = str(checkpoints[0])
            ckpts = ""
            laplace_state = str(states[0])
        elif spec.mode == "ensemble":
            ckpt = ""
            ckpts = ",".join(map(str, model_files))
        else:
            if len(model_files) != 1:
                raise RuntimeError(f"{spec.model} requires exactly one checkpoint")
            ckpt = str(model_files[0])
            ckpts = ""

        config = EvalConfig(
            dataset_tsv=str(massspecgym_data / "MassSpecGym.tsv"),
            helper_dir=str(massspecgym_data),
            architecture=spec.architecture,
            candidate_setting=spec.candidate_setting,
            label_mode="inchikey" if spec.candidate_setting == "mass" else "fingerprint",
            mode=spec.mode,
            ckpt=ckpt,
            ckpts=ckpts,
            passes=spec.samples,
            laplace_samples=spec.samples,
            laplace_state=laplace_state,
            laplace_tune_prior=False if laplace_state else True,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            overwrite=overwrite,
            seed=seed,
        )
        cache_key = (spec.architecture, spec.candidate_setting)
        if cache_key not in loaders:
            train_loader, val_loader, test_loader = make_train_val_test_loaders(
                config.dataset_tsv,
                config.helper_dir,
                config.bin_width,
                config.batch_size,
                config.num_workers,
                **_loader_kwargs(config),
            )
            loaders[cache_key] = {
                "train": train_loader,
                "val": val_loader,
                "test": test_loader,
            }
        loader = loaders[cache_key][spec.dataset_split]
        prediction_path, ranker_path = generate_predictions(output.parent, config, loader=loader)
        if ranker_path is not None and Path(ranker_path).is_file() and not shared_ranker.exists():
            shared_ranker.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ranker_path, shared_ranker)
        rows.append({
            "model": spec.model,
            "split": spec.output_split,
            "path": prediction_path.relative_to(data_dir).as_posix(),
            "size_bytes": prediction_path.stat().st_size,
            "samples": spec.samples,
            "seed": seed,
            "stochastic": spec.mode in {"mcdo", "laplace"},
        })

    index_path = data_dir / "models/predictions.json"
    index_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows
