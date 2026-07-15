"""
Helper functions for data loading, preprocessing, and common utilities.
"""

from __future__ import annotations
import json
import re
from typing import Iterable, Optional, Tuple, Union, List
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from ms_uq.data import RetrievalDataset_PrecompFPandInchi
from massspecgym.data.transforms import SpecBinner, MolFingerprinter, SpecTokenizer
from massspecgym.data.data_module import MassSpecDataModule
from ms_uq.models.registry import normalize_architecture


CANDIDATE_SETTINGS = (
    "formula",
    "mass",
    "formula_uncapped",
    "formula_pubchem_capped256",
    "formula_pubchem_record_capped256",
    "formula_pubchem_natural_capped256",
)


def normalize_candidate_setting(candidate_setting: str | None) -> str:
    setting = (candidate_setting or "formula").lower()
    if setting not in CANDIDATE_SETTINGS:
        raise ValueError(
            f"Unknown candidate_setting '{candidate_setting}'. "
            f"Expected one of {CANDIDATE_SETTINGS}."
        )
    return setting


def _first_existing(
    helper_dir: Path,
    names: List[str],
    description: str,
    candidate_setting: str,
) -> Path:
    for name in names:
        path = helper_dir / name
        if path.exists():
            return path
    tried = ", ".join(str(helper_dir / name) for name in names)
    raise FileNotFoundError(
        f"Missing {description} for candidate_setting='{candidate_setting}'. "
        f"Tried: {tried}"
    )


def resolve_candidate_paths(
    helper_dir: Union[str, Path],
    candidate_setting: str = "formula",
) -> Tuple[Path, Path, Path]:
    """Resolve candidate JSON, fingerprint, and InChI helper files."""
    helper_dir = Path(helper_dir)
    setting = normalize_candidate_setting(candidate_setting)

    candidates_pth = _first_existing(
        helper_dir,
        [f"MassSpecGym_retrieval_candidates_{setting}.json"],
        "candidate JSON",
        setting,
    )

    fp_names = [
        f"MassSpecGym_retrieval_candidates_{setting}_fps.npz",
        f"morgan_2_4096_{setting}cands.npz",
    ]
    if setting == "formula":
        fp_names.append("MassSpecGym_retrieval_candidates_formula_fps_old.npz")

    candidates_fp_pth = _first_existing(
        helper_dir,
        fp_names,
        "candidate fingerprint NPZ",
        setting,
    )

    candidates_inchi_pth = _first_existing(
        helper_dir,
        [
            f"MassSpecGym_retrieval_candidates_{setting}_inchi.npz",
            f"Inchis_{setting}cands.npz",
        ],
        "candidate InChI NPZ",
        setting,
    )
    return candidates_pth, candidates_fp_pth, candidates_inchi_pth


def make_spec_transform(
    architecture: str = "mlp",
    bin_width: float = 0.1,
    max_mz: float = 1005.0,
    n_peaks: int = 128,
    prec_mz_intensity: Optional[float] = 1.1,
):
    """Create the spectrum transform required by the selected architecture."""
    arch = normalize_architecture(architecture)
    if arch == "mlp":
        return SpecBinner(max_mz=max_mz, bin_width=bin_width, to_rel_intensities=True)
    return SpecTokenizer(
        n_peaks=n_peaks,
        prec_mz_intensity=prec_mz_intensity,
        matchms_kwargs={"mz_to": max_mz},
    )


def create_dataset(
    dataset_tsv: Union[str, Path],
    helper_dir: Union[str, Path],
    bin_width: float = 0.1,
    fp_size: int = 4096,
    max_mz: float = 1005.0,
    architecture: str = "mlp",
    candidate_setting: str = "formula",
    n_peaks: int = 128,
    prec_mz_intensity: Optional[float] = 1.1,
    label_mode: str = "fingerprint",
    query_identity_source: str = "precomputed",
    missing_target_policy: str = "error",
    lazy_candidate_helpers: bool = False,
) -> RetrievalDataset_PrecompFPandInchi:
    """Create retrieval dataset with precomputed fingerprints and candidates."""
    helper_dir = Path(helper_dir)
    candidates_pth, candidates_fp_pth, candidates_inchi_pth = resolve_candidate_paths(
        helper_dir, candidate_setting=candidate_setting
    )

    return RetrievalDataset_PrecompFPandInchi(
        spec_transform=make_spec_transform(
            architecture=architecture,
            bin_width=bin_width,
            max_mz=max_mz,
            n_peaks=n_peaks,
            prec_mz_intensity=prec_mz_intensity,
        ),
        mol_transform=MolFingerprinter(fp_size=fp_size),
        pth=str(dataset_tsv),
        fp_pth=helper_dir / "fp_4096.npy",
        inchi_pth=helper_dir / "inchis.npy",
        candidates_pth=candidates_pth,
        candidates_fp_pth=candidates_fp_pth,
        candidates_inchi_pth=candidates_inchi_pth,
        label_mode=label_mode,
        query_identity_source=query_identity_source,
        missing_target_policy=missing_target_policy,
        lazy_candidate_helpers=lazy_candidate_helpers,
    )

def _worker_init_fn(worker_id: int):
    """
    Worker initialization function for deterministic data loading.
    
    Sets the numpy and random seeds for each worker based on the base seed
    to ensure consistent iteration order across multiple runs.
    """
    import numpy as np
    import random
    # Use a fixed base seed + worker_id for reproducibility
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_test_loader(
    dataset_tsv: Union[str, Path],
    helper_dir: Union[str, Path],
    bin_width: float = 0.1,
    batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = False,
    subset_size: Optional[int] = None,
    architecture: str = "mlp",
    candidate_setting: str = "formula",
    max_mz: float = 1005.0,
    n_peaks: int = 128,
    prec_mz_intensity: Optional[float] = 1.1,
    label_mode: str = "fingerprint",
    query_identity_source: str = "precomputed",
    missing_target_policy: str = "error",
    lazy_candidate_helpers: bool = False,
) -> DataLoader:
    """
    Create test dataloader with memory-efficient and DETERMINISTIC settings.
    
    IMPORTANT: This loader is designed to yield batches in the same order
    every time it is iterated, which is essential for alignment between
    prediction generation and scoring.
    """
    ds = create_dataset(
        dataset_tsv, helper_dir, bin_width,
        architecture=architecture,
        candidate_setting=candidate_setting,
        max_mz=max_mz,
        n_peaks=n_peaks,
        prec_mz_intensity=prec_mz_intensity,
        label_mode=label_mode,
        query_identity_source=query_identity_source,
        missing_target_policy=missing_target_policy,
        lazy_candidate_helpers=lazy_candidate_helpers,
    )
    dm = MassSpecDataModule(dataset=ds, batch_size=batch_size, num_workers=num_workers)
    dm.prepare_data()
    dm.setup(stage="test")
    
    dl_orig = dm.test_dataloader()
    dataset = dl_orig.dataset
    if subset_size is not None:
        n = min(int(subset_size), len(dataset))
        dataset = Subset(dataset, range(n))
    
    # Create a generator with fixed seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(42)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=dl_orig.collate_fn,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=generator,
    )


def make_train_val_test_loaders(
    dataset_tsv: Union[str, Path],
    helper_dir: Union[str, Path],
    bin_width: float = 0.1,
    batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = False,
    architecture: str = "mlp",
    candidate_setting: str = "formula",
    max_mz: float = 1005.0,
    n_peaks: int = 128,
    prec_mz_intensity: Optional[float] = 1.1,
    label_mode: str = "fingerprint",
    query_identity_source: str = "precomputed",
    missing_target_policy: str = "error",
    lazy_candidate_helpers: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders for methods requiring fitting."""
    ds = create_dataset(
        dataset_tsv, helper_dir, bin_width,
        architecture=architecture,
        candidate_setting=candidate_setting,
        max_mz=max_mz,
        n_peaks=n_peaks,
        prec_mz_intensity=prec_mz_intensity,
        label_mode=label_mode,
        query_identity_source=query_identity_source,
        missing_target_policy=missing_target_policy,
        lazy_candidate_helpers=lazy_candidate_helpers,
    )
    dm = MassSpecDataModule(dataset=ds, batch_size=batch_size, num_workers=num_workers)
    dm.prepare_data()
    
    dm.setup(stage="fit")
    train_dl_orig = dm.train_dataloader()
    val_dl_orig = dm.val_dataloader()
    
    dm.setup(stage="test")
    test_dl_orig = dm.test_dataloader()
    
    def _wrap(dl, shuffle=False):
        generator = torch.Generator()
        generator.manual_seed(42)
        return DataLoader(
            dl.dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=False, collate_fn=dl.collate_fn,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            generator=generator,
        )
    
    return _wrap(train_dl_orig, True), _wrap(val_dl_orig), _wrap(test_dl_orig)


# =============================================================================
# Prediction Loading
# =============================================================================

def load_predictions(
    pred_dir: Union[str, Path],
    metric: str = "cosine",
    aggregation: str = None,
    temperature: float = None,
    require_scores: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Load fingerprint probabilities and optionally scores.
    
    Parameters
    ----------
    pred_dir : Path
        Directory containing prediction files.
    metric : str
        Similarity metric used.
    aggregation : str, optional
        Aggregation method: 'score', 'fingerprint', or 'probability'.
    temperature : float, optional
        Temperature used (only relevant for 'probability' aggregation).
    require_scores : bool
        If True, raise error when no scores found. If False, return None for scores.
    
    Returns
    -------
    Pbits : (N, S, K) bit probabilities or None
    scores_agg : (M,) aggregated scores or None
    scores_stack : (S, M) per-sample scores or None
    ptr : (N+1,) pointers or None
    """
    pred_dir = Path(pred_dir)
    
    # Load bit probabilities
    fp_path = pred_dir / "fp_probs.pt"
    Pbits = None
    if fp_path.exists():
        data = torch.load(fp_path, map_location="cpu")
        Pbits = (data["stack"] if isinstance(data, dict) else data).float()
        print(f"  Loaded fingerprints: {fp_path.name} (shape: {Pbits.shape})")
    
    # Find scores file (try most specific first)
    candidates = []
    if aggregation == "probability" and temperature is not None:
        candidates.append(f"scores_ragged_{metric}_{aggregation}_T{temperature}.pt")
    if aggregation:
        candidates.append(f"scores_ragged_{metric}_{aggregation}.pt")
    candidates.extend([f"scores_ragged_{metric}.pt", f"scores_{metric}.pt"])
    
    scores_path = None
    for name in candidates:
        if (pred_dir / name).exists():
            scores_path = pred_dir / name
            break
    
    if scores_path is None:
        if require_scores:
            raise FileNotFoundError(f"No scores file found in {pred_dir}. Tried: {candidates}")
        else:
            print(f"  No scores file found in {pred_dir} - will compute on demand")
            return Pbits, None, None, None
    
    print(f"  Loading scores from: {scores_path.name}")
    data = torch.load(scores_path, map_location="cpu")
    
    if isinstance(data, dict):
        # New format from scores_from_loader
        scores_agg = data.get("scores_flat", data.get("scores_stack_flat"))
        scores_stack = data.get("scores_stack_flat")
        ptr = data["ptr"]
        
        # If scores_flat is 2D (old format), it's the stack
        if scores_agg.dim() == 2:
            scores_stack = scores_agg
            scores_agg = scores_agg.mean(dim=0)
    else:
        # Legacy format
        scores_agg = data.mean(dim=0) if data.dim() == 2 else data
        scores_stack = data if data.dim() == 2 else None
        ptr = torch.load(pred_dir / "ptr.pt")
    
    return Pbits, scores_agg.float(), scores_stack, ptr.long()


def load_ground_truth(gt_path: Union[str, Path]) -> Tuple[Tensor, Optional[Tensor]]:
    """Load ground-truth fingerprints and optional legacy retrieval labels."""
    GT = torch.load(gt_path, map_location="cpu")
    return GT["y_bits"], GT.get("labels_flat")


def load_candidate_stats(stats_path: Union[str, Path]) -> Tensor:
    """Load candidate set statistics."""
    stats = torch.load(stats_path, map_location="cpu")
    if isinstance(stats, dict):
        return stats.get('n_candidates', stats.get('n_cands'))
    return stats


def discover_ensemble_ckpts(
    ens_dir: Union[str, Path],
    metric: str,
    prefer: str = "best"
) -> List[Path]:
    """Discover ensemble member checkpoints."""
    root = Path(ens_dir)
    
    members_dir = root / "members"
    if not members_dir.exists():
        if re.search(r"(member_\d{3}|model)$", root.as_posix()):
            members_dir = root.parent
        else:
            members_dir = root

    member_dirs = sorted([p for p in members_dir.glob("member_*") if p.is_dir()])
    if not member_dirs:
        member_dirs = sorted([p for p in root.glob("member_*") if p.is_dir()])

    ckpts = []
    for md in member_dirs:
        ckdir = md / "ckpts" / metric
        if not ckdir.exists():
            continue
        
        best = ckdir / "best.ckpt"
        if prefer == "best" and best.exists():
            ckpts.append(best)
        else:
            candidates = sorted(ckdir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                ckpts.append(candidates[0])

    return ckpts


def best_ckpt(
    run_dir: Union[str, Path],
    prefer: Iterable[str] = ("contiou", "tanim", "cossim", "reranker"),
    fallback_prefix: Optional[str] = None
) -> Path:
    """Select best checkpoint for a run."""
    run_dir = Path(run_dir)
    manifest = run_dir / "best_ckpts.json"
    
    if manifest.exists():
        d = json.loads(manifest.read_text())
        for key in prefer:
            p = d.get(key)
            if p and Path(p).exists():
                return Path(p)

    ckpt_dir = run_dir / "checkpoints"
    if fallback_prefix:
        matches = sorted(ckpt_dir.glob(f"{fallback_prefix}*.ckpt"))
        if matches:
            return _newest_by_epoch_step(matches)

    matches = sorted(ckpt_dir.glob("*.ckpt"))
    if matches:
        return _newest_by_epoch_step(matches)

    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


_epoch_step_re = re.compile(r".*?(\d+).*(\d+).ckpt$")

def _newest_by_epoch_step(paths):
    def key(p: Path):
        m = _epoch_step_re.match(p.name)
        return (int(m.group(1)) if m else -1, int(m.group(2)) if m else -1)
    return max(paths, key=key)


def save_tensor(tensor: Tensor, path: Union[str, Path]):
    """Save tensor to disk with directory creation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


# Confidence score detection keywords (shared across modules)
CONFIDENCE_KEYWORDS = ["prob", "margin", "confidence", "sim", "top1", "topk", "score_gap", "gap", "meta", "top_score", "s1"]
UNCERTAINTY_KEYWORDS = ["entropy", "epistemic", "aleatoric", "total", "var", "unc"]


def is_confidence_score(name: str) -> bool:
    """
    Check if a metric name indicates a confidence score (higher = better).
    
    Used for:
    - SGR analysis (determines score direction)
    - Rejection curves (determines sort order)
    - Meta-predictor feature engineering
    """
    name_lower = name.lower()
    # Check for explicit confidence indicators
    if any(kw in name_lower for kw in CONFIDENCE_KEYWORDS):
        return True
    # Check it's not an uncertainty indicator
    if any(kw in name_lower for kw in UNCERTAINTY_KEYWORDS):
        return False
    # Default: assume uncertainty (safer for rejection)
    return False


def negate_if_uncertainty(scores: Tensor, name: str) -> Tensor:
    """
    Negate scores if they represent uncertainty (for consistent 'higher=better' ordering).
    """
    if is_confidence_score(name):
        return scores
    return -scores