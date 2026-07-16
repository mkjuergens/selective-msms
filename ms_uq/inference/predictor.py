from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Tuple, Optional, Dict, Any
import gc

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_ranker_info(model: nn.Module) -> Optional[Dict[str, Any]]:
    """
    Extract ranker configuration and weights from a trained model.
    
    Parameters
    ----------
    model : nn.Module
        Trained FingerprintPredicter model
    
    Returns
    -------
    ranker_data : dict or None
        Dictionary with ranker type, config, and state_dict.
        None if model doesn't have a ranker.
    """
    if not hasattr(model, 'loss') or not hasattr(model.loss, 'rankwise_loss'):
        return None
    
    if not model.loss.rankwise_loss:
        return None
    
    # Find the ranker module
    loss_module = model.loss
    ranker_module = None
    ranker_type = None
    
    for loss_fn in loss_module.losses:
        if hasattr(loss_fn, 'reranker'):
            ranker_module = loss_fn
            # Determine type from class name
            class_name = loss_fn.__class__.__name__
            if 'Cross' in class_name:
                ranker_type = 'cross'
            else:
                ranker_type = 'bienc'
            break
    
    if ranker_module is None:
        return None
    
    # Extract configuration
    has_projector = getattr(ranker_module, 'proj', False)
    
    # Build state dict for standalone ranker
    state_dict = {}
    
    if ranker_type == 'cross':
        # CrossEncoderRankLearner has: projector (optional), cross_encoder
        if has_projector and hasattr(ranker_module, 'projector'):
            for name, param in ranker_module.projector.named_parameters():
                state_dict[f'projector.{name}'] = param.data.clone().cpu()
        
        if hasattr(ranker_module, 'cross_encoder'):
            for name, param in ranker_module.cross_encoder.named_parameters():
                state_dict[f'cross_encoder.{name}'] = param.data.clone().cpu()
    
    else:  # bienc
        # BiencoderRankLearner has: projector (optional), sim_func
        if has_projector and hasattr(ranker_module, 'projector'):
            for name, param in ranker_module.projector.named_parameters():
                state_dict[f'projector.{name}'] = param.data.clone().cpu()
    
    # Infer n_bits from model
    n_bits = 4096  # default
    if hasattr(loss_module, 'fp_pred_head'):
        n_bits = loss_module.fp_pred_head.out_features
    
    # Get sim_func for biencoder
    sim_func = 'cossim'
    if ranker_type == 'bienc' and hasattr(ranker_module, 'sim_func'):
        # Try to infer from the lambda - this is a bit hacky
        # Default to cossim
        pass
    
    return {
        'type': ranker_type,
        'n_bits': n_bits,
        'has_projector': has_projector,
        'sim_func': sim_func if ranker_type == 'bienc' else None,
        'dropout': 0.2,  # default
        'state_dict': state_dict,
    }


def save_ranker_from_model(model: nn.Module, out_path: Path) -> bool:
    """
    Extract ranker from model and save to file.
    
    Parameters
    ----------
    model : nn.Module
        Trained model with potential ranker
    out_path : Path
        Output path for ranker weights
    
    Returns
    -------
    success : bool
        True if ranker was saved, False if model has no ranker
    """
    ranker_info = extract_ranker_info(model)
    if ranker_info is None:
        return False
    
    torch.save(ranker_info, out_path)
    print(f"[ranker] Saved {ranker_info['type']} ranker to {out_path}")
    return True

@dataclass
class EnsembleSampler:
    """Iterator over checkpoints, yielding one eval model per iteration."""
    ckpts: Iterable[Path]
    model_cls: type
    mc_dropout_eval: bool = False
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        self.ckpts = list(map(Path, self.ckpts))

    def __len__(self) -> int:
        return len(self.ckpts)

    def __iter__(self):
        for i, ck in enumerate(self.ckpts):
            m = self.model_cls.load_from_checkpoint(ck, mc_dropout_eval=self.mc_dropout_eval)
            m = m.to(self.device).eval()
            yield (f"ens{i}", m)
            del m
            torch.cuda.empty_cache()
            gc.collect()  # Added !!



@dataclass
class MCDropoutSampler:
    """Iterator over MC passes for a single checkpoint, reusing one eval model with dropout active."""
    ckpt: Path
    model_cls: type
    passes: int
    device: str = "cuda:0"
    seed: int = 42

    def __len__(self) -> int:
        return self.passes

    def __iter__(self):
        m = self.model_cls.load_from_checkpoint(self.ckpt, mc_dropout_eval=True)
        m = m.to(self.device).eval()
        for s in range(self.passes):
            torch.manual_seed(self.seed + s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed + s)
            yield (f"mc{s}", m)
        # clean up
        del m
        torch.cuda.empty_cache()
        gc.collect()


@dataclass
class Predictor:
    """Run predict_fn over a sampler and stack outputs along the sample axis."""
    sampler: Iterable[Tuple[str, nn.Module]]
    predict_fn: Callable[[nn.Module, dict], torch.Tensor]
    dtype_disk: torch.dtype = torch.float16

    @torch.no_grad()
    def predict_stack(self, dl: DataLoader, outfile: str | Path, 
                     save_every: int = 100, overwrite: bool = False) -> Path:
        """
        Memory-efficient inference with chunked saving.
        
        Parameters
        ----------
        dl : DataLoader
            Data loader to iterate through
        outfile : str or Path
            Output file path
        save_every : int
            Save checkpoint every N batches
        overwrite : bool
            If True, regenerate even if files exist (clears all caches)
        """
        outfile = Path(outfile).with_suffix(".pt")
        checkpoint_dir = outfile.parent / ".checkpoints"
        
        # Handle overwrite - clear ALL caches
        if overwrite:
            if outfile.exists():
                outfile.unlink()
                print(f"[inference] Deleted existing {outfile}")
            if checkpoint_dir.exists():
                import shutil
                shutil.rmtree(checkpoint_dir)
                print(f"[inference] Cleared checkpoint directory {checkpoint_dir}")
        
        if outfile.exists():
            print(f"[inference] using cached {outfile}")
            return outfile

        S = len(self.sampler)
        checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"[inference] Processing {S} samples with chunked saving...")
        
        # Process each sample (model/pass) separately
        for s_idx, (_name, model) in enumerate(tqdm(self.sampler, total=S, desc="samples")):
            checkpoint_path = checkpoint_dir / f"sample_{s_idx:03d}.pt"
            
            if checkpoint_path.exists():
                print(f"[inference] using cached sample {s_idx}")
                continue
            
            # Chunked saving to avoid RAM accumulation
            rows = []
            chunk_size = 50  # Save every 50 batches
            chunk_idx = 0  # Use sequential chunk index, not batch_idx
            
            for batch_idx, batch in enumerate(tqdm(dl, desc=f"sample {s_idx}/{S}", leave=False)):
                with torch.cuda.amp.autocast():
                    pred = self.predict_fn(model, batch)  # (B, *D)

                # Move to CPU immediately
                rows.append(pred.detach().cpu().to(self.dtype_disk))

                # Clear GPU cache frequently
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                
                # Save chunk and clear RAM every `chunk_size` batches
                if len(rows) >= chunk_size:
                    chunk_data = torch.cat(rows, dim=0)
                    # Use zero-padded chunk index for correct lexicographic sorting
                    chunk_path = checkpoint_dir / f"sample_{s_idx:03d}_chunk_{chunk_idx:04d}.pt"
                    torch.save(chunk_data, chunk_path)
                    chunk_idx += 1
                    
                    # Clear memory
                    del rows, chunk_data
                    rows = []
                    gc.collect()
                    torch.cuda.empty_cache()

            # Save final chunk if any remaining
            if rows:
                chunk_data = torch.cat(rows, dim=0)
                # Use 9999 to ensure 'final' sorts last
                chunk_path = checkpoint_dir / f"sample_{s_idx:03d}_chunk_{chunk_idx:04d}.pt"
                torch.save(chunk_data, chunk_path)
                del rows, chunk_data
                gc.collect()
            
            # ✅ Merge all chunks for this sample
            chunk_files = sorted(checkpoint_dir.glob(f"sample_{s_idx:03d}_chunk_*.pt"))
            sample_chunks = []
            for chunk_file in chunk_files:
                chunk = torch.load(chunk_file, map_location="cpu")
                sample_chunks.append(chunk)
                chunk_file.unlink()  # Delete chunk after loading
                del chunk
                gc.collect()
            
            # Concatenate all chunks for this sample
            pred_all = torch.cat(sample_chunks, dim=0)  # (N, *D)
            torch.save(pred_all, checkpoint_path)

            # Clear RAM aggressively
            del sample_chunks, pred_all
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"[inference] Saved sample {s_idx} → {checkpoint_path}")
        
        # Merge all sample checkpoints into final tensor
        print("[inference] Merging sample checkpoints...")
        sample_files = sorted(checkpoint_dir.glob("sample_*.pt"))
        
        if not sample_files:
            raise RuntimeError("No sample checkpoints found!")
        
        # Load first to get dimensions
        first_sample = torch.load(sample_files[0], map_location="cpu")
        N = first_sample.shape[0]
        feat_shape = first_sample.shape[1:]
        
        # Pre-allocate final tensor
        stacked = torch.empty((N, S, *feat_shape), dtype=self.dtype_disk)
        
        # Fill in all samples
        for s_idx, sample_file in enumerate(tqdm(sample_files, desc="merge")):
            sample_data = torch.load(sample_file, map_location="cpu")
            stacked[:, s_idx] = sample_data
            
            # Delete checkpoint after loading
            sample_file.unlink()
            del sample_data
            gc.collect()
        
        # Clean up checkpoint directory
        try:
            checkpoint_dir.rmdir()
        except:
            pass
        
        # Save final result
        meta = {"samples": S}
        outfile.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"stack": stacked, "meta": meta}, outfile)
        print(f"[inference] wrote {outfile} – shape {tuple(stacked.shape)}")
        
        del stacked
        gc.collect()
        
        return outfile


@torch.no_grad()
def save_prestacked_predictions(
    dl: DataLoader,
    outfile: str | Path,
    predict_batch_fn: Callable[[dict], torch.Tensor],
    dtype_disk: torch.dtype = torch.float16,
    overwrite: bool = False,
    chunk_size_batches: int = 50,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save predictions that already include a sample axis.

    This is useful for methods that efficiently produce an (B, S, *D) tensor per
    batch (e.g., Laplace with logit sampling) without re-running the backbone S
    times. The function uses the same chunked-on-disk strategy as Predictor.

    Parameters
    ----------
    dl : DataLoader
        Data loader to iterate through.
    outfile : str or Path
        Output file path (will be saved as .pt).
    predict_batch_fn : Callable[[dict], Tensor]
        Function mapping batch -> Tensor of shape (B, S, *D) on CPU or GPU.
    dtype_disk : torch.dtype
        Dtype used when saving tensors to disk.
    overwrite : bool
        If True, overwrite any existing cached file and checkpoints.
    chunk_size_batches : int
        Number of batches per temporary chunk saved to disk.

    Returns
    -------
    Path
        Path to the saved .pt file.
    """
    outfile = Path(outfile).with_suffix(".pt")
    checkpoint_dir = outfile.parent / ".checkpoints_prestacked"

    if overwrite:
        if outfile.exists():
            outfile.unlink()
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)

    if outfile.exists():
        print(f"[inference] using cached {outfile}")
        return outfile

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    chunk_idx = 0
    for batch_idx, batch in enumerate(tqdm(dl, desc="prest...", leave=False)):
        pred = predict_batch_fn(batch)  # (B, S, *D)
        rows.append(pred.detach().cpu().to(dtype_disk))

        if len(rows) >= chunk_size_batches:
            chunk_data = torch.cat(rows, dim=0)
            chunk_path = checkpoint_dir / f"chunk_{chunk_idx:04d}.pt"
            torch.save(chunk_data, chunk_path)
            chunk_idx += 1
            del rows, chunk_data
            rows = []
            gc.collect()

    if rows:
        chunk_data = torch.cat(rows, dim=0)
        chunk_path = checkpoint_dir / f"chunk_{chunk_idx:04d}.pt"
        torch.save(chunk_data, chunk_path)
        del rows, chunk_data
        gc.collect()

    # Merge chunks
    chunk_files = sorted(checkpoint_dir.glob("chunk_*.pt"))
    if not chunk_files:
        raise RuntimeError("No prestacked chunks were saved.")

    parts = []
    for f in tqdm(chunk_files, desc="merge prestacked"):
        part = torch.load(f, map_location="cpu")
        parts.append(part)
        f.unlink()
        del part
        gc.collect()

    stacked = torch.cat(parts, dim=0)
    del parts
    gc.collect()

    try:
        checkpoint_dir.rmdir()
    except Exception:
        pass

    meta = {"samples": int(stacked.shape[1])}
    if meta_extra:
        meta.update(meta_extra)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stack": stacked, "meta": meta}, outfile)
    print(f"[inference] wrote {outfile} – shape {tuple(stacked.shape)}")
    del stacked
    gc.collect()
    return outfile



def head_probs_fn(
    pred_head: str = "loss.fp_pred_head",
    activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
) -> Callable[[nn.Module, dict], torch.Tensor]:
    """Build a predict_fn that applies a named head to model(x) and an activation.

    Parameters
    ----------
    pred_head : str
        Dotted attribute path to the prediction head (e.g. "loss.fp_pred_head").
    activation : Callable[[Tensor], Tensor]
        Activation applied to head outputs (e.g. torch.sigmoid).

    Returns
    -------
    Callable[[nn.Module, dict], torch.Tensor]
        A function mapping (model, batch) → Tensor of shape (B, *D).
    """
    path = pred_head.split(".")

    def _fn(model: nn.Module, batch: dict) -> torch.Tensor:
        x = batch["spec"].to(next(model.parameters()).device, non_blocking=True)
        emb = model(x)
        head = model
        for p in path:
            head = getattr(head, p)
        return activation(head(emb))

    return _fn