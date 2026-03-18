"""
Retrieval scoring with integrated aggregation.

Supports multiple aggregation strategies:
- score: Average similarity scores across ensemble members
- fingerprint: Average fingerprints, then compute similarity
- probability: Average softmax probabilities
- max_score_topk: Apply topk to each member, take max score (best for retrieval)

Also supports learned rankers (biencoder/cross-encoder) via the `ranker` parameter.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Iterator, Literal, Callable, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm import tqdm

from ms_uq.core import similarity_matrix


AggregationMethod = Literal["score", "fingerprint", "probability", "max_score_topk"]


class CrossEncoderRanker(nn.Module):
    """
    Standalone cross-encoder ranker for scoring.
    
    Loads weights saved from FPCrossEncoderRankLearner and applies
    the learned similarity function.
    """
    def __init__(
        self,
        n_bits: int = 4096,
        has_projector: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.has_projector = has_projector
        
        if has_projector:
            self.projector = nn.Sequential(
                nn.Linear(n_bits, n_bits // 8),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(n_bits // 8),
            )
            dim_in = n_bits // 8 * 3
        else:
            dim_in = n_bits * 3
        
        self.cross_encoder = nn.Sequential(
            nn.Linear(dim_in, n_bits // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(n_bits // 8),
            nn.Linear(n_bits // 8, n_bits // 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(n_bits // 16),
            nn.Linear(n_bits // 16, 1),
        )
    
    def forward(self, fp_pred: Tensor, fp_cand: Tensor) -> Tensor:
        """
        Compute learned similarity scores.
        
        Parameters
        ----------
        fp_pred : (N, K) or (K,) predicted fingerprints
        fp_cand : (M, K) candidate fingerprints
        
        Returns
        -------
        scores : (N, M) or (M,) similarity scores
        """
        squeeze = fp_pred.dim() == 1
        if squeeze:
            fp_pred = fp_pred.unsqueeze(0)
        
        N, K = fp_pred.shape
        M = fp_cand.shape[0]
        
        # Expand for pairwise computation
        fp_pred_exp = fp_pred.unsqueeze(1).expand(N, M, K)  # (N, M, K)
        fp_cand_exp = fp_cand.unsqueeze(0).expand(N, M, K)  # (N, M, K)
        
        if self.has_projector:
            # Flatten, project, reshape
            fp_pred_flat = fp_pred_exp.reshape(N * M, K)
            fp_cand_flat = fp_cand_exp.reshape(N * M, K)
            fp_pred_proj = self.projector(fp_pred_flat)
            fp_cand_proj = self.projector(fp_cand_flat)
            combined = torch.cat([
                fp_pred_proj,
                fp_cand_proj,
                fp_pred_proj * fp_cand_proj
            ], dim=-1)
        else:
            fp_pred_flat = fp_pred_exp.reshape(N * M, K)
            fp_cand_flat = fp_cand_exp.reshape(N * M, K)
            combined = torch.cat([
                fp_pred_flat,
                fp_cand_flat,
                fp_pred_flat * fp_cand_flat
            ], dim=-1)
        
        scores = self.cross_encoder(combined).squeeze(-1)  # (N * M,)
        scores = scores.reshape(N, M)
        
        if squeeze:
            scores = scores.squeeze(0)
        
        return scores


class BiencoderRanker(nn.Module):
    """
    Standalone biencoder ranker for scoring.
    
    Uses cosine similarity or IoU, optionally with a projector.
    """
    def __init__(
        self,
        n_bits: int = 4096,
        has_projector: bool = False,
        sim_func: str = "cossim",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.has_projector = has_projector
        self.sim_func_name = sim_func
        
        if sim_func == "cossim":
            self._sim_func = lambda x, y: F.cosine_similarity(x, y, dim=-1)
        elif sim_func == "iou":
            self._sim_func = self._cont_iou
        else:
            self._sim_func = lambda x, y: F.cosine_similarity(x, y, dim=-1)
        
        if has_projector:
            self.projector = nn.Sequential(
                nn.Linear(n_bits, n_bits // 8),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(n_bits // 8),
            )
    
    @staticmethod
    def _cont_iou(a: Tensor, b: Tensor) -> Tensor:
        total = (a + b).sum(dim=-1)
        diff = (a - b).abs().sum(dim=-1)
        return (total - diff) / (total + diff + 1e-8)
    
    def forward(self, fp_pred: Tensor, fp_cand: Tensor) -> Tensor:
        """
        Compute similarity scores.
        
        Parameters
        ----------
        fp_pred : (N, K) or (K,) predicted fingerprints
        fp_cand : (M, K) candidate fingerprints
        
        Returns
        -------
        scores : (N, M) or (M,) similarity scores
        """
        squeeze = fp_pred.dim() == 1
        if squeeze:
            fp_pred = fp_pred.unsqueeze(0)
        
        N, K = fp_pred.shape
        M = fp_cand.shape[0]
        
        if self.has_projector:
            fp_pred = self.projector(fp_pred)
            fp_cand = self.projector(fp_cand)
        
        # Expand for pairwise computation
        fp_pred_exp = fp_pred.unsqueeze(1).expand(N, M, -1)  # (N, M, K')
        fp_cand_exp = fp_cand.unsqueeze(0).expand(N, M, -1)  # (N, M, K')
        
        # Reshape for row-wise similarity
        fp_pred_flat = fp_pred_exp.reshape(N * M, -1)
        fp_cand_flat = fp_cand_exp.reshape(N * M, -1)
        
        scores = self._sim_func(fp_pred_flat, fp_cand_flat)
        scores = scores.reshape(N, M)
        
        if squeeze:
            scores = scores.squeeze(0)
        
        return scores


def load_ranker(ranker_path: Union[str, Path], device: str = "cpu") -> Optional[nn.Module]:
    """
    Load a saved ranker from file.
    
    Parameters
    ----------
    ranker_path : path
        Path to ranker weights file (.pt)
    device : str
        Device to load ranker to
    
    Returns
    -------
    ranker : nn.Module or None
        Loaded ranker module, or None if file doesn't exist
    """
    ranker_path = Path(ranker_path)
    if not ranker_path.exists():
        return None
    
    data = torch.load(ranker_path, map_location=device)
    
    ranker_type = data.get("type", "cross")
    n_bits = data.get("n_bits", 4096)
    has_projector = data.get("has_projector", False)
    
    if ranker_type == "cross":
        ranker = CrossEncoderRanker(
            n_bits=n_bits,
            has_projector=has_projector,
            dropout=data.get("dropout", 0.2),
        )
    else:  # bienc
        ranker = BiencoderRanker(
            n_bits=n_bits,
            has_projector=has_projector,
            sim_func=data.get("sim_func", "cossim"),
            dropout=data.get("dropout", 0.2),
        )
    
    ranker.load_state_dict(data["state_dict"])
    ranker.eval()
    ranker.to(device)
    
    return ranker


def extract_ranker_from_model(model: nn.Module) -> Optional[Dict[str, Any]]:
    """
    Extract ranker configuration and weights from a trained model.
    
    Parameters
    ----------
    model : nn.Module
        Trained FingerprintPredicter model
    
    Returns
    -------
    ranker_data : dict or None
        Dictionary with ranker type, config, and state_dict
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
                state_dict[f'projector.{name}'] = param.data.clone()
        
        if hasattr(ranker_module, 'cross_encoder'):
            for name, param in ranker_module.cross_encoder.named_parameters():
                state_dict[f'cross_encoder.{name}'] = param.data.clone()
    
    else:  # bienc
        # BiencoderRankLearner has: projector (optional), sim_func
        if has_projector and hasattr(ranker_module, 'projector'):
            for name, param in ranker_module.projector.named_parameters():
                state_dict[f'projector.{name}'] = param.data.clone()
    
    # Infer n_bits from model
    n_bits = 4096  # default
    if hasattr(loss_module, 'fp_pred_head'):
        n_bits = loss_module.fp_pred_head.out_features
    
    return {
        'type': ranker_type,
        'n_bits': n_bits,
        'has_projector': has_projector,
        'sim_func': getattr(ranker_module, 'sim_func', 'cossim') if ranker_type == 'bienc' else None,
        'dropout': 0.2,  # default
        'state_dict': state_dict,
    }


def topk_binarize(fp: Tensor, k: int, temperature: float = 0.1) -> Tensor:
    """
    Soft top-k binarization: amplify top-k bits, suppress rest.
    
    Parameters
    ----------
    fp : (*, K) fingerprint probabilities
    k : number of top bits to keep
    temperature : sharpness (lower = harder threshold)
    
    Returns
    -------
    fp_transformed : same shape, with top-k amplified
    """
    K = fp.shape[-1]
    ranks = fp.argsort(dim=-1, descending=True).argsort(dim=-1)
    mask = torch.sigmoid((k - ranks.float()) / temperature)
    return fp * mask


def scores_from_loader(
    fp_probs: Tensor,
    loader: Iterator,
    metric: Literal["cosine", "tanimoto", "iou"] = "cosine",
    aggregation: AggregationMethod = "score",
    temperature: float = 1.0,
    topk_k: int = 80,
    topk_temp: float = 0.1,
    return_labels: bool = True,
    return_per_sample: bool = False,
    show_progress: bool = True,
    ranker: Optional[nn.Module] = None,
    device: str = "cpu",
) -> dict:
    """
    Compute candidate scores from a dataloader with integrated aggregation.
    
    Parameters
    ----------
    fp_probs : (N, S, K) or (N, K)
        Predicted fingerprint probabilities.
    loader : DataLoader
        Yields batches with 'candidates' and optionally 'labels'.
    metric : str
        Similarity metric: 'cosine', 'tanimoto', or 'iou'.
        Ignored if ranker is provided.
    aggregation : str
        - 'score': Compute per-sample scores, then average
        - 'fingerprint': Average fingerprints first, then compute similarity
        - 'probability': Compute per-sample probs, then average
        - 'max_score_topk': Apply topk to each member, take max score
    temperature : float
        Softmax temperature (for 'probability' aggregation).
    topk_k : int
        Number of top bits for 'max_score_topk' aggregation.
    topk_temp : float
        Temperature for topk soft thresholding.
    return_labels : bool
        Include ground truth labels in output.
    return_per_sample : bool
        Also return per-sample scores (S, M) in addition to aggregated.
    show_progress : bool
        Show progress bar.
    ranker : nn.Module, optional
        Learned ranker module (CrossEncoderRanker or BiencoderRanker).
        If provided, uses learned similarity instead of metric.
    device : str
        Device for ranker computation.
    
    Returns
    -------
    dict with:
        - scores_flat : (M,) aggregated scores
        - ptr : (N+1,) pointers
        - labels_flat : (M,) labels (if return_labels)
        - scores_stack_flat : (S, M) per-sample scores (if applicable)
    """
    if fp_probs.dim() == 2:
        fp_probs = fp_probs.unsqueeze(1)
    
    N, S, K = fp_probs.shape
    
    # Determine scoring function
    use_ranker = ranker is not None
    if use_ranker:
        ranker.eval()
        ranker.to(device)
        
        def score_fn(fp: Tensor, cands: Tensor) -> Tensor:
            """Score using learned ranker."""
            with torch.no_grad():
                fp_dev = fp.to(device)
                cands_dev = cands.to(device)
                scores = ranker(fp_dev, cands_dev)
                return scores.cpu()
    else:
        def score_fn(fp: Tensor, cands: Tensor) -> Tensor:
            """Score using similarity matrix."""
            return similarity_matrix(fp, cands, metric=metric)
    
    # Pre-compute based on aggregation method
    if aggregation == "fingerprint":
        fp_mean = fp_probs.mean(dim=1)
    elif aggregation == "max_score_topk":
        fp_topk = torch.stack([
            topk_binarize(fp_probs[:, s], k=topk_k, temperature=topk_temp)
            for s in range(S)
        ], dim=1)
    
    all_scores = []
    all_scores_agg = []
    all_labels = []
    ptr_list = [0]
    
    query_idx = 0
    desc = f"Scoring ({'ranker' if use_ranker else metric}, {aggregation})"
    if aggregation == "max_score_topk":
        desc += f" k={topk_k}"
    iterator = tqdm(loader, desc=desc) if show_progress else loader
    
    for batch in iterator:
        candidates = batch["candidates"]
        batch_ptr = batch.get("batch_ptr")
        
        if isinstance(candidates, list):
            batch_cands = candidates
        elif batch_ptr is not None:
            cumsum = torch.cat([torch.tensor([0]), batch_ptr.cumsum(0)])
            batch_cands = [candidates[cumsum[i]:cumsum[i+1]] for i in range(len(batch_ptr))]
        else:
            batch_cands = [candidates]
        
        batch_labels = None
        if return_labels and "labels" in batch:
            labels = batch["labels"]
            if isinstance(labels, list):
                batch_labels = labels
            elif batch_ptr is not None:
                cumsum = torch.cat([torch.tensor([0]), batch_ptr.cumsum(0)])
                batch_labels = [labels[cumsum[i]:cumsum[i+1]] for i in range(len(batch_ptr))]
            else:
                batch_labels = [labels]
        
        for i, cands in enumerate(batch_cands):
            if query_idx >= N:
                break
            
            n_cand = cands.shape[0] if cands.dim() > 0 else 0
            if n_cand == 0:
                ptr_list.append(ptr_list[-1])
                query_idx += 1
                continue
            
            cands = cands.float()
            
            if aggregation == "fingerprint":
                scores_agg = score_fn(
                    fp_mean[query_idx].unsqueeze(0), cands
                ).squeeze(0)
                all_scores_agg.append(scores_agg)
                
                if return_per_sample:
                    scores_per_sample = score_fn(
                        fp_probs[query_idx], cands
                    )
                    all_scores.append(scores_per_sample)
            
            elif aggregation == "probability":
                scores_per_sample = score_fn(
                    fp_probs[query_idx], cands
                )
                logits = scores_per_sample / max(temperature, 1e-8)
                logits = logits - logits.max(dim=-1, keepdim=True).values
                probs_per_sample = F.softmax(logits, dim=-1)
                probs_mean = probs_per_sample.mean(dim=0)
                scores_agg = torch.log(probs_mean + 1e-12)
                
                all_scores_agg.append(scores_agg)
                if return_per_sample:
                    all_scores.append(scores_per_sample)
            
            elif aggregation == "max_score_topk":
                scores_per_sample = score_fn(
                    fp_topk[query_idx], cands
                )
                scores_agg = scores_per_sample.max(dim=0).values
                
                all_scores_agg.append(scores_agg)
                all_scores.append(scores_per_sample)
            
            else:  # score
                scores_per_sample = score_fn(
                    fp_probs[query_idx], cands
                )
                scores_agg = scores_per_sample.mean(dim=0)
                
                all_scores_agg.append(scores_agg)
                all_scores.append(scores_per_sample)
            
            ptr_list.append(ptr_list[-1] + n_cand)
            
            if batch_labels is not None and i < len(batch_labels):
                lbl = batch_labels[i]
                all_labels.append(lbl if isinstance(lbl, Tensor) else torch.tensor(lbl))
            
            query_idx += 1
    
    result = {
        "scores_flat": torch.cat(all_scores_agg) if all_scores_agg else torch.zeros(0),
        "ptr": torch.tensor(ptr_list, dtype=torch.long),
        "aggregation": aggregation,
        "used_ranker": use_ranker,
    }
    
    if aggregation == "max_score_topk":
        result["topk_k"] = topk_k
        result["topk_temp"] = topk_temp
    
    if all_scores:
        result["scores_stack_flat"] = torch.cat(all_scores, dim=1)
    
    if return_labels and all_labels:
        result["labels_flat"] = torch.cat(all_labels).float()
    
    return result


def scores_ragged_from_loader(
    fp_probs: Union[str, Path, Tensor],
    loader: Iterator,
    metric: str = "cosine",
    aggregation: AggregationMethod = "score",
    temperature: float = 1.0,
    topk_k: int = 80,
    topk_temp: float = 0.1,
    outfile: Optional[Union[str, Path]] = None,
    return_labels: bool = True,
    ranker: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Union[dict, Path]:
    """
    Compute and optionally save ragged scores with aggregation.
    
    Parameters
    ----------
    fp_probs : path or Tensor
        Fingerprint probabilities (N, S, K) or path to .pt file
    loader : DataLoader
        Test dataloader
    metric : str
        Similarity metric (ignored if ranker provided)
    aggregation : str
        Aggregation method
    temperature : float
        Softmax temperature
    topk_k, topk_temp : int, float
        Parameters for max_score_topk aggregation
    outfile : path, optional
        If provided, saves results to this file
    return_labels : bool
        Include labels in output
    ranker : nn.Module, optional
        Learned ranker for scoring
    device : str
        Device for ranker
    
    Returns
    -------
    dict or Path
        Results dict or path to saved file
    """
    if not isinstance(fp_probs, Tensor):
        data = torch.load(fp_probs, map_location="cpu")
        fp_probs = data["stack"] if isinstance(data, dict) else data
    
    result = scores_from_loader(
        fp_probs, loader, 
        metric=metric,
        aggregation=aggregation,
        temperature=temperature,
        topk_k=topk_k,
        topk_temp=topk_temp,
        return_labels=return_labels,
        return_per_sample=True,
        show_progress=True,
        ranker=ranker,
        device=device,
    )
    
    if outfile is not None:
        torch.save(result, outfile)
        print(f"Saved scores to {outfile}")
        return Path(outfile)
    
    return result


def ragged_softmax(scores: Tensor, ptr: Tensor, temperature: float = 1.0) -> Tensor:
    """Apply softmax within each ragged segment."""
    squeeze = scores.dim() == 1
    if squeeze:
        scores = scores.unsqueeze(0)
    
    S, M = scores.shape
    N = ptr.numel() - 1
    probs = torch.zeros_like(scores)
    temp = max(temperature, 1e-8)
    
    for i in range(N):
        start, end = int(ptr[i]), int(ptr[i + 1])
        if start < end:
            logits = scores[:, start:end] / temp
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs[:, start:end] = F.softmax(logits, dim=-1)
    
    return probs.squeeze(0) if squeeze else probs

