"""
Distance-based epistemic uncertainty estimation.

Implements distance-based OOD detection methods for selective prediction:
- k-NN distance (Sun et al., 2022): Distance to k-th nearest training neighbor
- Mahalanobis distance (Lee et al., 2018): Distance accounting for feature covariance
- Relative Mahalanobis (Ren et al., 2021): Class-conditional minus global distance

References:
    Lee et al. (2018): A Simple Unified Framework for Detecting OOD Samples
    Sun et al. (2022): Out-of-Distribution Detection with Deep Nearest Neighbors
    Ren et al. (2021): A Simple Fix to Mahalanobis Distance for Near-OOD Detection
    Mueller & Hein (2025): Mahalanobis++: Feature Normalization for OOD Detection
"""

from __future__ import annotations
from typing import Dict, Optional, Union, Literal
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ms_uq.unc_measures.base import BaseUncertainty


class DistanceUncertainty(BaseUncertainty):
    """
    Distance-based epistemic uncertainty estimation.
    
    Implements multiple distance measures from the OOD detection literature:
    - k-NN distance: Distance to k nearest training neighbors
    - Mahalanobis distance: Distance accounting for feature covariance
    - Centroid distance: Euclidean/cosine distance to training mean
    - Relative Mahalanobis: Mahalanobis minus background (global) distance
    
    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors for k-NN distance.
    metric : str
        Distance metric for k-NN: 'euclidean' or 'cosine'.
    normalize : bool
        Whether to L2-normalize embeddings before distance computation.
        Recommended: True (see Mahalanobis++ paper).
    covariance : str
        Covariance estimation for Mahalanobis: 'full', 'diagonal', or 'shrinkage'.
        'shrinkage' (Ledoit-Wolf) is recommended for high-dimensional embeddings.
    knn_aggregation : str
        How to aggregate k-NN distances:
        - 'kth': Use k-th nearest neighbor distance (Sun et al., 2022)
        - 'mean': Use mean distance to k nearest neighbors
    """
    
    def __init__(
        self,
        n_neighbors: int = 10,
        metric: str = "cosine",
        normalize: bool = True,
        covariance: Literal["full", "diagonal", "shrinkage"] = "shrinkage",
        knn_aggregation: Literal["kth", "mean"] = "kth",
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.normalize = normalize
        self.covariance = covariance
        self.knn_aggregation = knn_aggregation
        
        self._train_embeddings: Optional[Tensor] = None
        self._train_centroid: Optional[Tensor] = None
        self._precision_matrix: Optional[Tensor] = None  # Σ⁻¹ for Mahalanobis
        self._is_fitted = False
    
    def fit(self, train_embeddings: Tensor) -> "DistanceUncertainty":
        """
        Fit on training embeddings.
        
        Computes:
        - Training centroid (mean)
        - Precision matrix (inverse covariance) for Mahalanobis distance
        
        Parameters
        ----------
        train_embeddings : (N_train, D)
            Encoder embeddings from training set.
        """
        train_embeddings = train_embeddings.float().cpu()
        
        if self.normalize:
            train_embeddings = _l2_normalize(train_embeddings)
        
        self._train_embeddings = train_embeddings
        self._train_centroid = train_embeddings.mean(dim=0, keepdim=True)
        
        # Compute precision matrix for Mahalanobis
        self._precision_matrix = self._compute_precision_matrix(train_embeddings)
        
        self._is_fitted = True
        return self
    
    def _compute_precision_matrix(self, X: Tensor) -> Tensor:
        """
        Compute precision matrix (inverse covariance) with regularization.
        
        Parameters
        ----------
        X : (N, D) centered or uncentered embeddings
        
        Returns
        -------
        precision : (D, D) precision matrix
        """
        N, D = X.shape
        
        # Center the data
        X_centered = X - X.mean(dim=0, keepdim=True)
        
        if self.covariance == "diagonal":
            # Diagonal covariance (ignores correlations)
            var = (X_centered ** 2).mean(dim=0) + 1e-6
            precision = torch.diag(1.0 / var)
            
        elif self.covariance == "shrinkage":
            # Ledoit-Wolf shrinkage estimator
            # Shrinks toward diagonal, good for high-D / small N
            cov = (X_centered.T @ X_centered) / (N - 1)
            
            # Compute optimal shrinkage (simplified Ledoit-Wolf)
            trace_cov = torch.trace(cov)
            trace_cov_sq = torch.trace(cov @ cov)
            mu = trace_cov / D  # Shrinkage target: scaled identity
            
            # Shrinkage intensity (simplified formula)
            delta = trace_cov_sq / D + mu ** 2 - 2 * mu * trace_cov / D
            delta = max(delta.item(), 1e-10)
            
            # Estimate optimal alpha (shrinkage intensity)
            # Using a fixed reasonable value since exact LW requires more computation
            alpha = min(1.0, (1.0 / N) / (delta + 1e-10))
            alpha = max(0.1, min(alpha, 0.9))  # Clamp to reasonable range
            
            # Shrunk covariance
            shrunk_cov = (1 - alpha) * cov + alpha * mu * torch.eye(D)
            
            # Invert with regularization
            precision = torch.linalg.inv(shrunk_cov + 1e-5 * torch.eye(D))
            
        else:  # full
            # Full covariance with ridge regularization
            cov = (X_centered.T @ X_centered) / (N - 1)
            # Add regularization for numerical stability
            reg = 1e-5 * torch.trace(cov) / D * torch.eye(D)
            precision = torch.linalg.inv(cov + reg)
        
        return precision
    
    def forward(self, test_embeddings: Tensor) -> Dict[str, Tensor]:
        """
        Compute distance-based uncertainty.
        
        Parameters
        ----------
        test_embeddings : (N_test, D)
        
        Returns
        -------
        dict with:
            - knn_distance: Distance to k(-th) nearest neighbors (uncertainty)
            - mahalanobis: Mahalanobis distance to training distribution (uncertainty)
            - relative_mahalanobis: Mahalanobis - background Mahalanobis (Ren et al., 2021)
            - centroid_distance: Distance to training centroid (uncertainty)
            - local_density: 1 / (1 + knn_distance) (confidence)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before forward()")
        
        test_embeddings = test_embeddings.float()
        if self.normalize:
            test_embeddings = _l2_normalize(test_embeddings)
        
        device = test_embeddings.device
        train_emb = self._train_embeddings.to(device)
        centroid = self._train_centroid.to(device)
        precision = self._precision_matrix.to(device)
        
        # k-NN distance (with configurable aggregation)
        knn_dist = _compute_knn_distances(
            test_embeddings, train_emb, 
            self.n_neighbors, self.metric,
            aggregation=self.knn_aggregation,
        )
        
        # Centroid distance (Euclidean)
        centroid_dist = _pairwise_distance(
            test_embeddings, centroid, "euclidean"
        ).squeeze(-1)
        
        # Mahalanobis distance: sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
        mahalanobis = _mahalanobis_distance(test_embeddings, centroid, precision)
        
        # Relative Mahalanobis distance (Ren et al., 2021)
        # RMD = MD - MD_background
        # For background, we use identity precision (spherical Gaussian)
        # This effectively subtracts the Euclidean distance scaled by global variance
        background_mahal = _mahalanobis_distance(
            test_embeddings, centroid, 
            torch.eye(test_embeddings.shape[1], device=device)
        )
        relative_mahalanobis = mahalanobis - background_mahal
        
        return {
            "knn_distance": knn_dist,
            "mahalanobis": mahalanobis,
            "relative_mahalanobis": relative_mahalanobis,
            "centroid_distance": centroid_dist,
            "local_density": 1.0 / (1.0 + knn_dist),
        }
    
    def compute(self, test_embeddings: Tensor) -> Dict[str, Tensor]:
        return self.forward(test_embeddings)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save fitted model."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "train_embeddings": self._train_embeddings,
            "train_centroid": self._train_centroid,
            "precision_matrix": self._precision_matrix,
            "n_neighbors": self.n_neighbors,
            "metric": self.metric,
            "normalize": self.normalize,
            "covariance": self.covariance,
            "knn_aggregation": self.knn_aggregation,
        }, path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "DistanceUncertainty":
        """Load fitted model."""
        data = torch.load(path, map_location="cpu")
        model = cls(
            n_neighbors=data["n_neighbors"],
            metric=data["metric"],
            normalize=data["normalize"],
            covariance=data.get("covariance", "shrinkage"),
            knn_aggregation=data.get("knn_aggregation", "kth"),
        )
        model._train_embeddings = data["train_embeddings"]
        model._train_centroid = data["train_centroid"]
        model._precision_matrix = data.get("precision_matrix")
        model._is_fitted = True
        
        # Backward compatibility: compute precision if not saved
        if model._precision_matrix is None:
            model._precision_matrix = model._compute_precision_matrix(model._train_embeddings)
        
        return model


# =============================================================================
# Embedding extraction (memory-efficient streaming)
# =============================================================================

def extract_embeddings_from_loader(
    ckpt_path: Union[str, Path],
    loader: DataLoader,
    device: str = "cuda",
    show_progress: bool = True,
    embedding_type: str = "encoder",
    model_cls=None,
) -> Tensor:
    """
    Extract embeddings by streaming through a DataLoader.
    
    Parameters
    ----------
    ckpt_path : Path
        Single model checkpoint (any ensemble member works).
    loader : DataLoader
        Train or test DataLoader.
    device : str
        Device for inference.
    show_progress : bool
        Show tqdm progress bar.
    embedding_type : str
        "encoder": Use encoder output (512D typically) - captures input similarity
        "fingerprint": Use predicted fingerprints (4096D) - captures prediction similarity
    
    Returns
    -------
    embeddings : (N, D) where D depends on embedding_type
        - encoder: D = encoder output dimension (e.g., 512)
        - fingerprint: D = fingerprint dimension (e.g., 4096)
    """
    import torch.nn.functional as F
    if model_cls is None:
        from ms_uq.models.fingerprint_mlp import FingerprintPredicter
        model_cls = FingerprintPredicter
    
    # Handle PyTorch 2.6+ weights_only default
    try:
        from torch.serialization import add_safe_globals
        from massspecgym.models.base import Stage
        add_safe_globals([Stage])
    except Exception:
        pass
    
    model = model_cls.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    
    embeddings = []
    desc = f"Extracting {embedding_type} embeddings"
    iterator = tqdm(loader, desc=desc) if show_progress else loader
    
    with torch.no_grad():
        for batch in iterator:
            x = batch["spec"].to(device)
            
            if embedding_type == "encoder":
                # Encoder output (penultimate layer)
                emb = model.encode_spectrum(x) if hasattr(model, "encode_spectrum") else model.mlp(x)
            else:
                # Fingerprint predictions (output layer)
                hidden = model.encode_spectrum(x) if hasattr(model, "encode_spectrum") else model.mlp(x)
                logits = model.loss.fp_pred_head(hidden)
                emb = F.sigmoid(logits)  # Convert to probabilities
            
            embeddings.append(emb.cpu())
    
    return torch.cat(embeddings, dim=0)


def extract_embeddings_from_pbits(Pbits: Tensor) -> Tensor:
    """
    Fallback: use mean predicted fingerprints as embeddings.
    
    Less ideal than encoder embeddings but works without re-running model.
    
    Parameters
    ----------
    Pbits : (N, S, K) or (N, K)
    
    Returns
    -------
    embeddings : (N, K)
    """
    if Pbits.dim() == 3:
        return Pbits.mean(dim=1)
    return Pbits


def _l2_normalize(X: Tensor, eps: float = 1e-8) -> Tensor:
    return X / X.norm(dim=-1, keepdim=True).clamp_min(eps)


def _pairwise_distance(X: Tensor, Y: Tensor, metric: str) -> Tensor:
    """Pairwise distance: (N, D) x (M, D) -> (N, M)"""
    if metric == "euclidean":
        X_sq = (X ** 2).sum(-1, keepdim=True)
        Y_sq = (Y ** 2).sum(-1, keepdim=True)
        sq_dist = X_sq + Y_sq.T - 2 * (X @ Y.T)
        return sq_dist.clamp_min(0).sqrt()
    elif metric == "cosine":
        X_n, Y_n = _l2_normalize(X), _l2_normalize(Y)
        return 1.0 - (X_n @ Y_n.T)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _mahalanobis_distance(X: Tensor, mean: Tensor, precision: Tensor) -> Tensor:
    """
    Mahalanobis distance: sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
    
    Parameters
    ----------
    X : (N, D) test points
    mean : (1, D) or (D,) distribution mean
    precision : (D, D) precision matrix (inverse covariance)
    
    Returns
    -------
    distances : (N,)
    """
    if mean.dim() == 2:
        mean = mean.squeeze(0)
    
    diff = X - mean  # (N, D)
    
    # (x - μ)ᵀ Σ⁻¹ (x - μ) = sum_i [diff @ precision @ diff.T]_ii
    # Efficient: (diff @ precision) * diff, then sum
    mahal_sq = ((diff @ precision) * diff).sum(dim=-1)  # (N,)
    
    return mahal_sq.clamp_min(0).sqrt()


def _compute_knn_distances(
    test_emb: Tensor,
    train_emb: Tensor,
    k: int,
    metric: str,
    chunk_size: int = 500,
    aggregation: Literal["kth", "mean"] = "kth",
) -> Tensor:
    """
    Compute k-NN distance, chunked for memory efficiency.
    
    Parameters
    ----------
    test_emb : (N_test, D)
    train_emb : (N_train, D)
    k : int
        Number of neighbors
    metric : str
        "euclidean" or "cosine"
    chunk_size : int
        Process in chunks for memory efficiency
    aggregation : str
        - "kth": k-th nearest neighbor distance (Sun et al., 2022)
        - "mean": Mean distance to k nearest neighbors
    
    Returns
    -------
    knn_dist : (N_test,)
    """
    N = test_emb.shape[0]
    k = min(k, train_emb.shape[0])
    knn_dist = torch.zeros(N, device=test_emb.device)
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dists = _pairwise_distance(test_emb[start:end], train_emb, metric)
        topk, _ = dists.topk(k, dim=-1, largest=False)
        
        if aggregation == "kth":
            # k-th nearest neighbor (Sun et al., 2022)
            knn_dist[start:end] = topk[:, -1]
        else:
            # Mean of k nearest neighbors
            knn_dist[start:end] = topk.mean(dim=-1)
    
    return knn_dist