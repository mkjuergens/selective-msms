"""Diagonal last-layer Laplace approximation for multi-label BCE heads.

Implements a factorised Bernoulli Laplace approximation over the last linear
fingerprint prediction head. Supports:
  - Diagonal GGN curvature accumulation
  - Prior tuning via marginal likelihood or validation BCE
  - MC logit sampling and probit-mean prediction
  - Built-in diagnostics for debugging low sample diversity

References:
  - Daxberger et al., "Laplace Redux" (NeurIPS 2021)
  - Immer et al., "Improving predictions of Bayesian neural nets via local linearization" (AISTATS 2021)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _logistic_normal_mean(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """E[sigmoid(Z)] ≈ sigmoid(mu / sqrt(1 + π·var/8)) for Z ~ N(mu, var)."""
    return torch.sigmoid(mu / torch.sqrt(1.0 + (math.pi / 8.0) * var.clamp_min(0.0)))


class _FeatureExtractor(nn.Module):
    """Thin wrapper: extracts penultimate spectrum features."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.head: nn.Linear = model.loss.fp_pred_head

    @torch.inference_mode()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "encode_spectrum"):
            return self.model.encode_spectrum(x)
        return self.model.mlp(x)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LaplaceConfig:
    """All Laplace-related hyperparameters."""
    tau_w: float = 1.0
    tau_b: float = 1.0
    n_samples: int = 50
    tune_prior: bool = True
    tune_method: str = "marglik"          # "marglik" or "val_bce"
    max_batches: Optional[int] = 200      # limit training batches for speed
    tau_w_grid: Sequence[float] = field(
        default_factory=lambda: (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
    )
    tau_b_grid: Sequence[float] = field(
        default_factory=lambda: (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
    )
    tune_max_batches: int = 200
    tune_bit_subsample: int = 512
    tune_seed: int = 0
    out_chunk: int = 512                  # bit-chunk size for prediction
    diagnostics: bool = True              # print diagnostics after fit
    prediction_seed: int = 42              # MC posterior-sampling seed


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class DiagLaplaceBCEHead:
    """Diagonal Laplace posterior over a multi-label BCE last layer.

    Posterior: p(W,b | D) ≈ N((W_map, b_map), diag(curv + τ)^{-1})
    """

    def __init__(self, model: nn.Module, device: str = "cuda", cfg: Optional[LaplaceConfig] = None):
        self.cfg = cfg or LaplaceConfig()
        self.device = torch.device(device)

        model = model.eval().to(self.device)
        self.fe = _FeatureExtractor(model).eval().to(self.device)
        self.head = self.fe.head

        W = self.head.weight.detach().to(self.device, torch.float32)
        b = self.head.bias.detach().to(self.device, torch.float32) if self.head.bias is not None else None
        self.K, self.H = W.shape
        self.W_map = W.clone()
        self.b_map = b.clone() if b is not None else None

        self.curv_W = torch.zeros_like(self.W_map)
        self.curv_b = torch.zeros(self.K, device=self.device, dtype=torch.float32) if b is not None else None

        self.tau_w = self.cfg.tau_w
        self.tau_b = self.cfg.tau_b
        self.var_W: Optional[torch.Tensor] = None
        self.var_b: Optional[torch.Tensor] = None
        self._fitted = False

    # ── Fit ────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def fit(self, train_loader, show_pbar: bool = True) -> None:
        """Accumulate diagonal GGN curvature: H_{k,j} = Σ_i p_ik(1-p_ik) h_ij²."""
        self.curv_W.zero_()
        if self.curv_b is not None:
            self.curv_b.zero_()

        max_b = self.cfg.max_batches
        total = min(len(train_loader), max_b) if max_b else len(train_loader)
        it = enumerate(train_loader)
        if show_pbar:
            it = tqdm(it, total=total, desc="[Laplace] fitting curvature")

        n_seen = 0
        for bi, batch in it:
            if max_b is not None and bi >= max_b:
                break
            x = batch["spec"].to(self.device, non_blocking=True)
            h = self.fe.features(x).to(torch.float32)
            p = torch.sigmoid(self.head(h))
            r = p * (1.0 - p)  # (B, K)

            self.curv_W.add_(r.t() @ (h * h))   # (K, H)
            if self.curv_b is not None:
                self.curv_b.add_(r.sum(0))
            n_seen += x.size(0)

        # Scale up if we subsampled
        if max_b is not None:
            try:
                n_total = len(train_loader.dataset)
            except Exception:
                n_total = None
            if n_total and n_seen > 0 and n_total > n_seen:
                scale = n_total / n_seen
                self.curv_W.mul_(scale)
                if self.curv_b is not None:
                    self.curv_b.mul_(scale)

        self._fitted = True
        self._set_prior(self.tau_w, self.tau_b)

    # ── Prior setting ─────────────────────────────────────────────────────

    def _set_prior(self, tau_w: float, tau_b: float) -> None:
        """Recompute posterior variances from stored curvature + prior."""
        assert self._fitted
        self.tau_w, self.tau_b = float(tau_w), float(tau_b)
        self.var_W = 1.0 / (self.curv_W + self.tau_w).clamp_min(1e-12)
        if self.curv_b is not None:
            self.var_b = 1.0 / (self.curv_b + self.tau_b).clamp_min(1e-12)

    # ── Prior tuning ──────────────────────────────────────────────────────

    def tune_prior(self, val_loader=None) -> Tuple[float, float]:
        """Tune (tau_w, tau_b) using the method specified in cfg."""
        assert self._fitted
        cfg = self.cfg
        if cfg.tune_method == "marglik":
            return self._tune_marglik()
        else:
            assert val_loader is not None, "val_bce tuning requires a validation loader"
            return self._tune_val_bce(val_loader)

    def _tune_marglik(self) -> Tuple[float, float]:
        """Tune prior via log marginal likelihood (no data pass needed)."""
        best_lml, best_tw, best_tb = float("-inf"), self.tau_w, self.tau_b
        for tw in self.cfg.tau_w_grid:
            for tb in self.cfg.tau_b_grid:
                lml = self._log_marginal_likelihood(tw, tb)
                if lml > best_lml:
                    best_lml, best_tw, best_tb = lml, float(tw), float(tb)
        self._set_prior(best_tw, best_tb)
        return best_tw, best_tb

    def _log_marginal_likelihood(self, tau_w: float, tau_b: float) -> float:
        """log p(D|τ) ∝ log p(W|τ) - ½ log det(curv + τ)  (constant data term omitted)."""
        tw, tb = float(tau_w), float(tau_b)
        d_w = self.K * self.H
        lml = 0.5 * d_w * math.log(tw) - 0.5 * tw * (self.W_map ** 2).sum().item()
        lml -= 0.5 * torch.log((self.curv_W + tw).clamp_min(1e-30)).sum().item()
        if self.b_map is not None:
            lml += 0.5 * self.K * math.log(tb) - 0.5 * tb * (self.b_map ** 2).sum().item()
        if self.curv_b is not None:
            lml -= 0.5 * torch.log((self.curv_b + tb).clamp_min(1e-30)).sum().item()
        return lml

    def _tune_val_bce(self, val_loader) -> Tuple[float, float]:
        """Tune prior via validation BCE on a random bit subset."""
        cfg = self.cfg
        g = torch.Generator(device="cpu").manual_seed(cfg.tune_seed)
        bit_idx = torch.randperm(self.K, generator=g)[:min(cfg.tune_bit_subsample, self.K)].to(self.device)

        Wm = self.W_map[bit_idx]
        bm = self.b_map[bit_idx] if self.b_map is not None else None
        curvW = self.curv_W[bit_idx]
        curvB = self.curv_b[bit_idx] if self.curv_b is not None else None

        best_loss, best_tw, best_tb = float("inf"), self.tau_w, self.tau_b
        for tw in cfg.tau_w_grid:
            varW = 1.0 / (curvW + float(tw)).clamp_min(1e-12)
            for tb in cfg.tau_b_grid:
                varB = 1.0 / (curvB + float(tb)).clamp_min(1e-12) if curvB is not None else None
                total, count = 0.0, 0
                for bi, batch in enumerate(val_loader):
                    if bi >= cfg.tune_max_batches:
                        break
                    x = batch["spec"].to(self.device, non_blocking=True)
                    y = batch["mol"].to(self.device, non_blocking=True).float()[:, bit_idx]
                    h = self.fe.features(x).to(torch.float32)
                    mu = h @ Wm.t() + (bm.unsqueeze(0) if bm is not None else 0)
                    var = (h * h) @ varW.t() + (varB.unsqueeze(0) if varB is not None else 0)
                    total += float(F.binary_cross_entropy(_logistic_normal_mean(mu, var), y)) * x.size(0)
                    count += x.size(0)
                if count > 0 and total / count < best_loss:
                    best_loss, best_tw, best_tb = total / count, float(tw), float(tb)

        self._set_prior(best_tw, best_tb)
        return best_tw, best_tb

    # ── Prediction ────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict_batch(self, batch: dict, n_samples: Optional[int] = None,
                      method: str = "logit_mc") -> torch.Tensor:
        """Predict probabilities for one batch. Returns (B, S, K) float16 on CPU."""
        assert self.var_W is not None
        S = 1 if method == "probit_mean" else (n_samples or self.cfg.n_samples)
        x = batch["spec"].to(self.device, non_blocking=True)
        h = self.fe.features(x).to(torch.float32)
        B = h.size(0)
        chunk = self.cfg.out_chunk

        slices = []
        for k0 in range(0, self.K, chunk):
            k1 = min(k0 + chunk, self.K)
            Wm, vW = self.W_map[k0:k1], self.var_W[k0:k1]
            mu = h @ Wm.t()
            if self.b_map is not None:
                mu = mu + self.b_map[k0:k1].unsqueeze(0)
            var = (h * h) @ vW.t()
            if self.var_b is not None:
                var = var + self.var_b[k0:k1].unsqueeze(0)

            if method == "probit_mean":
                p = _logistic_normal_mean(mu, var).unsqueeze(1)  # (B,1,k)
            else:
                eps = torch.randn(S, B, k1 - k0, device=self.device)
                z = mu.unsqueeze(0) + eps * var.clamp_min(0).sqrt().unsqueeze(0)
                p = torch.sigmoid(z).permute(1, 0, 2)  # (B,S,k)
            slices.append(p.half())

        return torch.cat(slices, dim=2).cpu()

    # ── Diagnostics ───────────────────────────────────────────────────────

    @torch.inference_mode()
    def diagnostics(self, test_loader, topk_k: int = 80) -> Dict[str, float]:
        """Run diagnostics on one batch to understand sample diversity.

        Prints and returns stats about logit variance, probability variance,
        bit-flip rates, and top-k stability.
        """
        assert self.var_W is not None
        batch = next(iter(test_loader))
        S = self.cfg.n_samples

        # Logit-level stats
        x = batch["spec"].to(self.device, non_blocking=True)
        h = self.fe.features(x).to(torch.float32)
        mu = h @ self.W_map.t()
        if self.b_map is not None:
            mu = mu + self.b_map.unsqueeze(0)
        var = (h * h) @ self.var_W.t()
        if self.var_b is not None:
            var = var + self.var_b.unsqueeze(0)
        std = var.sqrt()

        frac_flippable = (mu.abs() < 2 * std).float().mean().item()

        # Sample-level stats
        probs = self.predict_batch(batch, n_samples=S).float()  # (B,S,K)
        prob_std = probs.std(dim=1)
        frac_varying = (prob_std > 0.01).float().mean().item()

        binary = (probs > 0.5).float()
        bit_disagree = (binary.std(dim=1) > 0).float().mean().item()

        # Top-k Jaccard between sample pairs
        B = probs.size(0)
        jaccards = []
        for b in range(min(B, 100)):
            s0 = set(probs[b, 0].topk(topk_k).indices.tolist())
            s1 = set(probs[b, 1].topk(topk_k).indices.tolist())
            jaccards.append(len(s0 & s1) / len(s0 | s1) if s0 | s1 else 1.0)

        stats = {
            "logit_mu_mean": mu.mean().item(),
            "logit_mu_std": mu.std().item(),
            "logit_std_mean": std.mean().item(),
            "logit_std_median": std.median().item(),
            "frac_flippable_2std": frac_flippable,
            "prob_std_mean": prob_std.mean().item(),
            "prob_std_max": prob_std.max().item(),
            "frac_bits_varying": frac_varying,
            "frac_bits_disagreeing": bit_disagree,
            f"topk{topk_k}_jaccard_mean": np.mean(jaccards),
            f"topk{topk_k}_jaccard_min": np.min(jaccards),
        }

        print(f"\n  [Laplace Diagnostics]")
        print(f"    tau_w={self.tau_w:.4g}, tau_b={self.tau_b:.4g}")
        print(f"    curv_W: mean={self.curv_W.mean():.4g}, median={self.curv_W.median():.4g}, max={self.curv_W.max():.4g}")
        print(f"    var_W:  mean={self.var_W.mean():.4g}, median={self.var_W.median():.4g}, max={self.var_W.max():.4g}")
        print(f"    Logit: mu_mean={stats['logit_mu_mean']:.3f}, mu_std={stats['logit_mu_std']:.3f}, "
              f"std_mean={stats['logit_std_mean']:.4f}")
        print(f"    Flippable bits (|μ|<2σ): {frac_flippable:.4f}")
        print(f"    Prob std across {S} samples: mean={stats['prob_std_mean']:.5f}, max={stats['prob_std_max']:.4f}")
        print(f"    Bits with any variation (p>0.01): {frac_varying:.4f}")
        print(f"    Bits with binary disagreement: {bit_disagree:.4f}")
        print(f"    Top-{topk_k} Jaccard (s0 vs s1): mean={stats[f'topk{topk_k}_jaccard_mean']:.4f}, "
              f"min={stats[f'topk{topk_k}_jaccard_min']:.4f}")
        return stats

    # ── Serialization ─────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, object]:
        return {
            "tau_w": self.tau_w, "tau_b": self.tau_b,
            "curv_W": self.curv_W.cpu(),
            "curv_b": self.curv_b.cpu() if self.curv_b is not None else None,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore fitted curvature and prior parameters."""
        curv_W = torch.as_tensor(state["curv_W"], device=self.device, dtype=torch.float32)
        if tuple(curv_W.shape) != tuple(self.curv_W.shape):
            raise ValueError(
                f"Laplace weight-curvature shape {tuple(curv_W.shape)} does not match "
                f"model head {tuple(self.curv_W.shape)}"
            )
        self.curv_W.copy_(curv_W)

        stored_b = state.get("curv_b")
        if self.curv_b is None:
            if stored_b is not None:
                raise ValueError("Laplace state has bias curvature but model head has no bias")
        else:
            if stored_b is None:
                raise ValueError("Laplace state has no bias curvature for a biased model head")
            curv_b = torch.as_tensor(stored_b, device=self.device, dtype=torch.float32)
            if tuple(curv_b.shape) != tuple(self.curv_b.shape):
                raise ValueError(
                    f"Laplace bias-curvature shape {tuple(curv_b.shape)} does not match "
                    f"model head {tuple(self.curv_b.shape)}"
                )
            self.curv_b.copy_(curv_b)

        self._fitted = True
        self._set_prior(float(state["tau_w"]), float(state["tau_b"]))

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)
        print(f"  [Laplace] Saved state: {path}")


# ---------------------------------------------------------------------------
# Top-level entry point for evaluation pipeline
# ---------------------------------------------------------------------------

def generate_laplace_predictions(
    out_dir: Path,
    ckpt: str,
    test_loader,
    train_loader=None,
    val_loader=None,
    device: str = "cuda",
    overwrite: bool = False,
    cfg: Optional[LaplaceConfig] = None,
    make_loaders_fn=None,
    save_ranker_fn=None,
    model_cls=None,
    state_path: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    """End-to-end: load model → fit Laplace → tune prior → diagnose → save predictions.

    Parameters
    ----------
    out_dir : Path
        Directory for outputs (fp_probs.pt, laplace_state.pt, ranker.pt).
    ckpt : str
        Path to model checkpoint.
    test_loader : DataLoader
        Test data loader.
    train_loader, val_loader : DataLoader, optional
        If None and needed, created via make_loaders_fn.
    device : str
    overwrite : bool
    cfg : LaplaceConfig, optional
    make_loaders_fn : callable, optional
        () -> (train_loader, val_loader, _) for when loaders aren't provided.
    save_ranker_fn : callable, optional
        (model, path) -> bool, saves ranker if present.
    model_cls : type, optional
        Model class with .load_from_checkpoint (default: FingerprintPredicter).
    state_path : Path, optional
        Released fitted Laplace state. When supplied, curvature fitting and prior tuning are skipped.
    """
    cfg = cfg or LaplaceConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_path = out_dir / "fp_probs.pt"
    ranker_path = out_dir / "ranker.pt"
    laplace_path = out_dir / "laplace_state.pt"

    if fp_path.exists() and not overwrite:
        print(f"  [Laplace] Using cached {fp_path}")
        return fp_path, ranker_path if ranker_path.exists() else None

    print(f"  [Laplace] {cfg.n_samples} samples, tune={cfg.tune_method if cfg.tune_prior else 'off'}")

    # Load model
    if model_cls is None:
        from ms_uq.models.fingerprint_mlp import FingerprintPredicter
        model_cls = FingerprintPredicter
    model = model_cls.load_from_checkpoint(ckpt, map_location=device)
    model.eval()

    if save_ranker_fn:
        try:
            save_ranker_fn(model, ranker_path)
        except Exception:
            pass

    # Build Laplace
    laplace = DiagLaplaceBCEHead(model, device=device, cfg=cfg)

    if state_path is not None:
        state_path = Path(state_path)
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        print(f"  [Laplace] Loading fitted state: {state_path}")
        laplace.load_state_dict(torch.load(state_path, map_location=device))
    else:
        if train_loader is None:
            assert make_loaders_fn, "Need train_loader or make_loaders_fn"
            train_loader, val_loader_new, _ = make_loaders_fn()
            if val_loader is None:
                val_loader = val_loader_new

        print("  [Laplace] Fitting curvature...")
        laplace.fit(train_loader)
        if cfg.tune_prior:
            print(f"  [Laplace] Tuning prior ({cfg.tune_method})...")
            tw, tb = laplace.tune_prior(val_loader)
            print(f"  [Laplace] Best prior: tau_w={tw:.4g}, tau_b={tb:.4g}")
        laplace.save(laplace_path)

    # Diagnostics
    if cfg.diagnostics:
        torch.manual_seed(cfg.prediction_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.prediction_seed)
        laplace.diagnostics(test_loader)

    # Reset after diagnostics so they cannot alter the released sampling sequence.
    torch.manual_seed(cfg.prediction_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.prediction_seed)

    # Generate predictions
    print("  [Laplace] Generating predictions...")
    from ms_uq.inference.predictor import save_prestacked_predictions
    save_prestacked_predictions(
        test_loader, fp_path,
        lambda batch: laplace.predict_batch(batch),
        overwrite=overwrite,
        meta_extra={"method": "laplace", "tau_w": laplace.tau_w, "tau_b": laplace.tau_b},
    )

    del model, laplace
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  [Laplace] Saved: {fp_path}")
    return fp_path, ranker_path if ranker_path.exists() else None