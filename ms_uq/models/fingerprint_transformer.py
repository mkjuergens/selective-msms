from __future__ import annotations

import torch
import torch.nn as nn

from ms_uq.models.fingerprint_mlp import FingerprintPredicter


class AttnAggregator(nn.Module):
    """Attention pooling over transformer sequence outputs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(hidden).squeeze(-1)
        if padding_mask is not None:
            logits = logits.masked_fill(padding_mask.bool(), -torch.inf)
        weights = torch.softmax(logits, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.sum(hidden * weights.unsqueeze(-1), dim=1)


class FingerprintPredicterTransformer(FingerprintPredicter):
    """Fingerprint predictor with a peak-transformer spectrum encoder."""

    def __init__(
        self,
        transformer_d_model: int = 256,
        transformer_nhead: int = 8,
        transformer_ff_dim: int = 1024,
        transformer_n_layers: int = 4,
        transformer_dropout: float = 0.25,
        **kwargs,
    ) -> None:
        try:
            from depthcharge.transformers import SpectrumTransformerEncoder
        except ImportError as exc:
            raise ImportError(
                "The transformer architecture requires the 'depthcharge-ms' package. "
                "Install it with `pip install depthcharge-ms`."
            ) from exc

        kwargs = dict(kwargs)
        kwargs["layer_dims"] = [transformer_d_model]
        kwargs.setdefault("n_in", transformer_d_model)
        super().__init__(**kwargs)

        self.transformer_d_model = transformer_d_model
        self.transformer_nhead = transformer_nhead
        self.transformer_ff_dim = transformer_ff_dim
        self.transformer_n_layers = transformer_n_layers
        self.transformer_dropout = transformer_dropout

        self.mlp = SpectrumTransformerEncoder(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_ff_dim,
            n_layers=transformer_n_layers,
            dropout=transformer_dropout,
        )
        self.aggregator = AttnAggregator(transformer_d_model)

    def encode_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != 2:
            raise ValueError(
                "Transformer spectra must have shape (batch, peaks, 2). "
                "Use architecture='transformer' so SpecTokenizer is selected."
            )

        self._enable_mc_dropout()
        # SpecTokenizer prepends the precursor m/z row. Depthcharge adds its own
        # global token, so we follow ms-mole and pass only fragment peaks here.
        mz_array = x[:, 1:, 0]
        intensity_array = x[:, 1:, 1]
        hidden, padding_mask = self.mlp(mz_array, intensity_array)
        return self.aggregator(hidden, padding_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_spectrum(x)
