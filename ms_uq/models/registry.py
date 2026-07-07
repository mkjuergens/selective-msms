from __future__ import annotations


ARCHITECTURES = ("mlp", "transformer")


def normalize_architecture(architecture: str | None) -> str:
    arch = (architecture or "mlp").lower()
    aliases = {
        "fingerprint_mlp": "mlp",
        "mlp": "mlp",
        "peak_transformer": "transformer",
        "transformer": "transformer",
    }
    if arch not in aliases:
        raise ValueError(f"Unknown architecture '{architecture}'. Expected one of {ARCHITECTURES}.")
    return aliases[arch]


def get_model_class(architecture: str | None):
    arch = normalize_architecture(architecture)
    if arch == "mlp":
        from ms_uq.models.fingerprint_mlp import FingerprintPredicter

        return FingerprintPredicter

    from ms_uq.models.fingerprint_transformer import FingerprintPredicterTransformer

    return FingerprintPredicterTransformer
