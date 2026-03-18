"""
Core mathematical functions for MS-UQ.
"""

from ms_uq.core.entropy import (
    binary_entropy,
    categorical_entropy,
    entropy_from_probs,
    entropy_normalized,
    kl_divergence
)

from ms_uq.core.similarity import (
    normalize,
    cosine_similarity_matrix,
    cosine_similarity_pairwise,
    tanimoto_similarity,
    hamming_distance,
    similarity_matrix
)

__all__ = [
    # Entropy
    "binary_entropy",
    "categorical_entropy", 
    "entropy_from_probs",
    "entropy_normalized",
    "kl_divergence",
    # Similarity
    "normalize",
    "cosine_similarity_matrix",
    "cosine_similarity_pairwise",
    "tanimoto_similarity",
    "hamming_distance",
    "similarity_matrix",
]