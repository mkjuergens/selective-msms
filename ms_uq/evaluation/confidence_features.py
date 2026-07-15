from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from ms_uq.data import candidate_fps_to_dense
from ms_uq.evaluation.candidate_sets import (
    canonical_candidate_indices,
    canonical_candidate_view,
    normalize_inchikey,
)
from ms_uq.utils import resolve_candidate_paths
from ms_uq.unc_measures.eval_measures import compute_fingerprint_uncertainties



_EPS = 1e-12


def load_score_bundle(path: Path):
    data = torch.load(path, map_location="cpu")
    labels = data.get("labels_flat")
    if labels is None:
        raise ValueError(f"{path} does not contain labels_flat; recompute with return_labels=True")
    scores_stack = data.get("scores_stack_flat")
    if scores_stack is None:
        scores_stack = data["scores_flat"].unsqueeze(0)
    aggregate = data["scores_flat"].double()
    stack = scores_stack.double()
    tolerance = 1e-6 if data["scores_flat"].dtype == torch.float32 else 1e-10
    if not torch.allclose(aggregate, stack.mean(dim=0), atol=tolerance, rtol=tolerance):
        raise ValueError(f"{path}: aggregate scores are not the arithmetic member mean")
    return aggregate, stack, data["ptr"].long(), labels.float()


def load_fingerprint_predictions(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", mmap=True)
    return data["stack"] if isinstance(data, dict) else data


def split_metadata(dataset_tsv: Path, fold: str) -> pd.DataFrame:
    frame = pd.read_csv(dataset_tsv, sep="\t")
    output = frame[frame["fold"] == fold].copy().reset_index(drop=True)
    if output.empty:
        raise ValueError(f"No rows for fold={fold}")
    output["query_id"] = output["identifier"].astype(str)
    output["molecule_group_id"] = output["inchikey"].map(normalize_inchikey)
    return output


def candidate_views_for_metadata(
    candidate_inchis,
    candidate_fps,
    metadata: pd.DataFrame,
    include_fps: bool = True,
    record_policy: str = "deduplicate",
):
    if record_policy not in {"preserve", "deduplicate"}:
        raise ValueError("record_policy must be preserve or deduplicate")
    cache = {}
    ids_by_query = []
    for smiles in metadata["smiles"]:
        if smiles not in cache:
            if record_policy == "preserve":
                ids = np.asarray([normalize_inchikey(value) for value in candidate_inchis[smiles]], dtype=object)
                dense = candidate_fps_to_dense(
                    candidate_fps[smiles], n_candidates=len(ids), fp_size=4096,
                ) if include_fps else None
            elif include_fps:
                ids, selected_fps, _ = canonical_candidate_view(candidate_inchis[smiles], candidate_fps[smiles])
                dense = candidate_fps_to_dense(selected_fps, n_candidates=len(ids), fp_size=4096)
            else:
                ids, _ = canonical_candidate_indices(candidate_inchis[smiles], candidate_fps[smiles])
                dense = None
            cache[smiles] = (ids, dense)
        ids_by_query.append(cache[smiles][0])
    candidate_fp_views = {smiles: values[1] for smiles, values in cache.items() if values[1] is not None}
    return ids_by_query, candidate_fp_views


def build_deployment_features(
    score_path: Path,
    fp_probs_path: Path,
    metadata: pd.DataFrame,
    helper_dir: Path,
    candidate_setting: str,
    top_ks: Iterable[int],
    temperature: float,
    include_cardinality: bool = True,
    candidate_record_policy: str = "deduplicate",
    include_fingerprint_uncertainty: bool = False,
    candidate_tie_break: str = "candidate_id",
    feature_convention: str = "canonical",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], pd.DataFrame]:
    scores_flat, scores_stack, ptr, labels = load_score_bundle(score_path)
    n_queries = ptr.numel() - 1
    if len(metadata) != n_queries:
        raise ValueError("Metadata and score query counts do not align")
    pbits = None
    if include_cardinality or include_fingerprint_uncertainty:
        pbits = load_fingerprint_predictions(fp_probs_path)
        if pbits.shape[0] < n_queries:
            raise ValueError("Prediction and score query counts do not align")
        pbits = pbits[:n_queries]

    _, candidate_fp_path, candidate_inchi_path = resolve_candidate_paths(helper_dir, candidate_setting)
    with np.load(candidate_fp_path) as candidate_fps, np.load(candidate_inchi_path) as candidate_inchis:
        ids_by_query, candidate_fp_views = candidate_views_for_metadata(
            candidate_inchis, candidate_fps, metadata, include_fps=include_cardinality,
            record_policy=candidate_record_policy,
        )
        if candidate_tie_break not in {"source_order", "candidate_id"}:
            raise ValueError("candidate_tie_break must be source_order or candidate_id")
        ranking_ids = ids_by_query if candidate_tie_break == "candidate_id" else None
        hits = hit_arrays(scores_flat, labels, ptr, top_ks, candidate_ids=ranking_ids)
        score_features, imputations = score_position_features(scores_flat, ptr, top_ks)
        temperature_features = retrieval_temperature_features(
            scores_stack, scores_flat, ptr, temperature, top_ks,
            candidate_ids=ranking_ids, feature_convention=feature_convention,
        )
        cardinality = {}
        if include_cardinality:
            cardinality = cardinality_features(
                pbits, scores_flat, ptr, candidate_fp_views,
                metadata["smiles"].tolist(), ranking_ids,
            )
    fingerprint_features = {}
    if include_fingerprint_uncertainty:
        fingerprint_uncertainty = compute_fingerprint_uncertainties(
            pbits, ["bitwise_total", "bitwise_aleatoric", "bitwise_epistemic"], batch_size=64,
        )
        fingerprint_features = {
            name: -np.asarray(values, dtype=np.float64)
            for name, values in fingerprint_uncertainty.items()
        }
    features = {
        **score_features, **temperature_features, **fingerprint_features,
        **metadata_features(metadata), **cardinality,
    }
    for name, values in features.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Feature {name} contains {int((~np.isfinite(values)).sum())} non-finite values")
    return features, hits, pd.DataFrame([imputation.__dict__ for imputation in imputations])


def temperature_tag(temperature: float) -> str:
    return f"T{float(temperature):g}".replace("-", "m").replace(".", "p")


def score_cache_name(prefix: str, aggregation: str, temperature: float) -> str:
    if aggregation == "probability":
        return f"scores_{prefix}_{aggregation}_{temperature_tag(temperature)}.pt"
    return f"scores_{prefix}_{aggregation}.pt"


def _as_ids(values: Optional[Sequence], n: int, prefix: str) -> np.ndarray:
    if values is None:
        return np.asarray([f"{prefix}{i:012d}" for i in range(n)], dtype=object)
    out = np.asarray(values, dtype=object)
    if len(out) != n:
        raise ValueError(f"Expected {n} identifiers, got {len(out)}")
    return out.astype(str)


def _confidence_order(
    confidence: np.ndarray,
    query_ids: Optional[Sequence] = None,
    tie_break: str = "query_id",
) -> np.ndarray:
    values = np.asarray(confidence, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Confidence values must all be finite")
    if tie_break == "source_order":
        return np.argsort(-values, kind="stable")
    if tie_break != "query_id":
        raise ValueError("tie_break must be query_id or source_order")
    ids = _as_ids(query_ids, len(values), "query-")
    return np.lexsort((ids, -values))


def _candidate_order(scores: np.ndarray, candidate_ids: Optional[Sequence] = None) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    ids = _as_ids(candidate_ids, len(values), "candidate-")
    return np.lexsort((ids, -values))


def _segment_candidate_ids(
    candidate_ids: Optional[Sequence], query_index: int, start: int, end: int, n_queries: int
) -> np.ndarray:
    if candidate_ids is None:
        return _as_ids(None, end - start, "candidate-")
    if len(candidate_ids) == n_queries and (
        len(candidate_ids) == 0 or not isinstance(candidate_ids[0], (str, bytes, np.str_, np.bytes_))
    ):
        return _as_ids(candidate_ids[query_index], end - start, "candidate-")
    return _as_ids(candidate_ids[start:end], end - start, "candidate-")


def hit_arrays(
    scores_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    ptr: torch.Tensor,
    top_ks: Iterable[int],
    candidate_ids: Optional[Sequence] = None,
) -> Dict[str, np.ndarray]:
    """Deterministic Hit@K using candidate identity to resolve exact ties."""
    scores = scores_flat.detach().cpu().numpy().astype(np.float64, copy=False)
    labels = labels_flat.detach().cpu().numpy().astype(bool, copy=False)
    n_queries = ptr.numel() - 1
    top_ks = [int(k) for k in top_ks]
    out = {f"hit@{k}": np.zeros(n_queries, dtype=np.float64) for k in top_ks}
    for i in range(n_queries):
        start, end = int(ptr[i]), int(ptr[i + 1])
        local_ids = _segment_candidate_ids(candidate_ids, i, start, end, n_queries)
        order = _candidate_order(scores[start:end], local_ids)
        for k in top_ks:
            out[f"hit@{k}"][i] = float(labels[start:end][order[:min(k, end - start)]].any())
    return out


def target_ranks(
    scores_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    ptr: torch.Tensor,
    candidate_ids: Optional[Sequence] = None,
) -> np.ndarray:
    """One-based best target rank, or infinity when the target is absent."""
    scores = scores_flat.detach().cpu().numpy().astype(np.float64, copy=False)
    labels = labels_flat.detach().cpu().numpy().astype(bool, copy=False)
    n_queries = ptr.numel() - 1
    ranks = np.full(n_queries, np.inf, dtype=np.float64)
    for i in range(n_queries):
        start, end = int(ptr[i]), int(ptr[i + 1])
        local_ids = _segment_candidate_ids(candidate_ids, i, start, end, n_queries)
        order = _candidate_order(scores[start:end], local_ids)
        target_positions = np.flatnonzero(labels[start:end][order])
        if target_positions.size:
            ranks[i] = float(target_positions[0] + 1)
    return ranks


def prefix_risk_curve(
    confidence: np.ndarray,
    loss: np.ndarray,
    query_ids: Optional[Sequence] = None,
    tie_break: str = "query_id",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return right-continuous empirical coverage/risk prefixes."""
    loss = np.asarray(loss, dtype=np.float64)
    if not np.isfinite(loss).all():
        raise ValueError("Loss values must all be finite")
    order = _confidence_order(confidence, query_ids, tie_break=tie_break)
    ordered_loss = loss[order]
    counts = np.arange(1, len(loss) + 1, dtype=np.float64)
    return counts / len(loss), np.cumsum(ordered_loss) / counts


def discrete_aurc(
    confidence: np.ndarray,
    loss: np.ndarray,
    query_ids: Optional[Sequence] = None,
    tie_break: str = "query_id",
) -> float:
    """AURC of the empirical right-continuous risk-coverage step function."""
    _, risk = prefix_risk_curve(confidence, loss, query_ids=query_ids, tie_break=tie_break)
    return float(np.mean(risk, dtype=np.float64))


def manuscript_trapezoid_aurc(confidence: np.ndarray, loss: np.ndarray) -> float:
    """Submitted-paper float32 rejection curve and trapezoidal integration."""
    confidence_t = torch.as_tensor(np.asarray(confidence), dtype=torch.float32)
    loss_t = torch.as_tensor(np.asarray(loss), dtype=torch.float32)
    uncertainty = -confidence_t
    rejected_first = torch.argsort(uncertainty, descending=True)
    ordered_loss = loss_t[rejected_first]
    tail_sum = torch.cumsum(ordered_loss.flip(0), dim=0).flip(0)
    kept_risk = tail_sum / torch.arange(len(loss_t), 0, -1, dtype=torch.float32)
    rejection = torch.arange(0, len(loss_t), dtype=torch.float32) / float(len(loss_t))
    coverage = 1.0 - rejection
    order = torch.argsort(coverage)
    return float(torch.trapz(kept_risk[order], coverage[order]).item())


def canonical_aurc_table(
    metrics: Dict[str, np.ndarray],
    hits: Dict[str, np.ndarray],
    query_ids: Optional[Sequence] = None,
    convention: str = "discrete_prefix_mean",
    tie_break: str = "query_id",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute canonical AURC and relAURC with all metrics oriented as confidence."""
    aurc_rows: Dict[str, Dict[str, float]] = {name: {} for name in metrics}
    aurc_rows["oracle"] = {}
    aurc_rows["random"] = {}
    ids = None
    for loss_name, hit in hits.items():
        loss = 1.0 - np.asarray(hit, dtype=np.float64)
        if ids is None:
            ids = _as_ids(query_ids, len(loss), "query-")
        oracle_conf = -loss
        if convention == "manuscript_trapezoid_seed42":
            oracle = manuscript_trapezoid_aurc(oracle_conf, loss)
            rng = np.random.default_rng(random_seed)
            random_conf = -rng.random(len(loss)).astype(np.float32)
            random = manuscript_trapezoid_aurc(random_conf, loss)
            for name, confidence in metrics.items():
                aurc_rows[name][loss_name] = manuscript_trapezoid_aurc(confidence, loss)
        elif convention == "discrete_prefix_mean":
            oracle = discrete_aurc(oracle_conf, loss, query_ids=ids, tie_break=tie_break)
            random = float(loss.mean(dtype=np.float64))
            for name, confidence in metrics.items():
                aurc_rows[name][loss_name] = discrete_aurc(
                    confidence, loss, query_ids=ids, tie_break=tie_break
                )
        else:
            raise ValueError(f"Unknown AURC convention: {convention}")
        aurc_rows["oracle"][loss_name] = oracle
        aurc_rows["random"][loss_name] = random
    aurc = pd.DataFrame.from_dict(aurc_rows, orient="index")
    denominator = aurc.loc["random"] - aurc.loc["oracle"]
    rel = (aurc.loc[list(metrics)] - aurc.loc["oracle"]) / denominator
    # A constant loss has identical random and oracle AURC, so relAURC is undefined.
    rel.loc[:, denominator <= 0] = np.nan
    return aurc, rel


def relative_aurc(
    metrics: Dict[str, np.ndarray],
    hits: Dict[str, np.ndarray],
    query_ids: Optional[Sequence] = None,
    convention: str = "discrete_prefix_mean",
    tie_break: str = "query_id",
) -> pd.DataFrame:
    return canonical_aurc_table(
        metrics, hits, query_ids=query_ids, convention=convention, tie_break=tie_break
    )[1]


@dataclass
class GapImputation:
    name: str
    n_imputed: int
    fill_value: float


def score_gap_at_k(scores_flat: torch.Tensor, ptr: torch.Tensor, k: int) -> Tuple[np.ndarray, GapImputation]:
    """Return s_k - s_(k+1), with the closed-world guaranteed-Hit@K edge value."""
    values = np.full(ptr.numel() - 1, 2.0, dtype=np.float64)
    n_imputed = 0
    for i in range(ptr.numel() - 1):
        start, end = int(ptr[i]), int(ptr[i + 1])
        seg = scores_flat[start:end].double()
        n_cand = int(end - start)
        if n_cand > k:
            sorted_scores = torch.sort(seg, descending=True).values
            values[i] = float(sorted_scores[k - 1] - sorted_scores[k])
        else:
            n_imputed += 1
    return values, GapImputation(name=f"score_gap_at_{k}", n_imputed=n_imputed, fill_value=2.0)


def score_position_features(scores_flat: torch.Tensor, ptr: torch.Tensor,
                            top_ks: Iterable[int]) -> Tuple[Dict[str, np.ndarray], List[GapImputation]]:
    n = ptr.numel() - 1
    features = {
        "s1": np.zeros(n, dtype=np.float64),
        "score_gap": np.full(n, 2.0, dtype=np.float64),
        "log_n_candidates": np.zeros(n, dtype=np.float64),
        "n_candidates": np.zeros(n, dtype=np.float64),
    }
    for i in range(n):
        start, end = int(ptr[i]), int(ptr[i + 1])
        seg = scores_flat[start:end].double()
        n_cand = int(end - start)
        log_n = np.log(max(n_cand, 1))
        features["n_candidates"][i] = -log_n
        features["log_n_candidates"][i] = log_n
        if n_cand == 0:
            continue
        sorted_scores = torch.sort(seg, descending=True).values
        features["s1"][i] = float(sorted_scores[0])
        if n_cand > 1:
            features["score_gap"][i] = float(sorted_scores[0] - sorted_scores[1])
    imputations = []
    for k in top_ks:
        gap, imp = score_gap_at_k(scores_flat, ptr, int(k))
        features[imp.name] = gap
        features[f"candidate_count_le_{int(k)}"] = (-features["n_candidates"] <= np.log(int(k))).astype(np.float64)
        imputations.append(imp)
    return features, imputations


def retrieval_temperature_features(scores_stack: torch.Tensor, scores_flat: torch.Tensor,
                                   ptr: torch.Tensor, temperature: float,
                                   top_ks: Iterable[int],
                                   candidate_ids: Optional[Sequence] = None,
                                   feature_convention: str = "canonical") -> Dict[str, np.ndarray]:
    return softmax_temperature_features(
        scores_stack,
        scores_flat,
        ptr,
        temperature,
        top_ks=top_ks,
        candidate_ids=candidate_ids,
        feature_convention=feature_convention,
    )


def softmax_temperature_features(
    scores_stack: torch.Tensor,
    scores_flat: torch.Tensor,
    ptr: torch.Tensor,
    temperature: float,
    top_ks: Iterable[int] = (1, 5, 20),
    candidate_ids: Optional[Sequence] = None,
    feature_convention: str = "canonical",
) -> Dict[str, np.ndarray]:
    """Confidence-oriented retrieval features at one temperature."""
    if feature_convention == "manuscript":
        from ms_uq.unc_measures.retrieval_unc import RetrievalUncertainty

        uncertainty = RetrievalUncertainty(
            temperature=temperature, normalize_entropy=False, top_k_list=list(top_ks)
        ).compute(scores_stack.float(), ptr, scores_flat.float())
        result = {
            "confidence": uncertainty["confidence_top1"].numpy().astype(np.float64),
            "retrieval_total": -uncertainty["entropy_total"].numpy().astype(np.float64),
            "retrieval_aleatoric": -uncertainty["entropy_aleatoric"].numpy().astype(np.float64),
            "retrieval_epistemic": -uncertainty["entropy_epistemic"].numpy().astype(np.float64),
            "normalized_entropy": 1.0 - uncertainty["normalized_entropy"].numpy().astype(np.float64),
        }
        result.update({
            f"rank_var_{int(k)}": -uncertainty[f"rank_var_{int(k)}"].numpy().astype(np.float64)
            for k in top_ks
        })
        return result
    if feature_convention != "canonical":
        raise ValueError("feature_convention must be canonical or manuscript")
    if scores_stack.dim() == 1:
        scores_stack = scores_stack.unsqueeze(0)
    scores_stack_np = scores_stack.detach().cpu().double().numpy()
    scores_flat_np = scores_flat.detach().cpu().double().numpy()
    n = ptr.numel() - 1
    top_ks = [int(k) for k in top_ks]
    out = {
        "confidence": np.zeros(n, dtype=np.float64),
        "retrieval_total": np.zeros(n, dtype=np.float64),
        "retrieval_aleatoric": np.zeros(n, dtype=np.float64),
        "retrieval_epistemic": np.zeros(n, dtype=np.float64),
        "normalized_entropy": np.ones(n, dtype=np.float64),
    }
    out.update({f"rank_var_{k}": np.zeros(n, dtype=np.float64) for k in top_ks})
    temp = max(float(temperature), _EPS)
    for i in range(n):
        start, end = int(ptr[i]), int(ptr[i + 1])
        n_cand = end - start
        if n_cand == 0:
            continue
        if n_cand == 1:
            out["confidence"][i] = 1.0
            continue
        member_scores = scores_stack_np[:, start:end]
        aggregate_scores = scores_flat_np[start:end]
        member_logits = member_scores / temp
        member_logits -= member_logits.max(axis=1, keepdims=True)
        member_probs = np.exp(member_logits)
        member_probs /= member_probs.sum(axis=1, keepdims=True)
        aggregate_logits = aggregate_scores / temp
        aggregate_logits -= aggregate_logits.max()
        aggregate_probs = np.exp(aggregate_logits)
        aggregate_probs /= aggregate_probs.sum()
        pbar = member_probs.mean(axis=0)
        h_total = -np.sum(pbar * np.log(np.clip(pbar, _EPS, None)))
        h_aleatoric = -np.mean(np.sum(member_probs * np.log(np.clip(member_probs, _EPS, None)), axis=1))
        out["confidence"][i] = float(aggregate_probs.max())
        out["retrieval_total"][i] = -float(h_total)
        out["retrieval_aleatoric"][i] = -float(h_aleatoric)
        out["retrieval_epistemic"][i] = float(-h_total + h_aleatoric)
        h_aggregate = -np.sum(aggregate_probs * np.log(np.clip(aggregate_probs, _EPS, None)))
        out["normalized_entropy"][i] = 1.0 - float(h_aggregate / np.log(float(n_cand)))

        local_ids = _segment_candidate_ids(candidate_ids, i, start, end, n)
        aggregate_order = _candidate_order(aggregate_scores, local_ids)
        member_ranks = np.empty((member_scores.shape[0], n_cand), dtype=np.int64)
        for member_idx, values in enumerate(member_scores):
            order = _candidate_order(values, local_ids)
            member_ranks[member_idx, order] = np.arange(n_cand)
        for k in top_ks:
            reference = aggregate_order[:min(k, n_cand)]
            rank_variance = np.var(member_ranks[:, reference], axis=0, ddof=0)
            out[f"rank_var_{k}"][i] = -float(rank_variance.mean()) if len(reference) else 0.0
    return out


def spearman_confidence_log_candidates(confidence: np.ndarray, log_n_candidates: np.ndarray) -> Tuple[float, float]:
    rho, p_value = spearmanr(confidence, log_n_candidates, nan_policy="omit")
    return float(rho), float(p_value)


def peak_count(value) -> int:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text:
            return 0
        sep = ";" if ";" in text else "," if "," in text else " "
        return len([x for x in text.split(sep) if x.strip()])
    try:
        return len(value)
    except TypeError:
        return 0


def metadata_features(metadata: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if "precursor_mz" in metadata.columns:
        out["precursor_mz"] = pd.to_numeric(metadata["precursor_mz"], errors="coerce").to_numpy(dtype=np.float32)
    if "mzs" in metadata.columns:
        out["n_peaks"] = metadata["mzs"].map(peak_count).to_numpy(dtype=np.float32)
    return out


def cardinality_features(
    Pbits: torch.Tensor,
    scores_flat: torch.Tensor,
    ptr: torch.Tensor,
    candidate_fps: Mapping[str, np.ndarray],
    query_smiles: List[str],
    candidate_ids_by_query: Optional[Sequence] = None,
) -> Dict[str, np.ndarray]:
    if Pbits.dim() == 3:
        fp_mean = torch.cat([
            Pbits[start:start + 256].float().mean(dim=1)
            for start in range(0, Pbits.shape[0], 256)
        ], dim=0)
    else:
        fp_mean = Pbits.float()
    pred_card = (fp_mean >= 0.5).sum(dim=1).numpy().astype(np.float64)
    top_card = np.zeros(len(query_smiles), dtype=np.float64)
    for i, smiles in enumerate(query_smiles):
        start, end = int(ptr[i]), int(ptr[i + 1])
        if end <= start:
            top_card[i] = np.nan
            continue
        local_scores = scores_flat[start:end].detach().cpu().double().numpy()
        local_ids = _segment_candidate_ids(candidate_ids_by_query, i, start, end, len(query_smiles))
        local_top = int(_candidate_order(local_scores, local_ids)[0])
        raw = np.asarray(candidate_fps[smiles])
        dense = raw if raw.ndim == 2 else np.unpackbits(raw, bitorder="big")[: (end - start) * fp_mean.shape[-1]].reshape(end - start, fp_mean.shape[-1])
        top_card[i] = float(dense[local_top].sum())
    return {
        "pred_fp_cardinality": pred_card,
        "top_candidate_fp_cardinality": top_card,
        "cardinality_mismatch": np.abs(pred_card - top_card).astype(np.float64),
    }
