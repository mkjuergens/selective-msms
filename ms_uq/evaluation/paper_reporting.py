"""Canonical metric, bootstrap, paired-comparison, and SGR reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ms_uq.evaluation.confidence_features import canonical_aurc_table, prefix_risk_curve
from ms_uq.evaluation.selective_risk import SelectiveGuaranteedRisk, attach_eval_result, make_cal_eval_split


BASE_SELECTORS = [
    "s1", "score_gap", "confidence", "normalized_entropy", "retrieval_total",
    "retrieval_aleatoric", "retrieval_epistemic", "bitwise_total", "bitwise_aleatoric",
    "bitwise_epistemic", "n_candidates",
]
GROUP_COLUMNS = [
    "run_id", "split", "run_label", "architecture", "training_candidate_setting",
    "evaluation_candidate_setting", "pool_variant", "model_hash", "candidate_pool_hash",
    "query_mask_id", "T_train", "T_eval", "aggregation", "aurc_convention",
    "candidate_record_policy", "candidate_tie_break", "feature_convention",
]
SGR_SINGLE_MEASURES = [
    "confidence", "score_gap", "retrieval_total", "retrieval_aleatoric",
    "retrieval_epistemic", "rank_var_1", "rank_var_5", "rank_var_20",
]


def selector_arrays(frame: pd.DataFrame, k: int) -> Dict[str, np.ndarray]:
    selectors = {
        name: frame[name].to_numpy(dtype=np.float64)
        for name in BASE_SELECTORS if name in frame and frame[name].notna().all()
    }
    selectors[f"score_gap_at_{k}"] = frame[f"score_gap_at_{k}"].to_numpy(dtype=np.float64)
    selectors[f"rank_var_{k}"] = frame[f"rank_var_{k}"].to_numpy(dtype=np.float64)
    for name in ["meta_full", "meta_no_gap_at_k"]:
        if name in frame and frame[name].notna().all():
            selectors[name] = frame[name].to_numpy(dtype=np.float64)
    for name, values in selectors.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Selector {name} contains non-finite values")
    return selectors


def _aurc_kwargs(frame: pd.DataFrame) -> dict:
    convention = str(frame["aurc_convention"].iloc[0]) if "aurc_convention" in frame else "discrete_prefix_mean"
    tie_break = str(frame["candidate_tie_break"].iloc[0]) if "candidate_tie_break" in frame else "query_id"
    tie_break = "source_order" if tie_break == "source_order" else "query_id"
    return {"convention": convention, "tie_break": tie_break}


def evaluate_frame(frame: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    if _aurc_kwargs(frame)["tie_break"] == "source_order":
        frame = frame.reset_index(drop=True)
    else:
        frame = frame.sort_values("query_id").reset_index(drop=True)
    selectors = selector_arrays(frame, k)
    hits = {f"hit@{k}": frame["hit"].to_numpy(dtype=np.float64)}
    aurc, rel = canonical_aurc_table(
        selectors, hits, query_ids=frame["query_id"].astype(str), **_aurc_kwargs(frame)
    )
    return aurc, rel, selectors


def point_metric_rows(frame: pd.DataFrame, k: int, bootstrap_replicates: int = 2000) -> List[dict]:
    aurc, rel, selectors = evaluate_frame(frame, k)
    common = {column: frame.iloc[0][column] for column in GROUP_COLUMNS if column in frame}
    common.update({
        "K": int(k), "n_spectra": int(len(frame)),
        "n_molecules": int(frame["molecule_group_id"].nunique()),
        "bootstrap_unit": "molecule_group_id", "bootstrap_replicates": int(bootstrap_replicates),
    })
    rows = [{**common, "measure": "retrieval", "metric": "hit_rate", "value": float(frame["hit"].mean())}]
    for measure in selectors:
        rows.append({**common, "measure": measure, "metric": "aurc", "value": float(aurc.loc[measure, f"hit@{k}"])})
        rows.append({**common, "measure": measure, "metric": "rel_aurc", "value": float(rel.loc[measure, f"hit@{k}"])})
    rows.append({**common, "measure": "oracle", "metric": "aurc", "value": float(aurc.loc["oracle", f"hit@{k}"])})
    rows.append({**common, "measure": "random", "metric": "aurc", "value": float(aurc.loc["random", f"hit@{k}"])})
    return rows


def _cluster_indices(frame: pd.DataFrame) -> Tuple[np.ndarray, List[np.ndarray]]:
    groups = np.asarray(sorted(frame["molecule_group_id"].astype(str).unique()), dtype=object)
    values = frame["molecule_group_id"].astype(str).to_numpy()
    return groups, [np.flatnonzero(values == group) for group in groups]


def _bootstrap_sample(
    frame: pd.DataFrame, clusters: List[np.ndarray], draw: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    pieces, ids = [], []
    query_ids = frame["query_id"].astype(str).to_numpy()
    for occurrence, cluster_position in enumerate(draw):
        index = clusters[int(cluster_position)]
        pieces.append(index)
        ids.extend(f"{query_ids[row]}#occ{occurrence:05d}" for row in index)
    sampled = frame.iloc[np.concatenate(pieces)].reset_index(drop=True)
    return sampled, np.asarray(ids, dtype=object)


def bootstrap_frame(frame: pd.DataFrame, k: int, n_replicates: int = 2000, seed: int = 42) -> pd.DataFrame:
    if _aurc_kwargs(frame)["tie_break"] == "source_order":
        frame = frame.reset_index(drop=True)
    else:
        frame = frame.sort_values("query_id").reset_index(drop=True)
    groups, clusters = _cluster_indices(frame)
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(n_replicates):
        draw = rng.integers(0, len(groups), size=len(groups))
        sampled, bootstrap_ids = _bootstrap_sample(frame, clusters, draw)
        selectors = selector_arrays(sampled, k)
        hits = {f"hit@{k}": sampled["hit"].to_numpy(dtype=np.float64)}
        aurc, rel = canonical_aurc_table(
            selectors, hits, query_ids=bootstrap_ids, **_aurc_kwargs(sampled)
        )
        rows.append({"replicate": replicate, "K": k, "measure": "retrieval", "metric": "hit_rate", "value": float(sampled["hit"].mean())})
        for measure in selectors:
            rows.append({"replicate": replicate, "K": k, "measure": measure, "metric": "aurc", "value": float(aurc.loc[measure, f"hit@{k}"])})
            rows.append({"replicate": replicate, "K": k, "measure": measure, "metric": "rel_aurc", "value": float(rel.loc[measure, f"hit@{k}"])})
    return pd.DataFrame(rows)


def attach_intervals(point: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in GROUP_COLUMNS + ["K", "measure", "metric"] if column in bootstrap]
    if bootstrap.empty:
        point["ci_low"], point["ci_high"] = np.nan, np.nan
        return point
    intervals = bootstrap.groupby(keys, dropna=False)["value"].quantile([0.025, 0.975]).unstack().reset_index()
    intervals = intervals.rename(columns={0.025: "ci_low", 0.975: "ci_high"})
    return point.merge(intervals, on=keys, how="left", validate="one_to_one")


def risk_coverage_rows(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    selectors = selector_arrays(frame, k)
    loss = 1.0 - frame["hit"].to_numpy(dtype=np.float64)
    query_ids = frame["query_id"].astype(str).to_numpy()
    common = {column: frame.iloc[0][column] for column in GROUP_COLUMNS if column in frame}
    rows = []
    for measure, confidence in selectors.items():
        tie_break = _aurc_kwargs(frame)["tie_break"]
        coverage, risk = prefix_risk_curve(
            confidence, loss, query_ids=query_ids, tie_break=tie_break
        )
        rows.extend({**common, "K": k, "measure": measure, "coverage": float(c), "risk": float(r)} for c, r in zip(coverage, risk))
    return pd.DataFrame(rows)


def candidate_summary_rows(frame: pd.DataFrame, k: int) -> List[dict]:
    counts = frame["candidate_count"].to_numpy(dtype=np.float64)
    common = {column: frame.iloc[0][column] for column in GROUP_COLUMNS if column in frame}
    return [{
        **common, "K": k, "n_spectra": len(frame),
        "n_molecules": frame["molecule_group_id"].nunique(),
        "target_present_fraction": float(frame["target_present"].mean()),
        "candidate_mean": float(counts.mean()), "candidate_median": float(np.median(counts)),
        "candidate_q25": float(np.quantile(counts, 0.25)), "candidate_q75": float(np.quantile(counts, 0.75)),
        "candidate_p90": float(np.quantile(counts, 0.90)), "candidate_p95": float(np.quantile(counts, 0.95)),
        "candidate_p99": float(np.quantile(counts, 0.99)), "candidate_max": float(counts.max()),
    }]


def merge_meta_predictions(query_scores: pd.DataFrame, specifications: Sequence[Tuple[str, str, Path]]) -> pd.DataFrame:
    result = query_scores.copy()
    for run_label, output_name, path in specifications:
        predictions = pd.read_parquet(path)
        predictions = predictions.loc[predictions["split"] == "test", ["query_id", "K", "score"]]
        predictions = predictions.rename(columns={"score": output_name})
        mask = (result["run_label"] == run_label) & (result["split"] == "test") & (result["evaluation_candidate_setting"] == "formula_official_capped")
        target = result.loc[mask, ["query_id", "K"]].merge(
            predictions, on=["query_id", "K"], how="left", validate="one_to_one"
        )
        if target[output_name].isna().any():
            raise ValueError(f"Missing {output_name} predictions for {run_label}")
        if output_name not in result:
            result[output_name] = np.nan
        result.loc[mask, output_name] = target[output_name].to_numpy()
    return result


def run_sgr_stability(
    query_scores: pd.DataFrame,
    out_dir: Path,
    seeds: Iterable[int] = range(100),
    target_risks: Sequence[float] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
    delta: float = 0.001,
    cal_fraction: float = 0.5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selections, partitions, thresholds, evaluations = [], [], [], []
    for run_label in ["mlp_formula", "transformer_formula"]:
        val_all = query_scores[(query_scores["run_label"] == run_label) & (query_scores["split"] == "val") & (query_scores["evaluation_candidate_setting"] == "formula_official_capped")]
        test_all = query_scores[(query_scores["run_label"] == run_label) & (query_scores["split"] == "test") & (query_scores["evaluation_candidate_setting"] == "formula_official_capped")]
        for k in [1, 5, 20]:
            val = val_all[val_all["K"] == k].reset_index(drop=True)
            test = test_all[test_all["K"] == k].reset_index(drop=True)
            val_selectors = {
                name: val[name].to_numpy(dtype=np.float64)
                for name in SGR_SINGLE_MEASURES if name in val
            }
            _, rel = canonical_aurc_table(
                val_selectors, {f"hit@{k}": val["hit"].to_numpy()},
                query_ids=val["query_id"], **_aurc_kwargs(val),
            )
            eligible = list(val_selectors)
            best_single = min(eligible, key=lambda name: (float(rel.loc[name, f"hit@{k}"]), name))
            selections.append({"run_label": run_label, "K": k, "measure": best_single, "validation_rel_aurc": float(rel.loc[best_single, f"hit@{k}"])})
            measures = {
                name: test[name].to_numpy(dtype=np.float64)
                for name in SGR_SINGLE_MEASURES if name in test
            }
            if "meta_full" in test and test["meta_full"].notna().all():
                measures["meta_full"] = test["meta_full"].to_numpy(dtype=np.float64)
            losses = 1.0 - test["hit"].to_numpy(dtype=np.float64)
            for seed in seeds:
                cal_idx, eval_idx = make_cal_eval_split(len(test), cal_fraction=cal_fraction, seed=int(seed))
                assignment = np.full(len(test), "eval", dtype=object)
                assignment[cal_idx] = "cal"
                partitions.extend({
                    "run_label": run_label, "K": k, "seed": int(seed),
                    "query_id": query_id, "partition": part,
                } for query_id, part in zip(test["query_id"], assignment))
                for measure, confidence in measures.items():
                    for target in target_risks:
                        sgr = SelectiveGuaranteedRisk(higher_is_confident=True, binary_loss=True)
                        result = sgr.fit(confidence[cal_idx], losses[cal_idx], float(target), delta=delta)
                        attach_eval_result(sgr, result, confidence[eval_idx], losses[eval_idx])
                        common = {
                            "run_label": run_label, "K": k, "measure": measure, "seed": int(seed),
                            "target_risk": float(target), "delta": delta,
                            "n_cal": len(cal_idx), "n_eval": len(eval_idx),
                        }
                        thresholds.append({
                            **common, "threshold": result.threshold, "cal_coverage": result.coverage,
                            "cal_empirical_risk": result.empirical_risk, "cal_risk_bound": result.risk_bound,
                            "feasible": result.feasible,
                        })
                        evaluations.append({
                            **common, "eval_coverage": result.eval_coverage,
                            "eval_empirical_risk": result.eval_empirical_risk,
                            "eval_n_selected": result.eval_n_selected, "feasible": result.feasible,
                        })
    pd.DataFrame(selections).to_csv(out_dir / "sgr_score_selection.csv", index=False)
    pd.DataFrame(partitions).to_csv(out_dir / "sgr_partitions.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out_dir / "sgr_thresholds.csv", index=False)
    pd.DataFrame(evaluations).to_csv(out_dir / "sgr_evaluation.csv", index=False)


def paired_selector_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    k: int,
    measure: str,
    comparison: str,
    n_replicates: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    keys = ["query_id", "molecule_group_id"]
    columns = keys + ["hit", measure]
    paired = left[columns].merge(right[columns], on=keys, suffixes=("_left", "_right"), validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(f"Paired comparison {comparison} does not use identical query IDs")
    groups, clusters = _cluster_indices(paired.rename(columns={"molecule_group_id": "molecule_group_id"}))
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(n_replicates):
        draw = rng.integers(0, len(groups), size=len(groups))
        sampled, ids = _bootstrap_sample(paired, clusters, draw)
        values = {}
        for side in ["left", "right"]:
            loss = 1.0 - sampled[f"hit_{side}"].to_numpy(dtype=np.float64)
            confidence = sampled[f"{measure}_{side}"].to_numpy(dtype=np.float64)
            aurc, rel = canonical_aurc_table(
                {measure: confidence}, {f"hit@{k}": 1.0 - loss}, query_ids=ids,
                **_aurc_kwargs(left),
            )
            values[f"hit_{side}"] = float(1.0 - loss.mean())
            values[f"aurc_{side}"] = float(aurc.loc[measure, f"hit@{k}"])
            values[f"rel_aurc_{side}"] = float(rel.loc[measure, f"hit@{k}"])
        rows.extend([
            {"comparison": comparison, "replicate": replicate, "K": k, "measure": measure, "metric": "hit_rate", "difference": values["hit_left"] - values["hit_right"]},
            {"comparison": comparison, "replicate": replicate, "K": k, "measure": measure, "metric": "aurc", "difference": values["aurc_left"] - values["aurc_right"]},
            {"comparison": comparison, "replicate": replicate, "K": k, "measure": measure, "metric": "rel_aurc", "difference": values["rel_aurc_left"] - values["rel_aurc_right"]},
        ])
    return pd.DataFrame(rows)
