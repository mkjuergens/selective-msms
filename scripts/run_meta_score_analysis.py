#!/usr/bin/env python3
"""Train validation-only COSMIC-inspired linear meta-scores and evaluate frozen on test."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ms_uq.evaluation.confidence_features import (
    build_deployment_features,
    canonical_aurc_table,
    load_fingerprint_predictions,
    load_score_bundle,
    split_metadata,
)
from ms_uq.evaluation.selective_risk import SelectiveGuaranteedRisk, attach_eval_result, make_cal_eval_split


DEFAULT_TOP_KS = [1, 5, 20]
DEFAULT_TARGET_RISKS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
DEFAULT_CS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]


def feature_names_for_k(k: int, available: Dict[str, np.ndarray], exclude_gap_at_k: bool = False) -> List[str]:
    names = [
        "s1", "score_gap", f"score_gap_at_{k}", "log_n_candidates", "confidence",
        "normalized_entropy", f"rank_var_{k}", "retrieval_total", "retrieval_aleatoric",
        "retrieval_epistemic", "pred_fp_cardinality", "top_candidate_fp_cardinality",
        "cardinality_mismatch", "precursor_mz", "n_peaks",
    ]
    if exclude_gap_at_k:
        names.remove(f"score_gap_at_{k}")
    return [name for name in names if name in available]


def matrix(features: Dict[str, np.ndarray], names: List[str]) -> np.ndarray:
    values = np.column_stack([np.asarray(features[name], dtype=np.float64) for name in names])
    if not np.isfinite(values).all():
        raise ValueError("Meta-model feature matrix contains non-finite values")
    return values


def make_pipeline(c: float = 1.0, meta_model: str = "logistic") -> Pipeline:
    if meta_model == "logistic":
        estimator = LogisticRegression(
            C=c, penalty="l2", solver="lbfgs", max_iter=10000, tol=1e-8,
            class_weight=None, fit_intercept=True, random_state=42,
        )
        step_name = "logreg"
    elif meta_model == "linear_svm":
        estimator = LinearSVC(
            C=c, penalty="l2", loss="squared_hinge", dual="auto",
            max_iter=10000, tol=1e-8, random_state=42,
        )
        step_name = "linearsvc"
    else:
        raise ValueError(f"Unknown meta_model={meta_model}")
    return Pipeline([("scaler", StandardScaler()), (step_name, estimator)])


def create_group_folds(y: np.ndarray, groups: np.ndarray, seed: int = 42, n_splits: int = 5):
    n_splits = min(n_splits, len(np.unique(groups)))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
        if any(len(np.unique(y[test])) < 2 for _, test in splits):
            raise ValueError("A stratified fold contains one class")
        strategy = "StratifiedGroupKFold"
    except ValueError:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
        strategy = "GroupKFold"
    return splits, strategy


def fold_mapping_from_splits(groups: np.ndarray, splits: Sequence[Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for fold, (_, test_idx) in enumerate(splits):
        for group in np.unique(groups[test_idx]):
            rows.append({"molecule_group_id": str(group), "cv_fold": fold})
    mapping = pd.DataFrame(rows).drop_duplicates("molecule_group_id")
    if mapping["molecule_group_id"].nunique() != len(np.unique(groups)):
        raise ValueError("CV fold mapping does not cover every molecule group")
    return mapping.sort_values("molecule_group_id").reset_index(drop=True)


def splits_from_mapping(groups: np.ndarray, mapping: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    fold_by_group = dict(zip(mapping["molecule_group_id"].astype(str), mapping["cv_fold"].astype(int)))
    missing = sorted(set(map(str, np.unique(groups))) - set(fold_by_group))
    if missing:
        raise ValueError(f"CV fold mapping misses {len(missing)} validation molecule groups")
    row_folds = np.asarray([fold_by_group[str(group)] for group in groups], dtype=int)
    return [(np.flatnonzero(row_folds != fold), np.flatnonzero(row_folds == fold)) for fold in sorted(set(row_folds))]


def binary_neg_log_loss(estimator, X: np.ndarray, y: np.ndarray) -> float:
    """Binary log-loss scorer with a fixed class universe for grouped CV folds."""
    return -float(log_loss(y, estimator.predict_proba(X), labels=[0, 1]))


def train_meta_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    c_values: List[float],
    meta_model: str = "logistic",
    cv_splits: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[Pipeline, pd.DataFrame, float]:
    if not np.isfinite(X).all():
        raise ValueError("Meta-model input contains non-finite values")
    if cv_splits is None:
        cv_splits, _ = create_group_folds(y, groups)
    step_name = "logreg" if meta_model == "logistic" else "linearsvc"
    scoring = binary_neg_log_loss if meta_model == "logistic" else "roc_auc"
    scoring_name = "neg_log_loss_labels_0_1" if meta_model == "logistic" else "roc_auc"
    grid = GridSearchCV(
        make_pipeline(meta_model=meta_model),
        param_grid={f"{step_name}__C": list(c_values)},
        scoring=scoring,
        cv=list(cv_splits),
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    grid.fit(X, y, groups=groups)
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df["cv_scoring"] = scoring_name
    return grid.best_estimator_, cv_df, float(grid.best_params_[f"{step_name}__C"])


def model_confidence(model: Pipeline, X: np.ndarray, meta_model: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if meta_model == "logistic" else model.decision_function(X)


def coefficient_rows(model: Pipeline, feature_names: List[str], task: str, best_c: float, meta_model: str):
    step_name = "logreg" if meta_model == "logistic" else "linearsvc"
    estimator = model.named_steps[step_name]
    scaler = model.named_steps["scaler"]
    rows = [{
        "task": task, "feature": "intercept", "coefficient": float(estimator.intercept_[0]),
        "best_C": best_c, "meta_model": meta_model, "scaler_mean": np.nan, "scaler_scale": np.nan,
    }]
    rows.extend({
        "task": task, "feature": name, "coefficient": float(coef), "best_C": best_c,
        "meta_model": meta_model, "scaler_mean": float(mean), "scaler_scale": float(scale),
    } for name, coef, mean, scale in zip(feature_names, estimator.coef_[0], scaler.mean_, scaler.scale_))
    return rows


def usable_binary_splits(y, cv_splits):
    return [
        (train_idx, test_idx)
        for train_idx, test_idx in cv_splits
        if len(np.unique(y[train_idx])) == 2 and len(test_idx) > 0
    ]


def out_of_fold_scores(X, y, best_c, meta_model, cv_splits, fallback_model=None):
    scores = np.full(len(y), np.nan, dtype=np.float64)
    for train_idx, test_idx in cv_splits:
        model = make_pipeline(best_c, meta_model=meta_model)
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = model_confidence(model, X[test_idx], meta_model)
    missing = ~np.isfinite(scores)
    if missing.any() and fallback_model is not None:
        scores[missing] = model_confidence(fallback_model, X[missing], meta_model)
    if not np.isfinite(scores).all():
        raise ValueError("Out-of-fold predictions are incomplete")
    return scores, int(missing.sum())


def sgr_rows(metrics, hits, top_ks, target_risks, seed, cal_fraction, delta):
    n = len(next(iter(hits.values())))
    cal_idx, eval_idx = make_cal_eval_split(n, cal_fraction=cal_fraction, seed=seed)
    rows = []
    for k in top_ks:
        loss_name = f"hit@{k}"
        losses = 1.0 - hits[loss_name]
        for measure, confidence in metrics.items():
            for target in target_risks:
                sgr = SelectiveGuaranteedRisk(higher_is_confident=True, binary_loss=True)
                result = sgr.fit(confidence[cal_idx], losses[cal_idx], float(target), delta=delta)
                attach_eval_result(sgr, result, confidence[eval_idx], losses[eval_idx])
                rows.append({
                    "loss": loss_name, "measure": measure, "target_risk": float(target), "delta": delta,
                    "split_seed": seed, "cal_coverage": result.coverage,
                    "cal_empirical_risk": result.empirical_risk, "cal_risk_bound": result.risk_bound,
                    "eval_coverage": result.eval_coverage, "eval_empirical_risk": result.eval_empirical_risk,
                    "feasible": result.feasible,
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_label", required=True)
    ap.add_argument("--dataset_tsv", type=Path, required=True)
    ap.add_argument("--helper_dir", type=Path, required=True)
    ap.add_argument("--candidate_setting", default="formula")
    ap.add_argument("--val_score", type=Path, required=True)
    ap.add_argument("--val_fp_probs", type=Path, required=True)
    ap.add_argument("--test_score", type=Path, required=True)
    ap.add_argument("--test_fp_probs", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--temperature", type=float, default=0.003)
    ap.add_argument("--top_ks", type=int, nargs="+", default=DEFAULT_TOP_KS)
    ap.add_argument("--c_values", type=float, nargs="+", default=DEFAULT_CS)
    ap.add_argument("--target_risks", type=float, nargs="+", default=DEFAULT_TARGET_RISKS)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--cal_fraction", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.001)
    ap.add_argument("--meta_model", choices=["logistic", "linear_svm"], default="logistic")
    ap.add_argument("--cv_fold_assignments", type=Path)
    ap.add_argument("--exclude_score_gap_at_k", action="store_true")
    ap.add_argument("--candidate_record_policy", choices=["preserve", "deduplicate"], default="deduplicate")
    ap.add_argument("--candidate_tie_break", choices=["source_order", "candidate_id"], default="candidate_id")
    ap.add_argument("--aurc_convention", choices=["discrete_prefix_mean", "manuscript_trapezoid_seed42"], default="discrete_prefix_mean")
    ap.add_argument("--feature_convention", choices=["canonical", "manuscript"], default="canonical")
    ap.add_argument("--max_queries", type=int)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_meta = split_metadata(args.dataset_tsv, "val")
    test_meta = split_metadata(args.dataset_tsv, "test")
    if args.max_queries is not None:
        val_meta = val_meta.iloc[:args.max_queries].copy()
        test_meta = test_meta.iloc[:args.max_queries].copy()
    val_features, val_hits, val_imp = build_deployment_features(
        args.val_score, args.val_fp_probs, val_meta, args.helper_dir,
        args.candidate_setting, args.top_ks, args.temperature,
        candidate_record_policy=args.candidate_record_policy,
        candidate_tie_break=args.candidate_tie_break,
        feature_convention=args.feature_convention,
    )
    test_features, test_hits, test_imp = build_deployment_features(
        args.test_score, args.test_fp_probs, test_meta, args.helper_dir,
        args.candidate_setting, args.top_ks, args.temperature,
        candidate_record_policy=args.candidate_record_policy,
        candidate_tie_break=args.candidate_tie_break,
        feature_convention=args.feature_convention,
    )

    groups = val_meta["molecule_group_id"].astype(str).to_numpy()
    fold_path = args.cv_fold_assignments or args.out_dir / "meta_cv_fold_assignments.csv"
    if fold_path.exists():
        fold_mapping = pd.read_csv(fold_path)
        fold_strategy = "reused"
    else:
        reference_y = val_hits[f"hit@{args.top_ks[0]}"].astype(int)
        initial_splits, fold_strategy = create_group_folds(reference_y, groups, seed=args.split_seed)
        fold_mapping = fold_mapping_from_splits(groups, initial_splits)
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        fold_mapping.to_csv(fold_path, index=False)
    cv_splits = splits_from_mapping(groups, fold_mapping)

    meta_metrics, coeff_rows, cv_rows, prediction_rows, oof_rows = {}, [], [], [], []
    suffix = "_no_gap_at_k" if args.exclude_score_gap_at_k else ""
    for k in args.top_ks:
        task = f"meta_hit{k}{suffix}"
        names = feature_names_for_k(k, val_features, exclude_gap_at_k=args.exclude_score_gap_at_k)
        X_val, X_test = matrix(val_features, names), matrix(test_features, names)
        y_val = val_hits[f"hit@{k}"].astype(int)
        task_cv_splits = usable_binary_splits(y_val, cv_splits)
        if len(task_cv_splits) != len(cv_splits) and args.max_queries is None:
            raise ValueError(f"Fixed CV assignment has a single-class training fold for Hit@{k}")
        if len(task_cv_splits) < 2:
            raise ValueError(f"Too few usable grouped CV folds for Hit@{k}")
        model, cv_df, best_c = train_meta_model(
            X_val, y_val, groups, args.c_values, args.meta_model, cv_splits=task_cv_splits
        )
        score = model_confidence(model, X_test, args.meta_model).astype(np.float64)
        oof, n_oof_fallback = out_of_fold_scores(
            X_val, y_val, best_c, args.meta_model, task_cv_splits,
            fallback_model=model if args.max_queries is not None else None,
        )
        meta_metrics[task] = score
        coeff_rows.extend(coefficient_rows(model, names, task, best_c, args.meta_model))
        cv_df = cv_df.copy()
        cv_df["task"], cv_df["meta_model"] = task, args.meta_model
        cv_df["n_fixed_cv_folds"] = len(cv_splits)
        cv_df["n_usable_cv_folds"] = len(task_cv_splits)
        cv_df["n_oof_smoke_fallback"] = n_oof_fallback
        cv_rows.append(cv_df)
        joblib.dump(model, args.out_dir / f"{task}.joblib")
        prediction_rows.extend({
            "query_id": query_id, "molecule_group_id": group, "K": k, "measure": task,
            "score": float(value), "split": "test",
        } for query_id, group, value in zip(test_meta["query_id"], test_meta["molecule_group_id"], score))
        oof_rows.extend({
            "query_id": query_id, "molecule_group_id": group, "K": k, "measure": task,
            "score": float(value), "label": int(label), "split": "validation_oof",
        } for query_id, group, value, label in zip(val_meta["query_id"], groups, oof, y_val))

    baseline_names = [
        "s1", "confidence", "score_gap", "retrieval_total", "retrieval_aleatoric",
        "retrieval_epistemic", "normalized_entropy", "n_candidates",
    ]
    baseline_metrics = {name: test_features[name] for name in baseline_names if name in test_features}
    for k in args.top_ks:
        for name in [f"score_gap_at_{k}", f"rank_var_{k}"]:
            baseline_metrics[name] = test_features[name]
    metrics = {**baseline_metrics, **meta_metrics}
    aurc, rel = canonical_aurc_table(
        metrics, test_hits, query_ids=test_meta["query_id"],
        convention=args.aurc_convention, tie_break=args.candidate_tie_break,
    )
    aurc.to_csv(args.out_dir / "meta_aurc.csv")
    rel.to_csv(args.out_dir / "meta_rel_aurc.csv")
    pd.DataFrame(coeff_rows).to_csv(args.out_dir / "meta_coefficients.csv", index=False)
    pd.concat(cv_rows, ignore_index=True).to_csv(args.out_dir / "meta_cv_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_parquet(args.out_dir / "meta_predictions.parquet", index=False)
    pd.DataFrame(oof_rows).to_parquet(args.out_dir / "meta_oof_predictions.parquet", index=False)
    pd.DataFrame(sgr_rows(
        metrics, test_hits, args.top_ks, args.target_risks,
        args.split_seed, args.cal_fraction, args.delta,
    )).to_csv(args.out_dir / "meta_sgr_results.csv", index=False)
    np.savez(args.out_dir / "meta_scores_test.npz", **meta_metrics)

    fold_counts = []
    row_folds = {str(row.molecule_group_id): int(row.cv_fold) for row in fold_mapping.itertuples()}
    for k in args.top_ks:
        y = val_hits[f"hit@{k}"].astype(int)
        for fold in sorted(fold_mapping["cv_fold"].unique()):
            mask = np.asarray([row_folds[group] == fold for group in groups])
            fold_counts.append({
                "K": k, "cv_fold": fold, "n_spectra": int(mask.sum()),
                "n_molecules": int(np.unique(groups[mask]).size),
                "n_positive": int(y[mask].sum()), "n_negative": int(mask.sum() - y[mask].sum()),
            })
    pd.DataFrame(fold_counts).to_csv(args.out_dir / "meta_cv_fold_counts.csv", index=False)
    pd.DataFrame([{
        "model": args.model_label, "meta_model": args.meta_model, "temperature": args.temperature,
        "candidate_setting": args.candidate_setting, "split_seed": args.split_seed,
        "cal_fraction": args.cal_fraction, "delta": args.delta,
        "cv_strategy": fold_strategy, "cv_fold_assignments": str(fold_path.resolve()),
        "exclude_score_gap_at_k": args.exclude_score_gap_at_k,
        "candidate_record_policy": args.candidate_record_policy,
        "candidate_tie_break": args.candidate_tie_break,
        "aurc_convention": args.aurc_convention,
        "feature_convention": args.feature_convention,
        "score_output": "predict_proba" if args.meta_model == "logistic" else "decision_function",
        "c_values": " ".join(map(str, args.c_values)),
    }]).to_csv(args.out_dir / "meta_run_config.csv", index=False)
    pd.concat([val_imp.assign(split="val"), test_imp.assign(split="test")], ignore_index=True).to_csv(
        args.out_dir / "meta_gap_imputations.csv", index=False
    )
    print(f"Saved meta-score outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
