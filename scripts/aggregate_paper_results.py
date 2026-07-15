#!/usr/bin/env python3
"""Aggregate canonical paper query scores into metrics, intervals, comparisons, and tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from ms_uq.evaluation.paper_reporting import (
    BASE_SELECTORS,
    GROUP_COLUMNS,
    attach_intervals,
    bootstrap_frame,
    candidate_summary_rows,
    evaluate_frame,
    merge_meta_predictions,
    paired_selector_difference,
    point_metric_rows,
    risk_coverage_rows,
)


def parse_meta(value: str) -> Tuple[str, str, Path]:
    if "=" not in value or "," not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("Meta specifications must be run_label,output_name=/path/predictions.parquet")
    left, path = value.split("=", 1)
    run_label, output_name = left.split(",", 1)
    return run_label, output_name, Path(path)


def add_group_columns(rows: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    for column in GROUP_COLUMNS:
        if column in frame:
            rows[column] = frame.iloc[0][column]
    rows["n_spectra"] = len(frame)
    rows["n_molecules"] = frame["molecule_group_id"].nunique()
    rows["bootstrap_unit"] = "molecule_group_id"
    return rows


def selector_comparison(frame, k, left_measure, right_measure, name, n_replicates, seed):
    left = frame[["query_id", "molecule_group_id", "hit", left_measure]].rename(columns={left_measure: "comparison_score"})
    right = frame[["query_id", "molecule_group_id", "hit", right_measure]].rename(columns={right_measure: "comparison_score"})
    result = paired_selector_difference(left, right, k, "comparison_score", name, n_replicates, seed)
    result["left_measure"], result["right_measure"] = left_measure, right_measure
    return result


def pool_comparisons(scores, n_replicates, seed):
    outputs = []
    for run_label in ["mlp_formula", "transformer_formula"]:
        capped = scores[(scores.run_label == run_label) & (scores.split == "test") & (scores.evaluation_candidate_setting == "formula_pubchem_capped256")]
        uncapped = scores[(scores.run_label == run_label) & (scores.split == "test") & (scores.evaluation_candidate_setting == "formula_pubchem_uncapped")]
        for k in [1, 5, 20]:
            left, right = uncapped[uncapped.K == k], capped[capped.K == k]
            for measure in ["confidence", "retrieval_total", "score_gap", f"score_gap_at_{k}", f"rank_var_{k}"]:
                if not left.empty and not right.empty:
                    outputs.append(paired_selector_difference(
                        left, right, k, measure, f"{run_label}: uncapped minus paired capped",
                        n_replicates, seed,
                    ).assign(left_measure=measure, right_measure=measure))

    formula_official = scores[(scores.run_label == "mlp_formula") & (scores.split == "test") & (scores.evaluation_candidate_setting == "formula_official_capped")]
    formula_mass = scores[(scores.run_label == "mlp_formula") & (scores.split == "test") & (scores.evaluation_candidate_setting == "mass_existing_capped256")]
    mass_model = scores[(scores.run_label == "mlp_mass") & (scores.split == "test") & (scores.evaluation_candidate_setting == "mass_existing_capped256")]
    for k in [1, 5, 20]:
        mass_eval = formula_mass[formula_mass.K == k]
        common_ids = set(mass_eval.query_id)
        official = formula_official[(formula_official.K == k) & (formula_official.query_id.isin(common_ids))]
        trained_mass = mass_model[(mass_model.K == k) & (mass_model.query_id.isin(common_ids))]
        for measure in ["confidence", "retrieval_total", "score_gap", f"score_gap_at_{k}", f"rank_var_{k}"]:
            if not mass_eval.empty and not official.empty:
                outputs.append(paired_selector_difference(
                    mass_eval, official, k, measure, "MLP formula model: existing mass minus official formula",
                    n_replicates, seed,
                ).assign(left_measure=measure, right_measure=measure))
            if not mass_eval.empty and not trained_mass.empty:
                outputs.append(paired_selector_difference(
                    trained_mass, mass_eval, k, measure, "Mass-trained minus formula-trained MLP on existing mass",
                    n_replicates, seed,
                ).assign(left_measure=measure, right_measure=measure))
    return outputs


def summarize_differences(replicates: pd.DataFrame) -> pd.DataFrame:
    keys = ["comparison", "K", "left_measure", "right_measure", "metric"]
    summary = replicates.groupby(keys)["difference"].agg(
        value="mean", ci_low=lambda value: value.quantile(0.025), ci_high=lambda value: value.quantile(0.975)
    ).reset_index()
    summary["sign_convention"] = "left minus right; positive favors left for Hit@K, negative favors left for AURC/relAURC"
    return summary


def build_tables(metrics, candidate_summary, out_dir):
    supervised = metrics[
        (metrics.split == "test")
        & (metrics.evaluation_candidate_setting == "formula_official_capped")
        & (metrics.metric.isin(["aurc", "rel_aurc"]))
        & (metrics.run_label.isin(["mlp_formula", "transformer_formula"]))
    ].copy()
    supervised.to_csv(out_dir / "table_supervised_combination.csv", index=False)
    (out_dir / "table_supervised_combination.tex").write_text(
        supervised[["architecture", "K", "measure", "metric", "value", "ci_low", "ci_high"]].to_latex(index=False, float_format="%.4f")
    )

    uq_runs = ["mlp_formula", "mlp_mc_dropout", "mlp_laplace", "transformer_formula"]
    paper_measures = {
        "retrieval", "bitwise_total", "bitwise_aleatoric", "bitwise_epistemic",
        "retrieval_total", "retrieval_aleatoric", "retrieval_epistemic",
        "confidence", "score_gap", "score_gap_at_1", "score_gap_at_5",
        "score_gap_at_20", "rank_var_1", "rank_var_5", "rank_var_20", "n_candidates",
    }
    uq_methods = metrics[
        (metrics["split"] == "test")
        & (metrics["evaluation_candidate_setting"] == "formula_official_capped")
        & (metrics["run_label"].isin(uq_runs))
        & (metrics["measure"].isin(paper_measures))
        & (metrics["metric"].isin(["hit_rate", "aurc", "rel_aurc"]))
    ].copy()
    uq_methods.to_csv(out_dir / "table_uq_methods.csv", index=False)
    (out_dir / "table_uq_methods.tex").write_text(
        uq_methods[["run_label", "K", "measure", "metric", "value", "ci_low", "ci_high"]]
        .to_latex(index=False, float_format="%.4f")
    )

    rel = metrics[(metrics.split == "test") & (metrics.metric == "rel_aurc")].copy()
    non_meta = rel[~rel.measure.str.startswith("meta_") & rel["value"].notna()]
    best_idx = non_meta.groupby(["run_label", "evaluation_candidate_setting", "K"])["value"].idxmin()
    best = non_meta.loc[best_idx, [
        "run_label", "evaluation_candidate_setting", "K", "measure", "value", "ci_low", "ci_high"
    ]].rename(columns={"measure": "best_selector", "value": "best_rel_aurc", "ci_low": "best_rel_aurc_ci_low", "ci_high": "best_rel_aurc_ci_high"})
    hit = metrics[(metrics.split == "test") & (metrics.metric == "hit_rate")][[
        "run_label", "evaluation_candidate_setting", "K", "value", "ci_low", "ci_high"
    ]].rename(columns={"value": "hit_rate", "ci_low": "hit_rate_ci_low", "ci_high": "hit_rate_ci_high"})
    table = candidate_summary.merge(hit, on=["run_label", "evaluation_candidate_setting", "K"], how="left")
    table = table.merge(best, on=["run_label", "evaluation_candidate_setting", "K"], how="left")
    table.to_csv(out_dir / "table_candidate_settings.csv", index=False)
    (out_dir / "table_candidate_settings.tex").write_text(table.to_latex(index=False, float_format="%.4f"))


def validation_report(scores, metrics):
    checks = []
    def check(name, passed, observed):
        checks.append({"name": name, "passed": bool(passed), "observed": observed})
    duplicate_key = ["run_label", "evaluation_candidate_setting", "split", "query_id", "K"]
    check("unique query/K rows per cell", not scores.duplicated(duplicate_key).any(), int(scores.duplicated(duplicate_key).sum()))
    monotonic = scores.pivot_table(index=["run_label", "evaluation_candidate_setting", "split", "query_id"], columns="K", values="hit")
    check("Hit@1 <= Hit@5 <= Hit@20", bool(((monotonic[1] <= monotonic[5]) & (monotonic[5] <= monotonic[20])).all()), len(monotonic))
    selector_columns = [name for name in BASE_SELECTORS if name in scores] + [f"score_gap_at_{k}" for k in [1,5,20]] + [f"rank_var_{k}" for k in [1,5,20]]
    finite = all(np.isfinite(scores[column].dropna()).all() for column in selector_columns)
    check("all canonical selectors finite", finite, selector_columns)
    official = scores[
        (scores["split"] == "test")
        & (scores["evaluation_candidate_setting"] == "formula_official_capped")
        & (scores["run_label"].isin([
            "mlp_formula", "transformer_formula", "mlp_mc_dropout", "mlp_laplace",
        ]))
    ]
    bitwise = ["bitwise_total", "bitwise_aleatoric", "bitwise_epistemic"]
    check("official manuscript bitwise selectors complete", all(name in official and official[name].notna().all() for name in bitwise), bitwise)
    random_rows = metrics[(metrics.measure == "random") & (metrics.metric == "aurc")]
    check("random AURC rows present", len(random_rows) > 0, len(random_rows))
    return {"passed": all(row["passed"] for row in checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_score", type=Path, action="append", required=True)
    parser.add_argument("--meta", type=parse_meta, action="append", default=[])
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.concat([pd.read_parquet(path) for path in args.query_score], ignore_index=True)
    scores = merge_meta_predictions(scores, args.meta)
    scores.to_parquet(args.out_dir / "query_scores.parquet", index=False, compression="zstd")

    point_rows, bootstrap_rows, rc_frames, candidate_rows = [], [], [], []
    group_keys = ["run_label", "evaluation_candidate_setting", "pool_variant", "split", "query_mask_id", "K"]
    for _, frame in scores.groupby(group_keys, sort=True, dropna=False):
        k = int(frame.K.iloc[0])
        point_rows.extend(point_metric_rows(frame, k, args.bootstrap_replicates))
        rc_frames.append(risk_coverage_rows(frame, k))
        candidate_rows.extend(candidate_summary_rows(frame, k))
        if frame.split.iloc[0] == "test" and args.bootstrap_replicates > 0:
            boot = bootstrap_frame(frame, k, args.bootstrap_replicates, args.bootstrap_seed)
            bootstrap_rows.append(add_group_columns(boot, frame))

    point = pd.DataFrame(point_rows)
    bootstrap = pd.concat(bootstrap_rows, ignore_index=True) if bootstrap_rows else pd.DataFrame()
    metrics = attach_intervals(point, bootstrap)
    metrics.to_csv(args.out_dir / "metrics_tidy.csv", index=False)
    if not bootstrap.empty:
        bootstrap.to_parquet(args.out_dir / "bootstrap_replicates.parquet", index=False, compression="zstd")
    pd.concat(rc_frames, ignore_index=True).to_parquet(args.out_dir / "risk_coverage_points.parquet", index=False, compression="zstd")
    candidate_summary = pd.DataFrame(candidate_rows)
    candidate_summary.to_csv(args.out_dir / "candidate_distribution_summary.csv", index=False)

    comparisons = []
    if args.bootstrap_replicates > 0:
        comparisons = pool_comparisons(scores, args.bootstrap_replicates, args.bootstrap_seed)
        for run_label in ["mlp_formula", "transformer_formula"]:
            official = scores[(scores.run_label == run_label) & (scores.split == "test") & (scores.evaluation_candidate_setting == "formula_official_capped")]
            for k in [1, 5, 20]:
                frame = official[official.K == k]
                if frame.empty or "meta_full" not in frame or frame.meta_full.isna().any():
                    continue
                _, rel, selectors = evaluate_frame(frame, k)
                eligible = [name for name in selectors if not name.startswith("meta_")]
                best = min(eligible, key=lambda name: (float(rel.loc[name, f"hit@{k}"]), name))
                comparisons.append(selector_comparison(frame, k, "meta_full", best, f"{run_label}: logistic minus best single", args.bootstrap_replicates, args.bootstrap_seed))
                comparisons.append(selector_comparison(frame, k, "meta_full", f"score_gap_at_{k}", f"{run_label}: logistic minus score_gap_at_K", args.bootstrap_replicates, args.bootstrap_seed))
    if comparisons:
        comparison_replicates = pd.concat(comparisons, ignore_index=True)
        comparison_replicates.to_parquet(args.out_dir / "paired_difference_replicates.parquet", index=False, compression="zstd")
        summarize_differences(comparison_replicates).to_csv(args.out_dir / "paired_differences.csv", index=False)

    build_tables(metrics, candidate_summary, args.out_dir)
    report = validation_report(scores, metrics)
    (args.out_dir / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("Canonical validation report contains failed checks")
    print(f"Saved canonical metrics and tables to {args.out_dir}")


if __name__ == "__main__":
    main()
