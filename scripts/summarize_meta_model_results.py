#!/usr/bin/env python3
"""Summarize T=0.003 logistic/SVM meta-score results for manuscript notes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("outputs/revision_meta_t0003")
TOP_KS = [1, 5, 20]
FEATURES = [
    "s1",
    "score_gap",
    "score_gap_at_K",
    "log_n_candidates",
    "confidence",
    "normalized_entropy",
    "rank_var_K",
    "retrieval_total",
    "retrieval_aleatoric",
    "retrieval_epistemic",
    "pred_fp_cardinality",
    "top_candidate_fp_cardinality",
    "cardinality_mismatch",
    "precursor_mz",
    "n_peaks",
]


def run_specs(root: Path) -> List[Dict[str, Path | str]]:
    return [
        {
            "architecture": "MLP",
            "meta_model": "logistic",
            "run_dir": root / "mlp_logistic",
            "plot_dir": root / "joint_plots_logistic",
            "aurc_glob": "mlp*_aurc.csv",
        },
        {
            "architecture": "Transformer",
            "meta_model": "logistic",
            "run_dir": root / "transformer_logistic",
            "plot_dir": root / "joint_plots_logistic",
            "aurc_glob": "transformer*_aurc.csv",
        },
        {
            "architecture": "MLP",
            "meta_model": "linear_svm",
            "run_dir": root / "mlp_linear_svm",
            "plot_dir": root / "joint_plots_linear_svm",
            "aurc_glob": "mlp*_aurc.csv",
        },
        {
            "architecture": "Transformer",
            "meta_model": "linear_svm",
            "run_dir": root / "transformer_linear_svm",
            "plot_dir": root / "joint_plots_linear_svm",
            "aurc_glob": "transformer*_aurc.csv",
        },
    ]


def first_match(plot_dir: Path, pattern: str) -> Path:
    matches = sorted(plot_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} in {plot_dir}")
    return matches[0]


def best_cv_row(cv: pd.DataFrame, task: str) -> pd.Series:
    sub = cv[cv["task"] == task].copy()
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.sort_values("rank_test_score").iloc[0]


def coefficient_summary(coef: pd.DataFrame, architecture: str, meta_model: str, n_top: int = 8) -> pd.DataFrame:
    rows = []
    for task, sub in coef[coef["feature"] != "intercept"].groupby("task"):
        sub = sub.copy()
        sub["abs_coefficient"] = sub["coefficient"].abs()
        top = sub.sort_values("abs_coefficient", ascending=False).head(n_top)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            rows.append({
                "architecture": architecture,
                "meta_model": meta_model,
                "task": task,
                "rank": rank,
                "feature": row["feature"],
                "coefficient": row["coefficient"],
                "abs_coefficient": row["abs_coefficient"],
                "best_C": row["best_C"],
            })
    return pd.DataFrame(rows)


def summarize_run(spec: Dict[str, Path | str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    architecture = str(spec["architecture"])
    meta_model = str(spec["meta_model"])
    run_dir = Path(spec["run_dir"])
    rel = pd.read_csv(run_dir / "meta_rel_aurc.csv", index_col=0)
    raw_reference = pd.read_csv(first_match(Path(spec["plot_dir"]), str(spec["aurc_glob"])), index_col=0)
    raw = rel.copy()
    for col in [f"hit@{k}" for k in TOP_KS]:
        oracle = float(raw_reference.loc["oracle", col])
        random = float(raw_reference.loc["random", col])
        raw[col] = pd.to_numeric(rel[col], errors="coerce") * (random - oracle) + oracle
    raw.loc["oracle", [f"hit@{k}" for k in TOP_KS]] = raw_reference.loc["oracle", [f"hit@{k}" for k in TOP_KS]]
    raw.loc["random", [f"hit@{k}" for k in TOP_KS]] = raw_reference.loc["random", [f"hit@{k}" for k in TOP_KS]]
    coef = pd.read_csv(run_dir / "meta_coefficients.csv")
    cv = pd.read_csv(run_dir / "meta_cv_results.csv")
    config = pd.read_csv(run_dir / "meta_run_config.csv").iloc[0].to_dict()

    rows = []
    for k in TOP_KS:
        col = f"hit@{k}"
        task = f"meta_hit{k}"

        rel_candidates = [idx for idx in rel.index if not str(idx).startswith("meta_hit")]
        rel_values = rel.loc[rel_candidates, col].dropna().astype(float)
        best_single = rel_values.idxmin()

        raw_candidates = [idx for idx in raw.index if not str(idx).startswith("meta_hit") and idx not in {"oracle", "random"}]
        raw_values = raw.loc[raw_candidates, col].dropna().astype(float)
        best_raw = raw_values.idxmin()

        coeff_task = coef[coef["task"] == task]
        cv_best = best_cv_row(cv, task)
        rows.append({
            "architecture": architecture,
            "meta_model": meta_model,
            "temperature": float(config["temperature"]),
            "score_output": config["score_output"],
            "task": task,
            "hit_k": k,
            "meta_rel_aurc": float(rel.loc[task, col]),
            "best_single_measure_rel": best_single,
            "best_single_rel_aurc": float(rel.loc[best_single, col]),
            "rel_aurc_gain": float(rel.loc[best_single, col] - rel.loc[task, col]),
            "meta_raw_aurc": float(raw.loc[task, col]),
            "best_single_measure_raw": best_raw,
            "best_single_raw_aurc": float(raw.loc[best_raw, col]),
            "raw_aurc_gain": float(raw.loc[best_raw, col] - raw.loc[task, col]),
            "best_C": float(coeff_task["best_C"].iloc[0]) if not coeff_task.empty else np.nan,
            "cv_scoring": cv_best.get("cv_scoring", ""),
            "cv_best_score": float(cv_best["mean_test_score"]) if "mean_test_score" in cv_best else np.nan,
        })

    sgr_rows = []
    sgr = pd.read_csv(run_dir / "meta_sgr_results.csv")
    for k in TOP_KS:
        loss = f"hit@{k}"
        task = f"meta_hit{k}"
        for target, sub in sgr[sgr["loss"] == loss].groupby("target_risk"):
            meta_row = sub[sub["measure"] == task]
            singles = sub[~sub["measure"].str.startswith("meta_hit")].copy()
            singles = singles[singles["feasible"].astype(bool)]
            best_single = singles.sort_values("eval_coverage", ascending=False).head(1)
            row = {
                "architecture": architecture,
                "meta_model": meta_model,
                "loss": loss,
                "target_risk": target,
                "meta_feasible": bool(meta_row["feasible"].iloc[0]) if not meta_row.empty else False,
                "meta_eval_coverage": float(meta_row["eval_coverage"].iloc[0]) if not meta_row.empty else np.nan,
                "meta_eval_empirical_risk": float(meta_row["eval_empirical_risk"].iloc[0]) if not meta_row.empty else np.nan,
            }
            if not best_single.empty:
                best = best_single.iloc[0]
                row.update({
                    "best_single_measure": best["measure"],
                    "best_single_eval_coverage": float(best["eval_coverage"]),
                    "best_single_eval_empirical_risk": float(best["eval_empirical_risk"]),
                    "eval_coverage_gain": float(row["meta_eval_coverage"] - best["eval_coverage"]),
                })
            sgr_rows.append(row)

    return pd.DataFrame(rows), coefficient_summary(coef, architecture, meta_model), pd.DataFrame(sgr_rows)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        if pd.isna(value):
            return ""
        return str(value)

    cols = list(df.columns)
    rows = [[fmt(row[col]) for col in cols] for _, row in df.iterrows()]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep] + body)

def markdown_summary(summary: pd.DataFrame, coef_top: pd.DataFrame, out_dir: Path) -> str:
    compact = summary.pivot_table(
        index=["architecture", "meta_model"],
        columns="hit_k",
        values=["meta_rel_aurc", "rel_aurc_gain"],
        aggfunc="first",
    )
    compact.columns = [f"{metric}_hit{k}" for metric, k in compact.columns]
    compact = compact.reset_index()

    lines = [
        "# Meta-Score Summary",
        "",
        "All meta-scores were trained only on the official MassSpecGym validation fold and evaluated frozen on the official test fold. The softmax-derived features use evaluation temperature `T_eval = 0.003`, matching the rankwise training temperature.",
        "",
        "For each architecture and each Hit@K task, a separate linear meta-model was fit. Logistic regression uses L2-regularized `predict_proba` scores and tunes `C` by grouped validation-fold cross-validation with negative log-loss. The linear SVM uses `LinearSVC` with an L2 squared-hinge objective and uses the signed decision function as the confidence score; `C` is tuned with grouped cross-validation using ROC-AUC. Groups are validation-fold InChIKeys.",
        "",
        "Features used: " + ", ".join(f"`{name}`" for name in FEATURES) + ".",
        "",
        "Lower relAURC and AURC are better; positive gain means the meta-score improves over the best single non-meta selector in that column.",
        "",
        "## Compact relAURC Table",
        "",
        df_to_markdown(compact),
        "",
        "## Largest Coefficients",
        "",
        df_to_markdown(coef_top[coef_top["rank"] <= 5]),
        "",
        "## Output Files",
        "",
        f"- Summary CSV: `{out_dir / 'meta_model_summary.csv'}`",
        f"- Top coefficients CSV: `{out_dir / 'meta_model_top_coefficients.csv'}`",
        f"- SGR summary CSV: `{out_dir / 'meta_model_sgr_summary.csv'}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_ROOT / "summary")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries, coeffs, sgrs = [], [], []
    for spec in run_specs(args.root):
        summary, coef, sgr = summarize_run(spec)
        summaries.append(summary)
        coeffs.append(coef)
        sgrs.append(sgr)

    summary_df = pd.concat(summaries, ignore_index=True)
    coef_df = pd.concat(coeffs, ignore_index=True)
    sgr_df = pd.concat(sgrs, ignore_index=True)
    summary_df.to_csv(args.out_dir / "meta_model_summary.csv", index=False)
    coef_df.to_csv(args.out_dir / "meta_model_top_coefficients.csv", index=False)
    sgr_df.to_csv(args.out_dir / "meta_model_sgr_summary.csv", index=False)
    (args.out_dir / "meta_model_summary.md").write_text(markdown_summary(summary_df, coef_df, args.out_dir))
    print(f"Saved meta-model summary to {args.out_dir}")


if __name__ == "__main__":
    main()
