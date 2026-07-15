#!/usr/bin/env python3
"""Build appendix figure and tables for candidate-pool comparisons."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from ms_uq.evaluation.revision_features import hit_arrays
from ms_uq.evaluation.visualisation import _FA, _FH, _FL, _FT, _FW, _ROW_H, _setup_ax
from ms_uq.utils import is_confidence_score
from scripts.plot_candidate_size_distribution import resolve_sizes, summarize


COLORS = {
    "Formula capped": "#2878b5",
    "Formula uncapped": "#d97922",
    "Mass capped": "#2a9d8f",
}
MEASURE_STYLES = {
    "confidence": "-",
    "retrieval_total": (0, (5, 3)),
}
MEASURE_LABELS = {
    "confidence": r"$\kappa_{\rm conf}$",
    "retrieval_total": "Retrieval total uncertainty",
}
TOP_KS = [1, 5, 20]
SELECTED_MEASURES = [
    "confidence",
    "retrieval_total",
    "retrieval_aleatoric",
    "normalized_entropy",
    "score_gap",
    "rank_variance",
    "n_candidates",
]


RESULTS = {
    ("MLP", "Formula capped"): Path("outputs/revision_candidate_comparison/formula_capped/mlp"),
    ("MLP", "Formula uncapped"): Path("outputs/revision_uncapped/eval/mlp"),
    ("MLP", "Mass capped"): Path("outputs/mass_mlp_bienc/eval/ensemble/bienc_mass"),
    ("Transformer", "Formula capped"): Path("outputs/revision_candidate_comparison/formula_capped/transformer"),
    ("Transformer", "Formula uncapped"): Path("outputs/revision_uncapped/eval/transformer"),
}


def _risk_curve(loss: np.ndarray, values: np.ndarray, measure: str) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    loss = loss[valid]
    uncertainty = values[valid]
    if is_confidence_score(measure):
        uncertainty = -uncertainty
    order = np.argsort(uncertainty, kind="stable")
    ordered_loss = loss[order]
    cumulative_risk = np.cumsum(ordered_loss) / np.arange(1, ordered_loss.size + 1)
    coverage = np.arange(1, ordered_loss.size + 1) / ordered_loss.size
    keep = np.unique(np.linspace(0, ordered_loss.size - 1, 300).astype(int))
    return coverage[keep], cumulative_risk[keep]


def _load_hit1_and_uncertainties(result_dir: Path) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    score_data = torch.load(result_dir / "scores_ranker_score.pt", map_location="cpu")
    hits = hit_arrays(
        score_data["scores_flat"].float(),
        score_data["labels_flat"].float(),
        score_data["ptr"].long(),
        [1],
    )
    uncertainties_npz = np.load(result_dir / "uncertainties_score.npz")
    uncertainties = {name: uncertainties_npz[name] for name in uncertainties_npz.files}
    return 1.0 - hits["hit@1"].astype(float), uncertainties


def plot_risk_coverage(out_prefix: Path) -> None:
    panels = [
        ("(a) Formula cap: MLP", "MLP", ["Formula capped", "Formula uncapped"]),
        ("(b) Formula cap: Transformer", "Transformer", ["Formula capped", "Formula uncapped"]),
        ("(c) Matched training: MLP", "MLP", ["Formula capped", "Mass capped"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(_FW, _ROW_H + 0.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.72, wspace=0.22)

    for ax, (title, architecture, settings) in zip(axes, panels):
        _setup_ax(ax)
        for setting in settings:
            loss, uncertainties = _load_hit1_and_uncertainties(RESULTS[(architecture, setting)])
            for measure in MEASURE_STYLES:
                coverage, risk = _risk_curve(loss, uncertainties[measure], measure)
                ax.plot(
                    coverage,
                    risk,
                    color=COLORS[setting],
                    linestyle=MEASURE_STYLES[measure],
                    linewidth=2.8,
                )
            del loss, uncertainties
            gc.collect()

        ax.set_title(title, fontsize=_FH, pad=10)
        ax.set_xlabel("Coverage", fontsize=_FL)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 0.95)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.tick_params(axis="both", labelsize=_FT)
    axes[0].set_ylabel("Hit@1 error rate", fontsize=_FL)

    condition_handles = [
        Line2D([], [], color=COLORS[name], linewidth=3, label=name)
        for name in ["Formula capped", "Formula uncapped", "Mass capped"]
    ]
    measure_handles = [
        Line2D([], [], color="#333333", linestyle=MEASURE_STYLES[name], linewidth=2.8,
               label=MEASURE_LABELS[name])
        for name in MEASURE_STYLES
    ]
    fig.legend(
        condition_handles + measure_handles,
        [handle.get_label() for handle in condition_handles + measure_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=5,
        frameon=False,
        fontsize=_FA,
        handlelength=2.8,
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def candidate_statistics(dataset_tsv: Path, helper_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(dataset_tsv, sep="\t", usecols=["smiles", "fold"], dtype=str)
    queries = metadata.loc[metadata["fold"] == "test", "smiles"].drop_duplicates().tolist()
    files = {
        "Formula capped": helper_dir / "MassSpecGym_retrieval_candidates_formula.json",
        "Formula uncapped": helper_dir / "MassSpecGym_retrieval_candidates_formula_uncapped.json",
        "Mass capped": helper_dir / "MassSpecGym_retrieval_candidates_mass.json",
    }
    counts = {}
    rows = []
    for setting, path in files.items():
        with path.open() as handle:
            candidate_map = json.load(handle)
        sizes = resolve_sizes(candidate_map, queries)
        counts[setting] = sizes
        rows.append(summarize(setting, sizes))
    per_query = pd.DataFrame({"smiles": queries, **counts})
    return pd.DataFrame(rows), per_query


def retrieval_performance() -> pd.DataFrame:
    rows = []
    for (architecture, setting), result_dir in RESULTS.items():
        hit_rates = pd.read_csv(result_dir / "hit_rates_aggregate.csv")
        score_row = hit_rates.loc[hit_rates["aggregation"] == "score"].iloc[0]
        rows.append({
            "architecture": architecture,
            "candidate_setting": setting,
            "training_setting": "Mass capped" if setting == "Mass capped" else "Formula capped",
            "evaluation_setting": setting,
            "hit@1": float(score_row["hit@1"]),
            "hit@5": float(score_row["hit@5"]),
            "hit@20": float(score_row["hit@20"]),
            "temperature": 0.003,
            "label_mode": "inchikey_fallback",
            "result_dir": str(result_dir.resolve()),
        })
    return pd.DataFrame(rows)


def comparison_deltas(performance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cap_rows = []
    for architecture in ["MLP", "Transformer"]:
        subset = performance[performance["architecture"] == architecture].set_index("candidate_setting")
        row = {"architecture": architecture}
        for k in TOP_KS:
            col = f"hit@{k}"
            row[f"capped_{col}"] = subset.loc["Formula capped", col]
            row[f"uncapped_{col}"] = subset.loc["Formula uncapped", col]
            row[f"delta_{col}"] = row[f"uncapped_{col}"] - row[f"capped_{col}"]
        cap_rows.append(row)

    mlp = performance[performance["architecture"] == "MLP"].set_index("candidate_setting")
    rule_row = {"architecture": "MLP"}
    for k in TOP_KS:
        col = f"hit@{k}"
        rule_row[f"formula_{col}"] = mlp.loc["Formula capped", col]
        rule_row[f"mass_{col}"] = mlp.loc["Mass capped", col]
        rule_row[f"delta_{col}"] = rule_row[f"mass_{col}"] - rule_row[f"formula_{col}"]
    return pd.DataFrame(cap_rows), pd.DataFrame([rule_row])


def selected_rel_aurc() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (architecture, setting), result_dir in RESULTS.items():
        rel = pd.read_csv(result_dir / "rel_aurc_retrieval_score.csv", index_col=0)
        for k in TOP_KS:
            column = f"hit@{k}"
            for measure in SELECTED_MEASURES:
                source_measure = f"rank_var_{k}" if measure == "rank_variance" else measure
                rows.append({
                    "architecture": architecture,
                    "candidate_setting": setting,
                    "hit_k": k,
                    "measure": measure,
                    "source_measure": source_measure,
                    "rel_aurc": float(rel.loc[source_measure, column]),
                })
    long = pd.DataFrame(rows)
    wide = long.pivot_table(
        index=["architecture", "candidate_setting", "measure"],
        columns="hit_k",
        values="rel_aurc",
    ).rename(columns={k: f"hit@{k}" for k in TOP_KS}).reset_index()
    return long, wide


def _markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    formatted = df.copy()
    for column in formatted.select_dtypes(include=[np.number]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{digits}f}")
    headers = [str(column) for column in formatted.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in formatted.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_summary_markdown(
    out_path: Path,
    stats: pd.DataFrame,
    performance: pd.DataFrame,
    cap_delta: pd.DataFrame,
    rule_delta: pd.DataFrame,
) -> None:
    sections = [
        "# Candidate-Pool Appendix Tables",
        "",
        "## Candidate-Pool Statistics",
        "",
        _markdown_table(stats),
        "",
        "## Retrieval Performance",
        "",
        _markdown_table(performance[["architecture", "candidate_setting", "training_setting", "hit@1", "hit@5", "hit@20"]]),
        "",
        "## Formula Cap Effect",
        "",
        _markdown_table(cap_delta),
        "",
        "## Formula Versus Mass (Matched MLP Pipelines)",
        "",
        _markdown_table(rule_delta),
        "",
        "Transformer mass results are not yet available.",
    ]
    out_path.write_text("\n".join(sections) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_tsv", type=Path, default=Path("/data/home/mira/data/msuq/MassSpecGym.tsv"))
    parser.add_argument("--helper_dir", type=Path, default=Path("/data/home/mira/data/msuq"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/revision_candidate_comparison/appendix"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_risk_coverage(args.out_dir / "candidate_setting_hit1_risk_coverage")
    stats, per_query = candidate_statistics(args.dataset_tsv, args.helper_dir)
    performance = retrieval_performance()
    cap_delta, rule_delta = comparison_deltas(performance)
    rel_long, rel_wide = selected_rel_aurc()

    stats.to_csv(args.out_dir / "table_candidate_pool_statistics.csv", index=False)
    per_query.to_csv(args.out_dir / "candidate_counts_per_query.csv", index=False)
    performance.to_csv(args.out_dir / "table_retrieval_performance.csv", index=False)
    cap_delta.to_csv(args.out_dir / "table_formula_cap_effect.csv", index=False)
    rule_delta.to_csv(args.out_dir / "table_formula_vs_mass_mlp.csv", index=False)
    rel_long.to_csv(args.out_dir / "table_rel_aurc_selected_long.csv", index=False)
    rel_wide.to_csv(args.out_dir / "table_rel_aurc_selected_wide.csv", index=False)
    write_summary_markdown(args.out_dir / "appendix_tables.md", stats, performance, cap_delta, rule_delta)

    print(f"Saved appendix outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
