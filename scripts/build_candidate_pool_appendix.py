#!/usr/bin/env python3
"""Build manuscript-ready candidate-pool distribution figures and compact tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ms_uq.evaluation.visualisation import _FA, _FH, _FL, _FT, _FW, _ROW_H, _setup_ax
from scripts.build_candidate_comparison_appendix import RESULTS, TOP_KS
from scripts.plot_candidate_size_distribution import CAP, resolve_sizes, summarize


FILL_COLORS = {
    "Formula capped": "#9ecae1",
    "Formula paired capped": "#9ecae1",
    "Formula uncapped": "#fdc086",
    "Mass capped": "#8dd3c7",
}
LINE_COLORS = {
    "Formula capped": "#2878b5",
    "Formula paired capped": "#2878b5",
    "Formula uncapped": "#d97922",
    "Mass capped": "#238b7e",
}
LINESTYLES = {
    "Formula capped": (0, (5, 3)),
    "Formula paired capped": (0, (5, 3)),
    "Formula uncapped": "-",
    "Mass capped": "-",
}


def load_candidate_counts(
    dataset_tsv: Path, helper_dir: Path, max_queries: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(dataset_tsv, sep="\t", usecols=["identifier", "smiles", "inchikey", "fold"], dtype=str)
    test = metadata.loc[metadata["fold"] == "test"].copy()
    if max_queries is not None:
        test = test.iloc[:max_queries].copy()
    test["molecule_group_id"] = test["inchikey"].str.split("-").str[0]
    representatives = test.sort_values("identifier").drop_duplicates("molecule_group_id")
    queries = representatives["smiles"].tolist()
    files = {
        "Formula paired capped": helper_dir / "MassSpecGym_retrieval_candidates_formula_pubchem_record_capped256_inchi.npz",
        "Formula uncapped": helper_dir / "MassSpecGym_retrieval_candidates_formula_uncapped_inchi.npz",
        "Formula capped": helper_dir / "MassSpecGym_retrieval_candidates_formula_inchi.npz",
        "Mass capped": helper_dir / "MassSpecGym_retrieval_candidates_mass_inchi.npz",
    }
    counts: dict[str, np.ndarray] = {}
    rows = []
    for setting, path in files.items():
        with np.load(path) as candidate_map:
            sizes = np.asarray([len(candidate_map[query]) for query in queries], dtype=np.int64)
        counts[setting] = sizes
        rows.append(summarize(setting, sizes))
    return pd.DataFrame(rows), pd.DataFrame({"molecule_group_id": representatives["molecule_group_id"].to_numpy(), **counts})


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    return ordered, np.arange(1, ordered.size + 1) / ordered.size


def plot_candidate_pair(
    counts: Mapping[str, np.ndarray],
    settings: Sequence[str],
    out_prefix: Path,
    *,
    log_x: bool,
) -> None:
    first, second = settings
    values = np.concatenate([counts[first], counts[second]])
    maximum = int(values.max())
    if log_x:
        bins = np.logspace(0, np.ceil(np.log10(maximum)), 32)
        x_ticks = [tick for tick in [1, 10, 100, 1_000, 10_000] if tick <= maximum]
        x_limits = (1, maximum * 1.15)
    else:
        bins = np.arange(0, CAP + 17, 16)
        x_ticks = [0, 64, 128, 192, 256]
        x_limits = (0, CAP * 1.055)

    fig, axes = plt.subplots(1, 2, figsize=(_FW, _ROW_H + 1.0), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.72, wspace=0.25)
    for ax in axes:
        _setup_ax(ax)
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.set_xlim(*x_limits)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="both", labelsize=_FT)
        ax.axvline(CAP, color="#666666", linestyle=(0, (3, 2)), linewidth=1.8, zorder=1)
        ax.set_xlabel(r"Candidate set size $|\mathcal{C}|$", fontsize=_FL)

    # Draw the second distribution first and hatch the first so overlapping mass remains visible.
    for zorder, setting in enumerate([second, first], start=2):
        axes[0].hist(
            counts[setting],
            bins=bins,
            weights=np.full(counts[setting].size, 100.0 / counts[setting].size),
            histtype="stepfilled",
            color=FILL_COLORS[setting],
            edgecolor=LINE_COLORS[setting],
            linewidth=2.4 if setting == first else 2.1,
            alpha=0.34 if setting == first else 0.42,
            hatch="///" if setting == first else None,
            zorder=zorder,
        )

    for zorder, setting in enumerate([second, first], start=2):
        x, y = _ecdf(counts[setting])
        axes[1].step(
            x,
            y,
            where="post",
            color=LINE_COLORS[setting],
            linewidth=4.2 if setting == second else 2.8,
            linestyle=LINESTYLES[setting],
            zorder=zorder,
        )

    axes[0].set_title("(a) Candidate-set sizes", fontsize=_FH, pad=10)
    axes[0].set_ylabel("Query pools per bin (%)", fontsize=_FL)
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    axes[1].set_title("(b) Cumulative distribution", fontsize=_FH, pad=10)
    axes[1].set_ylabel("Cumulative fraction", fontsize=_FL)
    axes[1].set_ylim(0, 1.02)
    axes[1].yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    handles = [
        plt.Line2D([], [], color=LINE_COLORS[setting], linewidth=3.0,
                   linestyle=LINESTYLES[setting])
        for setting in settings
    ]
    cap_handle = plt.Line2D([], [], color="#666666", linewidth=1.8, linestyle=(0, (3, 2)))
    fig.legend(
        handles + [cap_handle],
        list(settings) + [r"Candidate cap ($|\mathcal{C}|=256$)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
        fontsize=_FA,
        handlelength=2.8,
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_histograms(counts: Mapping[str, np.ndarray], out_prefix: Path) -> None:
    """Plot the two candidate-pool comparisons without cumulative panels."""
    uncapped_max = int(counts["Formula uncapped"].max())
    formula_bins = np.logspace(0, np.ceil(np.log10(uncapped_max)), 32)
    capped_bins = np.arange(0, CAP + 17, 16)

    fig, axes = plt.subplots(1, 2, figsize=(_FW, _ROW_H + 0.8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.73, wspace=0.24)
    panels = [
        (axes[0], ["Formula uncapped", "Formula paired capped"], formula_bins, "(a) Formula candidates"),
        (axes[1], ["Mass capped", "Formula capped"], capped_bins, "(b) Capped candidates"),
    ]
    for ax, settings, bins, title in panels:
        _setup_ax(ax)
        for zorder, setting in enumerate(settings, start=2):
            ax.hist(
                counts[setting],
                bins=bins,
                weights=np.full(counts[setting].size, 100.0 / counts[setting].size),
                histtype="stepfilled",
                color=FILL_COLORS[setting],
                edgecolor=LINE_COLORS[setting],
                linewidth=2.4 if "capped" in setting.lower() and "uncapped" not in setting.lower() else 2.1,
                alpha=0.34 if "capped" in setting.lower() and "uncapped" not in setting.lower() else 0.42,
                hatch="///" if "capped" in setting.lower() and "uncapped" not in setting.lower() else None,
                zorder=zorder,
            )
        ax.axvline(CAP, color="#666666", linestyle=(0, (3, 2)), linewidth=1.8, zorder=4)
        ax.set_title(title, fontsize=_FH, pad=10)
        ax.set_xlabel(r"Candidate set size $|\mathcal{C}|$", fontsize=_FL)
        ax.set_ylabel("Query pools per bin (%)", fontsize=_FL)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.tick_params(axis="both", labelsize=_FT)

    axes[0].set_xscale("log")
    axes[0].set_xlim(1, uncapped_max * 1.15)
    axes[0].set_xticks([1, 10, 100, 1_000, 10_000])
    axes[0].xaxis.set_major_formatter(mticker.ScalarFormatter())
    axes[0].xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))

    axes[1].set_xlim(0, CAP * 1.055)
    axes[1].set_xticks([0, 64, 128, 192, 256])

    legend_handles = [
        Patch(
            facecolor=FILL_COLORS[setting],
            edgecolor=LINE_COLORS[setting],
            linewidth=2.0,
            hatch="///" if setting == "Formula capped" else None,
            label=setting,
        )
        for setting in ["Formula capped", "Formula uncapped", "Mass capped"]
    ]
    legend_handles.append(
        plt.Line2D(
            [], [], color="#666666", linewidth=1.8, linestyle=(0, (3, 2)),
            label=r"Cap ($|\mathcal{C}|=256$)",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=4,
        frameon=False,
        fontsize=_FA,
        handlelength=2.4,
        columnspacing=1.5,
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_model_results() -> pd.DataFrame:
    rows = []
    for (architecture, setting), result_dir in RESULTS.items():
        hits = pd.read_csv(result_dir / "hit_rates_aggregate.csv")
        score_row = hits.loc[hits["aggregation"] == "score"].iloc[0]
        rel = pd.read_csv(result_dir / "rel_aurc_retrieval_score.csv", index_col=0)
        row = {
            "architecture": architecture,
            "training_candidates": "Mass capped" if setting == "Mass capped" else "Formula capped",
            "evaluation_candidates": setting,
        }
        for k in TOP_KS:
            row[f"hit@{k}"] = float(score_row[f"hit@{k}"])
            row[f"relAURC_conf@{k}"] = float(rel.loc["confidence", f"hit@{k}"])
            row[f"relAURC_total@{k}"] = float(rel.loc["retrieval_total", f"hit@{k}"])
        rows.append(row)
    return pd.DataFrame(rows)


def _format_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_manuscript_tables(stats: pd.DataFrame, results: pd.DataFrame, out_path: Path) -> None:
    stats_display = stats.copy()
    stats_display["IQR"] = stats_display.apply(
        lambda row: f"{row['q25']:.1f}-{row['q75']:.1f}", axis=1
    )
    stats_display["At cap (%)"] = 100.0 * stats_display["fraction_equal_256"]
    stats_display["Above cap (%)"] = 100.0 * stats_display["fraction_above_256"]
    stats_columns = ["setting", "n_queries", "median", "IQR", "mean", "max", "At cap (%)", "Above cap (%)"]

    lines = [
        "# Recommended Candidate-Pool Appendix Tables",
        "",
        "## Table A. Candidate-pool characteristics on the official test fold",
        "",
        "| Candidate setting | Queries | Median | IQR | Mean | Maximum | At 256 (%) | Above 256 (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats_display[stats_columns].itertuples(index=False, name=None):
        setting, n_queries, median, iqr, mean, maximum, at_cap, above_cap = row
        lines.append(
            f"| {setting} | {int(n_queries)} | {median:.1f} | {iqr} | {mean:.1f} | "
            f"{int(maximum)} | {at_cap:.1f} | {above_cap:.1f} |"
        )

    lines.extend([
        "",
        "## Table B. Retrieval and selective-prediction performance",
        "",
        "All values use score aggregation, InChIKey-based correctness, and T_eval = 0.003. "
        "Lower relAURC is better.",
        "",
        "| Architecture | Training candidates | Evaluation candidates | Hit@1 | Hit@5 | Hit@20 | conf relAURC@1 | conf relAURC@5 | conf relAURC@20 | total relAURC@1 | total relAURC@5 | total relAURC@20 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in results.iterrows():
        values = [
            row["architecture"],
            row["training_candidates"],
            row["evaluation_candidates"],
            *[_format_number(row[f"hit@{k}"]) for k in TOP_KS],
            *[_format_number(row[f"relAURC_conf@{k}"]) for k in TOP_KS],
            *[_format_number(row[f"relAURC_total@{k}"]) for k in TOP_KS],
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "The mass-filtered MLP is separately trained on mass-filtered candidates; therefore, the "
        "formula-versus-mass rows compare matched end-to-end pipelines rather than an evaluation-only candidate swap.",
        "Transformer results for mass-filtered training are not yet available.",
    ])
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_tsv", type=Path, default=Path("/data/home/mira/data/msuq/MassSpecGym.tsv"))
    parser.add_argument("--helper_dir", type=Path, default=Path("/data/home/mira/data/msuq"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/revision_candidate_comparison/appendix_structured"))
    parser.add_argument("--distributions_only", action="store_true")
    parser.add_argument("--max_queries", type=int)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats, per_query = load_candidate_counts(
        args.dataset_tsv, args.helper_dir, max_queries=args.max_queries
    )
    counts = {column: per_query[column].to_numpy() for column in stats["setting"]}
    plot_candidate_pair(
        counts,
        ["Formula paired capped", "Formula uncapped"],
        args.out_dir / "candidate_distribution_formula_capped_vs_uncapped",
        log_x=True,
    )
    plot_candidate_pair(
        counts,
        ["Formula capped", "Mass capped"],
        args.out_dir / "candidate_distribution_formula_vs_mass_capped",
        log_x=False,
    )

    plot_combined_histograms(
        counts,
        args.out_dir / "candidate_distribution_histograms",
    )

    stats.to_csv(args.out_dir / "table_A_candidate_pool_statistics.csv", index=False)
    per_query.to_csv(args.out_dir / "candidate_counts_per_query.csv", index=False)
    if not args.distributions_only:
        results = build_model_results()
        results.to_csv(args.out_dir / "table_B_model_results.csv", index=False)
        write_manuscript_tables(stats, results, args.out_dir / "recommended_tables.md")
    print(f"Saved structured appendix outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
