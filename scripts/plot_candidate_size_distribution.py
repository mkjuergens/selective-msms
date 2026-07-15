#!/usr/bin/env python3
"""Paper-style comparison of capped and uncapped candidate-set sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from rdkit import Chem

from ms_uq.evaluation.visualisation import _FA, _FH, _FL, _FT, _FW, _ROW_H, _setup_ax


COLORS = {
    "Capped": "#7fbfff",
    "Uncapped": "#fdc086",
}
LINE_COLORS = {
    "Capped": "#2878b5",
    "Uncapped": "#d97922",
}
CAP = 256


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def resolve_sizes(candidate_map: Dict[str, Sequence[str]], queries: Sequence[str]) -> np.ndarray:
    """Return candidate counts aligned to original query SMILES."""
    missing = [query for query in queries if query not in candidate_map]
    canonical_map = None
    if missing:
        canonical_map = {}
        for key in candidate_map:
            canonical = canonical_smiles(key)
            if canonical is not None:
                canonical_map.setdefault(canonical, key)

    sizes = []
    unresolved = []
    for query in queries:
        key = query
        if key not in candidate_map and canonical_map is not None:
            key = canonical_map.get(canonical_smiles(query))
        if key is None or key not in candidate_map:
            unresolved.append(query)
            continue
        sizes.append(len(candidate_map[key]))

    if unresolved:
        raise ValueError(
            f"Could not resolve {len(unresolved)} query SMILES; examples: {unresolved[:3]}"
        )
    return np.asarray(sizes, dtype=np.int64)


def summarize(setting: str, sizes: np.ndarray) -> dict:
    return {
        "setting": setting,
        "n_queries": int(sizes.size),
        "min": int(sizes.min()),
        "q25": float(np.quantile(sizes, 0.25)),
        "median": float(np.median(sizes)),
        "mean": float(sizes.mean()),
        "q75": float(np.quantile(sizes, 0.75)),
        "max": int(sizes.max()),
        "n_equal_256": int(np.count_nonzero(sizes == CAP)),
        "fraction_equal_256": float(np.mean(sizes == CAP)),
        "n_above_256": int(np.count_nonzero(sizes > CAP)),
        "fraction_above_256": float(np.mean(sizes > CAP)),
    }


def plot_distributions(capped: np.ndarray, uncapped: np.ndarray, out_prefix: Path) -> None:
    max_size = int(max(capped.max(), uncapped.max()))
    bins = np.logspace(0, np.ceil(np.log10(max_size)), 32)
    x_ticks = [tick for tick in [1, 10, 100, 1_000, 10_000] if tick <= max_size]

    fig, axes = plt.subplots(1, 2, figsize=(_FW, _ROW_H + 1.0), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.72, wspace=0.25)

    for ax in axes:
        _setup_ax(ax)
        ax.set_xscale("log")
        ax.set_xlim(1, max_size * 1.15)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.tick_params(axis="both", labelsize=_FT)
        ax.axvline(CAP, color="#666666", linestyle="--", linewidth=2.0, zorder=1)
        ax.set_xlabel(r"Candidate set size $|\mathcal{C}|$", fontsize=_FL)

    axes[0].hist(
        uncapped,
        bins=bins,
        weights=np.full(uncapped.size, 100.0 / uncapped.size),
        histtype="stepfilled",
        color=COLORS["Uncapped"],
        edgecolor=LINE_COLORS["Uncapped"],
        linewidth=2.2,
        alpha=0.38,
        zorder=2,
    )
    axes[0].hist(
        capped,
        bins=bins,
        weights=np.full(capped.size, 100.0 / capped.size),
        histtype="stepfilled",
        color=COLORS["Capped"],
        edgecolor=LINE_COLORS["Capped"],
        linewidth=2.8,
        alpha=0.32,
        hatch="///",
        zorder=3,
    )

    uncapped_ordered = np.sort(uncapped)
    uncapped_cdf = np.arange(1, uncapped_ordered.size + 1) / uncapped_ordered.size
    axes[1].step(
        uncapped_ordered,
        uncapped_cdf,
        where="post",
        color=LINE_COLORS["Uncapped"],
        linewidth=4.5,
        zorder=2,
    )
    capped_ordered = np.sort(capped)
    capped_cdf = np.arange(1, capped_ordered.size + 1) / capped_ordered.size
    axes[1].step(
        capped_ordered,
        capped_cdf,
        where="post",
        color=LINE_COLORS["Capped"],
        linewidth=2.7,
        linestyle=(0, (5, 3)),
        zorder=4,
    )

    axes[0].set_title("(a) Candidate-set sizes", fontsize=_FH, pad=10)
    axes[0].set_ylabel("Query pools per bin (%)", fontsize=_FL)
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    axes[1].set_title("(b) Cumulative distribution", fontsize=_FH, pad=10)
    axes[1].set_ylabel("Cumulative fraction", fontsize=_FL)
    axes[1].set_ylim(0, 1.02)
    axes[1].yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    handles = [
        plt.Line2D([], [], color=LINE_COLORS["Capped"], linewidth=2.7,
                   linestyle=(0, (5, 3))),
        plt.Line2D([], [], color=LINE_COLORS["Uncapped"], linewidth=4.0),
    ]
    labels = ["Capped", "Uncapped"]
    cap_handle = plt.Line2D([], [], color="#666666", linestyle="--", linewidth=2.0)
    fig.legend(
        handles + [cap_handle],
        labels + [r"Original cap ($|\mathcal{C}|=256$)"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_tsv", type=Path, required=True)
    parser.add_argument("--capped_json", type=Path, required=True)
    parser.add_argument("--uncapped_json", type=Path, required=True)
    parser.add_argument("--out_prefix", type=Path, required=True)
    parser.add_argument("--fold", default="test")
    args = parser.parse_args()

    metadata = pd.read_csv(args.dataset_tsv, sep="\t", usecols=["smiles", "fold"], dtype=str)
    queries = metadata.loc[metadata["fold"] == args.fold, "smiles"].drop_duplicates().tolist()

    with args.capped_json.open() as handle:
        capped_map = json.load(handle)
    with args.uncapped_json.open() as handle:
        uncapped_map = json.load(handle)

    capped = resolve_sizes(capped_map, queries)
    uncapped = resolve_sizes(uncapped_map, queries)
    if capped.shape != uncapped.shape:
        raise ValueError("Capped and uncapped query counts do not match")

    per_query = pd.DataFrame({
        "smiles": queries,
        "capped_n_candidates": capped,
        "uncapped_n_candidates": uncapped,
        "expansion_factor": uncapped / capped,
    })
    per_query.to_csv(args.out_prefix.with_name(args.out_prefix.name + "_per_query.csv"), index=False)

    summary = pd.DataFrame([
        summarize("Capped", capped),
        summarize("Uncapped", uncapped),
    ])
    summary["median_expansion_factor"] = [np.nan, float(np.median(uncapped / capped))]
    summary.to_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), index=False)

    plot_distributions(capped, uncapped, args.out_prefix)
    print(summary.to_string(index=False))
    print(f"Saved {args.out_prefix.with_suffix('.pdf')}")
    print(f"Saved {args.out_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
