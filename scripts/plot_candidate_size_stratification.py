#!/usr/bin/env python3
"""Plot manuscript-style Hit@K and AURC stratified by candidate-set size."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ms_uq.evaluation.confidence_features import canonical_aurc_table
from ms_uq.evaluation.visualisation import (
    _BAR_PITCH,
    _FA,
    _FH,
    _FL,
    _FT,
    _FW,
    _format_bar_panel,
    _setup_ax,
)


TOP_KS = [1, 5, 20]
HIT_COLORS = {1: "#7fbfff", 5: "#fdc086", 20: "#c9b3d9"}
BINS = [0, 2, 10, 50, 100, 200, 256, 257]
BIN_LABELS = ["1", "2-9", "10-49", "50-99", "100-199", "200-255", "256"]
AURC_MEASURES = [
    "confidence",
    "score_gap",
    "rank_var_1",
    "retrieval_aleatoric",
    "retrieval_epistemic",
    "n_candidates",
]


def load_official_scores(path: Path, run_label: str) -> pd.DataFrame:
    scores = pd.read_parquet(path)
    scores = scores[
        (scores["run_label"] == run_label)
        & (scores["split"] == "test")
        & (scores["evaluation_candidate_setting"] == "formula_official_capped")
    ].copy()
    if scores.empty:
        raise ValueError(f"No official formula test rows for {run_label}")
    for k in TOP_KS:
        frame = scores[scores["K"] == k]
        if len(frame) != scores["query_id"].nunique():
            raise ValueError(f"Incomplete Hit@{k} rows for {run_label}")
    temperatures = scores["T_eval"].drop_duplicates().to_numpy(dtype=float)
    if len(temperatures) != 1 or not np.isclose(temperatures[0], 0.003):
        raise ValueError(f"Expected T_eval=0.003, found {temperatures.tolist()}")
    return scores


def hit_rate_table(scores: pd.DataFrame) -> pd.DataFrame:
    base = scores[scores["K"] == 1][["query_id", "candidate_count"]].copy()
    base["candidate_bin"] = np.digitize(base["candidate_count"].to_numpy(), BINS) - 1
    rows = []
    for k in TOP_KS:
        hit = scores[scores["K"] == k][["query_id", "hit"]]
        merged = base.merge(hit, on="query_id", how="left", validate="one_to_one")
        for bin_index, label in enumerate(BIN_LABELS):
            subset = merged[merged["candidate_bin"] == bin_index]
            rows.append({
                "candidate_bin": label,
                "bin_index": bin_index,
                "K": k,
                "n_spectra": len(subset),
                "hit_rate": float(subset["hit"].mean()) if len(subset) else np.nan,
            })
    return pd.DataFrame(rows)


def stratified_aurc(scores: pd.DataFrame) -> pd.DataFrame:
    frame = scores[scores["K"] == 1].reset_index(drop=True)
    masks = {
        "candidate_count_lt_256": frame["candidate_count"].to_numpy() < 256,
        "candidate_count_eq_256": frame["candidate_count"].to_numpy() == 256,
    }
    rows = []
    for stratum, mask in masks.items():
        subset = frame.loc[mask].reset_index(drop=True)
        selectors = {
            name: subset[name].to_numpy(dtype=np.float64)
            for name in AURC_MEASURES if name in subset
        }
        tie_break = "source_order" if subset["candidate_tie_break"].iloc[0] == "source_order" else "query_id"
        aurc, _ = canonical_aurc_table(
            selectors,
            {"hit@1": subset["hit"].to_numpy(dtype=np.float64)},
            query_ids=subset["query_id"].astype(str),
            convention=str(subset["aurc_convention"].iloc[0]),
            tie_break=tie_break,
        )
        for measure, value in aurc["hit@1"].items():
            rows.append({
                "stratum": stratum,
                "measure": measure,
                "aurc": float(value),
                "n_spectra": len(subset),
            })
    return pd.DataFrame(rows)


def plot(scores: pd.DataFrame, out_path: Path) -> None:
    hit_rates = hit_rate_table(scores)
    aurc = stratified_aurc(scores)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hit_rates.to_csv(out_path.with_name("candidate_size_hit_rates.csv"), index=False)
    aurc.to_csv(out_path.with_name("candidate_size_stratified_aurc.csv"), index=False)

    n_bars = int(aurc.groupby("stratum")["measure"].size().max())
    fig_h = _BAR_PITCH * n_bars + 0.5
    fig = plt.figure(figsize=(_FW, fig_h))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.15, 1.05, 1.05],
        wspace=0.38,
        left=0.06,
        right=0.97,
        top=0.88,
        bottom=0.18,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    ax = axes[0]
    _setup_ax(ax)
    x = np.arange(len(BIN_LABELS))
    width = 0.8 / len(TOP_KS)
    for position, k in enumerate(TOP_KS):
        values = hit_rates[hit_rates["K"] == k].sort_values("bin_index")
        offset = (position - (len(TOP_KS) - 1) / 2) * width
        ax.bar(
            x + offset,
            values["hit_rate"],
            width=width,
            color=HIT_COLORS[k],
            edgecolor="white",
            linewidth=0.5,
            label=f"Hit@{k}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=_FT, rotation=35, ha="right")
    ax.set_xlabel(r"$|\mathcal{C}|$", fontsize=_FL)
    ax.set_ylabel("Hit Rate", fontsize=_FL)
    ax.tick_params(axis="y", labelsize=_FT)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=_FA, loc="upper right", framealpha=0.9)

    for target_ax, stratum, title in [
        (axes[1], "candidate_count_lt_256", r"$|\mathcal{C}| < 256$"),
        (axes[2], "candidate_count_eq_256", r"$|\mathcal{C}| = 256$"),
    ]:
        values = aurc[aurc["stratum"] == stratum].set_index("measure")["aurc"].sort_values()
        _format_bar_panel(target_ax, values, "")
        target_ax.set_title(title, fontsize=_FH, loc="center")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_scores", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--run_label", default="mlp_formula")
    args = parser.parse_args()
    plot(load_official_scores(args.query_scores, args.run_label), args.out_path)
    print(f"Saved candidate-size stratification to {args.out_path}")


if __name__ == "__main__":
    main()
