"""
SGR coverage analysis figure (Section 3.4).

Coverage vs target risk as line plots, one panel per K.
Independent y-axes per panel so Hit@1 (max ~10%) is readable
alongside Hit@20 (max ~90%).

Usage:
    python plot_sgr_analysis.py \
        --sgr_csv <path>/sgr_results.csv \
        --out_path fig/sgr_coverage.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Try importing from ms_uq; fall back to inline constants ─────────
try:
    from ms_uq.evaluation.visualisation import (
        TEXTWIDTH, DESIGN_SCALE,
        _FW, _ROW_W, _ROW_H,
        _FL, _FT, _FA, _FH,
        _setup_ax,
        display_name, get_metric_color,
    )
except ImportError:
    TEXTWIDTH = 6.85
    DESIGN_SCALE = 2.0
    _FW = TEXTWIDTH * DESIGN_SCALE          # 13.7
    _ROW_W = _FW - 1.2                      # 12.5
    _ROW_H = 4.2
    _FL, _FT, _FA, _FH = 18, 16, 16, 20

    _COLOR_MAP = {
        "confidence": "#7fbfff", "score_gap": "#7fbfff",
        "retrieval_aleatoric": "#c9b3d9", "retrieval_epistemic": "#f4a4a4",
        "rank_var_1": "#fdc086", "rank_var_5": "#fdc086",
        "rank_var_20": "#fdc086",
    }
    _DISPLAY = {
        "confidence": r"$\kappa_{\rm conf}$",
        "score_gap": r"$\kappa_{\rm gap}$",
        "retrieval_aleatoric": r"$\kappa_{\rm ret}^{\rm al}$",
        "retrieval_epistemic": r"$\kappa_{\rm ret}^{\rm ep}$",
        "rank_var_1": r"$\kappa_{\rm rank}^{(1)}$",
        "rank_var_5": r"$\kappa_{\rm rank}^{(5)}$",
        "rank_var_20": r"$\kappa_{\rm rank}^{(20)}$",
    }
    def get_metric_color(name):
        return _COLOR_MAP.get(name, "#95a5a6")
    def display_name(name):
        return _DISPLAY.get(name, name)
    def _setup_ax(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# ── Per-measure (linestyle, marker) ──────────────────────────────────
METRIC_STYLES = {
    "confidence":          ("-",  "o"),
    "score_gap":           ("-",  "s"),
    "retrieval_epistemic": ("-.", "x"),
    "rank_var_1":          (":",  "D"),
    "rank_var_5":          (":",  "P"),
    "rank_var_20":         (":",  "H"),
    "retrieval_aleatoric": ("-.", "^"),
    "retrieval_total":     ("--", "v"),
}
_FALLBACK = [("-", "o"), ("--", "s"), ("-.", "^"), (":", "D")]

# Legend strip width (design inches)
_LEG_W = 1.6

# Best five single-score selectors by mean validation relAURC across the two
# formula-trained architectures and Hit@1/5/20 in the canonical T=0.003 run.
DEFAULT_MEASURES = [
    "retrieval_total",
    "retrieval_aleatoric",
    "confidence",
    "rank_var_20",
    "rank_var_5",
]


def _get_style(name: str, idx: int):
    return METRIC_STYLES.get(name, _FALLBACK[idx % len(_FALLBACK)])


# ── Core plot ────────────────────────────────────────────────────────

def plot_sgr_coverage(
    sgr_df: pd.DataFrame,
    out_path: Path,
    measures: Optional[List[str]] = None,
    loss_cols: Optional[List[str]] = None,
    category: str = "retrieval",
    run_label: Optional[str] = None,
) -> None:
    measures = measures or DEFAULT_MEASURES

    # Filter category
    if "category" in sgr_df.columns:
        sgr_df = sgr_df[sgr_df["category"] == category].copy()
    if run_label is not None:
        if "run_label" not in sgr_df.columns:
            raise ValueError("--run_label was provided but the SGR table has no run_label column")
        sgr_df = sgr_df[sgr_df["run_label"] == run_label].copy()
        if sgr_df.empty:
            raise ValueError(f"No SGR rows for run_label={run_label}")
    elif "run_label" in sgr_df.columns and sgr_df["run_label"].nunique() > 1:
        raise ValueError("SGR table contains multiple run labels; select one with --run_label")

    # Determine panels
    all_losses = sorted(
        sgr_df["loss"].unique(),
        key=lambda s: int(s.split("@")[1]) if "@" in s else 0,
    )
    loss_cols = loss_cols or all_losses
    n = len(loss_cols)

    # Filter to measures that exist
    available = set(sgr_df["measure"].unique())
    measures = [m for m in measures if m in available]

    # Shared x-range
    all_risks = sorted(sgr_df["target_risk"].unique())
    x_lo = min(all_risks) - 0.02
    x_hi = max(all_risks) + 0.02
    x_ticks = np.arange(min(all_risks), max(all_risks) + 1e-9, 0.1)

    # ── Layout: 2 rows × n panels + legend strip ────────────────────
    leg_ratio = _LEG_W / (_ROW_W / n)
    fig = plt.figure(figsize=(_FW, _ROW_H * 2))
    gs = fig.add_gridspec(
        2, n + 1,
        width_ratios=[1] * n + [leg_ratio],
        height_ratios=[1, 1],
        wspace=0.28, hspace=0.22,
        left=0.06, right=0.99, top=0.93, bottom=0.10,
    )
    cov_axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cal_axes = [fig.add_subplot(gs[1, i]) for i in range(n)]
    ax_leg = fig.add_subplot(gs[:, n])   # legend spans both rows
    ax_leg.axis("off")

    # ── Row 1: Coverage ──────────────────────────────────────────────
    for idx, loss_name in enumerate(loss_cols):
        ax = cov_axes[idx]
        _setup_ax(ax)

        sub = sgr_df[sgr_df["loss"] == loss_name]
        max_cov = 0.0

        for mi, mname in enumerate(measures):
            ms = sub[sub["measure"] == mname].sort_values("target_risk")
            if ms.empty:
                continue
            rs, cov = [], []
            for target_risk in all_risks:
                row = ms[ms["target_risk"] == target_risk]
                if row.empty or not bool(row.iloc[0]["feasible"]):
                    rs.append(target_risk)
                    cov.append(0.0)
                else:
                    rs.append(target_risk)
                    cov.append(float(row.iloc[0]["coverage"]))
            rs, cov = np.asarray(rs), np.asarray(cov)
            max_cov = max(max_cov, float(np.nanmax(cov)))
            color = get_metric_color(mname)
            ls, marker = _get_style(mname, mi)
            ax.plot(rs, cov, color=color, ls=ls, marker=marker,
                    markersize=8, markeredgecolor="white", markeredgewidth=0.7,
                    lw=2.4, alpha=0.92, label=display_name(mname))

        # Annotate best at max r*
        r_max = max(all_risks)
        feas = sub[sub["feasible"] == True]
        at_rmax = feas[(feas["target_risk"] == r_max) &
                       (feas["measure"].isin(measures))]
        if not at_rmax.empty:
            best = at_rmax.loc[at_rmax["coverage"].idxmax()]
            best_coverage = float(best["coverage"])
            near_ceiling = best_coverage >= 0.9
            ax.annotate(
                f"{best_coverage:.1%}",
                xy=(best["target_risk"], best_coverage),
                xytext=(-4, -10 if near_ceiling else 8), textcoords="offset points",
                fontsize=_FT - 2, fontweight="medium",
                color="#444444", ha="right",
                va="top" if near_ceiling else "bottom",
            )

        y_top = min(1.0, max(0.05, max_cov * 1.30))
        ax.set_ylim(-0.005, y_top)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="both", labelsize=_FT)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_xticklabels([])           # x-labels only on bottom row
        ax.set_ylabel("Coverage" if idx == 0 else "", fontsize=_FL)
        ax.set_title(loss_name.replace("hit@", "Hit@"), fontsize=_FH)
        ax.grid(axis="y", alpha=0.18, lw=0.5)

    # ── Row 2: Calibration ───────────────────────────────────────────
    for idx, loss_name in enumerate(loss_cols):
        ax = cal_axes[idx]
        _setup_ax(ax)

        sub = sgr_df[sgr_df["loss"] == loss_name]
        feas = sub[sub["feasible"] == True]

        # Diagonal reference
        ax.plot([0, 1], [0, 1], color="#bbbbbb", ls="--", lw=1.5, zorder=0)

        for mi, mname in enumerate(measures):
            ms = feas[feas["measure"] == mname].sort_values("target_risk")
            if ms.empty:
                continue
            rs, emp = ms["target_risk"].values, ms["empirical_risk"].values
            color = get_metric_color(mname)
            ls, marker = _get_style(mname, mi)
            ax.plot(rs, emp, color=color, ls=ls, marker=marker,
                    markersize=8, markeredgecolor="white", markeredgewidth=0.7,
                    lw=2.4, alpha=0.92)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.01, max(all_risks) + 0.02)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_xlabel(r"Target risk $r^*$", fontsize=_FL)
        ax.tick_params(axis="both", labelsize=_FT)
        ax.set_ylabel("Empirical risk" if idx == 0 else "", fontsize=_FL)
        ax.grid(axis="y", alpha=0.18, lw=0.5)

    # ── Legend ────────────────────────────────────────────────────────
    all_handles, all_labels = {}, {}
    for ax in cov_axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in all_labels:
                all_handles[l] = h
                all_labels[l] = l
    ordered_h = [all_handles[display_name(m)] for m in measures
                 if display_name(m) in all_handles]
    ordered_l = [display_name(m) for m in measures
                 if display_name(m) in all_handles]

    ax_leg.legend(
        ordered_h, ordered_l,
        loc="center left",
        fontsize=_FA,
        frameon=False,
        borderpad=0.3,
        labelspacing=0.9,
        handlelength=2.4,
    )

    # ── Save ─────────────────────────────────────────────────────────
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CSV loading ──────────────────────────────────────────────────────

def load_sgr_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    renames = {
        "target_risk_level": "target_risk",
        "loss_name": "loss",
        "uncertainty": "measure",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
    if "coverage" not in df.columns and "eval_coverage" in df.columns:
        df["coverage"] = df["eval_coverage"]
    if "empirical_risk" not in df.columns and "eval_empirical_risk" in df.columns:
        df["empirical_risk"] = df["eval_empirical_risk"]

    for col in ["loss", "measure", "target_risk", "coverage", "feasible"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    if df["feasible"].dtype != bool:
        df["feasible"] = df["feasible"].astype(str).str.lower().isin(["true", "1"])
    return df


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="SGR coverage analysis figure (Section 3.4)")
    ap.add_argument("--sgr_csv", type=Path, required=True)
    ap.add_argument("--out_path", type=Path, default=Path("sgr_coverage.pdf"))
    ap.add_argument("--measures", nargs="*", default=None)
    ap.add_argument("--losses", nargs="*", default=None)
    ap.add_argument("--category", type=str, default="retrieval")
    ap.add_argument("--run_label")
    args = ap.parse_args()

    df = load_sgr_csv(args.sgr_csv)
    plot_sgr_coverage(
        df, args.out_path,
        measures=args.measures,
        loss_cols=args.losses,
        category=args.category,
        run_label=args.run_label,
    )


if __name__ == "__main__":
    main()
