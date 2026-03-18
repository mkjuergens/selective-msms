"""
Visualization functions for evaluation results.

This module provides plotting utilities for:
- AURC bar charts
- Risk-coverage curves
- Rejection curves
- Correlation heatmaps
- Ensemble member vs aggregation comparisons
- SGR (Selective Guaranteed Risk) analysis plots
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from ms_uq.evaluation.rejection_curve import rejection_curve
from ms_uq.utils import is_confidence_score


DEFAULT_COLOR_MAP = {
    # ── Predictive / Total (pastel blue) ─────────────────────
    # Retrieval-level
    "confidence":               "#7fbfff",
    "score_gap":                "#7fbfff",
    "margin":                   "#7fbfff",
    "retrieval_total":          "#9ecae1",
    "rank_var_1":               "#7fbfff",
    "rank_var_5":               "#7fbfff",
    "rank_var_20":              "#7fbfff",
    "top1_disagreement":        "#7fbfff",
    "top1_agreement":           "#7fbfff",
    "top5_agreement":           "#7fbfff",
    # Fingerprint-level (lighter blue)
    "bitwise_total":            "#d0e4ff",
    # ── Aleatoric (pastel lavender / purple) ─────────────────
    # Retrieval-level
    "retrieval_aleatoric":      "#c9b3d9",
    # Fingerprint-level (lighter lavender)
    "bitwise_aleatoric":        "#e2d5ed",
    # ── Epistemic (pastel peach / salmon) ────────────────────
    # Retrieval-level
    "retrieval_epistemic":      "#f4a4a4",
    # Fingerprint-level (lighter peach)
    "bitwise_epistemic":        "#fdd0a2",
    "bitwise_epistemic_sparse": "#fdd0a2",
    "bitwise_epistemic_active": "#fdae6b",
    # ── External / Distance-based (greys) ────────────────────
    "n_candidates":             "#d9d9d9",
    "knn_distance":             "#bdbdbd",
    "mahalanobis":              "#bdbdbd",
    "relative_mahalanobis":     "#bdbdbd",
    "centroid_distance":        "#bdbdbd",
    # ── SpecBridge first-order (pastel blue, same family) ────
    "entropy":                  "#7fbfff",
    "ambiguity_ratio":          "#7fbfff",
    # ── Baselines ────────────────────────────────────────────
    "oracle":                   "#2d6a4f",
    "random":                   "#aaaaaa",
}

CATEGORY_COLORS = {
    "Predictive":       "#7fbfff",
    "Aleatoric":        "#c9b3d9",
    "Epistemic":        "#f4a4a4",
    "External":         "#d9d9d9",
    "Baseline":         "#aaaaaa",
}

AGGREGATION_COLORS = {
    "score": "#7fbfff",
    "fingerprint": "#7fc97f",
    "probability": "#beaed4",
    "max_score_topk": "#fb9a99",
    "weighted": "#fdc086",
}
METRIC_LINESTYLES = {
    "predictive": "-",
    "aleatoric": "--",
    "epistemic": "-.",
    "external": ":",
    "total": (0, (3, 1, 1, 1)),
}

METRIC_STYLES: Dict[str, Tuple[str, str]] = {
    # ── Predictive / Total ───────────────────────────────────
    "confidence":               ("-",            "o"),
    "score_gap":                ("-",            "s"),
    "retrieval_total":          ("--",           "v"),
    "margin":                   ("-",            "s"),
    "rank_var_1":               (":",            "D"),
    "rank_var_5":               (":",            "D"),
    "rank_var_20":              (":",            "D"),
    "top1_disagreement":        (":",            "D"),
    "top1_agreement":           (":",            "D"),
    "top5_agreement":           (":",            "D"),
    "bitwise_total":            ("--",           "v"),
    # ── Aleatoric ────────────────────────────────────────────
    "retrieval_aleatoric":      ("-.",           "^"),
    "bitwise_aleatoric":        ("-.",           "^"),
    # ── Epistemic ────────────────────────────────────────────
    "retrieval_epistemic":      ("-.",           "x"),
    "bitwise_epistemic":        (":",            "+"),
    "bitwise_epistemic_sparse": (":",            "*"),
    "bitwise_epistemic_active": (":",            "p"),
    # ── External / Distance ──────────────────────────────────
    "n_candidates":             ((0,(3,1,1,1)),  "h"),
    "knn_distance":             ((0,(5,2)),      "8"),
    "mahalanobis":              ((0,(5,2)),      "8"),
    "relative_mahalanobis":     ((0,(5,2)),      "8"),
    "centroid_distance":        ((0,(5,2)),      "8"),
    # ── SpecBridge first-order ───────────────────────────────
    "entropy":                  ("-",            "o"),
    "ambiguity_ratio":          ("-",            "s"),
}

# Fallback cycle for metrics not in METRIC_STYLES
_CURVE_STYLES = [
    ("-", "o"),
    ("--", "s"),
    ("-.", "^"),
    (":", "D"),
    ((0, (3, 1, 1, 1)), "v"),
    ((0, (5, 2)), "x"),
]


def get_metric_style(name: str) -> Tuple:
    """Return (linestyle, marker) for a metric, with fallback."""
    return METRIC_STYLES.get(name, ("-", "o"))

DISPLAY_NAMES = {
    # Retrieval-level
    "confidence":               r"$\kappa_{\rm conf}$",
    "score_gap":                r"$\kappa_{\rm gap}$",
    "retrieval_epistemic":      r"$\kappa_{\rm ret}^{\rm ep}$",
    "retrieval_aleatoric":      r"$\kappa_{\rm ret}^{\rm al}$",
    "retrieval_total":          r"$\kappa_{\rm ret}^{\rm tot}$",
    # Fingerprint-level
    "bitwise_epistemic":        r"$\kappa_{\rm bit}^{\rm ep}$",
    "bitwise_aleatoric":        r"$\kappa_{\rm bit}^{\rm al}$",
    "bitwise_total":            r"$\kappa_{\rm bit}^{\rm tot}$",
    "bitwise_epistemic_sparse": r"$\kappa_{\rm bit}^{\rm ep,logit}$",
    "bitwise_epistemic_active": r"$\kappa_{\rm bit}^{\rm ep,active}$",
    # Rank-based
    "rank_var_1":               r"$\kappa_{\rm rank}^{(1)}$",
    "rank_var_5":               r"$\kappa_{\rm rank}^{(5)}$",
    "rank_var_20":              r"$\kappa_{\rm rank}^{(20)}$",
    # Distance-based
    "knn_distance":             r"$\kappa_{\rm knn}$",
    "mahalanobis":              r"$\kappa_{\rm mah}$",
    "relative_mahalanobis":     r"$\kappa_{\rm mah,rel}$",
    "centroid_distance":        r"$\kappa_{\rm centroid}$",
    # Other
    "n_candidates":             r"$|\mathcal{C}|$",
    "margin":                   r"$\kappa_{\rm margin}$",
    "ambiguity_ratio":          r"$\kappa_{\rm ambig}$",
    # Baselines
    "oracle":                   "Oracle",
    "random":                   "Random",
}
def display_name(code_name: str) -> str:
    return DISPLAY_NAMES.get(code_name, code_name)

# ── Publication layout configuration ──────────────────────────────

TEXTWIDTH = 6.85          # inches (174 mm for sn-jnl)
DESIGN_SCALE = 2.0        # design at 2×; LaTeX scales to 1×

# ── Dimensions (inches, at design scale) ─────────────────────────
_LEG_W   = 1.2                            # legend strip width
_FW      = TEXTWIDTH * DESIGN_SCALE       # total figure width  (13.7)
_ROW_W   = _FW - _LEG_W                  # content panels width
_ROW_H   = 4.2                            # single row height (base)
_RC_H    = 4.8                            # risk-coverage panel height
_BAR_PITCH = 0.48                         # inches per bar (design scale)
_BAR_PAD   = 1.8                          # top/bottom padding for title + x-label
_R_FRAC  = _ROW_W / _FW                  # subplots_adjust(right=...)

# ── Font sizes (pt in matplotlib; final ≈ pt / DESIGN_SCALE) ────
_FL = 18   # axis labels     → 9 pt final
_FT = 16   # tick labels     → 8 pt final
_FA = 16   # annotations / legend → 8 pt final
_FH = 20   # panel titles    → 10 pt final

def _make_legend_entries(measures: List[str]) -> list:
    from matplotlib.lines import Line2D
    entries = [
        Line2D([0], [0], color=DEFAULT_COLOR_MAP["oracle"], lw=3,
               label=display_name("oracle")),
        Line2D([0], [0], color=DEFAULT_COLOR_MAP["random"], lw=3, ls="--",
               label=display_name("random")),
    ]
    seen = {"Oracle", "Random"}
    for m in measures:
        dn = display_name(m)
        if dn in seen:
            continue
        ls, marker = get_metric_style(m)
        entries.append(Line2D([0], [0], color=get_metric_color(m), lw=2.5,
                              ls=ls, marker=marker, markersize=8, label=dn))
        seen.add(dn)
    return entries


def _format_bar_panel(
    ax, df: "pd.Series", col_title: str,
) -> None:
    """Format a single AURC bar panel: bars, labels, annotations, x-axis.

    Annotation policy: white text inside bars by default.
    Only if the bar is too short to fit the label, place it outside in black.

    Handles proportional annotation offset and clean x-axis ticks
    regardless of value magnitude (works for both Hit@k ~0.5 and Hamming ~0.01).
    """
    import matplotlib.ticker as mticker

    _setup_ax(ax)
    y = np.arange(len(df))
    bars = ax.barh(y, df.values,
                   color=[get_metric_color(m) for m in df.index],
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([display_name(m) for m in df.index], fontsize=_FA)
    ax.set_xlabel("AURC", fontsize=_FL)
    ax.tick_params(axis="x", labelsize=_FT)
    ax.set_title(_fmt_title(col_title), fontsize=_FH)

    # Proportional annotation offset and headroom
    vmax = df.values.max()
    offset = vmax * 0.02
    ax.set_xlim(0, vmax * 1.18)

    # Place annotations: white inside by default.
    # Only if bar < 15% of max (too short to fit text), place outside in black.
    inside_min = 0.15 * vmax
    for bar, val in zip(bars, df.values):
        y_pos = bar.get_y() + bar.get_height() / 2
        if val >= inside_min:
            ax.text(val - offset, y_pos, f"{val:.3f}",
                    va="center", ha="right", fontsize=_FT,
                    color="white", fontweight="bold")
        else:
            ax.text(val + offset, y_pos, f"{val:.3f}",
                    va="center", ha="left", fontsize=_FT,
                    color="black")

    # Clean x-axis ticks: fewer ticks, adaptive decimal places
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="upper"))
    if vmax < 0.01:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    elif vmax < 0.1:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    else:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    if "random" in df.index:
        ax.axvline(x=df["random"], color="#888888", ls="--", alpha=0.9, lw=1.5)



def get_metric_color(name: str, color_map: Optional[Dict[str, str]] = None) -> str:
    """
    Get color for a metric name.
    
    Parameters
    ----------
    name : str
        Metric name.
    color_map : dict, optional
        Custom color mapping. Falls back to DEFAULT_COLOR_MAP.
    
    Returns
    -------
    str
        Hex color code.
    """
    if color_map and name in color_map:
        return color_map[name]
    return DEFAULT_COLOR_MAP.get(name, "#95a5a6")


def get_metric_category(name: str) -> str:
    """
    Categorize a metric by its uncertainty component.

    The classification follows the information-theoretic decomposition:
    * **Predictive** — measures of total / predictive uncertainty,
      including first-order scores (confidence, score_gap, margin)
      that cannot decompose, the total component of the
      entropy-based split, and rank variance (which measures total
      sensitivity of the ranking to parameter changes).
    * **Aleatoric** — the irreducible (data) component.
    * **Epistemic** — the reducible (model) component.
    * **External** — scores not derived from the model's predictions
      (candidate set size, distance-based).
    * **Baseline** — oracle / random reference lines.
    """
    n = name.lower()
    if n in ("oracle", "random"):
        return "Baseline"
    if n in ("n_candidates", "knn_distance", "mahalanobis",
             "relative_mahalanobis", "centroid_distance"):
        return "External"
    if "epistemic" in n:
        return "Epistemic"
    if "aleatoric" in n:
        return "Aleatoric"
    # Everything else is predictive / total:
    # confidence, score_gap, margin, retrieval_total, bitwise_total,
    # rank_var_*, top*_agreement, top*_disagreement, entropy, ambiguity_ratio
    return "Predictive"


def _fmt_title(name: str) -> str:
    """Format metric name for plot titles."""
    if name.startswith("hit@"):
        return name.replace("hit@", "Hit@")
    for n in ["hamming", "tanimoto", "cosine"]:
        if n in name.lower():
            return n.title()
    return name.replace("_", " ").title()


def _setup_ax(ax: plt.Axes) -> None:
    """Apply clean style to axes (remove top/right spines)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _setup_rc_ticks(ax: plt.Axes) -> None:
    """Adaptive tick formatting for risk-coverage panels.

    Limits both axes to ≤5 ticks and chooses decimal places based on
    the y-range magnitude.  This prevents label overlap in panels with
    small value ranges (e.g. Hamming loss ~0.01) while leaving panels
    with larger ranges (e.g. Hit@1 error ~0.9) unchanged.
    """
    import matplotlib.ticker as mticker

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="upper"))

    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    if y_range < 0.05:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    elif y_range < 0.5:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    # else: matplotlib default ("%.1f" style) is fine


def _axes_list(axes, n: int) -> List[plt.Axes]:
    """Ensure axes is always a list for consistent iteration."""
    return [axes] if n == 1 else list(axes)


def plot_risk_coverage_curves(
    losses: Dict[str, np.ndarray],
    uncertainties: Dict[str, np.ndarray],
    out_path: Path,
    title: str = "",
    loss_cols: Optional[List[str]] = None,
    measures: Optional[List[str]] = None,
    shared_ylim: Optional[bool] = None,
) -> None:
    """Risk-coverage curves with right-side legend and paper notation.

    Parameters
    ----------
    shared_ylim : bool or None
        If True, all panels share the same y-axis range and only the
        leftmost panel shows y-tick labels.  If None (default), auto-
        detected: True when all loss columns are hit@K (retrieval task),
        False otherwise (e.g. fingerprint losses on different scales).
    """
    loss_cols = loss_cols or list(losses.keys())
    measures = [m for m in (measures or list(uncertainties.keys())) if m in uncertainties]
    n = len(loss_cols)

    # Auto-detect: share y-axis when all columns are hit@K
    if shared_ylim is None:
        shared_ylim = all(c.startswith("hit@") for c in loss_cols)

    leg_ratio = _LEG_W / (_ROW_W / n)
    fig = plt.figure(figsize=(_FW, _RC_H))
    gs = fig.add_gridspec(
        1, n + 1,
        width_ratios=[1] * n + [leg_ratio],
        wspace=0.20,
        left=0.06, right=0.99, top=0.93, bottom=0.12,
    )

    # Place random line at ~93% of each panel's y-range for visual alignment
    _RAND_FRAC = 0.93

    # Pre-compute global y_top for shared mode
    if shared_ylim:
        global_y_top = max(float(losses[c].mean()) for c in loss_cols
                          if c in losses) / _RAND_FRAC

    for idx, col in enumerate(loss_cols):
        ax = fig.add_subplot(gs[0, idx])
        _setup_ax(ax)
        if col not in losses:
            continue
        lv = losses[col]
        base_err = float(lv.mean())
        y_top = global_y_top if shared_ylim else base_err / _RAND_FRAC
        cov = np.linspace(1.0, 0.01, 100)

        order = np.argsort(-lv)
        ax.plot(cov, [lv[order[-max(1, int(len(lv) * c)):]].mean() for c in cov],
                color=DEFAULT_COLOR_MAP["oracle"], lw=2.5)
        ax.axhline(base_err, color=DEFAULT_COLOR_MAP["random"],
                   ls="--", alpha=0.9, lw=2.5)

        for i, m in enumerate(measures):
            uv = uncertainties[m]
            valid = ~np.isnan(uv)
            if valid.sum() < 10:
                continue
            uv_sort = -uv[valid] if is_confidence_score(m) else uv[valid]
            si = np.argsort(uv_sort)
            lval, nv = lv[valid], len(si)
            ls, marker = get_metric_style(m)
            ax.plot(cov, [lval[si[:max(1, int(nv * c))]].mean() for c in cov],
                    color=get_metric_color(m), lw=2.3,
                    ls=ls, marker=marker, markevery=10, markersize=6, alpha=0.9)

        ax.set_xlabel("Coverage", fontsize=_FL)
        ax.set_xlim(1, 0)
        ax.set_ylim(0, y_top)
        if idx == 0:
            ax.set_ylabel("Error Rate" if col.startswith("hit@") else "Loss",
                          fontsize=_FL)
        elif shared_ylim:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylabel("", fontsize=_FL)
        ax.set_title(_fmt_title(col), fontsize=_FH)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=_FT)
        _setup_rc_ticks(ax)

    # Legend in dedicated column
    lax = fig.add_subplot(gs[0, n])
    lax.axis("off")
    lax.legend(handles=_make_legend_entries(measures), loc="center",
               fontsize=_FA, framealpha=0.95, borderaxespad=0)

    if title:
        fig.suptitle(title, fontsize=_FH, y=0.99)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── REPLACE plot_aurc_bars ───────────────────────────────────────────

def plot_aurc_bars(
    aurc_df: pd.DataFrame,
    loss_cols: List[str],
    out_path: Path,
    title: str = "",
    measures: Optional[List[str]] = None,
) -> None:
    """AURC horizontal bars with right-side legend and paper notation.

    Uses explicit gridspec with the same width ratios as
    plot_risk_coverage_curves, guaranteeing perfect column alignment
    when the two figures are stacked in the paper.

    Figure height scales linearly with the number of bars so that
    bar thickness is consistent regardless of how many scores are shown.
    """
    if measures:
        keep = [m for m in measures if m in aurc_df.index]
        for b in ["oracle", "random"]:
            if b in aurc_df.index and b not in keep:
                keep.append(b)
        aurc_df = aurc_df.loc[keep]

    n = len(loss_cols)
    n_bars = len(aurc_df)
    bar_h = _BAR_PITCH * n_bars + _BAR_PAD

    leg_ratio = _LEG_W / (_ROW_W / n)
    fig = plt.figure(figsize=(_FW, bar_h))
    gs = fig.add_gridspec(
        1, n + 1,
        width_ratios=[1] * n + [leg_ratio],
        wspace=0.20,
        left=0.06, right=0.99, top=0.93, bottom=0.10,
    )

    all_measures, seen = [], set()
    for idx, col in enumerate(loss_cols):
        ax = fig.add_subplot(gs[0, idx])
        if col not in aurc_df.columns:
            continue
        df = aurc_df[col].dropna().sort_values()
        _format_bar_panel(ax, df, col)
        for m in df.index:
            if m not in seen:
                all_measures.append(m)
                seen.add(m)

    # Legend in dedicated column (same position as RC legend)
    lax = fig.add_subplot(gs[0, n])
    lax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=get_metric_color(m), ec="white",
                              lw=0.5, label=display_name(m)) for m in all_measures]
    lax.legend(handles=handles, loc="center",
               fontsize=_FA, framealpha=0.95, borderaxespad=0)

    if title:
        fig.suptitle(title, fontsize=_FH, y=0.99)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")



def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    out_path: Optional[Union[str, Path]] = None,
    title: str = "",
    cmap: str = "RdBu_r",
) -> None:
    """
    Plot scoring-function correlation matrix as a publication-quality heatmap.

    Uses the shared DESIGN_SCALE layout system.  Rows and columns are
    reordered according to _HEATMAP_ORDER (retrieval → fingerprint →
    external) with display names from DISPLAY_NAMES.  Group boundaries
    are shown as thin separator lines.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Square Spearman correlation matrix with code-name index/columns.
    out_path : path-like, optional
        Output path (pdf recommended). If None, displays interactively.
    title : str
        Optional figure title (leave empty for caption-only figures).
    cmap : str
        Matplotlib diverging colormap.
    """
    # ── Reorder rows/columns by canonical grouping ───────────────
    ordered = [m for m in _HEATMAP_ORDER if m in corr_df.columns]
    extra = [m for m in corr_df.columns if m not in ordered]
    ordered += extra
    corr_df = corr_df.loc[ordered, ordered]

    n = len(corr_df)
    labels = [display_name(c) for c in corr_df.columns]

    # ── Figure sizing ────────────────────────────────────────────
    # Each cell needs room for a ".95" annotation at _FT-2 pt.
    cell = 0.72                               # inches per cell (design)
    grid = n * cell
    cbar_pad = 2.0                            # space for colorbar + label
    label_pad = 1.8                           # bottom margin for rotated labels
    fig_w = grid + cbar_pad
    fig_h = grid + label_pad

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Heatmap ──────────────────────────────────────────────────
    im = ax.imshow(
        corr_df.values, cmap=cmap, vmin=-1.0, vmax=1.0,
        aspect="equal", interpolation="nearest",
    )

    # ── Tick labels ──────────────────────────────────────────────
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=_FT)
    ax.set_yticklabels(labels, fontsize=_FT)
    ax.tick_params(length=0)                  # no tick marks

    # ── Cell annotations ─────────────────────────────────────────
    _ann_size = _FT - 2                       # slightly smaller than ticks
    for i in range(n):
        for j in range(n):
            v = corr_df.iloc[i, j]
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                fontsize=_ann_size,
                color="white" if abs(v) > 0.6 else "black",
            )

    # ── Group separator lines ────────────────────────────────────
    col_list = list(corr_df.columns)
    boundaries = []
    for group_members in _HEATMAP_GROUPS.values():
        idxs = [col_list.index(m) for m in group_members if m in col_list]
        if idxs:
            boundaries.append(max(idxs) + 0.5)
    # Drop boundary at the very edge (no line needed there)
    boundaries = sorted(set(b for b in boundaries if b < n - 0.5))

    for b in boundaries:
        ax.axhline(b, color="0.3", linewidth=0.8, clip_on=True)
        ax.axvline(b, color="0.3", linewidth=0.8, clip_on=True)

    # ── Colorbar ─────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Spearman $\rho$", fontsize=_FL)
    cbar.ax.tick_params(labelsize=_FT)

    # ── Title (optional) ─────────────────────────────────────────
    if title:
        ax.set_title(title, fontsize=_FH, pad=12)

    # ── Remove spines ────────────────────────────────────────────
    for sp in ax.spines.values():
        sp.set_visible(False)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_member_vs_agg(
    member_hits: Dict[str, np.ndarray],
    agg_hits: Dict[str, Dict[str, float]],
    out_path: Path,
    title: str = "",
) -> None:
    """
    Compare individual ensemble members vs aggregation strategies.

    Uses the same gridspec + explicit-margin + right-side legend strip
    architecture as plot_risk_coverage_curves and plot_aurc_bars, so
    that all pipeline figures share identical effective margins, font
    scaling, and legend positioning when LaTeX embeds them at textwidth.
    """
    if not member_hits:
        return

    ks = sorted([int(k.split("@")[1]) for k in member_hits.keys()])
    nm = len(member_hits[f"hit@{ks[0]}"])
    agg_names = list(agg_hits.keys())
    na = len(agg_names)

    # ── Layout: single content panel + right-side legend strip ───
    # Use n=1 content panel; leg_ratio computed the same way as
    # in the multi-panel functions so the legend strip width matches.
    leg_ratio = _LEG_W / _ROW_W          # single panel occupies full _ROW_W
    fig = plt.figure(figsize=(_FW, _ROW_H))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1, leg_ratio],
        wspace=0.20,
        left=0.06, right=0.99, top=0.93, bottom=0.12,
    )

    ax = fig.add_subplot(gs[0, 0])
    _setup_ax(ax)
    x = np.arange(len(ks))

    # --- Bar geometry ---
    bar_w = 0.10                         # uniform width for all bars
    member_step = bar_w * 0.35           # heavy overlap between members
    agg_step = bar_w * 1.05              # slight gap between aggregates
    gap = bar_w * 0.6                    # gap between member block and agg block

    member_block = (nm - 1) * member_step + bar_w
    agg_block = (na - 1) * agg_step + bar_w
    total_block = member_block + gap + agg_block
    group_start = -total_block / 2

    # Member bars (overlapping)
    for s in range(nm):
        vals = [member_hits[f"hit@{k}"][s] for k in ks]
        center = group_start + s * member_step + bar_w / 2
        ax.bar(x + center, vals, bar_w, color="#888888", alpha=0.35,
               edgecolor="#666666", lw=0.3, zorder=2 + s)

    # Member mean annotation
    member_center = group_start + member_block / 2
    _ANN = _FT - 4  # compact annotation font (→ 6 pt final)
    for i, k in enumerate(ks):
        mv = member_hits[f"hit@{k}"].mean()
        mx = member_hits[f"hit@{k}"].max()
        ax.annotate(f"{mv:.1%}", xy=(x[i] + member_center, mx),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=_ANN, color="#555555")

    # Aggregation bars — stagger label heights to avoid overlap
    agg_start = group_start + member_block + gap
    stagger_offsets = [4, 18, 4, 18]  # alternating vertical offsets (pts)
    for j, an in enumerate(agg_names):
        hits = agg_hits[an]
        center = agg_start + j * agg_step + bar_w / 2
        vals = [hits.get(f"hit@{k}", 0) for k in ks]
        c = AGGREGATION_COLORS.get(an, "#beaed4")
        bars = ax.bar(x + center, vals, bar_w, color=c, alpha=0.85,
                      edgecolor="white", lw=0.5, label=an, zorder=3)
        y_off = stagger_offsets[j % len(stagger_offsets)]
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.1%}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, y_off), textcoords="offset points",
                        ha="center", va="bottom", fontsize=_ANN)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Hit@{k}" for k in ks], fontsize=_FL)
    ax.set_ylabel("Hit Rate", fontsize=_FL, labelpad=6)
    ax.tick_params(axis="y", labelsize=_FT)
    ax.grid(axis="y", alpha=0.3)

    if title:
        ax.set_title(title, fontsize=_FH)

    # ── Right-side legend strip (matches RC / AURC figures) ──
    lax = fig.add_subplot(gs[0, 1])
    lax.axis("off")
    handles = [Patch(facecolor="#888888", alpha=0.35, edgecolor="#666666",
                     label=f"Members (1\u2013{nm})")]
    handles += [Patch(facecolor=AGGREGATION_COLORS.get(a, "#beaed4"), alpha=0.85,
                      label=a) for a in agg_names]
    lax.legend(handles=handles, loc="center",
               fontsize=_FA, framealpha=0.95, borderaxespad=0)

    if title:
        fig.suptitle(title, fontsize=_FH, y=0.99)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")



def _compute_rc_curve(
    confidence: np.ndarray,
    losses: np.ndarray,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute risk-coverage curve (internal helper)."""
    n = len(confidence)
    sorted_idx = np.argsort(-confidence)  # Highest confidence first
    sorted_losses = losses[sorted_idx]
    
    cumsum = np.cumsum(sorted_losses)
    counts = np.arange(1, n + 1)
    risks = cumsum / counts
    coverages = counts / n
    
    if n > n_points:
        idx = np.linspace(0, n - 1, n_points, dtype=int)
        return coverages[idx], risks[idx]
    return coverages, risks


# Bar hatching patterns for differentiation
_BAR_HATCHES = ['', '///', '...', 'xxx', '\\\\\\', '+++', 'ooo', '---']



def plot_sgr_coverage_combined(
    results: Dict[str, Dict],
    title: str = "",
    out_path: Optional[Union[str, Path]] = None,
    top_k: int = 6,
    sharey: bool = True,
) -> plt.Figure:
    """
    Coverage bar plot for SGR across multiple losses.

    Parameters
    ----------
    results : dict
        Mapping loss_name -> {"sgr": {measure: {r*: SGRResult}}, "aurcs": {...},
                              "base_error": float, "target_risks": list}
    title : str
        Figure title.
    out_path : path-like, optional
        Output path.
    top_k : int
        Number of top measures to show (by average AURC).
    sharey : bool
        Whether to share y-axis across subplots.
    """
    FIXED_MEASURE_ORDER = [
        "score_gap", "confidence", "retrieval_aleatoric", "rank_var_20",
        "rank_var_5", "rank_var_1", "retrieval_epistemic", "margin",
        "bitwise_total", "bitwise_aleatoric", "bitwise_epistemic",
    ]

    loss_names = list(results.keys())
    n = len(loss_names)

    # Select top_k measures by average AURC, then order by fixed list
    all_measures = set()
    for data in results.values():
        all_measures.update(data["aurcs"].keys())
    avg_aurcs = {}
    for m in all_measures:
        vals = [data["aurcs"].get(m, float("inf")) for data in results.values()]
        valid = [a for a in vals if a != float("inf")]
        avg_aurcs[m] = np.mean(valid) if valid else float("inf")
    top_by_aurc = sorted(avg_aurcs, key=avg_aurcs.get)[:top_k]
    global_measures = [m for m in FIXED_MEASURE_ORDER if m in top_by_aurc]
    for m in top_by_aurc:
        if m not in global_measures:
            global_measures.append(m)

    # ── Layout: n content panels + 1 legend strip ────────────────────
    leg_ratio = _LEG_W / (_ROW_W / n)
    fig = plt.figure(figsize=(_FW, _ROW_H))
    gs = fig.add_gridspec(
        1, n + 1,
        width_ratios=[1] * n + [leg_ratio],
        wspace=0.30,
        left=0.05, right=0.98, top=0.92, bottom=0.12,
    )

    for panel_idx, loss_name in enumerate(loss_names):
        ax = fig.add_subplot(gs[0, panel_idx])
        _setup_ax(ax)
        data = results[loss_name]
        target_risks = data["target_risks"]
        base_error = data["base_error"]
        sorted_measures = [m for m in global_measures if m in data["aurcs"]]

        feasible_risks = [r for r in target_risks if r < base_error]
        if not feasible_risks:
            ax.text(0.5, 0.5, f"No feasible targets\n(baseline: {base_error:.1%})",
                    ha="center", va="center", fontsize=_FT, color="#888",
                    transform=ax.transAxes)
            ax.set_title(_fmt_title(loss_name), fontsize=_FH)
            continue

        n_risks = len(feasible_risks)
        n_measures = len(sorted_measures)
        x = np.arange(n_risks)
        width = 0.88 / n_measures

        # Compute coverages and find best per risk level
        all_coverages = {}
        for measure in sorted_measures:
            covs = []
            for r_star in feasible_risks:
                res = data["sgr"][measure].get(r_star)
                covs.append(res.coverage if res and res.feasible else 0)
            all_coverages[measure] = covs
        best_at_risk = [
            max(sorted_measures, key=lambda m: all_coverages[m][j])
            for j in range(n_risks)
        ]

        # Draw bars
        for i, measure in enumerate(sorted_measures):
            fixed_idx = (FIXED_MEASURE_ORDER.index(measure)
                         if measure in FIXED_MEASURE_ORDER
                         else len(FIXED_MEASURE_ORDER) + i)
            offset = (i - n_measures / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, all_coverages[measure], width * 0.94,
                color=get_metric_color(measure),
                hatch=_BAR_HATCHES[fixed_idx % len(_BAR_HATCHES)],
                edgecolor="white", linewidth=1.0,
                label=display_name(measure), alpha=0.88,
            )
            # Annotate best measure at each risk level
            for j, (bar, cov) in enumerate(zip(bars, all_coverages[measure])):
                if cov > 0.02 and best_at_risk[j] == measure:
                    ax.annotate(
                        f"{cov:.1%}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=_FT - 2, fontweight="medium", rotation=90,
                    )

        # Axis formatting
        ax.set_xlabel(r"Target risk ($r^*$)", fontsize=_FL)
        ax.set_xticks(x)
        dec = ".1f" if all(round(r, 1) == r for r in feasible_risks) else ".2f"
        ax.set_xticklabels([f"{r:{dec}}" for r in feasible_risks],
                           fontsize=_FT, rotation=35, ha="right")
        ax.tick_params(axis="y", labelsize=_FT)

        if sharey:
            ax.set_ylim(0, 1.05)
        else:
            max_cov = max(max(c) for c in all_coverages.values()) if all_coverages else 0
            ax.set_ylim(0, max(0.1, min(1.05, max_cov * 1.25 + 0.05)))

        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.set_title(_fmt_title(loss_name), fontsize=_FH)

        if panel_idx == 0 or not sharey:
            ax.set_ylabel("Coverage", fontsize=_FL)

    # ── Right-side legend ────────────────────────────────────────────
    lax = fig.add_subplot(gs[0, n])
    lax.axis("off")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    lax.legend(handles, labels, loc="center left",
               bbox_to_anchor=(-0.3, 0.5),
               fontsize=_FA, framealpha=0.95, borderaxespad=0)

    if title:
        fig.suptitle(title, fontsize=_FH, y=0.99)

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    return fig



def plot_sgr_risk_calibration(
    results: Dict[str, Dict],
    title: str = "",
    out_path: Optional[Union[str, Path]] = None,
    top_k: int = 5,
    sharey: bool = True,
) -> plt.Figure:
    """
    Target risk vs actual risk scatter plot.

    Parameters
    ----------
    results : dict
        Same format as plot_sgr_coverage_combined.
    title : str
        Figure title.
    out_path : path-like, optional
        Output path.
    top_k : int
        Number of top measures to show.
    sharey : bool
        Whether to share y-axis across subplots.
    """
    FIXED_MEASURE_ORDER = [
        "score_gap", "confidence", "retrieval_aleatoric", "rank_var_20",
        "rank_var_5", "rank_var_1", "retrieval_epistemic", "margin",
        "bitwise_total", "bitwise_aleatoric", "bitwise_epistemic",
    ]

    loss_names = list(results.keys())
    n = len(loss_names)

    # Select top_k by average AURC
    all_measures = set()
    for data in results.values():
        all_measures.update(data["aurcs"].keys())
    avg_aurcs = {}
    for m in all_measures:
        vals = [data["aurcs"].get(m, float("inf")) for data in results.values()]
        valid = [a for a in vals if a != float("inf")]
        avg_aurcs[m] = np.mean(valid) if valid else float("inf")
    top_by_aurc = sorted(avg_aurcs, key=avg_aurcs.get)[:top_k]
    global_measures = [m for m in FIXED_MEASURE_ORDER if m in top_by_aurc]
    for m in top_by_aurc:
        if m not in global_measures:
            global_measures.append(m)

    # ── Layout ───────────────────────────────────────────────────────
    leg_ratio = _LEG_W / (_ROW_W / n)
    fig = plt.figure(figsize=(_FW, _ROW_H))
    gs = fig.add_gridspec(
        1, n + 1,
        width_ratios=[1] * n + [leg_ratio],
        wspace=0.30,
        left=0.05, right=0.98, top=0.92, bottom=0.12,
    )

    _MARKERS = ["s", "^", "D", "v", "o", "x", "+", "*"]

    for panel_idx, loss_name in enumerate(loss_names):
        ax = fig.add_subplot(gs[0, panel_idx])
        _setup_ax(ax)
        data = results[loss_name]
        target_risks = data["target_risks"]
        sorted_measures = [m for m in global_measures if m in data["aurcs"]]

        # Diagonal reference line
        max_r = max(target_risks) * 1.1
        ax.plot([0, max_r], [0, max_r], color="#888", ls="--", lw=1.5,
                alpha=0.6)

        for i, measure in enumerate(sorted_measures):
            fixed_idx = (FIXED_MEASURE_ORDER.index(measure)
                         if measure in FIXED_MEASURE_ORDER
                         else len(FIXED_MEASURE_ORDER) + i)
            target_vals, actual_vals = [], []
            for r_star in target_risks:
                res = data["sgr"][measure].get(r_star)
                if res and res.feasible and res.coverage > 0.01:
                    target_vals.append(r_star)
                    actual_vals.append(res.empirical_risk)
            if target_vals:
                ax.scatter(
                    target_vals, actual_vals,
                    color=get_metric_color(measure),
                    marker=_MARKERS[fixed_idx % len(_MARKERS)],
                    s=100, alpha=0.88, edgecolor="white", linewidth=0.8,
                    label=display_name(measure),
                )

        ax.set_xlabel(r"Target risk ($r^*$)", fontsize=_FL)
        ax.set_xlim(0, max_r)
        ax.set_ylim(0, max_r)
        ax.tick_params(labelsize=_FT)
        ax.grid(alpha=0.2)
        ax.set_title(_fmt_title(loss_name), fontsize=_FH)

        if panel_idx == 0 or not sharey:
            ax.set_ylabel("Actual Risk", fontsize=_FL)

    # ── Right-side legend ────────────────────────────────────────────
    lax = fig.add_subplot(gs[0, n])
    lax.axis("off")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    lax.legend(handles, labels, loc="center left",
               bbox_to_anchor=(-0.3, 0.5),
               fontsize=_FA, framealpha=0.95, borderaxespad=0)

    if title:
        fig.suptitle(title, fontsize=_FH, y=0.99)

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    return fig



def _fmt_loss_name(name: str) -> str:
    """Format loss name for display."""
    if name.startswith("hit@"):
        return name.replace("hit@", "Hit@")
    return name.replace("_loss", "").replace("_", " ").title()



def plot_rc_and_aurc_paired(
    losses: Dict[str, np.ndarray],
    uncertainties: Dict[str, np.ndarray],
    aurc_df: pd.DataFrame,
    out_path: Path,
    loss_cols: Optional[List[str]] = None,
    measures: Optional[List[str]] = None,
    title: str = "",
    shared_ylim: Optional[bool] = None,
) -> None:
    """Combined: risk-coverage (top) + AURC bars (bottom), shared legend.

    Parameters
    ----------
    shared_ylim : bool or None
        If True, all RC panels share the same y-axis range and only
        the leftmost panel shows y-tick labels.  If None (default),
        auto-detected: True when all loss columns are hit@K.
    """
    loss_cols = loss_cols or list(losses.keys())
    all_measures = [m for m in (measures or list(uncertainties.keys()))
                    if m in uncertainties]
    n = len(loss_cols)

    # Auto-detect: share y-axis when all columns are hit@K
    if shared_ylim is None:
        shared_ylim = all(c.startswith("hit@") for c in loss_cols)

    if measures:
        keep = [m for m in measures if m in aurc_df.index]
        for b in ["oracle", "random"]:
            if b in aurc_df.index and b not in keep:
                keep.append(b)
        aurc_df = aurc_df.loc[keep]

    n_bars = len(aurc_df)
    bar_h = _BAR_PITCH * n_bars + _BAR_PAD
    leg_ratio = _LEG_W / (_ROW_W / n)
    total_h = _RC_H + bar_h + 0.15      # minimal gap between rows

    fig = plt.figure(figsize=(_FW, total_h))
    gs = fig.add_gridspec(
        2, n + 1,
        width_ratios=[1] * n + [leg_ratio],
        height_ratios=[_RC_H, bar_h],
        hspace=0.12, wspace=0.20,
        left=0.06, right=0.99, top=0.96, bottom=0.05,
    )

    # Place random line at ~93% of each panel's y-range
    _RAND_FRAC = 0.93

    # Pre-compute global y_top for shared mode
    if shared_ylim:
        global_y_top = max(float(losses[c].mean()) for c in loss_cols
                          if c in losses) / _RAND_FRAC

    # ── Top: risk-coverage ──
    for idx, col in enumerate(loss_cols):
        ax = fig.add_subplot(gs[0, idx])
        _setup_ax(ax)
        if col not in losses:
            continue
        lv = losses[col]
        base_err = float(lv.mean())
        y_top = global_y_top if shared_ylim else base_err / _RAND_FRAC
        cov = np.linspace(1.0, 0.01, 100)
        N = len(lv)

        order = np.argsort(-lv)
        ax.plot(cov, [lv[order[-max(1, int(N * c)):]].mean() for c in cov],
                color=DEFAULT_COLOR_MAP["oracle"], lw=2.5)
        ax.axhline(base_err, color=DEFAULT_COLOR_MAP["random"],
                   ls="--", alpha=0.9, lw=2.5)

        for i, m in enumerate(all_measures):
            uv = uncertainties[m]
            valid = ~np.isnan(uv)
            if valid.sum() < 10:
                continue
            uv_sort = -uv[valid] if is_confidence_score(m) else uv[valid]
            si = np.argsort(uv_sort)
            lval, nv = lv[valid], len(si)
            ls, marker = get_metric_style(m)
            ax.plot(cov, [lval[si[:max(1, int(nv * c))]].mean() for c in cov],
                    color=get_metric_color(m), lw=2.3,
                    ls=ls, marker=marker, markevery=10, markersize=6, alpha=0.9)

        ax.set_xlim(1, 0)
        ax.set_ylim(0, y_top)
        if idx == 0:
            ax.set_ylabel("Error Rate" if col.startswith("hit@") else "Loss",
                          fontsize=_FL)
        elif shared_ylim:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylabel("", fontsize=_FL)
        ax.set_title(_fmt_title(col), fontsize=_FH)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=_FT)
        _setup_rc_ticks(ax)

    # ── Bottom: AURC bars (no titles — shared with RC row above) ──
    for idx, col in enumerate(loss_cols):
        ax = fig.add_subplot(gs[1, idx])
        if col not in aurc_df.columns:
            continue
        present = [m for m in aurc_df.index if pd.notna(aurc_df.loc[m, col])]
        vals = aurc_df.loc[present, col].sort_values(ascending=True)
        _format_bar_panel(ax, vals, col)
        ax.set_title("")                 # remove duplicate title

    # ── Shared legend ──
    lax = fig.add_subplot(gs[:, n])
    lax.axis("off")
    lax.legend(handles=_make_legend_entries(all_measures), loc="center",
               fontsize=_FA, framealpha=0.95, borderaxespad=0)

    if title:
        fig.suptitle(title, fontsize=_FH, y=0.995)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
