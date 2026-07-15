#!/usr/bin/env python3
"""Paper-style risk-coverage/AURC plots with task-matched meta scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ms_uq.evaluation import compute_aurc_table
from ms_uq.evaluation.revision_features import canonical_aurc_table, hit_arrays
from ms_uq.evaluation.visualisation import (
    DEFAULT_COLOR_MAP,
    DISPLAY_NAMES,
    METRIC_STYLES,
    _BAR_PAD,
    _BAR_PITCH,
    _FA,
    _FH,
    _FL,
    _FT,
    _FW,
    _LEG_W,
    _RC_H,
    _ROW_W,
    _format_bar_panel,
    _fmt_title,
    _make_legend_entries,
    _setup_ax,
    _setup_rc_ticks,
    get_metric_color,
    get_metric_style,
)
from ms_uq.utils import is_confidence_score


DEFAULT_TOP_KS = [1, 5, 20]
RAND_LINE_FRAC = 0.93
DEFAULT_MEASURES = [
    "bitwise_total",
    "bitwise_aleatoric",
    "bitwise_epistemic",
    "retrieval_epistemic",
    "retrieval_aleatoric",
    "retrieval_total",
    "confidence",
    "score_gap",
    "rank_var_1",
    "rank_var_5",
    "rank_var_20",
    "n_candidates",
    "meta",
]
MANUSCRIPT_MEASURES = [
    "bitwise_total",
    "bitwise_aleatoric",
    "bitwise_epistemic",
    "retrieval_epistemic",
    "retrieval_aleatoric",
    "retrieval_total",
    "confidence",
    "score_gap",
    "rank_var_1",
    "rank_var_5",
    "rank_var_20",
    "n_candidates",
]

DEFAULT_MODELS = {
    "mlp": {
        "label": "MLP ensemble",
        "eval_dir": Path("/data/home/mira/data/msuq/figures/eval_v6/ensemble/bienc"),
        "meta_dir": Path("outputs/revision_meta/mlp_meta"),
    },
    "transformer": {
        "label": "Transformer ensemble",
        "eval_dir": Path("outputs/revision_analysis/transformer_ensemble_formula"),
        "meta_dir": Path("outputs/revision_meta/transformer_meta"),
    },
}


def _register_meta_style() -> None:
    DEFAULT_COLOR_MAP.setdefault("meta", "#264653")
    METRIC_STYLES.setdefault("meta", ("-", "*"))
    DISPLAY_NAMES.setdefault("meta", r"$\kappa_{\rm meta}$")


def _load_score_data(score_path: Path, top_ks: Iterable[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    data = torch.load(score_path, map_location="cpu")
    labels = data.get("labels_flat")
    if labels is None:
        raise ValueError(f"{score_path} does not contain labels_flat")
    hits = hit_arrays(data["scores_flat"].float(), labels.float(), data["ptr"].long(), list(top_ks))
    losses = {name: 1.0 - values.astype(np.float32) for name, values in hits.items()}
    return hits, losses


def _load_uncertainties(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {name: data[name] for name in data.files}


def _load_meta_by_loss(path: Path, top_ks: Iterable[int]) -> Dict[str, np.ndarray]:
    data = np.load(path)
    out = {}
    for k in top_ks:
        key = f"meta_hit{k}"
        if key not in data:
            raise KeyError(f"{path} is missing {key}")
        out[f"hit@{k}"] = data[key].astype(np.float32)
    return out


def _augment_aurc(
    aurc_path: Path,
    hits: Dict[str, np.ndarray],
    meta_by_loss: Dict[str, np.ndarray],
    top_ks: Iterable[int],
) -> pd.DataFrame:
    aurc = pd.read_csv(aurc_path, index_col=0)
    if "meta" not in aurc.index:
        aurc.loc["meta"] = np.nan
    for k in top_ks:
        col = f"hit@{k}"
        meta_name = f"meta_hit{k}"
        meta_aurc = compute_aurc_table({meta_name: meta_by_loss[col]}, hit_rates={col: hits[col]})
        aurc.loc["meta", col] = float(meta_aurc.loc[meta_name, col])
    return aurc


def _curves_for_measure(
    loss_values: np.ndarray,
    score_values: np.ndarray,
    measure: str,
    cov: np.ndarray,
    confidence_oriented: bool = False,
) -> np.ndarray:
    valid = ~np.isnan(score_values)
    if valid.sum() < 10:
        return np.full_like(cov, np.nan, dtype=float)
    uv = score_values[valid]
    if confidence_oriented or is_confidence_score(measure):
        uv = -uv
    order = np.argsort(uv)
    losses_valid = loss_values[valid]
    n = len(order)
    return np.asarray([losses_valid[order[:max(1, int(n * c))]].mean() for c in cov])


def plot_meta_paired(
    losses: Dict[str, np.ndarray],
    uncertainties: Dict[str, np.ndarray],
    meta_by_loss: Dict[str, np.ndarray],
    aurc: pd.DataFrame,
    out_path: Path,
    top_ks: Iterable[int],
    measures: List[str],
    title: str = "",
    confidence_oriented: bool = False,
) -> None:
    _register_meta_style()
    loss_cols = [f"hit@{k}" for k in top_ks]
    all_measures = [m for m in measures if m == "meta" or m in uncertainties]
    aurc = aurc.loc[[m for m in all_measures + ["oracle", "random"] if m in aurc.index]]

    n_cols = len(loss_cols)
    bar_h = _BAR_PITCH * len(aurc) + _BAR_PAD
    leg_ratio = _LEG_W / (_ROW_W / n_cols)
    total_h = _RC_H + bar_h + 0.15

    fig = plt.figure(figsize=(_FW, total_h))
    gs = fig.add_gridspec(
        2,
        n_cols + 1,
        width_ratios=[1] * n_cols + [leg_ratio],
        height_ratios=[_RC_H, bar_h],
        hspace=0.12,
        wspace=0.20,
        left=0.06,
        right=0.99,
        top=0.96,
        bottom=0.05,
    )

    global_y_top = max(float(losses[col].mean()) for col in loss_cols) / RAND_LINE_FRAC
    cov = np.linspace(1.0, 0.01, 100)

    for idx, col in enumerate(loss_cols):
        ax = fig.add_subplot(gs[0, idx])
        _setup_ax(ax)
        lv = losses[col]
        n = len(lv)

        oracle_order = np.argsort(-lv)
        ax.plot(
            cov,
            [lv[oracle_order[-max(1, int(n * c)):]].mean() for c in cov],
            color=DEFAULT_COLOR_MAP["oracle"],
            lw=2.5,
        )
        ax.axhline(
            float(lv.mean()),
            color=DEFAULT_COLOR_MAP["random"],
            ls="--",
            alpha=0.9,
            lw=2.5,
        )

        for measure in all_measures:
            score_values = meta_by_loss[col] if measure == "meta" else uncertainties[measure]
            curve = _curves_for_measure(
                lv, score_values, measure, cov, confidence_oriented=confidence_oriented
            )
            if np.all(np.isnan(curve)):
                continue
            ls, marker = get_metric_style(measure)
            ax.plot(
                cov,
                curve,
                color=get_metric_color(measure),
                lw=2.3,
                ls=ls,
                marker=marker,
                markevery=10,
                markersize=6,
                alpha=0.9,
            )

        ax.set_xlim(1, 0)
        ax.set_ylim(0, global_y_top)
        ax.set_title(_fmt_title(col), fontsize=_FH)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=_FT)
        _setup_rc_ticks(ax)
        if idx == 0:
            ax.set_ylabel("Error Rate", fontsize=_FL)
        else:
            ax.tick_params(axis="y", labelleft=False)

    for idx, col in enumerate(loss_cols):
        ax = fig.add_subplot(gs[1, idx])
        present = [m for m in aurc.index if col in aurc.columns and pd.notna(aurc.loc[m, col])]
        vals = aurc.loc[present, col].sort_values(ascending=True)
        _format_bar_panel(ax, vals, col)
        ax.set_title("")

    lax = fig.add_subplot(gs[:, n_cols])
    lax.axis("off")
    lax.legend(
        handles=_make_legend_entries(all_measures),
        loc="center",
        fontsize=_FA,
        framealpha=0.95,
        borderaxespad=0,
    )

    if title:
        fig.suptitle(title, fontsize=_FH, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_for_model(
    label: str,
    eval_dir: Path,
    meta_dir: Path,
    out_dir: Path,
    top_ks: Iterable[int],
    measures: List[str],
) -> Dict[str, str]:
    score_path = eval_dir / "scores_ranker_score.pt"
    unc_path = eval_dir / "uncertainties_score.npz"
    aurc_path = eval_dir / "aurc_retrieval_score.csv"
    meta_path = meta_dir / "meta_scores_test.npz"

    hits, losses = _load_score_data(score_path, top_ks)
    uncertainties = _load_uncertainties(unc_path)
    meta_by_loss = _load_meta_by_loss(meta_path, top_ks)
    aurc = _augment_aurc(aurc_path, hits, meta_by_loss, top_ks)

    stem = label.lower().replace(" ", "_").replace("/", "_")
    out_path = out_dir / f"{stem}_rc_aurc_paired_retrieval_score_meta.pdf"
    plot_meta_paired(losses, uncertainties, meta_by_loss, aurc, out_path, top_ks, measures)
    aurc.to_csv(out_path.with_name(out_path.stem + "_aurc.csv"))

    return {
        "label": label,
        "figure_pdf": str(out_path),
        "figure_png": str(out_path.with_suffix(".png")),
        "aurc_csv": str(out_path.with_name(out_path.stem + "_aurc.csv")),
        "score_path": str(score_path),
        "uncertainty_path": str(unc_path),
        "meta_path": str(meta_path),
    }


def build_canonical_model(
    run_label: str,
    label: str,
    query_scores_path: Path,
    out_dir: Path,
    top_ks: Iterable[int],
    measures: List[str],
    include_meta: bool,
) -> Dict[str, str]:
    scores = pd.read_parquet(query_scores_path)
    scores = scores[
        (scores["run_label"] == run_label)
        & (scores["split"] == "test")
        & (scores["evaluation_candidate_setting"] == "formula_official_capped")
    ]
    if scores.empty:
        raise ValueError(f"No official formula test rows for {run_label}")

    losses, meta_by_loss, aurc_columns, rel_aurc_columns = {}, {}, {}, {}
    common_confidences = {}
    canonical_measures = [name for name in measures if name != "meta"]
    for k in top_ks:
        frame = scores[scores["K"] == k].reset_index(drop=True)
        tie_break = str(frame["candidate_tie_break"].iloc[0])
        if tie_break != "source_order":
            frame = frame.sort_values("query_id").reset_index(drop=True)
            tie_break = "query_id"
        col = f"hit@{k}"
        hits = frame["hit"].to_numpy(dtype=np.float64)
        losses[col] = 1.0 - hits
        confidences = {
            name: frame[name].to_numpy(dtype=np.float64)
            for name in canonical_measures
            if name in frame
        }
        if include_meta:
            if "meta_full" not in frame or frame["meta_full"].isna().any():
                raise ValueError(f"Missing canonical meta scores for {run_label} Hit@{k}")
            meta_by_loss[col] = frame["meta_full"].to_numpy(dtype=np.float64)
            confidences["meta"] = meta_by_loss[col]
        aurc_k, rel_aurc_k = canonical_aurc_table(
            confidences,
            {col: hits},
            query_ids=frame["query_id"].astype(str),
            convention=str(frame["aurc_convention"].iloc[0]),
            tie_break=tie_break,
        )
        aurc_columns[col] = aurc_k[col]
        rel_aurc_columns[col] = rel_aurc_k[col]
        if not common_confidences:
            common_confidences = {
                name: values for name, values in confidences.items() if name != "meta"
            }

    aurc = pd.DataFrame(aurc_columns)
    rel_aurc = pd.DataFrame(rel_aurc_columns)
    suffix = "_meta" if include_meta else ""
    out_path = out_dir / f"{run_label}_rc_aurc_paired_retrieval_score{suffix}.pdf"
    plotted_measures = [
        name for name in canonical_measures if name in common_confidences
    ] + (["meta"] if include_meta else [])
    plot_meta_paired(
        losses,
        common_confidences,
        meta_by_loss,
        aurc,
        out_path,
        top_ks,
        plotted_measures,
        title="",
        confidence_oriented=True,
    )
    aurc_path = out_path.with_name(out_path.stem + "_aurc.csv")
    aurc.to_csv(aurc_path)
    rel_aurc_path = out_path.with_name(out_path.stem + "_rel_aurc.csv")
    rel_aurc.to_csv(rel_aurc_path)
    return {
        "label": label,
        "figure_kind": "meta" if include_meta else "manuscript",
        "figure_pdf": str(out_path),
        "figure_png": str(out_path.with_suffix(".png")),
        "aurc_csv": str(aurc_path),
        "rel_aurc_csv": str(rel_aurc_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/revision_meta/joint_plots"))
    ap.add_argument("--query_scores", type=Path, help="Canonical query_scores.parquet from the rerun")
    ap.add_argument("--top_ks", type=int, nargs="+", default=DEFAULT_TOP_KS)
    ap.add_argument("--measures", nargs="+", default=DEFAULT_MEASURES)
    ap.add_argument("--mlp_eval_dir", type=Path, default=DEFAULT_MODELS["mlp"]["eval_dir"])
    ap.add_argument("--mlp_meta_dir", type=Path, default=DEFAULT_MODELS["mlp"]["meta_dir"])
    ap.add_argument("--transformer_eval_dir", type=Path, default=DEFAULT_MODELS["transformer"]["eval_dir"])
    ap.add_argument("--transformer_meta_dir", type=Path, default=DEFAULT_MODELS["transformer"]["meta_dir"])
    ap.add_argument("--mlp_label", default=DEFAULT_MODELS["mlp"]["label"])
    ap.add_argument("--transformer_label", default=DEFAULT_MODELS["transformer"]["label"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.query_scores is not None:
        meta_measures = ["confidence", "retrieval_total", "score_gap"]
        rows = []
        manuscript_models = [
            ("mlp_formula", "MLP ensemble"),
            ("transformer_formula", "Transformer ensemble"),
            ("mlp_mc_dropout", "MLP MC dropout"),
            ("mlp_laplace", "MLP Laplace"),
        ]
        for run_label, label in manuscript_models:
            rows.append(build_canonical_model(
                run_label, label, args.query_scores, args.out_dir, args.top_ks,
                MANUSCRIPT_MEASURES, include_meta=False,
            ))
        for run_label, label in manuscript_models[:2]:
            rows.append(build_canonical_model(
                run_label, label, args.query_scores, args.out_dir, args.top_ks,
                meta_measures, include_meta=True,
            ))
        pd.DataFrame(rows).to_csv(args.out_dir / "joint_plot_manifest.csv", index=False)
        print(f"Saved canonical manuscript and meta joint plots to {args.out_dir}")
        return
    rows = [
        build_for_model(
            args.mlp_label,
            args.mlp_eval_dir,
            args.mlp_meta_dir,
            args.out_dir,
            args.top_ks,
            args.measures,
        ),
        build_for_model(
            args.transformer_label,
            args.transformer_eval_dir,
            args.transformer_meta_dir,
            args.out_dir,
            args.top_ks,
            args.measures,
        ),
    ]
    pd.DataFrame(rows).to_csv(args.out_dir / "meta_joint_plot_manifest.csv", index=False)
    print(f"Saved meta joint plots to {args.out_dir}")


if __name__ == "__main__":
    main()
