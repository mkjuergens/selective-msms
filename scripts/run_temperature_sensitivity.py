#!/usr/bin/env python3
"""Temperature sensitivity for retrieval confidence/entropy selective scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from ms_uq.evaluation.visualisation import (
    _FA,
    _FH,
    _FL,
    _FT,
    _FW,
    _ROW_H,
    _setup_ax,
    display_name,
    get_metric_color,
    get_metric_style,
)

from ms_uq.evaluation.confidence_features import (
    hit_arrays,
    relative_aurc,
    softmax_temperature_features,
    score_position_features,
    spearman_confidence_log_candidates,
)
from ms_uq.utils import load_ground_truth, resolve_candidate_paths
from ms_uq.evaluation.candidate_sets import canonical_candidate_view, normalize_inchikey


DEFAULT_TEMPERATURES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
DEFAULT_TOP_KS = [1, 5, 20]
TEMPERATURE_DEPENDENT_MEASURES = {
    "confidence",
    "retrieval_total",
    "retrieval_aleatoric",
    "normalized_entropy",
}


def parse_model(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Models must be specified as label=/path/to/output_dir")
    label, path = value.split("=", 1)
    return label, Path(path)


def find_score_file(model_dir: Path, filename: str | None) -> Path:
    if filename:
        path = model_dir / filename
        if path.exists():
            return path
        path = Path(filename)
        if path.exists():
            return path
        raise FileNotFoundError(filename)
    for name in ["scores_ranker_score.pt", "scores_cosine_score.pt", "scores_ranker_fingerprint.pt"]:
        path = model_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No score file found in {model_dir}")


def load_score_data(score_path: Path, gt_path: Path | None):
    data = torch.load(score_path, map_location="cpu")
    scores_flat = data["scores_flat"].float()
    scores_stack = data.get("scores_stack_flat")
    if scores_stack is None:
        scores_stack = scores_flat.unsqueeze(0)
    scores_stack = scores_stack.float()
    ptr = data["ptr"].long()
    labels = data.get("labels_flat")
    if labels is None and gt_path is not None:
        _, labels = load_ground_truth(gt_path)
    if labels is None:
        raise ValueError(f"No labels_flat in {score_path}; pass --gt_path or recompute scores with labels.")
    return scores_flat, scores_stack, ptr, labels.float()


def metrics_for_panel(all_metrics: Dict[str, np.ndarray], k: int) -> Dict[str, np.ndarray]:
    names = [
        "confidence",
        "retrieval_total",
        "retrieval_aleatoric",
        "normalized_entropy",
        "score_gap",
        f"score_gap_at_{k}",
        f"rank_var_{k}",
    ]
    return {name: all_metrics[name] for name in names if name in all_metrics}


def run_model(
    label: str,
    model_dir: Path,
    score_filename: str | None,
    temperatures: List[float],
    top_ks: List[int],
    gt_path: Path | None,
    dataset_tsv: Path | None = None,
    helper_dir: Path | None = None,
    candidate_setting: str = "formula",
    candidate_record_policy: str = "deduplicate",
    candidate_tie_break: str = "candidate_id",
    aurc_convention: str = "discrete_prefix_mean",
    feature_convention: str = "canonical",
    max_queries: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    score_path = find_score_file(model_dir, score_filename)
    scores_flat, scores_stack, ptr, labels = load_score_data(score_path, gt_path)
    query_ids = None
    candidate_ids_by_query = None
    if dataset_tsv is not None:
        metadata = pd.read_csv(dataset_tsv, sep="\t", usecols=["identifier", "smiles", "fold"])
        metadata = metadata.loc[metadata["fold"] == "test"].reset_index(drop=True)
        if max_queries is not None:
            metadata = metadata.iloc[:max_queries].copy()
        if len(metadata) != ptr.numel() - 1:
            raise ValueError(f"Test metadata rows do not match score rows for {label}")
        query_ids = metadata["identifier"].astype(str).to_numpy()
        if helper_dir is not None and candidate_tie_break == "candidate_id":
            _, fp_path, inchi_path = resolve_candidate_paths(helper_dir, candidate_setting)
            cache = {}
            with np.load(fp_path) as candidate_fps, np.load(inchi_path) as candidate_inchis:
                for smiles in metadata["smiles"].drop_duplicates():
                    if candidate_record_policy == "preserve":
                        cache[smiles] = np.asarray([
                            normalize_inchikey(value) for value in candidate_inchis[smiles]
                        ], dtype=object)
                    else:
                        cache[smiles] = canonical_candidate_view(
                            candidate_inchis[smiles], candidate_fps[smiles]
                        )[0]
            candidate_ids_by_query = [cache[smiles] for smiles in metadata["smiles"]]
        elif candidate_tie_break != "source_order":
            raise ValueError("candidate_tie_break must be source_order or candidate_id")
    hits = hit_arrays(scores_flat, labels, ptr, top_ks, candidate_ids=candidate_ids_by_query)
    invariant, imputations = score_position_features(scores_flat, ptr, top_ks)
    imputation_df = pd.DataFrame([imp.__dict__ | {"model": label} for imp in imputations])

    rows = []
    for temperature in temperatures:
        temp_metrics = softmax_temperature_features(
            scores_stack, scores_flat, ptr, temperature,
            top_ks=top_ks, candidate_ids=candidate_ids_by_query,
            feature_convention=feature_convention,
        )
        all_metrics = {**invariant, **temp_metrics}
        rho, p_value = spearman_confidence_log_candidates(
            all_metrics["confidence"], invariant["log_n_candidates"]
        )
        for k in top_ks:
            panel_metrics = metrics_for_panel(all_metrics, k)
            rel = relative_aurc(
                panel_metrics, {f"hit@{k}": hits[f"hit@{k}"]}, query_ids=query_ids,
                convention=aurc_convention, tie_break=candidate_tie_break,
            )
            for measure, value in rel[f"hit@{k}"].items():
                rows.append({
                    "model": label, "temperature": float(temperature), "hit_k": int(k),
                    "measure": measure, "rel_aurc": float(value),
                    "spearman_conf_log_n": rho, "spearman_conf_log_n_p": p_value,
                    "score_file": str(score_path),
                })
    return pd.DataFrame(rows), imputation_df


def _metric_order(k: int) -> List[str]:
    names = [
        "confidence",
        "retrieval_total",
        "retrieval_aleatoric",
        "score_gap",
    ]
    if k != 1:
        names.append(f"score_gap_at_{k}")
    names.append(f"rank_var_{k}")
    return names


def _metric_display_name(name: str) -> str:
    if name.startswith("score_gap_at_"):
        k = name.rsplit("_", 1)[1]
        return rf"$\kappa_{{\rm gap}}^{{({k})}}$"
    return display_name(name)


def _metric_color(name: str) -> str:
    if name.startswith("score_gap_at_"):
        return get_metric_color("score_gap")
    return get_metric_color(name)


def _metric_style(name: str):
    if name.startswith("score_gap_at_"):
        return "-", "P"
    return get_metric_style(name)


def _model_display_name(name: str) -> str:
    lookup = {
        "mlp": "MLP",
        "transformer": "Transformer",
    }
    return lookup.get(name.lower(), name)


def _temperature_formatter(value, _pos):
    labelled_ticks = [0.001, 0.01, 0.1, 1.0]
    if not any(np.isclose(value, tick) for tick in labelled_ticks):
        return ""
    if value >= 1:
        return f"{value:g}"
    return f"{value:.3g}"


def _is_temperature_dependent(name: str) -> bool:
    return name in TEMPERATURE_DEPENDENT_MEASURES


def plot_temperature(df: pd.DataFrame, out_path: Path, rankwise_temp: float = 0.003) -> None:
    if df.empty:
        return
    models = list(df["model"].drop_duplicates())
    ks = sorted(df["hit_k"].unique())
    n_rows, n_cols = len(models), len(ks)
    fig_h = max(3.6, _ROW_H * 0.86 * n_rows)
    fig = plt.figure(figsize=(_FW, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1] * n_cols + [0.82],
        left=0.065,
        right=0.99,
        top=0.90,
        bottom=0.17,
        wspace=0.26,
        hspace=0.20,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col in range(n_cols):
            axes[row, col] = fig.add_subplot(gs[row, col])
    ax_leg = fig.add_subplot(gs[:, n_cols])
    ax_leg.axis("off")

    plot_df = df[df["measure"].isin({m for k in ks for m in _metric_order(k)})].copy()
    y_min = np.floor((plot_df["rel_aurc"].min() - 0.03) * 20) / 20
    y_max = np.ceil((plot_df["rel_aurc"].max() + 0.03) * 20) / 20
    y_min = max(0.0, y_min)
    temperatures = sorted(df["temperature"].unique())

    for row, model in enumerate(models):
        for col, k in enumerate(ks):
            ax = axes[row, col]
            _setup_ax(ax)
            sub = df[(df["model"] == model) & (df["hit_k"] == k)]
            for measure in _metric_order(k):
                ms = sub[sub["measure"] == measure].sort_values("temperature")
                if ms.empty:
                    continue
                ls, marker = _metric_style(measure)
                if _is_temperature_dependent(measure):
                    ax.plot(
                        ms["temperature"],
                        ms["rel_aurc"],
                        color=_metric_color(measure),
                        ls=ls,
                        marker=marker,
                        markersize=7,
                        markeredgecolor="white",
                        markeredgewidth=0.65,
                        lw=2.3,
                        alpha=0.95,
                    )
                else:
                    ax.hlines(
                        float(ms["rel_aurc"].iloc[0]),
                        xmin=min(temperatures),
                        xmax=max(temperatures),
                        color=_metric_color(measure),
                        ls=ls,
                        lw=2.15,
                        alpha=0.72,
                    )

            ax.axvline(
                rankwise_temp,
                color="#555555",
                ls=(0, (5, 2)),
                lw=1.6,
                alpha=0.75,
                zorder=0,
            )
            ax.set_xscale("log")
            ax.set_xlim(min(temperatures) * 0.82, max(temperatures) * 1.22)
            ax.set_ylim(y_min, y_max)
            ax.xaxis.set_major_locator(mticker.FixedLocator(temperatures))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_temperature_formatter))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
            ax.tick_params(axis="both", labelsize=_FT)
            ax.grid(axis="y", alpha=0.18, lw=0.5)
            if row == 0:
                ax.set_title(f"Hit@{k}", fontsize=_FH)
            if row == n_rows - 1:
                for tick_label in ax.get_xticklabels():
                    tick_label.set_rotation(35)
                    tick_label.set_ha("right")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f"{_model_display_name(model)}\nrelAURC", fontsize=_FL)
            else:
                ax.set_yticklabels([])

    legend_measures = []
    available = set(df["measure"])
    for k in ks:
        for measure in _metric_order(k):
            if measure not in legend_measures and measure in available:
                legend_measures.append(measure)
    handles = []
    labels = []
    for measure in legend_measures:
        ls, marker = _metric_style(measure)
        marker_kwargs = {}
        if _is_temperature_dependent(measure):
            marker_kwargs = {
                "marker": marker,
                "markersize": 8,
                "markeredgecolor": "white",
                "markeredgewidth": 0.65,
            }
        handles.append(Line2D(
            [0], [0],
            color=_metric_color(measure),
            lw=2.5,
            ls=ls,
            alpha=0.95 if _is_temperature_dependent(measure) else 0.72,
            **marker_kwargs,
        ))
        labels.append(_metric_display_name(measure))
    handles.append(Line2D([0], [0], color="#555555", lw=1.6, ls=(0, (5, 2))))
    labels.append(r"$T_{\rm train}=0.003$")
    ax_leg.legend(
        handles,
        labels,
        loc="center left",
        fontsize=_FA,
        frameon=False,
        borderpad=0.3,
        labelspacing=0.9,
        handlelength=2.4,
    )
    fig.text(
        0.43,
        0.055,
        r"Evaluation temperature $T_{\rm eval}$",
        ha="center",
        va="center",
        fontsize=_FL,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=parse_model, action="append", required=True,
                    help="Model output as label=/path/to/eval_dir")
    ap.add_argument("--score_filename", default=None,
                    help="Optional score filename inside every model directory")
    ap.add_argument("--gt_path", type=Path, default=None)
    ap.add_argument("--dataset_tsv", type=Path)
    ap.add_argument("--helper_dir", type=Path)
    ap.add_argument("--candidate_setting", default="formula")
    ap.add_argument("--candidate_record_policy", choices=["preserve", "deduplicate"], default="deduplicate")
    ap.add_argument("--candidate_tie_break", choices=["source_order", "candidate_id"], default="candidate_id")
    ap.add_argument("--aurc_convention", choices=["discrete_prefix_mean", "manuscript_trapezoid_seed42"], default="discrete_prefix_mean")
    ap.add_argument("--feature_convention", choices=["canonical", "manuscript"], default="canonical")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    ap.add_argument("--top_ks", type=int, nargs="+", default=DEFAULT_TOP_KS)
    ap.add_argument("--rankwise_temp", type=float, default=0.003)
    ap.add_argument("--max_queries", type=int)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, all_imputations = [], []
    for label, model_dir in args.model:
        rows, imputations = run_model(
            label, model_dir, args.score_filename, args.temperatures, args.top_ks, args.gt_path,
            dataset_tsv=args.dataset_tsv, helper_dir=args.helper_dir,
            candidate_setting=args.candidate_setting,
            candidate_record_policy=args.candidate_record_policy,
            candidate_tie_break=args.candidate_tie_break,
            aurc_convention=args.aurc_convention,
            feature_convention=args.feature_convention,
            max_queries=args.max_queries,
        )
        all_rows.append(rows)
        all_imputations.append(imputations)

    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    imp = pd.concat(all_imputations, ignore_index=True) if all_imputations else pd.DataFrame()
    df.to_csv(args.out_dir / "temperature_sensitivity_rel_aurc.csv", index=False)
    imp.to_csv(args.out_dir / "score_gap_at_k_imputations.csv", index=False)
    plot_temperature(df, args.out_dir / "temperature_sensitivity_rel_aurc.pdf", rankwise_temp=args.rankwise_temp)
    print(f"Saved temperature sensitivity outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
