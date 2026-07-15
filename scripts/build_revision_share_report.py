#!/usr/bin/env python3
"""Build a lightweight static HTML index for sharing revision outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_OUT = Path("outputs/revision_share")


def rel_link(target: Path, report_dir: Path) -> str:
    return escape(str(target.resolve().relative_to(report_dir.resolve().parent)))


def table_html(path: Path, report_dir: Path, n: int | None = None, index_col: int | None = None) -> str:
    if not path.exists():
        return f"<p class='missing'>Missing: {escape(str(path))}</p>"
    df = pd.read_csv(path, index_col=index_col)
    if n is not None:
        df = df.head(n)
    body = df.to_html(classes="data", border=0, float_format=lambda x: f"{x:.4f}")
    return f"<p><a href='../{rel_link(path, report_dir)}'>{escape(path.name)}</a></p>{body}"


def link_list(paths: Iterable[Path], report_dir: Path) -> str:
    items = []
    for path in paths:
        if path.exists():
            items.append(f"<li><a href='../{rel_link(path, report_dir)}'>{escape(str(path))}</a></li>")
        else:
            items.append(f"<li class='missing'>Missing: {escape(str(path))}</li>")
    return "<ul>" + "\n".join(items) + "</ul>"


def image_block(path: Path, report_dir: Path, caption: str) -> str:
    if not path.exists():
        return f"<p class='missing'>Missing image: {escape(str(path))}</p>"
    href = "../" + rel_link(path, report_dir)
    pdf = path.with_suffix(".pdf")
    pdf_link = f" <a href='../{rel_link(pdf, report_dir)}'>PDF</a>" if pdf.exists() else ""
    return (
        "<figure>"
        f"<a href='{href}'><img src='{href}' alt='{escape(caption)}'></a>"
        f"<figcaption>{escape(caption)}.{pdf_link}</figcaption>"
        "</figure>"
    )


def meta_summary(report_dir: Path, source_dir: Path | None = None) -> str:
    rows = []
    root = source_dir or Path("outputs")
    paths = [
        ("MLP", root / "figures/meta/mlp_formula_rc_aurc_paired_retrieval_score_meta_aurc.csv"),
        ("Transformer", root / "figures/meta/transformer_formula_rc_aurc_paired_retrieval_score_meta_aurc.csv"),
    ] if source_dir is not None else [
        ("MLP", Path("outputs/revision_meta/joint_plots/mlp_ensemble_rc_aurc_paired_retrieval_score_meta_aurc.csv")),
        ("Transformer", Path("outputs/revision_meta/joint_plots/transformer_ensemble_rc_aurc_paired_retrieval_score_meta_aurc.csv")),
    ]
    for model, path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        for measure in ["meta", "confidence", "score_gap", "retrieval_total"]:
            if measure in df.index:
                row = {"model": model, "measure": measure}
                row.update(df.loc[measure].to_dict())
                rows.append(row)
    if not rows:
        return "<p class='missing'>No meta AURC summary available.</p>"
    df = pd.DataFrame(rows)
    return df.to_html(classes="data", border=0, index=False, float_format=lambda x: f"{x:.4f}")


def build_canonical(out_dir: Path, source_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Selective MS/MS Canonical Revision Rerun</title><style>
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#222; background:#fafafa; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 24px 56px; }} h1 {{ font-size:30px; margin:0 0 8px; }}
h2 {{ font-size:21px; margin:34px 0 12px; border-top:1px solid #ddd; padding-top:8px; }} p,li {{ line-height:1.45; }}
.muted {{ color:#666; }} .missing {{ color:#a33; }} figure {{ margin:18px 0 28px; background:white; padding:12px; border:1px solid #ddd; }}
img {{ width:100%; max-height:760px; object-fit:contain; display:block; }} table.data {{ border-collapse:collapse; width:100%; background:white; font-size:13px; }}
table.data th,table.data td {{ border:1px solid #ddd; padding:6px 8px; text-align:right; }} a {{ color:#175a9c; text-decoration:none; }}
</style></head><body><main>
<h1>Selective MS/MS Canonical Revision Rerun</h1><p class="muted">Generated {created}. All figures and tables below originate from one frozen, validated result tree.</p>
<h2>Figures</h2>
{image_block(source_dir / "figures/temperature/temperature_sensitivity_rel_aurc.png", out_dir, "Temperature sensitivity at fixed candidate pools")}
{image_block(source_dir / "figures/meta/mlp_formula_rc_aurc_paired_retrieval_score_meta.png", out_dir, "MLP risk-coverage and AURC with logistic meta-score")}
{image_block(source_dir / "figures/meta/mlp_formula_rc_aurc_paired_retrieval_score.png", out_dir, "MLP manuscript risk-coverage and AURC")}
{image_block(source_dir / "figures/meta/transformer_formula_rc_aurc_paired_retrieval_score.png", out_dir, "Transformer manuscript risk-coverage and AURC")}
{image_block(source_dir / "figures/meta/transformer_formula_rc_aurc_paired_retrieval_score_meta.png", out_dir, "Transformer risk-coverage and AURC with logistic meta-score")}
{image_block(source_dir / "figures/meta/mlp_mc_dropout_rc_aurc_paired_retrieval_score.png", out_dir, "MLP MC-dropout risk-coverage and AURC at T=0.003")}
{image_block(source_dir / "figures/meta/mlp_laplace_rc_aurc_paired_retrieval_score.png", out_dir, "MLP Laplace risk-coverage and AURC at T=0.003")}
{image_block(source_dir / "figures/sgr_coverage_mlp_formula_seed42.png", out_dir, "MLP ensemble SGR coverage and empirical risk")}
{image_block(source_dir / "figures/sgr_coverage_transformer_formula_seed42.png", out_dir, "Transformer ensemble SGR coverage and empirical risk")}
{image_block(source_dir / "figures/candidates/candidate_size_stratification.png", out_dir, "Hit rate and AURC stratified by official formula candidate-set size")}
{image_block(source_dir / "figures/candidates/candidate_distribution_histograms.png", out_dir, "Candidate-set distributions")}
<h2>Tables</h2><h3>Supervised combination</h3>{table_html(source_dir / "results/table_supervised_combination.csv", out_dir, n=80)}
<h3>Uncertainty methods</h3>{table_html(source_dir / "results/table_uq_methods.csv", out_dir, n=200)}
<h3>Candidate settings</h3>{table_html(source_dir / "results/table_candidate_settings.csv", out_dir, n=100)}
<h3>Paired differences</h3>{table_html(source_dir / "results/paired_differences.csv", out_dir, n=100)}
<h2>Meta-model summary</h2>{meta_summary(out_dir, source_dir)}
<h2>Machine-readable outputs</h2>{link_list([
source_dir / "run_manifest.json", source_dir / "input_hashes.csv", source_dir / "results/query_scores.parquet",
source_dir / "results/metrics_tidy.csv", source_dir / "results/bootstrap_replicates.parquet",
source_dir / "results/risk_coverage_points.parquet", source_dir / "results/validation_report.json",
source_dir / "sgr/sgr_score_selection.csv", source_dir / "sgr/sgr_thresholds.csv", source_dir / "sgr/sgr_evaluation.csv",
source_dir / "figures/candidates/candidate_size_hit_rates.csv",
source_dir / "figures/candidates/candidate_size_stratified_aurc.csv",
source_dir / "figures/meta/mlp_formula_rc_aurc_paired_retrieval_score_aurc.csv",
source_dir / "figures/meta/transformer_formula_rc_aurc_paired_retrieval_score_aurc.csv",
source_dir / "figures/meta/mlp_formula_rc_aurc_paired_retrieval_score_meta_aurc.csv",
source_dir / "figures/meta/transformer_formula_rc_aurc_paired_retrieval_score_meta_aurc.csv",
source_dir / "figures/meta/mlp_formula_rc_aurc_paired_retrieval_score_rel_aurc.csv",
source_dir / "figures/meta/transformer_formula_rc_aurc_paired_retrieval_score_rel_aurc.csv",
source_dir / "figures/meta/mlp_formula_rc_aurc_paired_retrieval_score_meta_rel_aurc.csv",
source_dir / "figures/meta/transformer_formula_rc_aurc_paired_retrieval_score_meta_rel_aurc.csv",
source_dir / "figures/meta/mlp_mc_dropout_rc_aurc_paired_retrieval_score_rel_aurc.csv",
source_dir / "figures/meta/mlp_laplace_rc_aurc_paired_retrieval_score_rel_aurc.csv",
], out_dir)}</main></body></html>"""
    path = out_dir / "index.html"
    path.write_text(html)
    return path


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Selective MS/MS Revision Results</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #222; background: #fafafa; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ font-size: 30px; margin: 0 0 8px; }}
    h2 {{ font-size: 21px; margin: 34px 0 12px; padding-top: 4px; border-top: 1px solid #ddd; }}
    h3 {{ font-size: 17px; margin: 22px 0 8px; }}
    p, li {{ line-height: 1.45; }}
    .muted {{ color: #666; }}
    .missing {{ color: #a33; }}
    figure {{ margin: 18px 0 28px; background: white; padding: 12px; border: 1px solid #ddd; }}
    figcaption {{ margin-top: 8px; color: #555; font-size: 14px; }}
    img {{ width: 100%; max-height: 760px; object-fit: contain; display: block; background: white; }}
    table.data {{ border-collapse: collapse; width: 100%; background: white; font-size: 14px; }}
    table.data th, table.data td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    table.data th:first-child, table.data td:first-child {{ text-align: left; }}
    a {{ color: #175a9c; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #eee; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
<main>
  <h1>Selective MS/MS Revision Results</h1>
  <p class="muted">Generated {created}. Static report for quick internal review before moving figures/tables into the manuscript.</p>

  <h2>Recommendation</h2>
  <p>Use this static HTML bundle for routine sharing: it is reproducible, easy to archive, works from the filesystem, and links directly to source CSV/PDF artifacts. A Streamlit app is useful only if colleagues need interactive filtering or drill-down; otherwise it adds a server and dependency layer without much benefit.</p>

  <h2>Key Figures</h2>
  {image_block(Path("outputs/revision_temperature/temperature_sensitivity_rel_aurc.png"), out_dir, "Temperature sensitivity relAURC for MLP and transformer ensembles")}
  {image_block(Path("outputs/revision_meta/joint_plots/mlp_ensemble_rc_aurc_paired_retrieval_score_meta.png"), out_dir, "MLP ensemble risk-coverage/AURC with task-matched meta score")}
  {image_block(Path("outputs/revision_meta/joint_plots/transformer_ensemble_rc_aurc_paired_retrieval_score_meta.png"), out_dir, "Transformer ensemble risk-coverage/AURC with task-matched meta score")}

  <h2>Core Tables</h2>
  <h3>Transformer vs MLP Hit Rates</h3>
  {table_html(Path("outputs/revision_analysis/model_comparison_hit_rates.csv"), out_dir)}

  <h3>Meta AURC Summary</h3>
  {meta_summary(out_dir)}

  <h3>Dataset Audit</h3>
  {table_html(Path("outputs/revision_audit/dataset.csv"), out_dir)}
  {table_html(Path("outputs/revision_audit/candidates.csv"), out_dir)}

  <h2>Detailed Result Files</h2>
  {link_list([
      Path("outputs/revision_temperature/temperature_sensitivity_rel_aurc.csv"),
      Path("outputs/revision_meta/mlp_meta/meta_rel_aurc.csv"),
      Path("outputs/revision_meta/mlp_meta/meta_coefficients.csv"),
      Path("outputs/revision_meta/transformer_meta/meta_rel_aurc.csv"),
      Path("outputs/revision_meta/transformer_meta/meta_coefficients.csv"),
      Path("outputs/revision_meta/joint_plots/meta_joint_plot_manifest.csv"),
      Path("outputs/revision_sgr/fixed_split/mlp_bienc/sgr/sgr_results.csv"),
      Path("outputs/revision_sgr/fixed_split/transformer_ensemble/sgr/sgr_results.csv"),
      Path("outputs/revision_audit/audit_summary.md"),
  ], out_dir)}
</main>
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(html)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--source_dir", type=Path)
    args = ap.parse_args()
    path = build_canonical(args.out_dir, args.source_dir) if args.source_dir else build(args.out_dir)
    print(path)


if __name__ == "__main__":
    main()
