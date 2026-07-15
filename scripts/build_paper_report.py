#!/usr/bin/env python3
"""Build the static, data-free HTML index for canonical paper results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd


def _relative(target: Path, report_dir: Path) -> str:
    return escape(str(target.resolve().relative_to(report_dir.resolve().parent)))


def _table(path: Path, report_dir: Path, limit: int | None = None) -> str:
    if not path.is_file():
        return f"<p class='missing'>Missing: {escape(str(path))}</p>"
    frame = pd.read_csv(path)
    if limit is not None:
        frame = frame.head(limit)
    href = "../" + _relative(path, report_dir)
    return f"<p><a href='{href}'>{escape(path.name)}</a></p>" + frame.to_html(
        classes="data", border=0, index=False, float_format=lambda value: f"{value:.4f}",
    )


def _links(paths: Iterable[Path], report_dir: Path) -> str:
    rows = []
    for path in paths:
        if path.is_file():
            href = "../" + _relative(path, report_dir)
            rows.append(f"<li><a href='{href}'>{escape(str(path.name))}</a></li>")
    return "<ul>" + "\n".join(rows) + "</ul>"


def _figure(path: Path, report_dir: Path, caption: str) -> str:
    if not path.is_file():
        return f"<p class='missing'>Missing image: {escape(str(path))}</p>"
    href = "../" + _relative(path, report_dir)
    pdf = path.with_suffix(".pdf")
    pdf_link = f" <a href='../{_relative(pdf, report_dir)}'>PDF</a>" if pdf.is_file() else ""
    return (
        "<figure>"
        f"<a href='{href}'><img src='{href}' alt='{escape(caption)}'></a>"
        f"<figcaption>{escape(caption)}.{pdf_link}</figcaption>"
        "</figure>"
    )


def build_report(source_dir: Path, out_dir: Path) -> Path:
    source_dir = source_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = source_dir / "figures"
    tables = source_dir / "tables" if (source_dir / "tables").is_dir() else source_dir / "results"
    numerical = source_dir / "numerical" if (source_dir / "numerical").is_dir() else source_dir / "results"
    analyses = source_dir / "analyses" if (source_dir / "analyses").is_dir() else source_dir
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks = [
        _figure(figures / "temperature/temperature_sensitivity_rel_aurc.png", out_dir, "Temperature sensitivity of selective-prediction performance"),
        _figure(figures / "meta/mlp_formula_rc_aurc_paired_retrieval_score_meta.png", out_dir, "MLP ensemble risk-coverage curves with logistic meta-score"),
        _figure(figures / "meta/transformer_formula_rc_aurc_paired_retrieval_score_meta.png", out_dir, "Transformer ensemble risk-coverage curves with logistic meta-score"),
        _figure(figures / "meta/mlp_mc_dropout_rc_aurc_paired_retrieval_score.png", out_dir, "MC Dropout risk-coverage curves at T=0.003"),
        _figure(figures / "meta/mlp_laplace_rc_aurc_paired_retrieval_score.png", out_dir, "Laplace risk-coverage curves at T=0.003"),
        _figure(figures / "sgr_coverage_mlp_formula_seed42.png", out_dir, "MLP ensemble selective risk control"),
        _figure(figures / "sgr_coverage_transformer_formula_seed42.png", out_dir, "Transformer ensemble selective risk control"),
        _figure(figures / "candidates/candidate_size_stratification.png", out_dir, "Performance stratified by official formula candidate-set size"),
        _figure(figures / "candidates/candidate_distribution_histograms.png", out_dir, "Candidate-set size distributions"),
        _figure(figures / "correlation/mlp_formula_confidence_spearman_heatmap.png", out_dir, "Spearman correlation among MLP confidence scores"),
    ]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Selective MS/MS Paper Results</title><style>
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#222; background:#fafafa; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 24px 56px; }} h1 {{ font-size:30px; margin:0 0 8px; }}
h2 {{ font-size:21px; margin:34px 0 12px; border-top:1px solid #ddd; padding-top:8px; }} p,li {{ line-height:1.45; }}
.muted {{ color:#666; }} .missing {{ color:#a33; }} figure {{ margin:18px 0 28px; background:white; padding:12px; border:1px solid #ddd; }}
img {{ width:100%; max-height:760px; object-fit:contain; display:block; }} table.data {{ border-collapse:collapse; width:100%; background:white; font-size:13px; }}
table.data th,table.data td {{ border:1px solid #ddd; padding:6px 8px; text-align:right; }} a {{ color:#175a9c; text-decoration:none; }}
</style></head><body><main>
<h1>Selective MS/MS Paper Results</h1><p class="muted">Generated {created} from the frozen, validated result tree.</p>
<h2>Figures</h2>{''.join(blocks)}
<h2>Tables</h2><h3>Supervised score</h3>{_table(tables / 'table_supervised_combination.csv', out_dir, 100)}
<h3>Uncertainty methods</h3>{_table(tables / 'table_uq_methods.csv', out_dir, 250)}
<h3>Candidate settings</h3>{_table(tables / 'table_candidate_settings.csv', out_dir, 150)}
<h2>Machine-Readable Results</h2>{_links([
    numerical / 'metrics.csv', numerical / 'query_scores.parquet',
    analyses / 'temperature/temperature_sensitivity_rel_aurc.csv',
    analyses / 'sgr/sgr_results_seed42.csv',
    source_dir / 'provenance/evaluation_matrix.tsv',
    source_dir / 'provenance/validation_report.json',
], out_dir)}</main></body></html>"""
    output = out_dir / "index.html"
    output.write_text(html)
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.source_dir, args.out_dir)


if __name__ == "__main__":
    main()
