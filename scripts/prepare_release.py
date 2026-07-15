#!/usr/bin/env python3
"""Plan, build, finalize, and verify the paper's canonical artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ms_uq.evaluation.artifacts import (
    build_cleanup_manifest,
    build_inventory,
    build_paper_results,
    build_release,
    finalize_release,
    validate_paper_results,
    verify_release,
)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/paper_results"))
    parser.add_argument("--release-dir", type=Path, default=Path("releases/zenodo"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="Inventory the repository without modifying or deleting files")
    add_common(plan)
    plan.add_argument("--output", type=Path, default=Path("outputs/release_inventory.tsv"))

    build = commands.add_parser("build", help="Build canonical results and the seven release files")
    add_common(build)
    build.add_argument("--source-run", type=Path, required=True, help="Completed frozen run used only as the local packaging source")
    build.add_argument("--force-results", action="store_true")
    build.add_argument("--copy-results", action="store_true", help="Copy rather than hard-link local canonical results")
    build.add_argument("--skip-archives", action="store_true", help="Build and validate outputs/paper_results only")

    finalize = commands.add_parser(
        "finalize",
        help="Refresh source and release metadata without rewriting large payload archives",
    )
    add_common(finalize)

    verify = commands.add_parser("verify", help="Verify canonical results and all release archives")
    add_common(verify)

    args = parser.parse_args()
    repo = args.repo.resolve()
    results_dir = (repo / args.results_dir).resolve() if not args.results_dir.is_absolute() else args.results_dir.resolve()
    release_dir = (repo / args.release_dir).resolve() if not args.release_dir.is_absolute() else args.release_dir.resolve()

    if args.command == "plan":
        output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
        frame = build_inventory(repo, output)
        summary = frame.groupby("category").size().to_dict()
        print(json.dumps({"inventory": str(output), "files_by_category": summary}, indent=2))
        return

    if args.command == "build":
        source_run = args.source_run.resolve()
        if args.force_results or not results_dir.exists():
            report = build_paper_results(
                source_run, results_dir, hardlink=not args.copy_results, force=args.force_results,
            )
        else:
            report = validate_paper_results(results_dir)
            if not report["passed"]:
                raise SystemExit("Existing canonical results failed validation; rebuild with --force-results")
        cleanup = build_cleanup_manifest(repo, repo / "outputs/cleanup_manifest.tsv")
        inventory = build_inventory(repo, repo / "outputs/release_inventory.tsv")
        print(f"Validated canonical results: {results_dir}")
        print(f"Cleanup candidates: {len(cleanup)}; inventoried files: {len(inventory)}")
        if not args.skip_archives:
            release_report = build_release(repo, source_run, results_dir, release_dir)
            print(json.dumps(release_report, indent=2))
            print(f"Seven-file release: {release_dir}")
        return

    if args.command == "finalize":
        release_report = finalize_release(repo, release_dir)
        print(json.dumps(release_report, indent=2))
        print(f"Finalized source and metadata in: {release_dir}")
        return

    result_report = validate_paper_results(results_dir)
    release_report = verify_release(release_dir)
    combined = {"passed": result_report["passed"] and release_report["passed"], "results": result_report, "release": release_report}
    print(json.dumps(combined, indent=2))
    if not combined["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
