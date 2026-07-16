#!/usr/bin/env python3
"""Build or verify the Zenodo data deposit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ms_uq.paper.release import (
    build_release,
    discover_checkpoint_sources,
    validate_paper_results,
    verify_release,
)


def _resolve(repo: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["build", "verify"]:
        command = commands.add_parser(name)
        command.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
        command.add_argument("--data", type=Path, default=Path("data"))
        command.add_argument("--output", type=Path, default=Path("releases/zenodo"))

    args = parser.parse_args()
    repo = args.repo.resolve()
    data_dir = _resolve(repo, args.data)
    release_dir = _resolve(repo, args.output)

    results = validate_paper_results(data_dir / "results")
    checkpoints = discover_checkpoint_sources(data_dir)
    models = {"checkpoints": len(checkpoints)}

    if args.command == "build":
        release = build_release(repo, data_dir, release_dir)
    else:
        release = verify_release(release_dir)

    report = {
        "passed": results["passed"] and release["passed"],
        "results": results,
        "models": models,
        "release": release,
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
