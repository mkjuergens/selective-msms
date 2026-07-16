#!/usr/bin/env python3
"""Prepare split-specific fp_probs and ragged score bundles for paper analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from ms_uq.inference import load_ranker
from ms_uq.inference.retrieve import scores_from_loader
from ms_uq.utils import load_predictions, make_train_val_test_loaders
from ms_uq.paper.evaluation import EvalConfig, generate_predictions, score_cache_name


def split_loader(config: EvalConfig, split: str):
    train_loader, val_loader, test_loader = make_train_val_test_loaders(
        config.dataset_tsv,
        config.helper_dir,
        config.bin_width,
        config.batch_size,
        config.num_workers,
        architecture=config.architecture,
        candidate_setting=config.candidate_setting,
        max_mz=config.max_mz,
        n_peaks=config.n_peaks,
        prec_mz_intensity=config.prec_mz_intensity,
        label_mode=config.label_mode,
        query_identity_source=config.query_identity_source,
        missing_target_policy=config.missing_target_policy,
        lazy_candidate_helpers=config.lazy_candidate_helpers,
    )
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    if split == "test":
        return test_loader
    raise ValueError("split must be one of train, val, test")


def limit_loader(loader, max_queries: int | None):
    if max_queries is None or max_queries >= len(loader.dataset):
        return loader
    dataset = Subset(loader.dataset, range(int(max_queries)))
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=False,
        prefetch_factor=2 if loader.num_workers > 0 else None,
        collate_fn=loader.collate_fn,
        worker_init_fn=loader.worker_init_fn,
        generator=torch.Generator().manual_seed(42),
    )


def write_metadata(dataset_tsv: Path, split: str, out_dir: Path) -> None:
    df = pd.read_csv(dataset_tsv, sep="\t")
    df[df["fold"] == split].reset_index(drop=True).to_csv(out_dir / f"metadata_{split}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_tsv", required=True)
    ap.add_argument("--helper_dir", required=True)
    ap.add_argument("--pred_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--architecture", choices=["mlp", "transformer"], default="mlp")
    ap.add_argument(
        "--candidate_setting",
        choices=[
            "formula",
            "mass",
            "formula_uncapped",
            "formula_pubchem_capped256",
            "formula_pubchem_record_capped256",
            "formula_pubchem_natural_capped256",
        ],
        default="formula",
    )
    ap.add_argument("--label_mode", choices=["fingerprint", "inchikey", "inchikey_fallback"], default="fingerprint")
    ap.add_argument("--query_identity_source", choices=["precomputed", "tsv"], default="precomputed")
    ap.add_argument("--missing_target_policy", choices=["error", "allow"], default="error")
    ap.add_argument("--lazy_candidate_helpers", action="store_true")
    ap.add_argument("--mode", choices=["ensemble", "single", "mcdo", "laplace"], default="ensemble")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--ckpts", default="")
    ap.add_argument("--ens_dir", default="")
    ap.add_argument("--ens_metric", default="reranker")
    ap.add_argument("--passes", type=int, default=50)
    ap.add_argument("--laplace_state", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--aggregation", default="score")
    ap.add_argument("--temperature", type=float, default=0.003)
    ap.add_argument("--topk_k", type=int, default=80)
    ap.add_argument("--topk_temp", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score_dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--bin_width", type=float, default=0.1)
    ap.add_argument("--max_mz", type=float, default=1005.0)
    ap.add_argument("--n_peaks", type=int, default=128)
    ap.add_argument("--prec_mz_intensity", type=float, default=1.1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_queries", type=int)
    args = ap.parse_args()

    args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = EvalConfig(
        dataset_tsv=args.dataset_tsv,
        helper_dir=args.helper_dir,
        architecture=args.architecture,
        candidate_setting=args.candidate_setting,
        label_mode=args.label_mode,
        query_identity_source=args.query_identity_source,
        missing_target_policy=args.missing_target_policy,
        lazy_candidate_helpers=args.lazy_candidate_helpers,
        mode=args.mode,
        ckpt=args.ckpt,
        ckpts=args.ckpts,
        ens_dir=args.ens_dir,
        ens_metric=args.ens_metric,
        passes=args.passes,
        laplace_state=args.laplace_state,
        seed=args.seed,
        temperature=args.temperature,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bin_width=args.bin_width,
        max_mz=args.max_mz,
        n_peaks=args.n_peaks,
        prec_mz_intensity=args.prec_mz_intensity,
        overwrite=args.overwrite,
    )
    loader = limit_loader(split_loader(config, args.split), args.max_queries)
    fp_path, ranker_path = generate_predictions(args.pred_dir, config, loader=loader)
    ranker = load_ranker(ranker_path, device=args.device) if ranker_path and Path(ranker_path).exists() else None
    Pbits, _, _, _ = load_predictions(args.pred_dir, metric=args.metric, aggregation="score", require_scores=False)
    if Pbits is None:
        raise FileNotFoundError(fp_path)
    score_prefix = "ranker" if ranker else args.metric
    score_path = args.out_dir / score_cache_name(score_prefix, args.aggregation, args.temperature)
    if score_path.exists() and not args.overwrite:
        print(f"Using cached {score_path}")
    else:
        result = scores_from_loader(
            Pbits,
            loader,
            metric=args.metric,
            aggregation=args.aggregation,
            temperature=args.temperature,
            topk_k=args.topk_k,
            topk_temp=args.topk_temp,
            return_labels=True,
            return_per_sample=True,
            ranker=ranker,
            device=args.device,
            score_dtype=torch.float64 if args.score_dtype == "float64" else torch.float32,
        )
        result.update({
            "candidate_setting": args.candidate_setting,
            "split": args.split,
            "label_mode": args.label_mode,
            "query_identity_source": args.query_identity_source,
            "candidate_record_policy": "preserve",
            "candidate_tie_break": "source_order",
            "score_dtype": args.score_dtype,
        })
        torch.save(result, score_path)
        print(f"Saved {score_path}")
    write_metadata(Path(args.dataset_tsv), args.split, args.out_dir)
    if args.max_queries is not None:
        metadata_path = args.out_dir / f"metadata_{args.split}.csv"
        pd.read_csv(metadata_path).iloc[:args.max_queries].to_csv(metadata_path, index=False)
    print(f"Prepared {args.split} bundle in {args.out_dir}")


if __name__ == "__main__":
    main()
