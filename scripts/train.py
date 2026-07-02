import argparse
import os
import ast
from pathlib import Path
import json
import shutil

import torch

from ms_uq.data import RetrievalDataset_PrecompFPandInchi
from massspecgym.data.transforms import MolFingerprinter, SpecBinner
from massspecgym.data.data_module import MassSpecDataModule
from ms_uq.models.fingerprint_mlp import FingerprintPredicter

try:
    from torch.serialization import add_safe_globals
    from massspecgym.models.base import Stage
    add_safe_globals([Stage])
except Exception:
    pass

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import LightningEnvironment

try:
    import pytorch_lightning as pl          # PL 0.10 – 2.x
    seed_everything = pl.seed_everything
except (ImportError, AttributeError):
    from lightning_fabric.utilities.seed import seed_everything


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def int_or_float(v):
    s = str(v).strip()
    if "." in s:
        return float(s)
    return int(s)


def make_parser() -> argparse.ArgumentParser:
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
    ):
        pass

    p = argparse.ArgumentParser(
        description="Training script (per-metric checkpoints + manifest).",
        formatter_class=CustomFormatter,
    )

    # Required paths
    p.add_argument("dataset_path", type=str, help="Dataset root.")
    p.add_argument("helper_files_dir", type=str, help="Directory with helper npy/npz/json files.")
    p.add_argument("logs_path", type=str, help="(Legacy) root under which to place this run.")  # ### CHANGED (kept for compatibility)

    # General toggles
    p.add_argument("--skip_test", type=boolean, default=True)
    p.add_argument("--df_test_path", type=str, default=None)
    p.add_argument("--try_harder", type=boolean, default=False)

    # Data/loader
    p.add_argument("--bin_width", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--devices", type=ast.literal_eval, default=[0])
    p.add_argument("--precision", type=str, default="bf16-mixed")
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--accelerator", type=str, default="gpu",
                   help='Lightning accelerator, e.g. "gpu", "cpu", or "auto".')

    # Model/optim
    p.add_argument("--layer_dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.0001)

    # Losses
    p.add_argument("--bitwise_loss", type=str, default=None, help='{"bce","fl", "hamm", None}')
    p.add_argument("--fpwise_loss", type=str, default=None, help='{"cossim","iou",None}')
    p.add_argument("--rankwise_loss", type=str, default=None, help='{"bienc","cross",None}')

    p.add_argument("--bitwise_lambd", type=float, default=1.0)
    p.add_argument("--fpwise_lambd", type=float, default=1.0)
    p.add_argument("--rankwise_lambd", type=float, default=1.0)

    p.add_argument("--bitwise_weighted", type=boolean, default=False)
    p.add_argument("--bitwise_fl_gamma", type=float, default=5.0)

    p.add_argument("--fpwise_iou_jml_v", type=boolean, default=False)

    p.add_argument("--rankwise_temp", type=float, default=1.0)
    p.add_argument("--rankwise_dropout", type=float, default=0.2)
    p.add_argument("--rankwise_sim_func", type=str, default="cossim")
    p.add_argument("--rankwise_projector", type=boolean, default=False)

    # Eval-time MC-dropout toggle (stored in ckpt; predictor may use it)
    p.add_argument("--mc_dropout_eval", type=boolean, default=False)

    # Repro
    p.add_argument("--seed", type=int, default=42)

    # ### NEW: explicit run directory + checkpoint behavior
    p.add_argument("--run_dir", type=str, default=None,
                   help="If set, use this as the run root (recommended). Else derive from logs_path.")
    p.add_argument("--save_top_k", type=int, default=1)
    p.add_argument("--save_last", type=boolean, default=True)
    p.add_argument("--max_epochs", type=int, default=None,
                   help="Override epoch count; useful for smoke tests.")
    p.add_argument("--limit_train_batches", type=int_or_float, default=None,
                   help="Lightning limit_train_batches; int for batches or float fraction.")
    p.add_argument("--limit_val_batches", type=int_or_float, default=None,
                   help="Lightning limit_val_batches; int for batches or float fraction.")
    p.add_argument("--limit_test_batches", type=int_or_float, default=None,
                   help="Lightning limit_test_batches; int for batches or float fraction.")
    p.add_argument("--num_sanity_val_steps", type=int, default=None,
                   help="Override Lightning sanity validation steps.")

    return p


def mk_checkpoint_cb(ckpt_root: Path, metric: str, folder: str,
                     save_top_k: int, save_last: bool) -> ModelCheckpoint:
    """Create a per-metric ModelCheckpoint writing into <ckpt_root>/<folder>/."""
    return ModelCheckpoint(
        dirpath=str(ckpt_root / folder),
        filename=f"{folder}" + "-{epoch:02d}-{step}",
        monitor=metric,
        mode="max",
        save_top_k=save_top_k,
        save_last=save_last,
        auto_insert_metric_name=False,
    )


def main():
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("is_available =", torch.cuda.is_available(), "count =", torch.cuda.device_count())
    
    parser = make_parser()
    args = parser.parse_args()

    # Reproducibility
    seed_everything(args.seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Resolve run directory  ### NEW
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.logs_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Logger (kept minimal; metrics go to TensorBoard under run_dir/tb/run)  ### CHANGED
    logger = TensorBoardLogger(save_dir=str(run_dir), name="tb", version="run")

    # Dataset + datamodule
    dataset = RetrievalDataset_PrecompFPandInchi(
        spec_transform=SpecBinner(max_mz=1005, bin_width=args.bin_width, to_rel_intensities=True),
        mol_transform=MolFingerprinter(fp_size=4096),
        pth=args.dataset_path,
        fp_pth=os.path.join(args.helper_files_dir, "fp_4096.npy"),
        inchi_pth=os.path.join(args.helper_files_dir, "inchis.npy"),
        candidates_pth=os.path.join(args.helper_files_dir, "MassSpecGym_retrieval_candidates_formula.json"),
        candidates_fp_pth=os.path.join(args.helper_files_dir, "MassSpecGym_retrieval_candidates_formula_fps.npz"),
        candidates_inchi_pth=os.path.join(args.helper_files_dir, "MassSpecGym_retrieval_candidates_formula_inchi.npz"),
    )

    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    data_module.prepare_data()
    data_module.setup()

    # Normalize "None" strings
    if args.bitwise_loss == "None":
        args.bitwise_loss = None
    if args.fpwise_loss == "None":
        args.fpwise_loss = None
    if args.rankwise_loss == "None":
        args.rankwise_loss = None

    loss_kwargs_dict = {
        "bce": {"weighted": args.bitwise_weighted},
        "fl": {"gamma": args.bitwise_fl_gamma, "weighted": args.bitwise_weighted},
        "hamm": {"weighted": args.bitwise_weighted},
        "cossim": {},
        "iou": {"jml_version": args.fpwise_iou_jml_v},
        "bienc": {
            "temp": args.rankwise_temp,
            "n_bits": 4096,
            "dropout": args.rankwise_dropout,
            "sim_func": args.rankwise_sim_func,
            "projector": args.rankwise_projector,
        },
        "cross": {
            "temp": args.rankwise_temp,
            "n_bits": 4096,
            "dropout": args.rankwise_dropout,
            "projector": args.rankwise_projector,
        },
        None: {},
    }

    model = FingerprintPredicter(
        n_in=int(1005 / args.bin_width),
        layer_dims=[args.layer_dim] * args.n_layers,
        n_bits=4096,
        layer_or_batchnorm="layer",
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=0,
        df_test_path=args.df_test_path,
        bitwise_loss=args.bitwise_loss,
        fpwise_loss=args.fpwise_loss,
        rankwise_loss=args.rankwise_loss,
        bitwise_lambd=args.bitwise_lambd,
        fpwise_lambd=args.fpwise_lambd,
        rankwise_lambd=args.rankwise_lambd,
        bitwise_kwargs=loss_kwargs_dict[args.bitwise_loss],
        fpwise_kwargs=loss_kwargs_dict[args.fpwise_loss],
        rankwise_kwargs=loss_kwargs_dict[args.rankwise_loss],
        # mc_dropout_eval flag is part of ckpt hyperparams for later inference if you wish
        mc_dropout_eval=args.mc_dropout_eval,
    )

    # Checkpoint layout under run_dir/ckpts/<metric>/...  ### NEW
    ckpt_root = run_dir / "ckpts"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    callbacks = []
    cbs = {}

    # Per-metric checkpoints
    cbs["cossim"] = mk_checkpoint_cb(
        ckpt_root, metric="val_cossim_hit_rate@20", folder="cossim",
        save_top_k=args.save_top_k, save_last=args.save_last
    )
    cbs["tanim"] = mk_checkpoint_cb(
        ckpt_root, metric="val_tanim_hit_rate@20", folder="tanim",
        save_top_k=args.save_top_k, save_last=args.save_last
    )
    cbs["contiou"] = mk_checkpoint_cb(
        ckpt_root, metric="val_contiou_hit_rate@20", folder="contiou",
        save_top_k=args.save_top_k, save_last=args.save_last
    )
    if args.rankwise_loss is not None:
        cbs["reranker"] = mk_checkpoint_cb(
            ckpt_root, metric="val_reranker_hit_rate@20", folder="reranker",
            save_top_k=args.save_top_k, save_last=args.save_last
        )

    callbacks.extend(cbs.values())

    max_epochs = args.max_epochs if args.max_epochs is not None else (250 if args.try_harder else 50)

    trainer_kwargs = {}
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    if args.limit_test_batches is not None:
        trainer_kwargs["limit_test_batches"] = args.limit_test_batches
    if args.num_sanity_val_steps is not None:
        trainer_kwargs["num_sanity_val_steps"] = args.num_sanity_val_steps

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="auto",
        gradient_clip_val=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
        plugins=[LightningEnvironment()],
        logger=logger,
        precision=args.precision,
        **trainer_kwargs,
    )

    # Pre-train validation
    trainer.validate(model, datamodule=data_module)

    # Train
    trainer.fit(model, datamodule=data_module)

    # Write manifest + ensure best.ckpt symlinks exist  ### NEW
    best = {name: cb.best_model_path for name, cb in cbs.items()}
    (run_dir / "best_ckpts.json").write_text(json.dumps(best, indent=2))

    for name, path in best.items():
        if not path:
            continue
        link = ckpt_root / name / "best.ckpt"
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(Path(path))
        except OSError:
            shutil.copy2(path, link)

    # Post-train: validate each best ckpt explicitly (no ambiguous ckpt_path="best")  ### NEW
    for name, path in best.items():
        if path:
            print(f"[validate] best '{name}': {path}")
            trainer.validate(model=None, datamodule=data_module, ckpt_path=path)

    # Optional test: run on each best ckpt (can be time-consuming)      ### NEW
    if not args.skip_test:
        for name, path in best.items():
            if path:
                print(f"[test] best '{name}': {path}")
                trainer.test(model=None, datamodule=data_module, ckpt_path=path)


if __name__ == "__main__":
    main()

