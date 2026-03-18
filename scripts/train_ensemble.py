#!/usr/bin/env python3
import argparse
import ast
import json
import os
import sys
import subprocess
import datetime as _dt
from pathlib import Path
import shutil
from typing import Iterable, List


def boolean(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("yes", "true", "t", "y", "1"): return True
    if v in ("no", "false", "f", "n", "0"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M")


def _train_py_path() -> str:
    env = os.environ.get("MSUQ_TRAIN_PY")
    if env:
        return env
    here = Path(__file__).resolve().parent
    cand = here / "train.py"
    if cand.exists():
        return str(cand)
    return "/data/home/mira/MS-UQ/ms_uq/train.py"


def _run_train(argv, env=None, tee_path: Path | None = None):
    cmd = [sys.executable, _train_py_path()] + argv
    print(f"[launcher] exec: {' '.join(cmd)}")
    if tee_path is None:
        subprocess.run(cmd, check=True, env=env)
    else:
        tee_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tee_path, "w") as logf:
            p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            p.wait()
            return p.returncode


def _finalize_best_by_metric(exp_root: Path, member_dirs: list[Path]) -> None:
    """Create best_by_metric/<metric>/model_*.ckpt symlinks (or copies)."""
    if not member_dirs:
        return
    first = member_dirs[0] / "ckpts"
    if not first.exists():
        return
    metrics = [p.name for p in first.iterdir() if p.is_dir()]
    out_root = exp_root / "best_by_metric"
    out_root.mkdir(parents=True, exist_ok=True)
    for m in metrics:
        dst_m = out_root / m
        dst_m.mkdir(parents=True, exist_ok=True)
        for i, md in enumerate(member_dirs):
            best = md / "ckpts" / m / "best.ckpt"
            if not best.exists():
                continue
            link = dst_m / f"model_{i:03d}.ckpt"
            try:
                if link.exists() or link.is_symlink(): link.unlink()
                link.symlink_to(best)
            except OSError:
                shutil.copy2(best, link)


def _normalize_devices(argval) -> List[int]:
    """
    Accepts values like:
      - 0                      (int)
      - "0"                    (str)
      - "[0,1,2]"              (str list literal)
      - "0,1,2"                (csv string)
      - [0,1,2]                (python list)
    Returns a list of unique ints, preserving order.
    """
    if isinstance(argval, list):
        devs = argval
    elif isinstance(argval, int):
        devs = [argval]
    else:
        s = str(argval).strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, int):
                devs = [parsed]
            elif isinstance(parsed, (list, tuple)):
                devs = list(parsed)
            else:
                # fallback to csv
                devs = [int(x) for x in s.split(",") if x.strip() != ""]
        except Exception:
            devs = [int(x) for x in s.split(",") if x.strip() != ""]
    # sanitize/unique while preserving order
    out, seen = [], set()
    for d in devs:
        di = int(d)
        if di not in seen:
            out.append(di); seen.add(di)
    if not out:
        raise ValueError("No GPU indices parsed from --devices.")
    return out


# MAIN LAUNCHER
def main():
    class Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter): pass
    p = argparse.ArgumentParser("Experiment launcher", formatter_class=Fmt)

    # Required
    p.add_argument("dataset_path", type=str)
    p.add_argument("helper_files_dir", type=str)
    p.add_argument("logs_root", type=str)

    # Method & grouping (not forwarded)
    p.add_argument("--method", type=str, default="ensemble",
                   choices=["ensemble", "mc_dropout", "single"])
    p.add_argument("--n_members", type=int, default=5, help="Used for ensemble.")
    p.add_argument("--tag", type=str, default=None)

    # Common training flags (forwarded)
    p.add_argument("--skip_test", type=boolean, default=True)
    p.add_argument("--df_test_path", type=str, default=None)
    p.add_argument("--try_harder", type=boolean, default=False)

    p.add_argument("--bin_width", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--devices", type=str, default="[0]")     # <— ADD type=str
    p.add_argument("--precision", type=str, default="bf16-mixed")

    p.add_argument("--layer_dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--bitwise_loss", type=str, default=None)
    p.add_argument("--fpwise_loss", type=str, default=None)
    p.add_argument("--rankwise_loss", type=str, default=None)

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

    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--mc_dropout_eval", type=boolean, default=False,
                   help="Store mc_dropout_eval=True in ckpt (used by mc_dropout/single).")
    
    p.add_argument("--save_top_k", type=int, default=1,
                   help="Number of best checkpoints to keep per metric.")
    p.add_argument("--save_last", type=boolean, default=True,
                   help="Whether to save the last checkpoint.")
    p.add_argument("--n_workers", type=int, default=4,
                   help="Number of dataloader workers.")
    
    p.add_argument("--max_parallel", type=int, default=None,
                   help="Max concurrent members. Default = len(--devices) for ensemble.")

    args = p.parse_args()

    # Normalize paths and devices
    dataset_path = os.path.abspath(args.dataset_path)
    helper_dir   = os.path.abspath(args.helper_files_dir)
    logs_root    = Path(os.path.abspath(args.logs_root))
    devices      = _normalize_devices(args.devices)     # <— robust parsing

    print(f"[launcher] parsed devices = {devices}")

    # Build experiment root: <logs_root>/<method>_<timestamp>[_tag]
    tag = f"_{args.tag}" if args.tag else ""
    exp_root = logs_root / f"{args.method}_{_ts()}{tag}"
    exp_root.mkdir(parents=True, exist_ok=True)

    # Record a minimal manifest
    manifest = {
        "method": args.method,
        "n_members": args.n_members if args.method == "ensemble" else 1,
        "base_seed": args.base_seed,
        "save_top_k": args.save_top_k,                        # ADD
        "save_last": args.save_last,  
        "args_forwarded": {
            "bin_width": args.bin_width,
            "batch_size": args.batch_size,
            "n_workers": args.n_workers,    
            "precision": args.precision,
            "layer_dim": args.layer_dim, "n_layers": args.n_layers, "dropout": args.dropout, "lr": args.lr,
            "bitwise_loss": args.bitwise_loss, "fpwise_loss": args.fpwise_loss, "rankwise_loss": args.rankwise_loss,
            "bitwise_lambd": args.bitwise_lambd, "fpwise_lambd": args.fpwise_lambd, "rankwise_lambd": args.rankwise_lambd,
            "bitwise_weighted": args.bitwise_weighted, "bitwise_fl_gamma": args.bitwise_fl_gamma,
            "fpwise_iou_jml_v": args.fpwise_iou_jml_v,
            "rankwise_temp": args.rankwise_temp, "rankwise_dropout": args.rankwise_dropout,
            "rankwise_sim_func": args.rankwise_sim_func, "rankwise_projector": args.rankwise_projector,
        },
    }
    (exp_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.method == "ensemble":
        max_parallel = args.max_parallel or len(devices)
        print(f"[launcher] max_parallel = {max_parallel}")

        member_dirs: list[Path] = []
        procs: list[tuple[subprocess.Popen, Path]] = []

        for i in range(args.n_members):
            seed = args.base_seed + i
            member_dir = exp_root / "members" / f"member_{i:03d}"
            member_dir.mkdir(parents=True, exist_ok=True)

            # Choose a PHYSICAL GPU for this member (global index)
            phys_gpu = devices[i % len(devices)]

            # Inside the child, we mask to that single physical GPU so it appears as local cuda:0
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = str(phys_gpu)

            argv = [
                dataset_path, helper_dir, str(member_dir),
                f"--skip_test={args.skip_test}",
                f"--df_test_path={args.df_test_path}",
                f"--try_harder={args.try_harder}",
                f"--bin_width={args.bin_width}",
                f"--batch_size={args.batch_size}",
                f"--devices={[0]}",
                f"--precision={args.precision}",
                f"--n_workers={args.n_workers}",              # ADD
                f"--layer_dim={args.layer_dim}",
                f"--n_layers={args.n_layers}",
                f"--dropout={args.dropout}",
                f"--lr={args.lr}",
                f"--bitwise_loss={args.bitwise_loss}",
                f"--fpwise_loss={args.fpwise_loss}",
                f"--rankwise_loss={args.rankwise_loss}",
                f"--bitwise_lambd={args.bitwise_lambd}",
                f"--fpwise_lambd={args.fpwise_lambd}",
                f"--rankwise_lambd={args.rankwise_lambd}",
                f"--bitwise_weighted={args.bitwise_weighted}",
                f"--bitwise_fl_gamma={args.bitwise_fl_gamma}",
                f"--fpwise_iou_jml_v={args.fpwise_iou_jml_v}",
                f"--rankwise_temp={args.rankwise_temp}",
                f"--rankwise_dropout={args.rankwise_dropout}",
                f"--rankwise_sim_func={args.rankwise_sim_func}",
                f"--rankwise_projector={args.rankwise_projector}",
                f"--seed={seed}",
                f"--run_dir={str(member_dir)}",
                f"--save_top_k={args.save_top_k}",            # ADD
                f"--save_last={args.save_last}",              # ADD
            ]

            log_path = member_dir / "train.out"
            cmd = [sys.executable, _train_py_path()] + argv
            print(f"[launcher] member {i} → PHYS_GPU={phys_gpu} (masked as cuda:0) :: {' '.join(cmd)}")
            p = subprocess.Popen(cmd, env=child_env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
            procs.append((p, log_path))
            member_dirs.append(member_dir)

            # throttle to max_parallel
            while len([q for q, _ in procs if q.poll() is None]) >= max_parallel:
                for q, _lp in list(procs):
                    if q.poll() is not None:
                        procs.remove((q, _lp))

        # wait for the rest
        for q, _lp in procs:
            q.wait()

        _finalize_best_by_metric(exp_root, member_dirs)
        return

   
    model_dir = exp_root / "single" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Pick a physical GPU to run this single job on
    phys_gpu = devices[0]
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = str(phys_gpu)  

    argv = [
        dataset_path, helper_dir, str(model_dir),
        f"--skip_test={args.skip_test}",
        f"--df_test_path={args.df_test_path}",
        f"--try_harder={args.try_harder}",
        f"--bin_width={args.bin_width}",
        f"--batch_size={args.batch_size}",
        f"--devices={[0]}",                              # <— matches the masked view
        f"--precision={args.precision}",
        f"--layer_dim={args.layer_dim}",
        f"--n_layers={args.n_layers}",
        f"--dropout={args.dropout}",
        f"--lr={args.lr}",
        f"--bitwise_loss={args.bitwise_loss}",
        f"--fpwise_loss={args.fpwise_loss}",
        f"--rankwise_loss={args.rankwise_loss}",
        f"--bitwise_lambd={args.bitwise_lambd}",
        f"--fpwise_lambd={args.fpwise_lambd}",
        f"--rankwise_lambd={args.rankwise_lambd}",
        f"--bitwise_weighted={args.bitwise_weighted}",
        f"--bitwise_fl_gamma={args.bitwise_fl_gamma}",
        f"--fpwise_iou_jml_v={args.fpwise_iou_jml_v}",
        f"--rankwise_temp={args.rankwise_temp}",
        f"--rankwise_dropout={args.rankwise_dropout}",
        f"--rankwise_sim_func={args.rankwise_sim_func}",
        f"--rankwise_projector={args.rankwise_projector}",
        f"--seed={args.base_seed}",
        f"--run_dir={str(model_dir)}",
        f"--save_top_k={args.save_top_k}",                    # ADD
        f"--save_last={args.save_last}",                      # ADD
        f"--n_workers={args.n_workers}", 
    ]
    if args.method == "mc_dropout":
        argv.append(f"--mc_dropout_eval={True}")

    print(f"[launcher] single/mc → PHYS_GPU={phys_gpu} (masked as cuda:0)")
    _run_train(argv, env=child_env, tee_path=model_dir / "train.out")


if __name__ == "__main__":
    main()

