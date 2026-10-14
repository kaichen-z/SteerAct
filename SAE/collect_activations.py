from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

if __package__:
    from .evaluate import PIPELINE, TASKS, WORKSPACE, configure
else:
    from evaluate import PIPELINE, TASKS, WORKSPACE, configure


BACKEND_TASKS = (
    "put_bottles_dustbin", "stack_blocks_three", "stack_bowls_three",
    "blocks_ranking_size", "place_dual_shoes", "dump_bin_bigbin",
    "open_microwave", "turn_switch", "place_mouse_pad", "pick_diverse_bottles",
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect frozen-policy activations for RoboTwin SAE training.\n\nThis is a thin driver around the existing SAE pipeline collector and merger.\nIt keeps the source rollouts separate from held-out evaluation seeds.\n')
    parser.add_argument("--backbone", required=True, choices=("pi05", "fastwam"))
    parser.add_argument("--stage", choices=("collect", "merge", "all"), default="all")
    parser.add_argument("--layer", help="default: paligemma_L14 for pi05, fastwam_video_L14 for Fast-WAM")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--setting", choices=("demo_clean", "demo_randomized"), default="demo_clean")
    parser.add_argument("--episodes", type=int, default=24, help="valid source episodes per task")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--gpu-ids", default="0", help="comma-separated physical GPU IDs")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--worker-python", type=Path, default=WORKSPACE / "MemGen/Olympus/environments/isaac50/.venv/bin/python")
    parser.add_argument("--out-dir", type=Path, required=True, help="root for merged activations/")
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--norm-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or args.workers_per_gpu <= 0:
        parser.error("episodes and workers-per-gpu must be positive")
    args.layer = args.layer or ("paligemma_L14" if args.backbone == "pi05" else "fastwam_video_L14")
    prefix = "paligemma_L" if args.backbone == "pi05" else "fastwam_video_L"
    if not args.layer.startswith(prefix):
        parser.error(f"this encoder-side recipe expects a layer named {prefix}<index>")
    args.out_dir = args.out_dir.expanduser().resolve()
    args.worker_python = args.worker_python.expanduser().absolute()
    for field in ("ckpt", "stats", "norm_dir"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    if args.stage in ("collect", "all") and not args.worker_python.is_file():
        raise FileNotFoundError(args.worker_python)
    module = configure(args)
    task_ids = [BACKEND_TASKS.index(task) for task in args.tasks]
    gpu_ids = [part.strip() for part in args.gpu_ids.split(",") if part.strip()]
    if not gpu_ids:
        parser.error("--gpu-ids must contain at least one GPU")
    act_dir = args.out_dir / "activations"

    batch_size = len(gpu_ids) * args.workers_per_gpu
    collect_commands = []
    for start in range(0, len(task_ids), batch_size):
        batch = task_ids[start:start + batch_size]
        collect_commands.append([
            sys.executable, str(PIPELINE / "scripts/collect_ft_activations.py"),
            "--tasks", ",".join(map(str, batch)), "--seeds-per-task", str(args.episodes),
            "--seed-start", str(args.seed_start), "--gpu-ids", args.gpu_ids,
            "--workers-per-gpu", str(args.workers_per_gpu), "--out-dir", str(act_dir),
            "--rollout-module", module, "--layers", args.layer,
            "--pi05-python", str(args.worker_python), "--ensure-valid-per-task",
        ])
    commands = {
        "collect": collect_commands,
        "merge": [[sys.executable, str(PIPELINE / "scripts/_merge_ft_activations.py"),
                  "--in-dir", str(act_dir), "--layers", args.layer]],
    }
    stages = ("collect", "merge") if args.stage == "all" else (args.stage,)
    environment = os.environ.copy()
    environment["PI05_VENV_PYTHON"] = str(args.worker_python)
    for stage in stages:
        for command in commands[stage]:
            print(f"[{stage}] {' '.join(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, env=environment, check=True)
        if not args.dry_run and stage == "merge" and not (act_dir / f"{args.layer}.pt").is_file():
            raise RuntimeError(f"merge did not produce {act_dir / f'{args.layer}.pt'}")
    print(f"merged activations: {act_dir / f'{args.layer}.pt'}")


if __name__ == "__main__":
    main()
