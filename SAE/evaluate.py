from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
PIPELINE = WORKSPACE / "SAE" / "sae_pipeline"
TASKS = (
    "put_bottles_dustbin",
    "stack_blocks_three",
    "blocks_ranking_size",
    "place_dual_shoes",
    "dump_bin_bigbin",
    "open_microwave",
    "turn_switch",
    "place_mouse_pad",
    "pick_diverse_bottles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='RoboTwin 2.0 SAE steering evaluations for pi0.5/Fast-WAM.\n\nUses the existing workspace rollout backends. One JSONL row\nis written per attempted episode; valid episodes count toward the requested\nevaluation budget, so interrupted runs can be resumed with the same output.\n')
    parser.add_argument("--backbone", required=True, choices=("pi05", "fastwam"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--setting", choices=("demo_clean", "demo_randomized"), default="demo_clean")
    parser.add_argument("--episodes", type=int, default=200, help="valid episodes per task")
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--max-attempts", type=int, default=1000, help="attempts per task, including invalid episodes")
    parser.add_argument("--out", required=True, type=Path, help="append-only JSONL output")
    parser.add_argument("--ckpt", type=Path, help="policy checkpoint; defaults to the local release checkpoint")
    parser.add_argument("--stats", type=Path, help="Fast-WAM dataset_stats.json override")
    parser.add_argument("--norm-dir", type=Path, help="pi0.5 normalization assets override")
    parser.add_argument("--d5-bank", type=Path, help="task D5 banks from build_d5_bank.py")
    parser.add_argument("--sae-checkpoint", type=Path, help="TopK SAE checkpoint for the selected layer")
    return parser.parse_args()


def configure(args: argparse.Namespace) -> str:
    if not PIPELINE.is_dir():
        raise FileNotFoundError(f"missing rollout backend directory: {PIPELINE}")
    sys.path.insert(0, str(PIPELINE))
    if args.backbone == "pi05":
        checkpoint = args.ckpt or WORKSPACE / "checkpoints/robotwin_pi05/35000"
        norm_dir = args.norm_dir or checkpoint / "assets/Robotwin_50tasks_lerobot_data_clean"
        if not checkpoint.exists() or not norm_dir.exists():
            raise FileNotFoundError(f"pi0.5 checkpoint or norm directory missing: {checkpoint}, {norm_dir}")
        os.environ.update({
            "ROBOTWIN_PI05_CKPT": str(checkpoint),
            "ROBOTWIN_PI05_NORM_DIR": str(norm_dir),
            "ROBOTWIN_PI05_CONFIG": "pi05_base_aloha_full_sim_arx-x5_seed_0",
            "ROBOTWIN_PI05_ACTION_HORIZON": "50",
            "ROBOTWIN_PI05_DELTA_ACTIONS": "1",
            "ROBOTWIN_ENV_CFG_TYPE": "aloha_agilex",
            "ROBOTWIN_TASK_CONFIG": args.setting,
            "ROBOTWIN_MAX_STEPS": "1700",
            "ROBOTWIN_OFFICIAL_PROTOCOL": "0",
            "ROBOTWIN_PROMPT_MODE": "description",
        })
        return "team_rollout_robotwin"

    checkpoint = args.ckpt or WORKSPACE / "checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
    stats = args.stats or WORKSPACE / "checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
    if not checkpoint.is_file() or not stats.is_file():
        raise FileNotFoundError(f"Fast-WAM checkpoint or statistics missing: {checkpoint}, {stats}")
    os.environ.update({
        "FASTWAM_CKPT": str(checkpoint),
        "FASTWAM_DATASET_STATS": str(stats),
        "FASTWAM_TASK_CONFIG": args.setting,
        "ROBOTWIN_ENV_CFG_TYPE": "arx_x5",
    })
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
    os.environ.setdefault("DIFFSYNTH_DIT_PATH", str(WORKSPACE / "FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B"))
    os.environ.setdefault("DIFFSYNTH_VAE_PATH", str(WORKSPACE / "checkpoints/robotwin_fastwam/Wan2.2_VAE.safetensors"))
    os.environ.setdefault("DIFFSYNTH_TEXT_ENCODER_PATH", str(WORKSPACE / "checkpoints/robotwin_fastwam/models_t5_umt5-xxl-enc-bf16.safetensors"))
    os.environ.setdefault("DIFFSYNTH_TOKENIZER_PATH", str(WORKSPACE / "checkpoints/robotwin_fastwam/google/umt5-xxl"))
    return "team_rollout_fastwam"


def summarize_hook(hook):
    trace = hook.controller.trace
    return {
        "calls": len(trace),
        "applied": sum(row["alpha"] != 0 for row in trace),
        "gate_on_decisions": sum(row["support_open"] and row["failure_open"] for row in trace),
        "matured_decisions": sum(row["support_open"] for row in trace),
        "amplify_decisions": sum(row["alpha"] > 0 for row in trace),
        "suppress_decisions": sum(row["alpha"] < 0 for row in trace),
        "sum_signed_alpha": sum(row["alpha"] for row in trace),
        "sum_abs_alpha": sum(abs(row["alpha"]) for row in trace),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_attempts < args.episodes:
        raise ValueError("require 0 < episodes <= max-attempts")
    if args.d5_bank is None or args.sae_checkpoint is None:
        raise ValueError("SteerAct requires --d5-bank and --sae-checkpoint")
    args.out = args.out.expanduser().resolve()
    for field in ("ckpt", "stats", "norm_dir", "d5_bank", "sae_checkpoint"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    configure(args)
    checkpoint = os.environ["ROBOTWIN_PI05_CKPT" if args.backbone == "pi05" else "FASTWAM_CKPT"]
    asset_path = os.environ["ROBOTWIN_PI05_NORM_DIR" if args.backbone == "pi05" else "FASTWAM_DATASET_STATS"]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from steeract.runtime import DecisionHook
    os.environ["STEERACT_ROBOTWIN_ROOT"] = str(WORKSPACE / "RoboTwin")
    os.environ["STEERACT_FASTWAM_ROOT"] = str(WORKSPACE / "FastWAM")
    os.environ["STEERACT_MAX_STEPS_MULTIPLIER"] = "2"
    backend = importlib.import_module(f"steeract.backends.{args.backbone}")
    bank = None
    sae = None
    if args.d5_bank is not None:
        import torch
        from core.sae_trainer import load_sae_checkpoint

        selection_path = args.d5_bank
        selection = json.loads(selection_path.read_text())
        if selection.get("backbone") != args.backbone:
            raise ValueError("steering artifact backbone does not match --backbone")
        expected_prefix = "paligemma_L" if args.backbone == "pi05" else "fastwam_video_L"
        if not str(selection.get("layer", "")).startswith(expected_prefix):
            raise ValueError(f"steering artifact layer must start with {expected_prefix}")
        if Path(selection.get("sae_checkpoint", "")).resolve() != args.sae_checkpoint:
            raise ValueError("steering artifact was built from another SAE checkpoint")
        if args.d5_bank:
            if selection.get("schema") != "steeract.robotwin.paper_bank.v1":
                raise ValueError("rebuild the bank with build_d5_bank.py: paper_bank.v1 required")
            bank = selection
        if not args.sae_checkpoint.is_file():
            raise FileNotFoundError(args.sae_checkpoint)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        sae = load_sae_checkpoint(str(args.sae_checkpoint), device=device)[0].eval()
    variant = "steeract_paper"
    from steeract.method import Settings, make_hook as make_steering_hook
    if __package__:
        from .collect_activations import BACKEND_TASKS
    else:
        from collect_activations import BACKEND_TASKS

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()] if args.out.exists() else []
    for row in existing:
        if (row.get("backbone") != args.backbone or row.get("setting") != args.setting
                or row.get("checkpoint") != checkpoint or row.get("asset_path") != asset_path
                or row.get("variant") != variant
                or row.get("d5_bank") != (str(args.d5_bank) if bank else None)
                or row.get("sae_checkpoint") != (str(args.sae_checkpoint) if sae else None)
                or row.get("max_steps_multiplier") != 2):
            raise ValueError(f"{args.out} contains another steering configuration")

    with args.out.open("a", buffering=1) as output, open(os.devnull, "w") as quiet:
        for task in args.tasks:
            hooks = []
            hook = None
            feature_id = None
            layer = None
            if bank:
                layer = str(bank["layer"])
                config = bank["groups"][f"D5__T{BACKEND_TASKS.index(task)}__{layer}"]
                settings = Settings(**bank["control_settings"])
                feature_id = int(config["feature_id"])
                source_seeds = set(config["source_seeds"])
                eval_seeds = range(args.seed_start, args.seed_start + args.max_attempts)
                if any(seed in eval_seeds for seed in source_seeds):
                    raise ValueError("evaluation seed range overlaps reference seeds")
                if not 0 <= feature_id < sae.decoder.weight.shape[1]:
                    raise ValueError(f"feature {feature_id} outside SAE dictionary")
            prior = [row for row in existing if row.get("task") == task]
            used_seeds = {int(row["seed"]) for row in prior}
            valid = sum(not row.get("invalid", True) for row in prior)
            success = sum(bool(row.get("success")) for row in prior if not row.get("invalid", True))
            consecutive_errors = 0
            for seed in range(args.seed_start, args.seed_start + args.max_attempts):
                if valid >= args.episodes:
                    break
                if seed in used_seeds:
                    continue
                started = time.monotonic()
                if bank:
                    hook = make_steering_hook(sae, config, settings=settings, seed=seed)
                    hooks = [(layer, DecisionHook(hook))]
                try:
                    with contextlib.redirect_stdout(quiet):
                        result = backend.run_pi05_with_hook(env_seed=seed, task_name=task, hooks=hooks)
                    if bank and result.get("num_decisions") is not None:
                        if len(hook.controller.history) != int(result["num_decisions"]):
                            raise ValueError("context hook cadence differs from robot decisions; adapt backend")
                    invalid = bool(result.get("invalid_rollout_flag", False))
                    error = None
                except Exception as exc:
                    result = {}
                    invalid = True
                    error = f"{type(exc).__name__}: {exc}"
                row = {
                    "backbone": args.backbone,
                    "setting": args.setting,
                    "checkpoint": checkpoint,
                    "asset_path": asset_path,
                    "variant": variant,
                    "d5_bank": str(args.d5_bank) if bank else None,
                    "sae_checkpoint": str(args.sae_checkpoint) if sae else None,
                    "layer": layer,
                    "feature_id": feature_id,
                    "max_steps_multiplier": 2,
                    "hook_calls": len(hook.controller.history),
                    "d5_state": summarize_hook(hook) if bank else None,
                    "task": task,
                    "seed": seed,
                    "success": bool(result.get("success", False)) if not invalid else False,
                    "invalid": invalid,
                    "steps": result.get("episode_length", result.get("n_steps")),
                    "mean_decision_inference_ms": result.get("mean_decision_inference_ms", result.get("mean_inference_ms")),
                    "seconds": round(time.monotonic() - started, 2),
                    "error": error,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                used_seeds.add(seed)
                if invalid:
                    consecutive_errors += 1
                else:
                    valid += 1
                    success += int(row["success"])
                    consecutive_errors = 0
                print(f"{args.backbone} {task}: {valid}/{args.episodes} valid; seed={seed} success={row['success']} invalid={invalid}", flush=True)
                if consecutive_errors >= 5:
                    raise RuntimeError(f"five consecutive invalid episodes for {task}; inspect {args.out}")
            if valid < args.episodes:
                raise RuntimeError(f"only {valid}/{args.episodes} valid episodes for {task}; inspect {args.out}")
            print(f"{args.backbone} {task}: {success}/{valid} = {100 * success / valid:.1f}%", flush=True)


if __name__ == "__main__":
    main()
