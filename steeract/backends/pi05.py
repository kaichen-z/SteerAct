from __future__ import annotations

import copy
import dataclasses
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(os.environ.get("STEERACT_WORKSPACE", Path(__file__).resolve().parents[3]))
ROBOTWIN = Path(os.environ["STEERACT_ROBOTWIN_ROOT"])
XPOLICY = ROBOTWIN / "XPolicyLab"
for p in (ROBOTWIN, ROBOTWIN / "scripts", XPOLICY, XPOLICY / "integrations"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.chdir(ROBOTWIN)

import torch


_policy = None
_model = None

from ..config import TASKS
from ..runtime import apply_step_limit

ROBOTWIN_TASKS = set(TASKS)
ROBOTWIN_TASK_ORDER = list(TASKS)
COAST_ROBOTWIN_TASK_ORDER = list(TASKS)


def _load_policy():
    global _policy, _model
    if _policy is not None:
        return _policy
    openpi_src = XPOLICY / "policy" / "Pi_05" / "openpi" / "src"
    sys.path.insert(0, str(openpi_src))
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config

    ckpt = Path(os.environ.get(
        "ROBOTWIN_PI05_CKPT",
        str(ROOT / "checkpoints/robotwin_pi05/35000"),
    ))
    norm_dir = Path(os.environ.get(
        "ROBOTWIN_PI05_NORM_DIR",
        str(ckpt / "assets/Robotwin_50tasks_lerobot_data_clean"),
    ))
    config_name = os.environ.get(
        "ROBOTWIN_PI05_CONFIG", "pi05_base_aloha_full_sim_arx-x5_seed_0"
    )
    cfg = config.get_config(config_name)
    horizon = os.environ.get("ROBOTWIN_PI05_ACTION_HORIZON")
    if horizon:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, action_horizon=int(horizon)))
    delta = os.environ.get("ROBOTWIN_PI05_DELTA_ACTIONS")
    if delta is not None and hasattr(cfg.data, "use_delta_joint_actions"):
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(
            cfg.data, use_delta_joint_actions=(delta.lower() in {"1", "true", "yes"})))
    norm_stats = normalize.load(norm_dir)
    old_compile = torch.compile
    torch.compile = lambda fn, *a, **kw: fn
    try:
        device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda"
        _policy = policy_config.create_trained_policy(
            cfg, str(ckpt), norm_stats=norm_stats, pytorch_device=device
        )
    finally:
        torch.compile = old_compile
    _model = _policy._model
    _model.eval().requires_grad_(False)
    return _policy


def validate_collection_layers():
    _load_policy()
    backbone = _model.paligemma_with_expert
    if (len(backbone.paligemma.model.language_model.layers) != 18
            or len(backbone.gemma_expert.model.layers) != 18):
        raise ValueError('fixed collection configuration requires 18 PaliGemma and 18 action expert layers')


def _find_module(layer: str):
    model = _model
    if layer.startswith("paligemma_L"):
        return model.paligemma_with_expert.paligemma.model.language_model.layers[
            int(layer.removeprefix("paligemma_L"))
        ]
    if layer.startswith("expert_L"):
        return model.paligemma_with_expert.gemma_expert.model.layers[
            int(layer.removeprefix("expert_L"))
        ]
    raise ValueError(f"Unsupported activation layer: {layer}")


def _tuple_hook(raw):
    def wrapped(module, inputs, output):
        if isinstance(output, tuple):
            return (raw(module, inputs, output[0]),) + output[1:]
        return raw(module, inputs, output)
    if hasattr(raw, "state"):
        wrapped.state = raw.state
    return wrapped


def run_pi05_with_hook(
    env_seed: int,
    hook=None,
    task_id: int | None = None,
    task_name: str | None = None,
    layer: str | None = None,
    *,
    hooks: list[tuple[str, Any]] | None = None,
    **_: Any,
) -> dict[str, Any]:
    if task_name is None and task_id is not None:
        try:
            order = (COAST_ROBOTWIN_TASK_ORDER if os.environ.get("ROBOTWIN_COAST_TASKS") == "1"
                     else ROBOTWIN_TASK_ORDER)
            task_name = order[int(task_id)]
        except (IndexError, ValueError):
            raise ValueError(f"Unknown RoboTwin task id: {task_id}") from None
    if task_name is None:
        raise ValueError("RoboTwin backend requires task_name or task_id")
    if hooks is None:
        if layer is None or hook is None:
            raise ValueError("pass hooks=[...] or (layer, hook)")
        hooks = [(layer, hook)]
    from eval_policy_xpolicylab import (
        class_decorator, load_task_args, normalize_action_chunk,
        xpolicylab_action_to_robotwin, is_episode_end, build_instruction,
    )
    from XPolicyLab.policy.Pi_05.model import encode_obs
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    policy = _load_policy()
    random.seed(int(env_seed))
    np.random.seed(int(env_seed))
    torch.manual_seed(int(env_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(env_seed))
    handles = [_find_module(name).register_forward_hook(_tuple_hook(fn)) for name, fn in hooks]
    task_env = None
    steps = 0
    success = False
    invalid = False
    termination_reason = "unknown"
    inference_latencies = []
    video_path = None
    video_dir = None
    try:
        usr = {
            "task_name": task_name, "policy_name": "Pi_05",
            "task_config": os.environ.get("ROBOTWIN_TASK_CONFIG", "demo_clean"),
            "ckpt_setting": "35000", "seed": int(env_seed),
            "env_cfg_type": os.environ.get("ROBOTWIN_ENV_CFG_TYPE", "arx_x5"),
            "eval_mode": True,
            "action_type": os.environ.get("ROBOTWIN_ACTION_TYPE", "joint"),
            "instruction_type": "seen", "expert_check": False,
        }
        args, _ = load_task_args(usr)
        args["env_cfg_type"] = usr["env_cfg_type"]
        args["eval_mode"] = True
        video_dir = os.environ.get("ROBOTWIN_VIDEO_DIR")
        if video_dir:
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            args["eval_video_save_dir"] = video_dir
        for _side in ("left_embodiment_config", "right_embodiment_config"):
            if isinstance(args.get(_side), dict):
                args[_side]["planner"] = "mplib_RRT"
        task_env = class_decorator(task_name)
        official_protocol = os.environ.get("ROBOTWIN_OFFICIAL_PROTOCOL") == "1"
        episode_info = {"info": {}}
        if official_protocol:
            task_env.setup_demo(now_ep_num=0, seed=int(env_seed), is_test=True, **args)
            episode_info = task_env.play_once()
            expert_success = bool(task_env.plan_success and task_env.check_success())
            task_env.close_env()
            if not expert_success:
                raise RuntimeError(f"expert_check_failed seed={env_seed}")
        task_env.setup_demo(now_ep_num=0, seed=int(env_seed), is_test=True, **args)
        eval_step_limit = os.environ.get("ROBOTWIN_EVAL_STEP_LIMIT")
        if eval_step_limit:
            task_env.step_lim = int(eval_step_limit)
        if video_dir:
            initial_obs = task_env.get_obs()
            frame = initial_obs["observation"]["head_camera"]["rgb"]
            height, width = np.asarray(frame).shape[:2]
            video_path = Path(video_dir) / "episode0.mp4"
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                 "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                 "-vcodec", "libx264", "-crf", "23", str(video_path)],
                stdin=subprocess.PIPE,
            )
            task_env._set_eval_video_ffmpeg(ffmpeg)
        if official_protocol:
            instruction = build_instruction(args, episode_info, args.get("eval_instruction"), 50)
        elif os.environ.get("ROBOTWIN_PROMPT_MODE") == "description":
            instruction_path = ROBOTWIN / "description" / "task_instruction" / f"{task_name}.json"
            instruction = json.loads(instruction_path.read_text())["full_description"].strip()
        else:
            instruction = task_name
        task_env.set_instruction(instruction=instruction)
        policy.reset()
        action_dim_info = get_robot_action_dim_info(args["env_cfg_type"])
        step_limit_setting = os.environ.get("ROBOTWIN_MAX_STEPS", "1200")
        max_steps = apply_step_limit(task_env, step_limit_setting)
        replan_steps = int(os.environ.get("ROBOTWIN_REPLAN_STEPS", "0"))
        if replan_steps < 0:
            raise ValueError("ROBOTWIN_REPLAN_STEPS must be nonnegative (0 executes the full chunk)")
        while not is_episode_end(task_env) and steps < max_steps:
            raw_obs = task_env.get_obs()
            obs = dict(raw_obs.get("observation", {}))
            obs["vision"] = obs
            obs["joint_action"] = raw_obs.get("joint_action", {})
            obs["endpose"] = raw_obs.get("endpose", {})
            obs["instruction"] = instruction
            ja = obs["joint_action"]
            obs["state"] = {
                "left_arm_joint_state": np.asarray(ja["left_arm"], dtype=np.float32),
                "right_arm_joint_state": np.asarray(ja["right_arm"], dtype=np.float32),
                "left_ee_joint_state": np.asarray([ja["left_gripper"]], dtype=np.float32),
                "right_ee_joint_state": np.asarray([ja["right_gripper"]], dtype=np.float32),
            }
            if os.environ.get("ROBOTWIN_DEBUG_ONCE") == "1" and steps == 0:
                print("[robotwin-debug] setting=%s egl=%s visible=%s" % (
                    usr["task_config"], os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                    os.environ.get("CUDA_VISIBLE_DEVICES")), flush=True)
                for key, val in obs.get("vision", {}).items():
                    if isinstance(val, dict):
                        if "rgb" in val:
                            arr = np.asarray(val["rgb"])
                            print(f"[robotwin-debug] image {key}.rgb shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()} mean={arr.mean():.3f}", flush=True)
                        else:
                            print(f"[robotwin-debug] group {key} keys={list(val)}", flush=True)
                        continue
                    arr = np.asarray(val)
                    if arr.dtype.kind not in "biufc":
                        continue
                    print(f"[robotwin-debug] image {key} shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()} mean={arr.mean():.3f}", flush=True)
                print("[robotwin-debug] raw_joint_state14", np.concatenate([
                    obs["state"]["left_arm_joint_state"], obs["state"]["left_ee_joint_state"],
                    obs["state"]["right_arm_joint_state"], obs["state"]["right_ee_joint_state"]]), flush=True)
            if not isinstance(obs, dict):
                raise TypeError(f"RoboTwin get_obs returned {type(obs).__name__}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _infer_t0 = time.perf_counter()
            try:
                policy_obs = encode_obs(obs, usr["action_type"], action_dim_info)
            except Exception as exc:
                raise RuntimeError(f"Pi05 observation encoding failed; keys={list(obs.keys())}") from exc
            if os.environ.get("ROBOTWIN_DEBUG_ONCE") == "1" and steps == 0:
                st = policy_obs.get("state") if isinstance(policy_obs, dict) else None
                if st is not None:
                    arr = np.asarray(st)
                    print(f"[robotwin-debug] encoded_state shape={arr.shape} values={arr.reshape(-1)}", flush=True)
            for _, fn in hooks:
                fn.begin_decision()
            actions = normalize_action_chunk(policy.infer(policy_obs)["actions"])
            for _, fn in hooks:
                fn.end_decision()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_latencies.append(time.perf_counter() - _infer_t0)
            if os.environ.get("ROBOTWIN_DEBUG_ONCE") == "1" and steps == 0:
                aa = np.asarray(actions)
                print(f"[robotwin-debug] actions shape={aa.shape} min={aa.min():.5f} max={aa.max():.5f} mean={aa.mean():.5f}", flush=True)
            if not actions:
                raise RuntimeError("RoboTwin pi05 returned an empty action chunk")
            for action in (actions[:replan_steps] if replan_steps else actions):
                if steps >= max_steps:
                    termination_reason = "max_steps"
                    break
                flat, action_type = xpolicylab_action_to_robotwin(
                    action, action_type=usr["action_type"], current_observation=obs
                )
                if os.environ.get("ROBOTWIN_DEBUG_ADAPTER") == "1" and steps == 0:
                    cur = np.concatenate([obs["state"]["left_arm_joint_state"], obs["state"]["right_arm_joint_state"], obs["state"]["left_ee_joint_state"], obs["state"]["right_ee_joint_state"]])
                    print(f"[robotwin-debug] adapter action_type={action_type} flat_shape={np.asarray(flat).shape} flat={np.asarray(flat)}", flush=True)
                    print(f"[robotwin-debug] adapter delta_vs_state={np.asarray(flat).reshape(-1)[:cur.size] - cur}", flush=True)
                task_env.take_action(flat, action_type=action_type)
                steps += 1
                if task_env.eval_success or is_episode_end(task_env):
                    termination_reason = "success" if task_env.eval_success else "environment_end"
                    break
                obs = task_env.get_obs()
            if termination_reason == "max_steps":
                break
        success = bool(getattr(task_env, "eval_success", False))
        if termination_reason == "unknown":
            termination_reason = "success" if success else ("max_steps" if steps >= max_steps else "environment_end")
    except Exception:
        invalid = True
        raise
    finally:
        for handle in handles:
            handle.remove()
        if task_env is not None:
            try:
                if video_dir and hasattr(task_env, "eval_video_ffmpeg"):
                    task_env._del_eval_video_ffmpeg()
                task_env.close_env(clear_cache=True)
            except Exception:
                pass
    return {
        "success": success,
        "episode_length": steps,
        "n_steps": steps,
        "task_name": task_name,
        "instruction": instruction,
        "invalid_rollout_flag": invalid or steps <= 0,
        "termination_reason": termination_reason,
        "inference_seconds": inference_latencies,
        "inference_ms": [x * 1000.0 for x in inference_latencies],
        "per_decision_ms": [x * 1000.0 for x in inference_latencies],
        "mean_per_decision_ms": float(np.mean(inference_latencies) * 1000.0) if inference_latencies else None,
        "mean_inference_seconds": float(np.mean(inference_latencies)) if inference_latencies else None,
        "mean_inference_ms": float(np.mean(inference_latencies) * 1000.0) if inference_latencies else None,
        "num_inference_calls": len(inference_latencies),
        "decision_count": len(inference_latencies),
        "num_decisions": len(inference_latencies),
        "execution_replan_steps": replan_steps,
        "max_steps": max_steps,
        "video_path": str(video_path) if video_path is not None else None,
    }
