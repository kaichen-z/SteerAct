from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(os.environ.get("STEERACT_WORKSPACE", Path(__file__).resolve().parents[3]))
ROBOTWIN = Path(os.environ["STEERACT_ROBOTWIN_ROOT"])
FASTWAM = Path(os.environ["STEERACT_FASTWAM_ROOT"])
for path in (
    ROBOTWIN,
    ROBOTWIN / "scripts",
    FASTWAM,
    FASTWAM / "src",
    FASTWAM / "experiments" / "robotwin",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.chdir(ROBOTWIN)

import torch

_policy = None

from ..config import TASKS
from ..runtime import apply_step_limit

ROBOTWIN_TASKS = set(TASKS)
ROBOTWIN_TASK_ORDER = list(TASKS)
COAST_ROBOTWIN_TASK_ORDER = list(TASKS)


def _load_policy():
    global _policy
    if _policy is not None:
        return _policy
    from fastwam_policy.deploy_policy import get_model

    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
    os.environ.setdefault(
        "DIFFSYNTH_MODEL_BASE_PATH",
        os.environ.get("FASTWAM_MODEL_BASE_PATH", str(ROOT / "checkpoints" / "robotwin_fastwam")),
    )
    asset_root = Path(os.environ["DIFFSYNTH_MODEL_BASE_PATH"])
    os.environ.setdefault("DIFFSYNTH_TEXT_ENCODER_PATH", str(asset_root / "models_t5_umt5-xxl-enc-bf16.safetensors"))
    os.environ.setdefault("DIFFSYNTH_VAE_PATH", str(asset_root / "Wan2.2_VAE.safetensors"))
    os.environ.setdefault("DIFFSYNTH_TOKENIZER_PATH", str(asset_root / "google" / "umt5-xxl"))

    _policy = get_model({
        "sim_cfg_path": str(FASTWAM / "configs" / "sim_robotwin.yaml"),
        "sim_task": os.environ.get(
            "FASTWAM_SIM_TASK", "robotwin_uncond_3cam_384_1e-4"
        ),
        "ckpt_setting": os.environ.get(
            "FASTWAM_CKPT",
            str(ROOT / "checkpoints" / "fastwam_release" / "robotwin_uncond_3cam_384.pt"),
        ),
        "dataset_stats_path": os.environ.get(
            "FASTWAM_DATASET_STATS",
            str(ROOT / "checkpoints" / "fastwam_release" /
                "robotwin_uncond_3cam_384_dataset_stats.json"),
        ),
        "device": "cuda",
        "mixed_precision": os.environ.get("FASTWAM_MIXED_PRECISION", "bf16"),
        "replan_steps": int(os.environ.get("FASTWAM_REPLAN_STEPS", "24")),
        "num_inference_steps": int(os.environ.get("FASTWAM_INFERENCE_STEPS", "10")),
        "rand_device": "cpu",
        "tiled": False,
        "timing_enabled": True,
    })
    _policy.model.eval().requires_grad_(False)
    return _policy


def validate_collection_layers():
    model = _load_policy().model
    for expert in ('video', 'action'):
        module = getattr(model, f'{expert}_expert', None)
        if module is None:
            module = model.mot.mixtures[expert]
        if len(module.blocks) != 30:
            raise ValueError(f'fixed collection configuration requires 30 {expert} expert layers')


def _find_module(model: torch.nn.Module, layer: str) -> torch.nn.Module:
    aliases = {
        f"fastwam_action_L{i}": (
            f"action_expert.blocks.{i}", f"mot.mixtures.action.blocks.{i}"
        ) for i in range(30)
    }
    aliases.update({
        f"fastwam_video_L{i}": (
            f"video_expert.blocks.{i}", f"mot.mixtures.video.blocks.{i}"
        ) for i in range(30)
    })
    aliases.update({
        "fastwam_action_encoder": ("action_encoder", "mot.action_encoder"),
        "fastwam_proprio_encoder": ("proprio_encoder",),
    })
    modules = dict(model.named_modules())
    candidates = aliases.get(layer, (layer,))
    for name in candidates:
        if name in modules:
            return modules[name]
    suffix_matches = [module for name, module in modules.items()
                      if any(name.endswith(candidate) for candidate in candidates)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise ValueError(
        f"Fast-WAM layer {layer!r} not found; tried {candidates}. "
        f"Available block roots: {[n for n in modules if '.blocks.14' in n][:8]}"
    )


def _tensor_hook(raw):
    def wrapped(module, inputs, output):
        if isinstance(output, tuple):
            replacement = raw(module, inputs, output[0])
            return (replacement,) + output[1:]
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
            order = (COAST_ROBOTWIN_TASK_ORDER if os.environ.get("FASTWAM_COAST_TASKS") == "1"
                     else ROBOTWIN_TASK_ORDER)
            task_name = order[int(task_id)]
        except (IndexError, ValueError):
            raise ValueError(f"Unknown RoboTwin task id: {task_id}") from None
    if task_name is None:
        raise ValueError("Fast-WAM backend requires task_name or task_id")
    if hooks is None:
        if layer is None or hook is None:
            raise ValueError("pass hooks=[...] or (layer, hook)")
        hooks = [(layer, hook)]

    from eval_policy_xpolicylab import class_decorator, load_task_args, is_episode_end

    policy = _load_policy()
    import random
    random.seed(int(env_seed))
    np.random.seed(int(env_seed))
    torch.manual_seed(int(env_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(env_seed))
    target_hooks = {}
    forward_handles = []
    for name, fn in hooks:
        mod = _find_module(policy.model, name)
        target_hooks[id(mod)] = _tensor_hook(fn)
        if not name.startswith(('fastwam_video_L', 'fastwam_action_L')) and 'blocks.' not in name:
            forward_handles.append(mod.register_forward_hook(target_hooks[id(mod)]))
    mot = getattr(policy.model, "mot", None)
    mot_cls = type(mot) if mot is not None else None
    original_post_block = getattr(mot_cls, "_apply_expert_post_block_tensor", None) if mot_cls else None
    if target_hooks and mot is not None and original_post_block is not None:
        def capture_post_block(*, block, **kwargs):
            result = original_post_block(block=block, **kwargs)
            fn = target_hooks.get(id(block))
            if fn is not None:
                result = fn(block, (), result)
            return result
        mot_cls._apply_expert_post_block_tensor = staticmethod(capture_post_block)
    task_env = None
    video_writer = None
    video_path = os.environ.get("FASTWAM_VIDEO_PATH")
    video_frames = 0
    steps = 0
    success = False
    invalid = False
    decision_inference_ms = []
    try:
        usr = {
            "task_name": task_name,
            "policy_name": "fastwam_policy",
            "task_config": os.environ.get("FASTWAM_TASK_CONFIG", "demo_clean"),
            "seed": int(env_seed),
            "env_cfg_type": os.environ.get("ROBOTWIN_ENV_CFG_TYPE", "arx_x5"),
            "eval_mode": True,
            "instruction_type": "unseen",
            "expert_check": False,
        }
        args, _ = load_task_args(usr)
        args["env_cfg_type"] = usr["env_cfg_type"]
        args["eval_mode"] = True
        for _side in ("left_embodiment_config", "right_embodiment_config"):
            if isinstance(args.get(_side), dict):
                args[_side]["planner"] = "mplib_RRT"
        task_env = class_decorator(task_name)
        task_env.setup_demo(now_ep_num=0, seed=int(env_seed), is_test=True, **args)
        max_steps = apply_step_limit(task_env)
        task_env.set_instruction(instruction=task_name)
        policy.reset()
        if video_path:
            import imageio.v2 as imageio
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(video_path, fps=10, codec="libx264")

        while not is_episode_end(task_env):
            observation = task_env.get_obs() if policy.should_request_observation() else None
            infer_before = float(policy.get_timing_rollout().get("infer_s", 0.0))
            requested = observation is not None
            if requested:
                for _, fn in hooks:
                    fn.begin_decision()
            policy.step(task_env, observation)
            if requested:
                for _, fn in hooks:
                    fn.end_decision()
            if video_writer is not None:
                frame_obs = task_env.get_obs()
                frame = np.asarray(
                    frame_obs["observation"]["head_camera"]["rgb"]
                )
                if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
                    raise ValueError(f"unexpected RoboTwin video frame shape: {frame.shape}")
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                if frame.dtype != np.uint8:
                    scale = 255.0 if np.issubdtype(frame.dtype, np.floating) and float(np.max(frame)) <= 1.0 else 1.0
                    frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
                video_writer.append_data(frame)
                video_frames += 1
            infer_after = float(policy.get_timing_rollout().get("infer_s", 0.0))
            if infer_after > infer_before:
                decision_inference_ms.append((infer_after - infer_before) * 1000.0)
            steps += 1
            if bool(getattr(task_env, "eval_success", False)):
                break
        success = bool(getattr(task_env, "eval_success", False))
    except Exception:
        invalid = True
        raise
    finally:
        if video_writer is not None:
            video_writer.close()
        for handle in forward_handles:
            handle.remove()
        if target_hooks and mot is not None and original_post_block is not None:
            mot_cls._apply_expert_post_block_tensor = staticmethod(original_post_block)
        if task_env is not None:
            try:
                task_env.close_env(clear_cache=True)
            except Exception:
                pass
    if video_path:
        output_video = Path(video_path)
        if video_frames <= 0 or not output_video.is_file() or output_video.stat().st_size <= 0:
            raise RuntimeError(
                f"RoboTwin video capture produced no usable output: "
                f"path={output_video}, frames={video_frames}"
            )
    return {
        "success": success,
        "episode_length": steps,
        "n_steps": steps,
        "max_steps": max_steps,
        "task_name": task_name,
        "invalid_rollout_flag": invalid or steps <= 0,
        "decision_inference_ms": decision_inference_ms,
        "num_decisions": len(decision_inference_ms),
        "mean_decision_inference_ms": (float(np.mean(decision_inference_ms))
                                        if decision_inference_ms else None),
        "p50_decision_inference_ms": (float(np.percentile(decision_inference_ms, 50))
                                      if decision_inference_ms else None),
        "p95_decision_inference_ms": (float(np.percentile(decision_inference_ms, 95))
                                      if decision_inference_ms else None),
        "fastwam_timing": policy.get_timing_rollout(),
        "video_path": str(Path(video_path).resolve()) if video_path else None,
        "num_video_frames": video_frames,
    }
