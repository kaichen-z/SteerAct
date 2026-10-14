from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .collect_activations import BACKEND_TASKS
    from .evaluate import PIPELINE, TASKS
else:
    from collect_activations import BACKEND_TASKS
    from evaluate import PIPELINE, TASKS


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steeract.method import Settings, ReferenceBank, descriptor, energy_score


def episode_calls(sae, activations, start, counts, batch_size, device, features=None):
    n = sum(counts)
    selected = None if features is None else torch.as_tensor(features, device=device)
    width = len(features) if features is not None else sae.decoder.weight.shape[1]
    totals = torch.zeros((len(counts), width), device=device)
    finite_counts = torch.zeros(len(counts), device=device)
    boundaries = np.cumsum([0, *counts])
    with torch.inference_mode():
        for local in range(0, n, batch_size):
            end = min(local + batch_size, n)
            rows = activations[start + local:start + end].to(device, dtype=torch.float32)
            finite = torch.isfinite(rows).all(dim=1)
            if not bool(finite.any()):
                continue
            codes = sae.encode(rows[finite]).clamp_min(0)
            if selected is not None:
                codes = codes.index_select(1, selected)
            call_ids = torch.as_tensor(np.searchsorted(boundaries[1:], np.arange(local, end), side="right"), device=device)
            totals.index_add_(0, call_ids[finite], codes)
            finite_counts.index_add_(0, call_ids[finite], torch.ones(int(finite.sum()), device=device))
    if bool((finite_counts == 0).any()):
        raise ValueError("an episode contains a hook call with no finite activation rows")
    return (totals / finite_counts[:, None]).cpu().numpy()


def choose_d5(sequences, labels, candidates):
    best = None
    for column, feature_id in enumerate(candidates):
        positive = np.concatenate([sequence[:, column] for sequence in sequences])
        positive = positive[positive > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        vectors = np.stack([descriptor(sequence[:, column], scale)[0] for sequence in sequences])
        score = energy_score(vectors[labels], vectors[~labels])
        if best is None or score > best[0]:
            best = (score, int(feature_id), scale, vectors, column)
    if best is None:
        raise ValueError("empty candidate feature set")
    return best


def main():
    parser = argparse.ArgumentParser(description='Build task-specific D5 descriptor banks from unsteered RoboTwin activations.')
    parser.add_argument("--backbone", required=True, choices=("pi05", "fastwam"))
    parser.add_argument("--layer", required=True)
    parser.add_argument("--act-dir", required=True, type=Path)
    parser.add_argument("--sae-checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--feature-ids", type=int, nargs="+",
                        help="explicit candidate set C; default: entire SAE dictionary")
    parser.add_argument("--control-config", type=Path, help="JSON Settings overrides, saved with bank")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    settings = Settings(**(json.loads(args.control_config.read_text()) if args.control_config else {}))
    expected = "paligemma_L" if args.backbone == "pi05" else "fastwam_video_L"
    if not args.layer.startswith(expected):
        parser.error(f"layer must start with {expected}")
    args.act_dir = args.act_dir.expanduser().resolve()
    args.sae_checkpoint = args.sae_checkpoint.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    sys.path.insert(0, str(PIPELINE))
    from core.sae_trainer import load_sae_checkpoint

    episodes = torch.load(args.act_dir / "_episode_index.pt", map_location="cpu", weights_only=False)
    acts = torch.load(args.act_dir / f"{args.layer}.pt", map_location="cpu", weights_only=False, mmap=True)
    sae = load_sae_checkpoint(str(args.sae_checkpoint), device=args.device)[0].eval()
    if acts.ndim != 2 or acts.shape[1] != sae.decoder.weight.shape[0]:
        raise ValueError("activation dimension does not match SAE")

    indexed, offset = [], 0
    requested = {BACKEND_TASKS.index(task) for task in args.tasks}
    for ep in episodes:
        count = ep["n_rows"]
        count = int(count.get(args.layer, 0) if isinstance(count, dict) else count)
        calls = ep.get("captured_rows_per_hook_call", {}).get(args.layer, [])
        if count < 0 or offset + count > len(acts):
            raise ValueError("episode index is misaligned with activation tensor")
        if int(ep["task"]) in requested and not ep.get("invalid_rollout_flag", False) and count:
            if not calls or any(int(c) <= 0 for c in calls) or sum(calls) != count:
                raise ValueError(f"task={ep['task']} seed={ep['seed']}: missing or invalid hook-call boundaries")
            indexed.append((ep, offset, [int(c) for c in calls]))
        offset += count
    if offset != len(acts):
        raise ValueError("episode index does not cover the activation tensor")

    width = sae.decoder.weight.shape[1]
    candidates = sorted(set(args.feature_ids)) if args.feature_ids is not None else list(range(width))
    if not candidates or min(candidates) < 0 or max(candidates) >= width:
        raise ValueError("candidate feature IDs outside SAE dictionary")
    grouped = {task_id: [] for task_id in requested}
    for ep, start, calls in indexed:
        grouped[int(ep["task"])].append((ep, start, calls))

    payload = {
        "schema": "steeract.robotwin.paper_bank.v1",
        "method": "SteerAct Eq. 1-20",
        "backbone": args.backbone,
        "layer": args.layer,
        "sae_checkpoint": str(args.sae_checkpoint),
        "source_activations": str(args.act_dir),
        "control_settings": settings.to_dict(),
        "candidate_features": candidates,
        "cadence": "one_context_hook_call_per_robot_decision",
        "groups": {},
    }
    for task in args.tasks:
        task_id = BACKEND_TASKS.index(task)
        records = []
        for ep, start, calls in grouped[task_id]:
            values = episode_calls(sae, acts, start, calls, args.batch_size, args.device, candidates)
            if len(values) < 5:
                raise ValueError("reference episodes need >=5 decisions for both ordinal spacings")
            records.append((ep, values))
        labels = np.asarray([bool(ep["success"]) for ep, _ in records])
        sequences = [values for _, values in records]
        if min(int(labels.sum()), int((~labels).sum())) < 2:
            raise ValueError(f"{task}: need >=2 valid episodes per outcome")
        score, fid, scale, full, column = choose_d5(sequences, labels, candidates)
        reference = ReferenceBank(full[labels], full[~labels], settings)
        decoder = sae.decoder.weight[:, fid].detach().float()
        encoder = sae.encoder.weight[fid].detach().float()
        gain = float(torch.dot(decoder, encoder).cpu())
        payload["groups"][f"D5__T{task_id}__{args.layer}"] = {
            "descriptor": "ordinal_pattern", "feature_id": fid,
            "intensity_scale": scale, "analytic_self_gain": gain,
            "success": full[labels].tolist(), "failure": full[~labels].tolist(),
            "support_radius": reference.radius, "coordinate_scales": reference.scales.tolist(),
            "d5_energy_distance": score,
            "success_episodes": int(labels.sum()), "failure_episodes": int((~labels).sum()),
            "source_seeds": [int(ep["seed"]) for ep, _ in records],
        }
        print(f"{task}: feature={fid} D5={score:.4f} ({labels.sum()}S/{(~labels).sum()}F)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
