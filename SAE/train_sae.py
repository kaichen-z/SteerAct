from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from evaluate import PIPELINE


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a shared TopK SAE for a frozen RoboTwin policy layer.\n\nUses the existing SAE pipeline trainer on activations collected by\ncollect_activations.py. The paper recipe uses dictionary size 8d and k=100.\n')
    parser.add_argument("--backbone", required=True, choices=("pi05", "fastwam"))
    parser.add_argument("--run-dir", type=Path, required=True, help="contains activations/; writes saes/")
    parser.add_argument("--layer", help="default: paligemma_L14 or fastwam_video_L14")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    args.run_dir = args.run_dir.expanduser().resolve()
    args.layer = args.layer or ("paligemma_L14" if args.backbone == "pi05" else "fastwam_video_L14")
    prefix = "paligemma_L" if args.backbone == "pi05" else "fastwam_video_L"
    if not args.layer.startswith(prefix):
        parser.error(f"expected a layer named {prefix}<index>")
    act_dir = args.run_dir / "activations"
    sae_dir = args.run_dir / "saes"
    activation = act_dir / f"{args.layer}.pt"
    checkpoint = sae_dir / args.layer / "topk_er8_k100.pt"
    if not args.dry_run and not activation.is_file():
        raise FileNotFoundError(activation)
    command = [sys.executable, str(PIPELINE / "scripts/train_ft_sae.py"),
               "--act-dir", str(act_dir), "--out-dir", str(sae_dir),
               "--layers", args.layer, "--device", args.device,
               "--expansion-ratio", "8", "--k", "100", "--steps", str(args.steps)]
    print("[train] " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)
        if not checkpoint.is_file():
            raise RuntimeError(f"training did not produce {checkpoint}")
    print(f"SAE checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
