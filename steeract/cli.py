import argparse
import json

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description='Release workflow. Run each command in its appropriate Python environment.')
    parser.add_argument('stage', choices=('collect', 'train', 'rank', 'infer'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0', help='SAE device for train/rank/infer')
    parser.add_argument('--dry-run', action='store_true', help='validate and print resolved recipe without loading models')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        print(json.dumps(dict(stage=args.stage, configuration=cfg), indent=2))
        return
    if args.stage == 'collect':
        from .data import collect
        collect(cfg)
    elif args.stage == 'train':
        from .sae import train
        train(cfg, args.device)
    elif args.stage == 'rank':
        from .ranking import rank
        rank(cfg, args.device)
    else:
        from .inference import infer
        infer(cfg, args.device)
