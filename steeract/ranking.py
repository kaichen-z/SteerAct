from pathlib import Path
import hashlib

import numpy as np
import torch

from .config import collection_identity, digest
from .method import Settings, ReferenceBank, descriptor, energy_score
from .data import atomic_json, records
from .sae import load_sae


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


@torch.inference_mode()
def episode_intensities(sae, root, row, device, batch_size):
    decisions = []
    for filename, count in zip(row['files'], row['counts']):
        hidden = torch.load(root / filename, map_location='cpu', mmap=True, weights_only=True)
        if hidden.ndim != 2 or len(hidden) != count or not torch.isfinite(hidden).all():
            raise ValueError('invalid activation shard')
        total = torch.zeros(sae.decoder.weight.shape[1], device=device)
        for batch in hidden.split(batch_size):
            total += sae.encode(batch.to(device=device, dtype=torch.float32)).clamp_min(0).sum(0)
        decisions.append((total / count).cpu().numpy())
    return np.stack(decisions)


def rank(cfg, device):
    root, rows = records(cfg)
    run = Path(cfg['paths']['run_dir'])
    path = run / 'sae/model.pt'
    sae, checkpoint = load_sae(path, device)
    identity = digest(collection_identity(cfg))
    if checkpoint['collection_identity'] != identity:
        raise ValueError('SAE was trained on a different collection')
    count = cfg['steering']['top_k']
    settings = Settings(**cfg['steering']['controller'])
    artifact = dict(schema='steeract.topk_bank.v1', backbone=cfg['backbone'], layer=cfg['layer'],
                    collection_identity=identity, checkpoint_sha256=file_hash(path),
                    steering_top_k=count, combination='uniform_mean',
                    control_settings=settings.to_dict(), tasks={})
    for task in cfg['tasks']:
        episodes = [r for r in rows if r['task'] == task]
        labels = np.array([r['success'] for r in episodes], dtype=bool)
        if min(int(labels.sum()), int((~labels).sum())) < 2:
            raise ValueError(f'{task}: reference set needs >=2 success and >=2 failure episodes; do not use evaluation data')
        sequences = [episode_intensities(sae, root, row, device, cfg['sae']['batch_size']) for row in episodes]
        ranking = []
        for feature in range(sae.decoder.weight.shape[1]):
            active = any(np.any(sequence[:, feature] > 0) for sequence in sequences)
            vectors = np.stack([descriptor(sequence[:, feature])[0] for sequence in sequences])
            score = energy_score(vectors[labels], vectors[~labels])
            ranking.append(dict(feature_id=feature, score=score, active=active))
        ranking.sort(key=lambda r: (-r['score'], r['feature_id']))
        candidates = [r for r in ranking if r['active']]
        if count > len(candidates):
            raise ValueError(f'{task}: requested {count} steering features, only {len(candidates)} active')
        selected = [r['feature_id'] for r in candidates[:count]]
        combined = [sequence[:, selected].mean(1) for sequence in sequences]
        positive = np.concatenate(combined)
        positive = positive[positive > 0]
        scale = float(np.median(positive)) if len(positive) else 1.0
        vectors = np.stack([descriptor(sequence, scale)[0] for sequence in combined])
        bank = ReferenceBank(vectors[labels], vectors[~labels], settings)
        artifact['tasks'][task] = dict(feature_ids=selected, intensity_scale=scale,
            success=vectors[labels].tolist(), failure=vectors[~labels].tolist(),
            support_radius=bank.radius, coordinate_scales=bank.scales.tolist(),
            source_seeds=[r['seed'] for r in episodes],
            success_episodes=int(labels.sum()), failure_episodes=int((~labels).sum()))
        atomic_json(run / 'ranking' / f'{task}.json', dict(task=task,
            checkpoint_sha256=artifact['checkpoint_sha256'], ranking=ranking, selected=selected))
        print(f'{task}: steering top-{count} = {selected}', flush=True)
    atomic_json(run / 'steering_bank.json', artifact)
    return run / 'steering_bank.json'
