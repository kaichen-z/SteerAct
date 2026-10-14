import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from .config import collection_identity, digest
from .runtime import DecisionHook, load_backend


def token_indices(hidden, limit):
    if hidden.ndim != 3 or hidden.shape[0] != 1 or not hidden.shape[1]:
        raise ValueError('expected [1, tokens, hidden_dim] context activation')
    n = hidden.shape[1]
    return torch.linspace(0, n - 1, min(n, limit), device=hidden.device).long()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def records(cfg, *, require_budget=True):
    root = Path(cfg['paths']['run_dir']) / 'activations'
    manifest = root / 'manifest.json'
    payload = json.loads(manifest.read_text())
    if payload.get('schema') != 'steeract.activations.v2':
        raise ValueError('activation schema changed; recollect into a new run directory')
    if payload['identity'] != digest(collection_identity(cfg)):
        raise ValueError('activation manifest belongs to a different collection configuration')
    rows = payload['episodes']
    seen = set()
    for row in rows:
        key = (row['task'], row['seed'])
        if key in seen or row['task'] not in cfg['tasks'] or not row['files']:
            raise ValueError('invalid or duplicate episode in manifest')
        seen.add(key)
        if len(row['files']) != row['num_decisions'] or len(row['counts']) != len(row['files']):
            raise ValueError('decision boundaries do not match shard list')
        if any(n <= 0 for n in row['counts']):
            raise ValueError('empty activation decision')
        if set(row.get('layers', {})) != set(cfg['collection']['layers']):
            raise ValueError('manifest does not contain the configured collection layers')
        for layer, shards in row['layers'].items():
            if (len(shards['files']) != row['num_decisions']
                    or len(shards['counts']) != row['num_decisions']
                    or any(n <= 0 for n in shards['counts'])):
                raise ValueError(f'{layer}: invalid per-decision shards')
        selected = row['layers'][cfg['layer']]
        if row['files'] != selected['files'] or row['counts'] != selected['counts']:
            raise ValueError('selected layer does not match episode shards')
    if require_budget:
        for task in cfg['tasks']:
            count = sum(row['task'] == task for row in rows)
            if count != cfg['collection']['episodes_per_task']:
                raise ValueError(f'{task}: expected {cfg["collection"]["episodes_per_task"]} episodes, found {count}')
    return root, rows


def collect(cfg):
    root = Path(cfg['paths']['run_dir']) / 'activations'
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'manifest.json'
    identity = digest(collection_identity(cfg))
    if manifest.exists():
        _, rows = records(cfg, require_budget=False)
    else:
        rows = []
    backend = load_backend(cfg)
    backend.validate_collection_layers()
    settings = cfg['collection']
    for task in cfg['tasks']:
        existing = [row for row in rows if row['task'] == task]
        used = {row['seed'] for row in existing}
        for seed in range(settings['seed_start'], settings['seed_start'] + settings['max_attempts']):
            if len(existing) >= settings['episodes_per_task']:
                break
            if seed in used:
                continue
            staging = Path(tempfile.mkdtemp(prefix='pending_', dir=root))
            layer_shards = {name: dict(files=[], counts=[]) for name in settings['layers']}
            def make_hook(name):
                shards = layer_shards[name]
                (staging / name).mkdir()
                def capture(_module, _inputs, hidden):
                    ids = token_indices(hidden, settings['tokens_per_decision'])
                    selected = hidden[0, ids].detach().to(device='cpu', dtype=getattr(torch, settings['storage_dtype']))
                    if not torch.isfinite(selected).all():
                        raise ValueError('nonfinite saved activation (including storage overflow)')
                    filename = f'{name}/{len(shards["files"]):06d}.pt'
                    torch.save(selected, staging / filename)
                    shards['files'].append(filename)
                    shards['counts'].append(len(selected))
                    return hidden
                action_layer = name.startswith(('expert_L', 'fastwam_action_L'))
                return DecisionHook(capture, first_call_only=action_layer)
            hooks = [(name, make_hook(name)) for name in settings['layers']]
            try:
                result = backend.run_pi05_with_hook(env_seed=seed, task_name=task, hooks=hooks)
                decisions = result['num_decisions']
                if result.get('invalid_rollout_flag') or decisions < 5:
                    raise ValueError('invalid rollout or fewer than five robot decisions')
                if any(hook.decisions != decisions for _, hook in hooks):
                    raise ValueError('backend decision count mismatch')
                destination = root / task / f'{seed:08d}'
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    shutil.rmtree(destination)
                staging.rename(destination)
                for shards in layer_shards.values():
                    shards['files'] = [str((destination / f).relative_to(root)) for f in shards['files']]
                selected = layer_shards[cfg['layer']]
                row = dict(task=task, seed=seed, success=bool(result['success']),
                           num_decisions=decisions, counts=selected['counts'],
                           files=selected['files'], layers=layer_shards)
                rows.append(row)
                existing.append(row)
                atomic_json(manifest, dict(schema='steeract.activations.v2', identity=identity,
                                          configuration=collection_identity(cfg), episodes=rows))
                print(f'{task}: {len(existing)}/{settings["episodes_per_task"]}, seed={seed}, success={row["success"]}', flush=True)
            except Exception as exc:
                with (root / 'attempts.jsonl').open('a') as output:
                    output.write(json.dumps(dict(task=task, seed=seed, error=f'{type(exc).__name__}: {exc}')) + '\n')
                raise
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        if len(existing) != settings['episodes_per_task']:
            raise RuntimeError(f'{task}: reference budget not reached')


class ActivationRows:
    def __init__(self, root, episodes):
        self.root = Path(root)
        self.files = [f for ep in episodes for f in ep['files']]
        counts = [n for ep in episodes for n in ep['counts']]
        self.ends = np.cumsum(counts)
        if not len(self.ends):
            raise ValueError('empty activation dataset')
        self.dimension = self.shard(0).shape[1]

    def shard(self, index):
        value = torch.load(self.root / self.files[index], map_location='cpu', mmap=True, weights_only=True)
        if value.ndim != 2 or not torch.isfinite(value).all():
            raise ValueError('activation shard must be a finite matrix')
        start = self.ends[index-1] if index else 0
        if len(value) != self.ends[index] - start:
            raise ValueError('activation shard length differs from manifest')
        return value

    def batch(self, size, rng):
        ids = rng.integers(0, self.ends[-1], size=size)
        shards = np.searchsorted(self.ends, ids, side='right')
        output = torch.empty((size, self.dimension), dtype=torch.float32)
        for index in np.unique(shards):
            mask = np.flatnonzero(shards == index)
            start = self.ends[index-1] if index else 0
            value = self.shard(index)
            if value.shape[1] != self.dimension:
                raise ValueError('mixed hidden dimensions')
            output[torch.as_tensor(mask)] = value[torch.as_tensor(ids[mask] - start)].float()
        return output
