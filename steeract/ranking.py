from pathlib import Path
import hashlib

import numpy as np
import torch

from .config import collection_identity, digest, layer_artifact_dir, sae_artifact_dir, steering_layers
from .method import Settings, ReferenceBank, descriptor, energy_score
from .data import atomic_json, records, token_indices
from .runtime import DecisionHook, LastCallHook, load_backend
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


def action_response_ranking(cfg, backend, sae, task, reference_seed, sequences, labels):
    summaries = np.stack([sequence.mean(0) for sequence in sequences])
    success, failure = summaries[labels], summaries[~labels]
    success_mean, failure_mean = success.mean(0), failure.mean(0)
    gap = success_mean - failure_mean
    spread = np.sqrt((success.var(0) + failure.var(0)) / 2 + 1e-8)
    separation = np.abs(gap) / spread
    rejected = (success_mean == 0) & (failure_mean > 0)
    ordered = np.argsort(-separation, kind='stable')
    candidates = ordered[~rejected[ordered]][:8]
    if not len(candidates):
        raise ValueError(f'{task}: no feature passed the success-activation filter')

    direction, alpha = None, 0.
    layer = cfg['layer']
    token_limit = cfg['collection']['tokens_per_decision']

    def edit(_module, _inputs, hidden):
        if direction is None:
            return hidden
        ids = token_indices(hidden, token_limit)
        edited = hidden.clone()
        edited[0, ids] += alpha * direction.to(hidden)
        if not torch.isfinite(edited[0, ids]).all():
            raise ValueError('action probe produced nonfinite activations')
        return edited

    if cfg['collection']['action_capture'] == 'last_denoising_call':
        hook = LastCallHook(edit, cfg['policy']['inference_steps'])
    else:
        hook = DecisionHook(edit, first_call_only=layer.startswith(('expert_L', 'fastwam_action_L')))

    @torch.inference_mode()
    def probe(infer):
        nonlocal direction, alpha

        def actions():
            torch.manual_seed(1701)
            hook.begin_decision()
            value = np.array(infer(), copy=True)
            hook.end_decision()
            if not value.size or not np.isfinite(value).all():
                raise ValueError('invalid action chunk in feature probe')
            return value

        baseline = actions()
        if not np.array_equal(baseline, actions()):
            raise ValueError('same-observation baseline actions are not reproducible')
        ranking = []
        for feature in map(int, candidates):
            direction = sae.decoder.weight[:, feature].detach()
            doses = []
            for dose in (10., 100., 300.):
                alpha = float(np.sign(gap[feature])) * dose
                action = actions()
                if action.shape != baseline.shape:
                    raise ValueError('action probe changed the action chunk shape')
                rms = float(np.sqrt(np.mean((action - baseline) ** 2)))
                doses.append(dict(alpha=alpha, action_rms=rms))
            score = float(separation[feature]) * max(row['action_rms'] for row in doses)
            ranking.append(dict(feature_id=feature, separation=float(separation[feature]),
                                gap=float(gap[feature]), doses=doses, score=score))
        return sorted(ranking, key=lambda row: -row['score'])

    ranking = backend.run_pi05_with_hook(env_seed=reference_seed, task_name=task,
                                       hooks=[(layer, hook)], action_probe=probe)
    if not isinstance(ranking, list):
        raise RuntimeError(f'{task}: reference action probe did not run')
    if ranking[0]['score'] <= 0:
        raise ValueError(f'{task}: no candidate with a positive action-response score at {layer}')
    return ranking


def rank(cfg, device):
    layers = steering_layers(cfg)
    destination = Path(cfg['paths']['run_dir']) / 'steering_bank.json'
    banks = {layer: rank_layer(cfg, device, layer) for layer in layers}
    if len(layers) == 1:
        artifact = banks[layers[0]]
    else:
        artifact = dict(schema='steeract.layer_banks.v1',
                        collection_identity=digest(collection_identity(cfg)), layers=banks)
    atomic_json(destination, artifact)
    return destination


def rank_layer(cfg, device, layer):
    root, rows = records(cfg, layer=layer)
    output = layer_artifact_dir(cfg, layer)
    identity = digest(collection_identity(cfg, layer=layer))
    count = cfg['steering']['top_k']
    selector = cfg['steering'].get('selector', 'ordinal')
    settings = Settings(**cfg['steering']['controller'])
    artifact = dict(schema='steeract.topk_bank.v1', backbone=cfg['backbone'], layer=layer,
                    collection_identity=identity,
                    steering_top_k=count, combination='uniform_mean',
                    control_settings=settings.to_dict(), tasks={})
    backend = None
    if selector == 'action_response':
        if count != 1:
            raise ValueError('action_response selection requires steering.top_k=1')
        artifact['selector'] = selector
        backend = load_backend(cfg)
    for task in cfg['tasks']:
        path = sae_artifact_dir(cfg, layer, task) / 'model.pt'
        sae, checkpoint = load_sae(path, device)
        if checkpoint['collection_identity'] != identity or checkpoint['tasks'] != [task]:
            raise ValueError(f'{task} {layer}: SAE training data mismatch')
        checkpoint_sha256 = file_hash(path)
        episodes = [r for r in rows if r['task'] == task]
        labels = np.array([r['success'] for r in episodes], dtype=bool)
        if min(int(labels.sum()), int((~labels).sum())) < 2:
            raise ValueError(f'{task}: reference set needs >=2 success and >=2 failure episodes; do not use evaluation data')
        sequences = [episode_intensities(sae, root, row, device, cfg['sae']['batch_size']) for row in episodes]
        selection_info = {}
        if backend is not None:
            reference_seed = episodes[0]['seed']
            ranking = candidates = action_response_ranking(
                {**cfg, 'layer': layer}, backend, sae, task, reference_seed, sequences, labels)
            selection_info = dict(selector=selector, training_observation_seed=reference_seed)
        else:
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
        if len(cfg['tasks']) > 1:
            artifact['tasks'][task]['checkpoint_sha256'] = checkpoint_sha256
        else:
            artifact['checkpoint_sha256'] = checkpoint_sha256
        atomic_json(output / 'ranking' / f'{task}.json', dict(task=task,
            checkpoint_sha256=checkpoint_sha256, ranking=ranking, selected=selected, **selection_info))
        atomic_json(output / 'ranking' / f'{task}_temporal.json', dict(
            task=task, feature_ids=selected, layer=layer,
            intensity_scale=scale, spacings=[1, 2],
            descriptor='six ordinal frequencies, entropy, tie rate, availability per spacing',
            episodes=[dict(seed=row['seed'], success=row['success'],
                           intensity=history.tolist(), descriptor=vector.tolist())
                      for row, history, vector in zip(episodes, combined, vectors)]))
        print(f'{task} {layer}: steering top-{count} = {selected}', flush=True)
    return artifact
