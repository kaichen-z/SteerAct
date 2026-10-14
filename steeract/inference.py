import json
from pathlib import Path

import torch

from .config import collection_identity, digest
from .method import Controller, ReferenceBank, Settings
from .data import atomic_json, token_indices
from .ranking import file_hash
from .runtime import DecisionHook, load_backend
from .sae import load_sae


def steering_hook(sae, group, settings, token_limit, seed):
    ids = group['feature_ids']
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= sae.decoder.weight.shape[1]:
        raise ValueError('invalid steering feature IDs')
    sae.eval().requires_grad_(False)
    feature_ids = torch.tensor(ids, device=sae.decoder.weight.device)
    direction = sae.decoder.weight[:, feature_ids].mean(1).detach()
    response = sae.encoder.weight[feature_ids] @ direction
    bank = ReferenceBank(group['success'], group['failure'], settings)
    controller = Controller(bank, settings=settings, scale=group['intensity_scale'], seed=seed)

    def measure(rows):
        code = sae.encode(rows.to(sae.encoder.weight))[:, feature_ids]
        if not torch.isfinite(code).all():
            raise ValueError('nonfinite SAE code')
        intensity = float(code.clamp_min(0).mean())
        active_fraction = (code > 0).float().mean(0)
        gain = float((active_fraction * response).mean())
        return intensity, gain

    @torch.inference_mode()
    def apply(_module, _inputs, hidden):
        indices = token_indices(hidden, token_limit)
        rows = hidden[0, indices]
        if not torch.isfinite(rows).all():
            raise ValueError('nonfinite context activation')
        intensity, gain = measure(rows)
        alpha = controller.step(intensity, gain)
        if alpha == 0:
            controller.feedback(intensity)
            return hidden
        edited = hidden.clone()
        edited[0, indices] = rows + alpha * direction.to(hidden)
        controller.feedback(measure(edited[0, indices])[0])
        return edited

    hook = DecisionHook(apply)
    hook.controller = controller
    return hook


def infer(cfg, device):
    run = Path(cfg['paths']['run_dir'])
    settings = cfg['evaluation']
    path = run / 'sae/model.pt'
    bank = json.loads((run / 'steering_bank.json').read_text())
    if bank['collection_identity'] != digest(collection_identity(cfg)):
        raise ValueError('steering bank collection configuration mismatch')
    if bank['checkpoint_sha256'] != file_hash(path):
        raise ValueError('SAE checkpoint changed; rebuild feature ranking and bank')
    if bank['steering_top_k'] != cfg['steering']['top_k']:
        raise ValueError('steering.top_k changed; rerun rank')
    control = Settings(**cfg['steering']['controller'])
    if bank['control_settings'] != control.to_dict():
        raise ValueError('controller settings changed; rerun rank')
    sae, _ = load_sae(path, device)
    identity = digest(dict(config=cfg, checkpoint=bank['checkpoint_sha256'], bank=bank))
    source_range = range(cfg['collection']['seed_start'],
                         cfg['collection']['seed_start'] + cfg['collection']['max_attempts'])
    evaluation_range = range(settings['seed_start'], settings['seed_start'] + settings['max_attempts'])
    if set(source_range).intersection(evaluation_range):
        raise ValueError('reference and evaluation seed ranges must be disjoint')
    destination = run / 'results' / f'top{bank["steering_top_k"]}'
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())['identity'] != identity:
        raise ValueError('output belongs to another configuration; use a new run/result directory')
    atomic_json(config_path, dict(identity=identity, configuration=cfg))
    output = destination / 'episodes.jsonl'
    prior = [json.loads(line) for line in output.read_text().splitlines() if line.strip()] if output.exists() else []
    backend = load_backend(cfg, inference=True)
    for task in cfg['tasks']:
        existing = [r for r in prior if r['task'] == task]
        used = {r['seed'] for r in existing}
        valid = sum(not r['invalid'] for r in existing)
        for seed in evaluation_range:
            if valid >= settings['episodes_per_task']:
                break
            if seed in used:
                continue
            hook = steering_hook(sae, bank['tasks'][task], control,
                                 cfg['collection']['tokens_per_decision'], seed)
            result = backend.run_pi05_with_hook(env_seed=seed, task_name=task,
                                               hooks=[(cfg['layer'], hook)])
            if hook.decisions != result['num_decisions']:
                raise RuntimeError('decision cadence mismatch')
            row = dict(task=task, seed=seed, success=bool(result['success']),
                       invalid=bool(result.get('invalid_rollout_flag', False)),
                       num_decisions=result['num_decisions'], variant='steeract',
                       top_k=bank['steering_top_k'],
                       max_steps=result.get('max_steps'),
                       max_steps_multiplier=settings['max_steps_multiplier'])
            trace = destination / 'traces' / f'{task}_{seed}.json'
            atomic_json(trace, hook.controller.trace)
            row['trace'] = str(trace.relative_to(destination))
            with output.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            prior.append(row)
            valid += not row['invalid']
            print(f'{task}: {valid}/{settings["episodes_per_task"]}, seed={seed}, success={row["success"]}', flush=True)
        if valid != settings['episodes_per_task']:
            raise RuntimeError(f'{task}: evaluation budget not reached')
    summary = {}
    for task in cfg['tasks']:
        rows = [r for r in prior if r['task'] == task and not r['invalid']]
        summary[task] = dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                             success_rate=sum(r['success'] for r in rows)/len(rows))
    atomic_json(destination / 'summary.json', dict(tasks=summary,
        mean_success_rate=sum(r['success_rate'] for r in summary.values())/len(summary)))
