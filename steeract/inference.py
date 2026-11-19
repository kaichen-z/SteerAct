import fcntl
import json
from contextlib import ExitStack
from pathlib import Path

import torch

from .config import collection_identity, digest, sae_artifact_dir, steering_layers
from .method import Controller, ReferenceBank, Settings
from .data import atomic_json, token_indices
from .ranking import file_hash
from .runtime import DecisionHook, LastCallHook, load_backend
from .sae import load_sae
from .trajectory import TrajectoryBank, TrajectoryController, TrajectorySettings
from .phase_alignment import AlignedTrajectoryBank, AlignedTrajectoryController, AlignmentSettings


def steering_hook(sae, group, settings, token_limit, seed, trajectory_settings=None, alignment_settings=None,
                  last_call_count=None):
    if alignment_settings is not None and trajectory_settings is None:
        raise ValueError('phase alignment requires trajectory settings')
    ids = group['feature_ids']
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= sae.decoder.weight.shape[1]:
        raise ValueError('invalid steering feature IDs')
    sae.eval().requires_grad_(False)
    feature_ids = torch.tensor(ids, device=sae.decoder.weight.device)
    direction = sae.decoder.weight[:, feature_ids].mean(1).detach()
    response = sae.encoder.weight[feature_ids] @ direction
    if trajectory_settings is None:
        bank = ReferenceBank(group['success'], group['failure'], settings)
        prefix_banks = {int(key): ReferenceBank(value['success'], value['failure'], settings)
                        for key, value in group.get('prefix_banks', {}).items()}
        controller = Controller(bank, settings=settings, scale=group['intensity_scale'], seed=seed,
                                prefix_banks=prefix_banks)
    elif alignment_settings is None:
        bank = TrajectoryBank(group['trajectory_success'], group['trajectory_failure'],
                              scale=group['intensity_scale'], settings=trajectory_settings)
        controller = TrajectoryController(bank, alpha_settings=settings)
    else:
        bank = AlignedTrajectoryBank(group['trajectory_success'], group['trajectory_failure'],
            scale=group['intensity_scale'], settings=trajectory_settings, alignment=alignment_settings)
        controller = AlignedTrajectoryController(bank, alpha_settings=settings)

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
        alpha = controller.step(intensity, gain,
            predict_intensity=lambda a: intensity if a == 0 else measure(rows + a * direction.to(hidden))[0])
        if alpha == 0:
            controller.feedback(intensity)
            return hidden
        edited = hidden.clone()
        edited[0, indices] = rows + alpha * direction.to(hidden)
        controller.feedback(measure(edited[0, indices])[0])
        return edited

    if last_call_count is not None:
        def last_apply(module, inputs, hidden):
            edited = apply(module, inputs, hidden)
            controller.trace[-1].update(capture_call=last_call_count, denoising_calls=last_call_count)
            return edited
        hook = LastCallHook(last_apply, last_call_count)
    else:
        hook = DecisionHook(apply)
    hook.controller = controller
    return hook


def load_layer(cfg, device, layer, task, bank, control):
    path = sae_artifact_dir(cfg, layer, task) / 'model.pt'
    if bank['collection_identity'] != digest(collection_identity(cfg, layer=layer)):
        raise ValueError(f'{layer}: steering bank collection configuration mismatch')
    checkpoint_bank = bank['tasks'][task] if len(cfg['tasks']) > 1 else bank
    if checkpoint_bank['checkpoint_sha256'] != file_hash(path):
        raise ValueError('SAE checkpoint changed; rebuild feature ranking and bank')
    if bank['steering_top_k'] != cfg['steering']['top_k']:
        raise ValueError('steering.top_k changed; rerun rank')
    if bank.get('selector', 'ordinal') != cfg['steering'].get('selector', 'ordinal'):
        raise ValueError('steering.selector changed; rerun rank')
    if Settings(**bank['control_settings']).to_dict() != control.to_dict():
        raise ValueError('controller settings changed; rerun rank')
    method = bank.get('method', 'ordinal')
    if method not in ('ordinal', 'trajectory', 'phase_aligned'):
        raise ValueError('unknown steering bank method')
    trajectory_control = None
    if method in ('trajectory', 'phase_aligned'):
        trajectory_control = TrajectorySettings(**cfg['steering']['trajectory'])
        if trajectory_control != TrajectorySettings(**bank['trajectory_settings']):
            raise ValueError('trajectory settings changed; rebuild bank')
    elif 'trajectory' in cfg['steering']:
        raise ValueError('trajectory config requires a trajectory bank')
    alignment_control = None
    if method == 'phase_aligned':
        alignment_control = AlignmentSettings(**cfg['steering']['alignment'])
        if alignment_control != AlignmentSettings(**bank['alignment_settings']):
            raise ValueError('alignment settings changed; rebuild bank')
    elif 'alignment' in cfg['steering']:
        raise ValueError('alignment config requires a phase-aligned bank')
    sae, checkpoint = load_sae(path, device)
    if checkpoint['tasks'] != [task]:
        raise ValueError(f'{task} {layer}: SAE training task mismatch')
    return sae, trajectory_control, alignment_control


def infer(cfg, device):
    run = Path(cfg['paths']['run_dir'])
    settings = cfg['evaluation']
    variant = settings.get('variant', 'both')
    if variant not in ('both', 'steeract', 'base'):
        raise ValueError('evaluation.variant must be both, steeract, or base')
    layers = steering_layers(cfg)
    multilayer = len(layers) > 1
    bank = json.loads(Path(settings.get('bank_path', run / 'steering_bank.json')).read_text())
    if bank['collection_identity'] != digest(collection_identity(cfg)):
        raise ValueError('steering bank collection configuration mismatch')
    if multilayer:
        if bank.get('schema') != 'steeract.layer_banks.v1' or set(bank['layers']) != set(layers):
            raise ValueError('steering bank layers differ from steering.layers; rerun rank')
        layer_banks = bank['layers']
    else:
        if bank.get('schema') == 'steeract.layer_banks.v1':
            raise ValueError('steering bank has multiple layers; rerun rank')
        layer_banks = {layers[0]: bank}
    for layer, layer_bank in layer_banks.items():
        if set(layer_bank['tasks']) != set(cfg['tasks']):
            raise ValueError(f'{layer}: steering bank tasks differ from configured tasks; rerun rank')
    if len(cfg['tasks']) > 1:
        checkpoints = {}
        for layer, layer_bank in layer_banks.items():
            if any('checkpoint_sha256' not in group for group in layer_bank['tasks'].values()):
                raise ValueError(f'{layer}: task SAE checkpoints are missing; rerun train and rank')
            checkpoints[layer] = {task: group['checkpoint_sha256'] for task, group in layer_bank['tasks'].items()}
    elif multilayer:
        checkpoints = {layer: layer_banks[layer]['checkpoint_sha256'] for layer in layers}
    else:
        checkpoints = bank['checkpoint_sha256']
    control = Settings(**cfg['steering']['controller'])
    identity = digest(dict(config=cfg, checkpoint=checkpoints, bank=bank))
    source_range = range(cfg['collection']['seed_start'],
                         cfg['collection']['seed_start'] + cfg['collection']['max_attempts'])
    evaluation_range = range(settings['seed_start'], settings['seed_start'] + settings['max_attempts'])
    if set(source_range).intersection(evaluation_range):
        raise ValueError('reference and evaluation seed ranges must be disjoint')
    top_k = cfg['steering']['top_k']
    default_result = f'top{top_k}_layers{len(layers)}' if multilayer else f'top{top_k}'
    result_name = settings.get('result_name', 'base' if variant == 'base' else default_result)
    if not isinstance(result_name, str) or not result_name.isascii() or not result_name.replace('_', '').isalnum():
        raise ValueError('evaluation.result_name must contain only letters, digits, and underscores')
    results = run / 'results'
    if variant == 'both':
        if result_name == 'base':
            raise ValueError('evaluation.result_name must differ from base in both mode')
        baseline_settings = {key: value for key, value in settings.items()
                             if key not in ('variant', 'result_name', 'bank_path')}
        baseline_identity = digest(dict(collection=collection_identity(cfg), evaluation=baseline_settings))
        arms = {'base': (results / 'base', baseline_identity),
                'steeract': (results / result_name, identity)}
    else:
        arms = {variant: (results / result_name, identity)}
    with ExitStack() as locks:
        for destination, _ in sorted(arms.values()):
            destination.mkdir(parents=True, exist_ok=True)
            lock = locks.enter_context((destination / 'inference.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX)
        outputs = {}
        for arm, (destination, arm_identity) in arms.items():
            config_path = destination / 'configuration.json'
            if config_path.exists() and json.loads(config_path.read_text())['identity'] != arm_identity:
                raise ValueError('output belongs to another configuration; use a new run/result directory')
            atomic_json(config_path, dict(identity=arm_identity, configuration=cfg))
            output = destination / 'episodes.jsonl'
            prior = ([json.loads(line) for line in output.read_text().splitlines() if line.strip()]
                     if output.exists() else [])
            outputs[arm] = (destination, prior)
        evaluate(cfg, device, layer_banks, control, evaluation_range, outputs)


def evaluate(cfg, device, layer_banks, control, evaluation_range, outputs):
    settings = cfg['evaluation']
    budget = settings['episodes_per_task']
    layers = steering_layers(cfg)
    multilayer = len(layers) > 1
    top_k = cfg['steering']['top_k']
    last_call_count = (cfg['policy']['inference_steps']
                       if cfg['collection']['action_capture'] == 'last_denoising_call' else None)
    backend = load_backend(cfg, inference=True)
    for task in cfg['tasks']:
        previous = {variant: [row for row in prior if row['task'] == task]
                    for variant, (_, prior) in outputs.items()}
        used = {variant: {row['seed'] for row in rows} for variant, rows in previous.items()}
        valid = {variant: sum(not row['invalid'] for row in rows) for variant, rows in previous.items()}
        models = {}
        if 'steeract' in outputs:
            models = {layer: load_layer(cfg, device, layer, task, layer_banks[layer], control)
                      for layer in layers}
        for seed in evaluation_range:
            if all(count >= budget for count in valid.values()):
                break
            for variant, (destination, prior) in outputs.items():
                if valid[variant] >= budget or seed in used[variant]:
                    continue
                hooks = []
                if variant == 'steeract':
                    for layer, (sae, trajectory_control, alignment_control) in models.items():
                        group = layer_banks[layer]['tasks'][task]
                        hook = steering_hook(sae, group, control, cfg['collection']['tokens_per_decision'],
                                             seed, trajectory_settings=trajectory_control,
                                             alignment_settings=alignment_control, last_call_count=last_call_count)
                        hooks.append((layer, hook))
                result = backend.run_pi05_with_hook(env_seed=seed, task_name=task, hooks=hooks)
                if any(hook.decisions != result['num_decisions'] for _, hook in hooks):
                    raise RuntimeError('decision cadence mismatch')
                row = dict(task=task, seed=seed, success=bool(result['success']),
                           invalid=bool(result.get('invalid_rollout_flag', False)),
                           invalid_reason=result.get('invalid_reason'),
                           num_decisions=result['num_decisions'], variant=variant,
                           top_k=top_k if hooks or len(outputs) == 1 else 0,
                           max_steps=result.get('max_steps'),
                           max_steps_multiplier=settings['max_steps_multiplier'])
                if hooks or len(outputs) == 1:
                    trace = destination / 'traces' / f'{task}_{seed}.json'
                    traces = {layer: hook.controller.trace for layer, hook in hooks}
                    atomic_json(trace, traces if multilayer else traces.get(layers[0], []))
                    row['trace'] = str(trace.relative_to(destination))
                with (destination / 'episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                prior.append(row)
                valid[variant] += not row['invalid']
                print(f'{task} {variant}: {valid[variant]}/{budget}, seed={seed}, success={row["success"]}', flush=True)
        for variant, count in valid.items():
            if count < budget:
                raise RuntimeError(f'{task} {variant}: evaluation budget not reached')
    for destination, prior in outputs.values():
        summarize(cfg['tasks'], prior, destination)


def summarize(tasks, prior, destination):
    summary = {}
    for task in tasks:
        rows = [r for r in prior if r['task'] == task and not r['invalid']]
        summary[task] = dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                             success_rate=sum(r['success'] for r in rows)/len(rows))
    atomic_json(destination / 'summary.json', dict(tasks=summary,
        mean_success_rate=sum(r['success_rate'] for r in summary.values())/len(summary)))
