import hashlib
import json
import os
from pathlib import Path

TASKS = (
    'put_bottles_dustbin', 'stack_blocks_three',
    'blocks_ranking_size', 'place_dual_shoes', 'dump_bin_bigbin',
    'open_microwave', 'turn_switch', 'place_mouse_pad', 'pick_diverse_bottles',
)


def collection_layers(backbone):
    if backbone == 'pi05':
        return [f'paligemma_L{i}' for i in range(15, 18)] + [f'expert_L{i}' for i in range(18)]
    if backbone == 'fastwam':
        return [f'fastwam_video_L{i}' for i in range(27, 30)] + [f'fastwam_action_L{i}' for i in range(30)]
    raise ValueError('unsupported backbone')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    path = Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text())
    if cfg['backbone'] not in ('pi05', 'fastwam'):
        raise ValueError('unsupported backbone')
    if cfg['sae']['scope'] not in ('single_task', 'all_tasks'):
        raise ValueError('scope must be single_task or all_tasks')
    tasks = cfg['tasks']
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in TASKS for t in tasks):
        raise ValueError('tasks must be distinct entries in the nine-task registry')
    if cfg['sae']['scope'] == 'single_task' and len(tasks) != 1:
        raise ValueError('single_task requires exactly one task')
    if cfg['sae']['scope'] == 'all_tasks' and set(tasks) != set(TASKS):
        raise ValueError('all_tasks requires all nine RoboTwin tasks')
    expected_layers = collection_layers(cfg['backbone'])
    layers = cfg['collection'].setdefault('layers', expected_layers)
    if (not isinstance(layers, list) or any(not isinstance(layer, str) for layer in layers)
            or len(set(layers)) != len(layers)
            or any(layer not in expected_layers for layer in layers)):
        raise ValueError('collection.layers must be a list of distinct supported layer names')
    if not set(expected_layers[:3]).issubset(layers):
        raise ValueError('collection.layers must contain all three required vision layers; action expert layers are optional')
    if cfg['collection'].get('action_capture') != 'first_denoising_call':
        raise ValueError('collection.action_capture is fixed at first_denoising_call')
    if cfg['layer'] not in expected_layers[:3]:
        raise ValueError('SAE/steering layer must be one of the three collected vision layers')
    for key in ('episodes_per_task', 'max_attempts', 'tokens_per_decision'):
        if cfg['collection'][key] < 1:
            raise ValueError(f'collection.{key} must be positive')
    if cfg['collection']['max_attempts'] < cfg['collection']['episodes_per_task']:
        raise ValueError('max_attempts must cover the collection budget')
    if cfg['collection']['storage_dtype'] not in ('float16', 'bfloat16', 'float32'):
        raise ValueError('unsupported activation storage dtype')
    if cfg['sae']['log_every'] < 1:
        raise ValueError('sae.log_every must be positive')
    top_k = cfg['steering'].setdefault('top_k', 4)
    if type(top_k) is not int or top_k < 1:
        raise ValueError('steering.top_k must be a positive integer')
    if cfg['evaluation']['episodes_per_task'] < 1 or cfg['evaluation']['max_attempts'] < cfg['evaluation']['episodes_per_task']:
        raise ValueError('invalid evaluation budget')
    factor = cfg['evaluation'].get('max_steps_multiplier', 2)
    if type(factor) is not int or factor < 1:
        raise ValueError('evaluation.max_steps_multiplier must be a positive integer')
    cfg['evaluation']['max_steps_multiplier'] = factor
    for section in ('paths',):
        for key, value in cfg[section].items():
            if value is not None:
                if value.startswith('hf://'):
                    if key not in ('dit', 'vae', 'text_encoder', 'tokenizer'):
                        raise ValueError(f'Hugging Face paths are not supported for {key}')
                    continue
                value = Path(os.path.expandvars(value)).expanduser()
                cfg[section][key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    return cfg


def collection_identity(cfg):
    return dict(parameters={key: cfg[key] for key in ('backbone', 'layer', 'tasks', 'collection', 'policy')},
                sources={key: value for key, value in cfg['paths'].items() if key != 'run_dir'})
