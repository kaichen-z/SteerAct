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


def steering_layers(cfg):
    return cfg.get('steering', {}).get('layers', [cfg['layer']])


def layer_artifact_dir(cfg, layer):
    run = Path(cfg['paths']['run_dir'])
    return run / 'layers' / layer if len(steering_layers(cfg)) > 1 else run


def sae_artifact_dir(cfg, layer, task):
    output = layer_artifact_dir(cfg, layer)
    return output / 'tasks' / task / 'sae' if len(cfg['tasks']) > 1 else output / 'sae'


def load_config(path):
    path = Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text())
    if cfg['backbone'] not in ('pi05', 'fastwam'):
        raise ValueError('unsupported backbone')
    scope = cfg['sae'].get('scope')
    if scope == 'all_tasks':
        del cfg['sae']['scope']
    elif scope not in (None, 'single_task'):
        raise ValueError('invalid sae.scope')
    tasks = cfg['tasks']
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in TASKS for t in tasks):
        raise ValueError('tasks must be distinct entries in the nine-task registry')
    if scope == 'single_task' and len(tasks) != 1:
        raise ValueError('single_task requires exactly one task')
    expected_layers = collection_layers(cfg['backbone'])
    vision = expected_layers[:3]
    if cfg['backbone'] == 'fastwam':
        vision = ['fastwam_video_L26', *vision]
    selected = steering_layers(cfg)
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(layer, str) for layer in selected)
            or len(set(selected)) != len(selected) or cfg['layer'] not in selected):
        raise ValueError('steering.layers must list distinct layers including layer')
    default_layers = list(dict.fromkeys([*expected_layers, *selected]))
    layers = cfg['collection'].setdefault('layers', default_layers)
    if (not isinstance(layers, list) or any(not isinstance(layer, str) for layer in layers)
            or len(set(layers)) != len(layers)
            or any(layer not in vision and layer not in expected_layers[3:] for layer in layers)):
        raise ValueError('collection.layers must be a list of distinct supported layer names')
    if not set(selected).issubset(layers):
        raise ValueError('collection.layers must include every steering layer')
    action_last = cfg['collection'].get('action_capture') == 'last_denoising_call'
    if action_last:
        if (cfg['backbone'] != 'fastwam' or cfg['layer'] not in expected_layers[3:]
                or layers != [cfg['layer']]):
            raise ValueError('last-call mode requires one FastWAM action layer')
        calls = cfg['policy']['inference_steps']
        if type(calls) is not int or calls < 1:
            raise ValueError('last-call mode requires positive integer inference_steps')
    elif 'layers' not in cfg['steering'] and not set(expected_layers[:3]).issubset(layers):
        raise ValueError('collection.layers must contain all three required vision layers; action expert layers are optional')
    if not action_last and cfg['collection'].get('action_capture') != 'first_denoising_call':
        raise ValueError('collection.action_capture is fixed at first_denoising_call')
    if not action_last and any(layer not in vision for layer in selected):
        raise ValueError('steering layers must be supported vision layers')
    for key in ('episodes_per_task', 'max_attempts', 'tokens_per_decision'):
        if cfg['collection'][key] < 1:
            raise ValueError(f'collection.{key} must be positive')
    if cfg['collection']['max_attempts'] < cfg['collection']['episodes_per_task']:
        raise ValueError('max_attempts must cover the collection budget')
    minimum = cfg['collection'].get('min_decisions', 5)
    if type(minimum) is not int or minimum < 1:
        raise ValueError('collection.min_decisions must be a positive integer')
    if cfg['collection']['storage_dtype'] not in ('float16', 'bfloat16', 'float32'):
        raise ValueError('unsupported activation storage dtype')
    if cfg['sae']['log_every'] < 1:
        raise ValueError('sae.log_every must be positive')
    top_k = cfg['steering'].setdefault('top_k', 4)
    if type(top_k) is not int or top_k < 1:
        raise ValueError('steering.top_k must be a positive integer')
    selector = cfg['steering'].get('selector', 'ordinal')
    if selector not in ('ordinal', 'action_response'):
        raise ValueError('steering.selector must be ordinal or action_response')
    if selector == 'action_response' and top_k != 1:
        raise ValueError('action_response selection requires steering.top_k=1')
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


def collection_identity(cfg, *, layer=None):
    parameters = {key: cfg[key] for key in ('backbone', 'layer', 'tasks', 'collection', 'policy')}
    if layer is not None:
        parameters['layer'] = layer
    return dict(parameters=parameters,
                sources={key: value for key, value in cfg['paths'].items() if key != 'run_dir'})
