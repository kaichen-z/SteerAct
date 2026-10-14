import importlib
import os
from pathlib import Path



def resolve_model_asset(value, kind):
    """Resolve an HF model URI to the local cache expected by Fast-WAM."""
    if not value.startswith('hf://'):
        return value
    parts = value[5:].strip('/').split('/')
    if len(parts) < 2 or any(part in ('', '.', '..') for part in parts):
        raise ValueError(f'invalid Hugging Face model URI: {value}')
    repo_id = '/'.join(parts[:2])
    subpath = '/'.join(parts[2:])
    from huggingface_hub import hf_hub_download, snapshot_download
    if kind in ('vae', 'text_encoder'):
        if not subpath:
            raise ValueError(f'{kind} requires a filename in its HF URI')
        return hf_hub_download(repo_id=repo_id, filename=subpath)
    if kind == 'dit':
        prefix = subpath + '/' if subpath else ''
        patterns = [prefix + 'diffusion_pytorch_model*.safetensors', prefix + '*.json']
    elif kind == 'tokenizer':
        if not subpath:
            raise ValueError('tokenizer requires a subdirectory in its HF URI')
        patterns = [subpath + '/*']
    else:
        raise ValueError(f'unsupported HF asset kind: {kind}')
    cached = snapshot_download(repo_id=repo_id, allow_patterns=patterns)
    return str(Path(cached) / subpath) if subpath else str(cached)


def load_backend(cfg, *, inference=False):
    paths, policy = cfg['paths'], cfg['policy']
    os.environ['STEERACT_MAX_STEPS_MULTIPLIER'] = str(cfg['evaluation']['max_steps_multiplier'] if inference else 1)
    for name in ('robotwin', 'checkpoint'):
        if not Path(paths[name]).exists():
            raise FileNotFoundError(f'{name}: {paths[name]}')
    os.environ['STEERACT_ROBOTWIN_ROOT'] = paths['robotwin']
    os.environ['STEERACT_WORKSPACE'] = str(Path(paths['robotwin']).parent)
    os.environ['ROBOTWIN_ENV_CFG_TYPE'] = policy['embodiment']
    os.environ['ROBOTWIN_TASK_CONFIG'] = policy['setting']
    if cfg['backbone'] == 'pi05':
        os.environ.update({
            'ROBOTWIN_PI05_CKPT': paths['checkpoint'], 'ROBOTWIN_PI05_NORM_DIR': paths['norm'],
            'ROBOTWIN_PI05_CONFIG': policy['config_name'],
            'ROBOTWIN_PI05_ACTION_HORIZON': str(policy['action_horizon']),
            'ROBOTWIN_PI05_DELTA_ACTIONS': str(int(policy['delta_actions'])),
            'ROBOTWIN_MAX_STEPS': str(policy['max_steps']),
            'ROBOTWIN_REPLAN_STEPS': str(policy['replan_steps']),
            'ROBOTWIN_OFFICIAL_PROTOCOL': '0', 'ROBOTWIN_PROMPT_MODE': 'description',
        })
    else:
        assets = {key: resolve_model_asset(paths[key], key)
                  for key in ('dit', 'vae', 'text_encoder', 'tokenizer')}
        os.environ.update({
            'STEERACT_FASTWAM_ROOT': paths['fastwam'],
            'FASTWAM_CKPT': paths['checkpoint'], 'FASTWAM_DATASET_STATS': paths['stats'],
            'FASTWAM_TASK_CONFIG': policy['setting'], 'FASTWAM_REPLAN_STEPS': str(policy['replan_steps']),
            'FASTWAM_INFERENCE_STEPS': str(policy['inference_steps']),
            'DIFFSYNTH_DIT_PATH': assets['dit'], 'DIFFSYNTH_VAE_PATH': assets['vae'],
            'DIFFSYNTH_TEXT_ENCODER_PATH': assets['text_encoder'],
            'DIFFSYNTH_TOKENIZER_PATH': assets['tokenizer'],
        })
    return importlib.import_module(f"steeract.backends.{cfg['backbone']}")


class DecisionHook:
    def __init__(self, callback, *, first_call_only=False):
        self.callback = callback
        self.first_call_only = first_call_only
        self.calls = 0
        self.decisions = 0
        self.active = False

    def begin_decision(self):
        if self.active:
            raise RuntimeError('nested policy decision')
        self.calls = 0
        self.active = True

    def __call__(self, module, inputs, hidden):
        if not self.active or (self.calls and not self.first_call_only):
            raise RuntimeError('configured context layer must run exactly once per decision')
        self.calls += 1
        if self.first_call_only and self.calls > 1:
            return hidden
        return self.callback(module, inputs, hidden)

    def end_decision(self):
        if not self.active or self.calls < 1 or (not self.first_call_only and self.calls != 1):
            raise RuntimeError('context activation missing at robot decision')
        self.active = False
        self.decisions += 1


def apply_step_limit(task_env, configured_max=None):
    factor = int(os.environ.get('STEERACT_MAX_STEPS_MULTIPLIER', '1'))
    if factor < 1:
        raise ValueError('max_steps_multiplier must be positive')
    task_env.step_lim = int(task_env.step_lim) * factor
    policy_cap = task_env.step_lim if configured_max in (None, 'native') else int(configured_max) * factor
    return min(task_env.step_lim, policy_cap)
