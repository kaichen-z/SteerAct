from pathlib import Path
import json

import numpy as np
import torch
from torch import nn

from .config import collection_identity, digest
from .data import ActivationRows, atomic_json, records


class TopKSAE(nn.Module):
    def __init__(self, hidden_dim, dict_size, top_k):
        super().__init__()
        if not 1 <= top_k <= dict_size:
            raise ValueError('sae.top_k must be within dictionary size')
        self.encoder = nn.Linear(hidden_dim, dict_size)
        self.decoder = nn.Linear(dict_size, hidden_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(hidden_dim))
        self.register_buffer('k', torch.tensor(top_k, dtype=torch.int32))
        self.register_buffer('threshold', torch.tensor(-1., dtype=torch.float32))
        with torch.no_grad():
            self.normalize_decoder()
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()

    def encode(self, x):
        dense = torch.relu(self.encoder(x - self.b_dec))
        values, indices = dense.topk(int(self.k), dim=-1, sorted=False)
        return torch.zeros_like(dense).scatter_(-1, indices, values)

    def forward(self, x):
        return self.decoder(self.encode(x)) + self.b_dec

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))


def load_sae(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = TopKSAE(config['hidden_dim'], config['dict_size'], config['k'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint


def train(cfg, device):
    root, episodes = records(cfg)
    options = cfg['sae']
    for key in ('steps', 'batch_size', 'expansion_ratio', 'top_k'):
        if options[key] < 1:
            raise ValueError(f'sae.{key} must be positive')
    if options['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    data = ActivationRows(root, episodes)
    torch.manual_seed(options['seed'])
    rng = np.random.default_rng(options['seed'])
    dictionary = data.dimension * options['expansion_ratio']
    model = TopKSAE(data.dimension, dictionary, options['top_k']).to(device)
    with torch.no_grad():
        model.b_dec.copy_(data.batch(options['batch_size'], rng).mean(0).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'])
    output = Path(cfg['paths']['run_dir']) / 'sae'
    output.mkdir(parents=True, exist_ok=True)
    losses = []
    model.train()
    for step in range(options['steps']):
        hidden = data.batch(options['batch_size'], rng).to(device)
        loss = (model(hidden) - hidden).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite SAE loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            weights = model.decoder.weight
            grad = weights.grad
            grad.sub_(weights * (weights * grad).sum(0, keepdim=True))
        optimizer.step()
        model.normalize_decoder()
        if step == 0 or (step + 1) % options['log_every'] == 0 or step + 1 == options['steps']:
            losses.append(dict(step=step+1, mse=float(loss.detach())))
            print(f'SAE step {step+1}/{options["steps"]}: mse={losses[-1]["mse"]:.6g}', flush=True)
    checkpoint = dict(schema='steeract.sae.v1',
                      config=dict(hidden_dim=data.dimension, dict_size=dictionary, k=options['top_k']),
                      state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                      collection_identity=digest(collection_identity(cfg)), training=options,
                      tasks=cfg['tasks'], source_episodes=[dict(task=r['task'], seed=r['seed']) for r in episodes])
    temp = output / 'model.tmp.pt'
    torch.save(checkpoint, temp)
    temp.replace(output / 'model.pt')
    atomic_json(output / 'training.json', dict(options=options, metrics=losses,
                                             episodes=len(episodes), token_rows=int(data.ends[-1])))
    return output / 'model.pt'
