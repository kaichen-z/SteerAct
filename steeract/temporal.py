from collections import deque

import numpy as np

from .method import Controller, Gate, descriptor


class TemporalTarget:
    def __init__(self, references, horizon, window=8, neighbors=3, weight=0.5):
        if (not references or not 1 <= horizon <= max(len(activity) for _, activity in references)
                or any(not np.isfinite(activity).all() or np.any(np.asarray(activity) < 0)
                       for _, activity in references)):
            raise ValueError('invalid successful reference histories')
        self.window = window
        self.weight = weight
        self.neighbors = neighbors
        self.steps = {}
        self.means = []
        for end in range(1, horizon + 1):
            available = [(seed, activity) for seed, activity in references if len(activity) >= end]
            self.means.append(float(np.mean([activity[end - 1] for _, activity in available])))
            if end < 5 or len(available) <= neighbors:
                continue
            vectors = np.stack([descriptor(activity[max(0, end - window):end])[0]
                                for _, activity in available])
            if len(np.unique(vectors, axis=0)) < 2:
                continue
            distances = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1)
            np.fill_diagonal(distances, np.inf)
            radii = np.partition(distances, neighbors - 1, axis=1)[:, neighbors - 1]
            self.steps[end] = available, vectors, float(np.quantile(radii, 0.9))

    def match(self, history, step):
        if step > len(self.means):
            return None, []
        mean = self.means[step - 1]
        if step not in self.steps:
            return mean, []
        available, vectors, radius = self.steps[step]
        distances = np.linalg.norm(vectors - descriptor(history)[0], axis=1)
        boundary = float(np.partition(distances, self.neighbors - 1)[self.neighbors - 1])
        picked = np.flatnonzero(distances <= boundary + 1e-12)
        if boundary > radius + 1e-12 or len(picked) == len(available):
            return mean, []
        matched = float(np.mean([available[i][1][step - 1] for i in picked]))
        return (1 - self.weight) * mean + self.weight * matched, [available[i][0] for i in picked]


def inverse_alpha(intensity, target, gain, settings):
    if target is None or not np.isfinite(gain) or abs(gain) < settings.min_effective_gain:
        return 0.0
    raw = (target - intensity) / gain
    return float(np.clip(raw, -settings.alpha_max, settings.alpha_max)) if np.isfinite(raw) else 0.0


class TemporalController(Controller):
    def __init__(self, *args, target, editing, **kwargs):
        self.target = target
        self.editing = editing
        super().__init__(*args, **kwargs)

    def reset(self, seed=0):
        super().reset(seed)
        self.recent = deque(maxlen=self.target.window)
        self.support_gate.is_open = self.editing

    def step(self, intensity, gain, *, predict_intensity=None):
        cfg = self.settings
        mass, ready, prefix_ready, ds, df = self.observe(intensity)
        step = len(self.history)
        self.recent.append(float(intensity))
        target, matched = self.target.match(self.recent, step) if self.editing else (None, [])
        if self.editing and ready:
            self.support_gate.update(mass >= 0.95 * cfg.support_mass)
        supported = self.editing and target is not None and self.support_gate.is_open
        alpha = inverse_alpha(intensity, target, gain, cfg) if supported else 0.0
        progress = 0.0
        if alpha:
            predicted = max(0.0, intensity + gain * alpha)
            vector, _ = descriptor([*self.history[:-1], predicted], self.scale)
            progress = ds - self.bank.distance(vector)
        self.trace.append(dict(decision=step, support_mass=mass, support_open=supported,
                               failure_open=self.editing, d_success=ds, d_failure=df,
                               alpha=alpha, predicted_progress=progress,
                               intensity_before=float(intensity), effective_gain=float(gain),
                               success_target=target, temporal_descriptor_used=bool(matched),
                               matched_reference_seeds=matched))
        if cfg.reference_mode == 'prefix':
            self.trace[-1]['matched_prefix'] = step if prefix_ready else None
        self.ready = ready and prefix_ready
        return alpha


class EpisodeGate:
    def __init__(self, layers, settings):
        self.settings = settings
        self.support = Gate(settings.support_open, settings.support_close)
        self.failures = {layer: Gate(settings.failure_open, settings.failure_close) for layer in layers}
        self.pending = {}

    def update(self, layer, controller):
        self.pending[layer] = controller
        if len(self.pending) < len(self.failures):
            return False
        mass = max(c.trace[-1]['support_mass'] for c in self.pending.values())
        ready = any(c.ready for c in self.pending.values())
        opened = self.support.update(ready and mass >= self.settings.support_mass)
        triggered = False
        for layer, controller in self.pending.items():
            row = controller.trace[-1]
            gate = self.failures[layer]
            if opened:
                gate.update(row['d_failure'] < row['d_success'])
            else:
                gate.reset()
            row.update(support_open=opened, failure_open=gate.is_open)
            triggered |= gate.is_open
        self.pending.clear()
        return triggered
