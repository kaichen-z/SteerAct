from dataclasses import asdict, dataclass
from itertools import permutations

import numpy as np

PATTERNS = np.asarray(list(permutations(range(3))))


def descriptor(history, scale=1.0):
    raw = np.asarray(history, dtype=np.float64)
    if raw.ndim != 1 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("history must be a finite nonnegative vector")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("intensity scale must be positive and finite")
    values = np.log1p(raw / scale)
    vector, counts = np.zeros(18), np.zeros((2, 6), dtype=np.int64)
    for block, lag in enumerate((1, 2)):
        n = len(values) - 2 * lag
        if n <= 0:
            continue
        triples = np.stack([values[i * lag:i * lag + n] for i in range(3)], axis=1)
        order = np.argsort(triples, axis=1, kind="stable")
        ids = (order[:, None, :] == PATTERNS[None, :, :]).all(2).argmax(1)
        counts[block] = np.bincount(ids, minlength=6)
        p = counts[block] / n
        tied = ((triples[:, 0] == triples[:, 1]) | (triples[:, 0] == triples[:, 2])
                | (triples[:, 1] == triples[:, 2]))
        vector[block * 9:(block + 1) * 9] = [*p, entropy(p), tied.mean(), 1.0]
    return vector, counts


def entropy(p):
    return -(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum(axis=-1) / np.log(6)


def energy_score(success, failure):
    success, failure = np.asarray(success), np.asarray(failure)
    if min(len(success), len(failure)) < 2:
        raise ValueError("need at least two reference episodes per outcome")
    def within(x):
        distances = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        return distances.sum() / (len(x) * (len(x) - 1))
    return float(2 * np.linalg.norm(success[:, None] - failure[None, :], axis=-1).mean()
                 - within(success) - within(failure))


@dataclass(frozen=True)
class Settings:
    posterior_samples: int = 500
    support_mass: float = 0.65
    alpha_max: float = 10.0
    alpha_step: float = 0.5
    support_k: int = 3
    support_quantile: float = 0.95
    outcome_k: int = 3
    min_windows: int = 3
    support_open: int = 1
    support_close: int = 2
    failure_open: int = 2
    failure_close: int = 2
    min_effective_gain: float = 1e-3

    def __post_init__(self):
        for name in ("posterior_samples", "support_k", "outcome_k", "min_windows",
                     "support_open", "support_close", "failure_open", "failure_close"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.support_mass <= 1 or not 0 < self.support_quantile < 1:
            raise ValueError("invalid support probability/quantile")
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.alpha_max, self.alpha_step, self.min_effective_gain)):
            raise ValueError("coefficient bounds and gain floor must be finite and positive")
        if not np.isclose(self.alpha_max / self.alpha_step,
                          round(self.alpha_max / self.alpha_step)):
            raise ValueError("alpha_max must be a multiple of alpha_step so grid includes zero")

    def to_dict(self):
        return asdict(self)


class ReferenceBank:
    def __init__(self, success, failure, settings):
        self.success, self.failure = (np.array(x, dtype=float, copy=True) for x in (success, failure))
        for x in (self.success, self.failure):
            if x.ndim != 2 or x.shape[1] != 18 or len(x) < 2 or not np.isfinite(x).all():
                raise ValueError("each outcome bank must contain >=2 finite 18D descriptors")
        self.pooled = np.concatenate((self.success, self.failure))
        if settings.support_k >= len(self.pooled):
            raise ValueError("support_k must be less than the pooled reference count")
        self.scales = np.maximum(1e-8, np.sqrt(
            (self.success.var(0) + self.failure.var(0)) / 2
            + (self.success.mean(0) - self.failure.mean(0)) ** 2 / 4))
        distances = np.linalg.norm(self.pooled[:, None] - self.pooled[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        kth = np.partition(distances, settings.support_k - 1, axis=1)[:, settings.support_k - 1]
        self.radius = float(3 * np.quantile(kth, settings.support_quantile))
        self.k = min(settings.outcome_k, len(self.success), len(self.failure))
        for x in (self.success, self.failure, self.pooled, self.scales):
            x.flags.writeable = False

    def distance(self, vector, outcome="success"):
        points = self.success if outcome == "success" else self.failure
        distances = np.linalg.norm((points - vector) / self.scales, axis=1)
        return float(np.partition(distances, self.k - 1)[:self.k].mean())

    def support(self, vector, counts, settings, rng):
        samples = np.tile(vector, (settings.posterior_samples, 1))
        for block in range(2):
            p = rng.dirichlet(counts[block] + 0.5, settings.posterior_samples)
            samples[:, block * 9:block * 9 + 6] = p
            samples[:, block * 9 + 6] = entropy(p)
        distances = np.linalg.norm(samples[:, None] - self.pooled[None, :], axis=-1)
        return float(((distances <= self.radius).sum(1) >= settings.support_k).mean())


class Gate:
    def __init__(self, opening, closing):
        self.opening, self.closing = opening, closing
        self.reset()

    def reset(self):
        self.is_open, self.passes, self.fails = False, 0, 0

    def update(self, passed):
        self.passes = self.passes + 1 if passed else 0
        self.fails = 0 if passed else self.fails + 1
        if self.passes >= self.opening:
            self.is_open = True
        if self.fails >= self.closing:
            self.is_open = False
        return self.is_open


class Controller:
    def __init__(self, bank, *, scale=1.0, settings=None, seed=0):
        self.settings = settings or Settings()
        self.bank, self.scale = bank, scale
        self.reset(seed)

    def reset(self, seed=0):
        cfg = self.settings
        self.rng = np.random.default_rng(seed)
        self.history, self.trace = [], []
        self.support_gate = Gate(cfg.support_open, cfg.support_close)
        self.failure_gate = Gate(cfg.failure_open, cfg.failure_close)

    def step(self, intensity, gain):
        cfg = self.settings
        self.history.append(float(intensity))
        vector, counts = descriptor(self.history, self.scale)
        mass = self.bank.support(vector, counts, cfg, self.rng)
        self.support_gate.update(mass >= cfg.support_mass and (counts.sum(1) >= cfg.min_windows).all())
        ds, df = self.bank.distance(vector), self.bank.distance(vector, "failure")
        if self.support_gate.is_open:
            self.failure_gate.update(df < ds)
        else:
            self.failure_gate.reset()
        alpha, best = 0.0, ds
        if self.support_gate.is_open and self.failure_gate.is_open and np.isfinite(gain) and abs(gain) >= cfg.min_effective_gain:
            half = int(round(cfg.alpha_max / cfg.alpha_step))
            grid = sorted(np.arange(-half, half + 1) * cfg.alpha_step, key=lambda a: (abs(a), a))
            for candidate in grid:
                predicted = max(0.0, intensity + candidate * gain)
                candidate_vector, _ = descriptor([*self.history[:-1], predicted], self.scale)
                distance = self.bank.distance(candidate_vector)
                if distance < best:
                    alpha, best = float(candidate), distance
        self.trace.append(dict(decision=len(self.history), support_mass=mass,
                               support_open=self.support_gate.is_open, failure_open=self.failure_gate.is_open,
                               d_success=ds, d_failure=df, alpha=alpha, predicted_progress=ds-best,
                               intensity_before=float(intensity), effective_gain=float(gain)))
        return alpha

    def feedback(self, intensity):
        if not self.history or not np.isfinite(intensity) or intensity < 0:
            raise ValueError("feedback requires a decision and a finite nonnegative intensity")
        self.history[-1] = float(intensity)
        self.trace[-1]["intensity_after"] = float(intensity)


def make_hook(sae, group, settings=None, seed=0):
    import torch
    cfg = settings or Settings()
    bank = ReferenceBank(group["success"], group["failure"], cfg)
    controller = Controller(bank, scale=group["intensity_scale"], settings=cfg, seed=seed)
    fid = int(group["feature_id"])
    if not 0 <= fid < sae.decoder.weight.shape[1]:
        raise ValueError("feature outside SAE dictionary")
    sae.eval()
    direction = sae.decoder.weight[:, fid].detach()
    self_gain = float(group["analytic_self_gain"])

    @torch.inference_mode()
    def measure(hidden):
        weight = sae.encoder.weight
        codes = sae.encode(hidden.reshape(-1, hidden.shape[-1]).to(device=weight.device, dtype=weight.dtype))[:, fid]
        if not torch.isfinite(codes).all():
            raise ValueError("nonfinite SAE activation")
        return float(codes.clamp_min(0).mean()), float((codes > 0).float().mean())

    @torch.inference_mode()
    def hook(_module, _inputs, hidden):
        if hidden.ndim not in (2, 3) or (hidden.ndim == 3 and hidden.shape[0] != 1):
            raise ValueError("expected [tokens, hidden] or [1, tokens, hidden]")
        if hidden.numel() == 0 or not torch.isfinite(hidden).all():
            raise ValueError("empty or nonfinite hidden state")
        intensity, active_fraction = measure(hidden)
        alpha = controller.step(intensity, active_fraction * self_gain)
        edited = hidden if alpha == 0 else hidden + alpha * direction.to(hidden)
        after = intensity if alpha == 0 else measure(edited)[0]
        controller.feedback(after)
        return edited

    hook.controller = controller
    return hook
