from dataclasses import asdict, dataclass

import numpy as np

from .method import Settings


@dataclass(frozen=True)
class TrajectorySettings:
    window: int = 3
    neighbors: int = 1
    coordinate_floor: float = .1
    support_quantile: float = .95
    support_multiplier: float = 3.
    support_floor: float = .5
    refinement_steps: int = 0

    def __post_init__(self):
        if type(self.refinement_steps) is not int or self.refinement_steps < 0:
            raise ValueError('refinement_steps must be a nonnegative integer')
        for key in ('window', 'neighbors'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not 0 < self.support_quantile < 1:
            raise ValueError('invalid support quantile')
        for key in ('coordinate_floor', 'support_multiplier', 'support_floor'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')

    def to_dict(self):
        return asdict(self)


def curve_vector(values, scale):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('expected a nonempty finite nonnegative trajectory')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('trajectory scale must be finite and positive')
    levels = np.log1p(values/scale)
    return np.concatenate((levels, np.diff(levels)))


def coordinate_scales(success, failure, floor):
    return np.maximum(floor, np.sqrt((success.var(0)+failure.var(0))/2
                      + (success.mean(0)-failure.mean(0))**2/4))


class TrajectoryBank:
    def __init__(self, success, failure, *, scale=1., settings=None):
        self.settings = settings or TrajectorySettings()
        self.scale = scale
        if min(len(success), len(failure)) < 2:
            raise ValueError('need at least two reference curves per outcome')
        curves = [np.array(h, dtype=float, copy=True) for h in [*success, *failure]]
        for h in curves:
            curve_vector(h, scale)
        self.horizon = min(map(len, curves))
        self.phases = {}
        count = len(success)
        self.k = min(self.settings.neighbors, len(success), len(failure))
        for phase in range(1, self.horizon+1):
            width = min(phase, self.settings.window)
            points = np.stack([curve_vector(h[phase-width:phase], scale) for h in curves])
            positive, negative = points[:count], points[count:]
            scales = coordinate_scales(positive, negative, self.settings.coordinate_floor)
            distances = np.linalg.norm((points[:, None]-points[None, :])/scales, axis=-1)
            np.fill_diagonal(distances, np.inf)
            radius = max(self.settings.support_floor, self.settings.support_multiplier *
                         np.quantile(distances.min(1), self.settings.support_quantile))
            self.phases[phase] = (positive, negative, scales, radius)

    def context(self, history):
        phase = min(len(history), self.horizon)
        if phase < 1:
            raise ValueError('empty observed history')
        width = min(phase, self.settings.window)
        return self.phases[phase], curve_vector(history[-width:], self.scale), phase

    def distance(self, history, outcome='success'):
        (success, failure, scales, _), vector, _ = self.context(history)
        points = success if outcome == 'success' else failure
        return float(np.sort(np.linalg.norm((points-vector)/scales, axis=1))[:self.k].mean())

    def supported(self, history):
        (success, failure, scales, radius), vector, _ = self.context(history)
        distance = np.linalg.norm((np.concatenate((success, failure))-vector)/scales, axis=1).min()
        return bool(distance <= radius)


class TrajectoryController:
    def __init__(self, bank, *, alpha_settings=None):
        self.bank = bank
        self.settings = alpha_settings or Settings(response_model='measured')
        if self.settings.response_model != 'measured':
            raise ValueError('trajectory controller requires measured response')
        self.history, self.trace = [], []

    def step(self, intensity, gain, *, predict_intensity=None):
        if predict_intensity is None:
            raise ValueError('trajectory controller needs a real candidate predictor')
        curve_vector([intensity], self.bank.scale)
        self.history.append(float(intensity))
        ds, df = self.bank.distance(self.history), self.bank.distance(self.history, 'failure')
        support = self.bank.supported(self.history)
        alpha, best = 0., ds
        if support and df < ds:
            half = int(round(self.settings.alpha_max/self.settings.alpha_step))
            for candidate in sorted(np.arange(-half, half+1)*self.settings.alpha_step,
                                    key=lambda a: (abs(a), a)):
                predicted = float(predict_intensity(float(candidate)))
                curve_vector([predicted], self.bank.scale)
                distance = self.bank.distance([*self.history[:-1], predicted])
                if distance < best:
                    alpha, best = float(candidate), distance
            for depth in range(self.bank.settings.refinement_steps):
                spacing = self.settings.alpha_step/(2**(depth+1))
                candidates = sorted({float(np.clip(alpha+offset*spacing,
                    -self.settings.alpha_max, self.settings.alpha_max)) for offset in (-1, 1)},
                    key=lambda a: (abs(a), a))
                for candidate in candidates:
                    predicted = float(predict_intensity(candidate))
                    curve_vector([predicted], self.bank.scale)
                    distance = self.bank.distance([*self.history[:-1], predicted])
                    if distance < best:
                        alpha, best = candidate, distance
        self.trace.append(dict(controller='trajectory', decision=len(self.history),
            reference_phase=min(len(self.history), self.bank.horizon), support_mass=float(support),
            support_open=support, failure_open=bool(df < ds), d_success=ds, d_failure=df,
            alpha=alpha, predicted_progress=ds-best, intensity_before=float(intensity),
            effective_gain=float(gain)))
        return alpha

    def feedback(self, intensity):
        curve_vector([intensity], self.bank.scale)
        if not self.history:
            raise ValueError('feedback before a decision')
        self.history[-1] = float(intensity)
        self.trace[-1]['intensity_after'] = float(intensity)
        self.trace[-1]['measured_progress'] = self.trace[-1]['d_success']-self.bank.distance(self.history)
