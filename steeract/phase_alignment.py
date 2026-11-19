from dataclasses import asdict, dataclass

import numpy as np

from .trajectory import TrajectoryBank, TrajectoryController, curve_vector


@dataclass(frozen=True)
class AlignmentSettings:
    max_advance: int = 2

    def __post_init__(self):
        if type(self.max_advance) is not int or self.max_advance < 1:
            raise ValueError('max_advance must be a positive integer')

    def to_dict(self):
        return asdict(self)


class AlignedTrajectoryBank(TrajectoryBank):
    def __init__(self, *args, alignment=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alignment = alignment or AlignmentSettings()
        self.reset_alignment()

    def reset_alignment(self):
        self.phase, self.observation = 0, 0

    def context(self, history):
        count = len(history)
        if count < 1:
            raise ValueError('empty observed history')
        width = min(count, self.settings.window, self.horizon)
        vector = curve_vector(history[-width:], self.scale)
        if count != self.observation:
            if count != self.observation+1:
                raise ValueError('phase matching requires contiguous observations; reset between episodes')
            if count <= min(self.settings.window, self.horizon):
                candidates = [count]
            else:
                candidates = range(max(width, self.phase),
                    min(self.horizon, self.phase+self.alignment.max_advance)+1)
            def pooled_distance(phase):
                success, failure, scales, _ = self.phases[phase]
                points = np.concatenate((success, failure))
                distance = np.linalg.norm((points-vector)/scales, axis=1).min()
                return float(distance), phase
            self.phase = min(candidates, key=pooled_distance)
            self.observation = count
        return self.phases[self.phase], vector, self.phase


class AlignedTrajectoryController(TrajectoryController):
    def step(self, *args, **kwargs):
        alpha = super().step(*args, **kwargs)
        self.trace[-1].update(controller='phase_aligned', reference_phase=self.bank.phase)
        return alpha
