"""The five target-motion generator families (rl_training_guide.md 16.3).

One family per evaluation scenario of the source paper [P Table 4], with
randomised amplitude/frequency/speed and a hard speed cap: both aircraft are
Tellos, so an uncatchable target measures nothing
(drl_ros2_reference_analysis.md 6.1). The cap is a *parameter of every
result* and the curriculum's knob (guide section 20).

Every generator is seeded from the environment's numpy Generator and world-
space (metres); the environment projects. `speed_cap` bounds the commanded
speed instantaneously, not just on average.
"""
import math
from typing import Dict, Type

import numpy as np

FAMILIES = ('static', 'constant_velocity', 'vertical_oscillation',
            'horizontal_oscillation', 'aggressive')


class TargetGenerator:
    """Base: holds position, advances it; subclasses shape the motion."""

    def __init__(self, rng: np.random.Generator, origin: np.ndarray,
                 speed_cap: float):
        self.rng = rng
        self.pos = np.asarray(origin, dtype=float).copy()
        self.speed_cap = float(speed_cap)
        self.t = 0.0

    def advance(self, dt: float) -> np.ndarray:
        vel = self._velocity(dt)
        speed = float(np.linalg.norm(vel))
        if speed > self.speed_cap > 0.0:
            vel = vel * (self.speed_cap / speed)
        elif self.speed_cap == 0.0:
            vel = np.zeros(3)
        self.pos += vel * dt
        self.t += dt
        return self.pos

    def _velocity(self, dt: float) -> np.ndarray:
        raise NotImplementedError


class Static(TargetGenerator):
    def _velocity(self, dt):
        return np.zeros(3)


class ConstantVelocity(TargetGenerator):
    """Mostly-horizontal straight line at a fixed random speed."""

    def __init__(self, rng, origin, speed_cap):
        super().__init__(rng, origin, speed_cap)
        heading = rng.uniform(0.0, 2.0 * math.pi)
        climb = rng.uniform(-0.15, 0.15)
        speed = rng.uniform(0.3, 1.0) * max(speed_cap, 1e-6)
        self._vel = speed * np.array(
            [math.cos(heading), math.sin(heading), climb])

    def _velocity(self, dt):
        return self._vel


class _Oscillation(TargetGenerator):
    """Sinusoid on one axis plus a slow drift -- the paper's scenarios 3/4."""

    AXIS = 2  # overridden

    def __init__(self, rng, origin, speed_cap):
        super().__init__(rng, origin, speed_cap)
        self.amp = rng.uniform(0.2, 0.8)                    # m
        self.freq = rng.uniform(0.1, 0.5)                   # Hz
        self.phase = rng.uniform(0.0, 2.0 * math.pi)
        heading = rng.uniform(0.0, 2.0 * math.pi)
        drift_speed = rng.uniform(0.0, 0.3) * max(speed_cap, 1e-6)
        self._drift = drift_speed * np.array(
            [math.cos(heading), math.sin(heading), 0.0])

    def _velocity(self, dt):
        # d/dt of amp*sin(2*pi*f*t + phase) on the oscillation axis.
        omega = 2.0 * math.pi * self.freq
        osc = self.amp * omega * math.cos(omega * self.t + self.phase)
        vel = self._drift.copy()
        vel[self.AXIS] += osc
        return vel


class VerticalOscillation(_Oscillation):
    AXIS = 2


class HorizontalOscillation(_Oscillation):
    """Oscillates on world-y: perpendicular to the follower's initial line of
    sight, which the environment aligns with world-x at reset."""
    AXIS = 1


class Aggressive(TargetGenerator):
    """Variable high-speed motion: sum of incommensurate sinusoids on all
    axes plus random velocity kicks with bounded acceleration -- the paper's
    scenario 5, minus the built-in uncatchability."""

    def __init__(self, rng, origin, speed_cap):
        super().__init__(rng, origin, speed_cap)
        self.freqs = rng.uniform(0.1, 0.7, size=3)
        self.amps = rng.uniform(0.2, 0.7, size=3)
        self.phases = rng.uniform(0.0, 2.0 * math.pi, size=3)
        self._kick = np.zeros(3)
        self._kick_timer = 0.0

    def _velocity(self, dt):
        omega = 2.0 * math.pi * self.freqs
        osc = self.amps * omega * np.cos(omega * self.t + self.phases)
        self._kick_timer -= dt
        if self._kick_timer <= 0.0:
            self._kick_timer = float(self.rng.uniform(0.5, 2.0))
            direction = self.rng.normal(size=3)
            direction /= max(np.linalg.norm(direction), 1e-9)
            self._kick = direction * self.rng.uniform(
                0.0, max(self.speed_cap, 1e-6))
        return osc + self._kick


GENERATORS: Dict[str, Type[TargetGenerator]] = {
    'static': Static,
    'constant_velocity': ConstantVelocity,
    'vertical_oscillation': VerticalOscillation,
    'horizontal_oscillation': HorizontalOscillation,
    'aggressive': Aggressive,
}


def make(family: str, rng: np.random.Generator, origin, speed_cap: float
         ) -> TargetGenerator:
    try:
        cls = GENERATORS[family]
    except KeyError:
        raise ValueError(
            f'unknown target family {family!r}; one of {FAMILIES}') from None
    return cls(rng, np.asarray(origin, dtype=float), speed_cap)
