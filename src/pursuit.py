from __future__ import annotations

import numpy as np

from .integrators import Event
from .systems import FirstOrderSystem

__all__ = ["PurePursuit", "MovingTargetPursuit", "capture_event"]

Array = np.ndarray


class PurePursuit(FirstOrderSystem):

    name = "pure pursuit"

    def __init__(self, speed: float = 1.0):
        if speed <= 0:
            raise ValueError("pursuit speed must be positive")
        self.speed = float(speed)

    def __repr__(self) -> str:
        return f"PurePursuit(speed={self.speed:g})"

    def rhs(self, t: float, y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        rho = np.linalg.norm(y, axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -self.speed * y / rho

    def separation(self, t: float, y: Array) -> float:
        r""":math:`\rho = \|r\|`; the target sits at the origin."""
        return float(np.linalg.norm(np.asarray(y, dtype=float)))

    def capture_time(self, y0: Array) -> float:
        r"""Exact capture time :math:`T = \rho_0/v`."""
        return float(np.linalg.norm(y0) / self.speed)

    def exact_radius(self, t: Array | float, y0: Array) -> Array | float:
        r"""Exact :math:`\rho(t) = \rho_0 - vt`, clipped at capture."""
        rho0 = float(np.linalg.norm(y0))
        return np.maximum(rho0 - self.speed * np.asarray(t, dtype=float), 0.0)

    def exact_solution(self, t: Array | float, y0: Array) -> Array:
        r"""Exact trajectory: a straight line inward along :math:`\hat r_0`."""
        y0 = np.asarray(y0, dtype=float)
        rho0 = float(np.linalg.norm(y0))
        rho = self.exact_radius(t, y0)
        return np.asarray(rho)[..., None] * (y0 / rho0)


class MovingTargetPursuit(FirstOrderSystem):

    name = "moving-target pursuit"

    def __init__(
        self,
        speed: float = 1.0,
        target_velocity: Array = (0.0, 0.5),
        target_position: Array = (0.0, 0.0),
    ):
        if speed <= 0:
            raise ValueError("pursuit speed must be positive")
        self.speed = float(speed)
        self.target_velocity = np.asarray(target_velocity, dtype=float)
        self.target_position = np.asarray(target_position, dtype=float)
        self.target_speed = float(np.linalg.norm(self.target_velocity))

    def __repr__(self) -> str:
        return (
            f"MovingTargetPursuit(speed={self.speed:g}, "
            f"target_speed={self.target_speed:g})"
        )

    def target(self, t: float) -> Array:
        return self.target_position + self.target_velocity * t

    def rhs(self, t: float, y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        d = self.target(t) - y
        rho = np.linalg.norm(d, axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.speed * d / rho

    def separation(self, t: float, y: Array) -> float:
        return float(np.linalg.norm(self.target(t) - np.asarray(y, dtype=float)))

    def invariant(self, t: float, y: Array) -> float:
        r""":math:`I = v\rho + wq`, which must equal :math:`I_0 + (w^2 - v^2)t`."""
        d = self.target(t) - np.asarray(y, dtype=float)
        rho = float(np.linalg.norm(d))
        if self.target_speed == 0.0:
            return self.speed * rho
        q = float(np.dot(self.target_velocity / self.target_speed, d))
        return self.speed * rho + self.target_speed * q

    def capture_time(self, y0: Array) -> float:
        """Exact capture time; ``inf`` if the target is not slower than the pursuer."""
        if self.target_speed >= self.speed:
            return float("inf")
        return float(self.invariant(0.0, y0) / (self.speed**2 - self.target_speed**2))

    def capture_point(self, y0: Array) -> Array:
        """Position at which capture occurs (on the target's straight-line path)."""
        return self.target(self.capture_time(y0))


def capture_event(system, rho_capture: float = 1e-3, name: str = "capture") -> Event:

    def g(t: float, y: Array) -> float:
        return system.separation(t, y) - rho_capture

    return Event(g, direction=-1.0, terminal=True, name=name)