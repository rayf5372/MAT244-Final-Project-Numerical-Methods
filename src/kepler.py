from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .systems import make_state, split_state

__all__ = ["KeplerOrbit", "elements_from_state", "radial_infall_time"]

Array = np.ndarray


@dataclass
class KeplerOrbit:
    mu: float
    a: float
    e: float

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError("mu must be positive")
        if self.a <= 0:
            raise ValueError("KeplerOrbit describes bound orbits only (a > 0)")
        if not 0.0 <= self.e < 1.0:
            raise ValueError("require 0 <= e < 1 for a bound orbit")

    # ------------------------------------------------------------- properties
    @property
    def period(self) -> float:
        r"""Orbital period :math:`T = 2\pi\sqrt{a^3/\mu}`."""
        return float(2.0 * np.pi * np.sqrt(self.a**3 / self.mu))

    @property
    def mean_motion(self) -> float:
        r""":math:`n = \sqrt{\mu/a^3}`."""
        return float(np.sqrt(self.mu / self.a**3))

    @property
    def rho_min(self) -> float:
        """Pericentre distance."""
        return float(self.a * (1.0 - self.e))

    @property
    def rho_max(self) -> float:
        """Apocentre distance."""
        return float(self.a * (1.0 + self.e))

    @property
    def energy(self) -> float:
        r"""Specific energy :math:`E = -\mu/(2a)`."""
        return float(-self.mu / (2.0 * self.a))

    @property
    def angular_momentum(self) -> float:
        r""":math:`L = \sqrt{\mu a(1-e^2)}` (counter-clockwise)."""
        return float(np.sqrt(self.mu * self.a * (1.0 - self.e**2)))

    # -------------------------------------------------------------- solution
    def initial_state(self) -> Array:
        rp = self.rho_min
        return make_state([rp, 0.0], [0.0, self.angular_momentum / rp])

    def eccentric_anomaly(self, t: Array | float, tol: float = 1e-14, max_iter: int = 60) -> Array:
        M = np.atleast_1d(self.mean_motion * np.asarray(t, dtype=float))
        M = np.mod(M + np.pi, 2.0 * np.pi) - np.pi
        E = M + self.e * np.sin(M)
        for _ in range(max_iter):
            f = E - self.e * np.sin(E) - M
            fp = 1.0 - self.e * np.cos(E)
            dE = -f / fp
            E = E + dE
            if np.max(np.abs(dE)) < tol:
                break
        else:  # pragma: no cover - only for pathological e
            raise RuntimeError("Kepler equation did not converge")
        return E

    def state(self, t: Array | float) -> Array:
        E = self.eccentric_anomaly(t)
        a, e, n = self.a, self.e, self.mean_motion
        b = a * np.sqrt(1.0 - e**2)
        one_minus = 1.0 - e * np.cos(E)
        x = a * (np.cos(E) - e)
        y = b * np.sin(E)
        vx = -a * n * np.sin(E) / one_minus
        vy = b * n * np.cos(E) / one_minus
        out = np.stack([x, y, vx, vy], axis=-1)
        return out[0] if np.ndim(t) == 0 else out

    def position(self, t: Array | float) -> Array:
        return self.state(t)[..., :2]

    def radius(self, t: Array | float) -> Array:
        r"""Exact :math:`\rho(t) = a(1 - e\cos E)`."""
        E = self.eccentric_anomaly(t)
        rho = self.a * (1.0 - self.e * np.cos(E))
        return float(rho[0]) if np.ndim(t) == 0 else rho

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_state(cls, y: Array, mu: float) -> "KeplerOrbit":
        """Recover the orbit containing a given state."""
        a, e, _ = elements_from_state(y, mu)
        return cls(mu=mu, a=a, e=e)


def radial_infall_time(mu: float, rho0: float, rho: Array | float = 0.0) -> Array | float:
    if mu <= 0 or rho0 <= 0:
        raise ValueError("mu and rho0 must be positive")
    rho = np.asarray(rho, dtype=float)
    if np.any(rho < 0) or np.any(rho > rho0):
        raise ValueError("require 0 <= rho <= rho0")
    eta = np.arccos(np.clip(2.0 * rho / rho0 - 1.0, -1.0, 1.0))
    t = np.sqrt(rho0**3 / (8.0 * mu)) * (eta + np.sin(eta))
    return float(t) if t.ndim == 0 else t


def elements_from_state(y: Array, mu: float) -> tuple[float, float, float]:
    x, v = split_state(np.asarray(y, dtype=float))
    rho = float(np.linalg.norm(x))
    L = float(x[0] * v[1] - x[1] * v[0])
    E = 0.5 * float(np.dot(v, v)) - mu / rho
    a = float("inf") if E == 0.0 else -mu / (2.0 * E)
    e = float(np.sqrt(max(0.0, 1.0 + 2.0 * E * L**2 / mu**2)))
    return a, e, L