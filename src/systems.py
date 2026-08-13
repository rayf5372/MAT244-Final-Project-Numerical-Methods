from __future__ import annotations

import numpy as np

__all__ = [
    "FirstOrderSystem",
    "SecondOrderSystem",
    "PowerLawCentralForce",
    "split_state",
    "make_state",
]

Array = np.ndarray


def split_state(y: Array) -> tuple[Array, Array]:
    """Split a second-order state ``(x, y, vx, vy)`` into position and velocity."""
    y = np.asarray(y, dtype=float)
    return y[..., :2], y[..., 2:]


def make_state(position: Array, velocity: Array) -> Array:
    """Assemble a second-order state from position and velocity vectors."""
    return np.concatenate([np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)])


class FirstOrderSystem:
    """A law of the form :math:`\\dot r = f(t, r)`; state dimension 2."""

    order = 1
    state_dim = 2

    def rhs(self, t: float, y: Array) -> Array:  # pragma: no cover - abstract
        raise NotImplementedError

    def angular_momentum(self, t: float, y: Array) -> float | Array:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            rdot = np.asarray(self.rhs(t, y), dtype=float)
            return float(y[0] * rdot[1] - y[1] * rdot[0])
        t_arr = np.broadcast_to(np.asarray(t, dtype=float), y.shape[:-1])
        out = np.empty(y.shape[:-1])
        for i in np.ndindex(*y.shape[:-1]):
            rdot = np.asarray(self.rhs(float(t_arr[i]), y[i]), dtype=float)
            out[i] = y[i][0] * rdot[1] - y[i][1] * rdot[0]
        return out


class SecondOrderSystem:
    """A law of the form :math:`\\ddot r = a(t, r)`; state dimension 4."""

    order = 2
    state_dim = 4

    def accel(self, t: float, x: Array) -> Array:  # pragma: no cover - abstract
        raise NotImplementedError

    def rhs(self, t: float, y: Array) -> Array:
        """First-order form of the second-order law, for Euler/RK4/``solve_ivp``."""
        x, v = split_state(y)
        return np.concatenate([v, np.atleast_1d(self.accel(t, x))], axis=-1)

    def angular_momentum(self, t: float, y: Array) -> float | Array:
        r""":math:`L = x v_y - y v_x`, conserved by any central law."""
        x, v = split_state(y)
        L = x[..., 0] * v[..., 1] - x[..., 1] * v[..., 0]
        return float(L) if np.ndim(L) == 0 else L


class PowerLawCentralForce(SecondOrderSystem):
    def __init__(self, s: float, k: float = 1.0):
        if k <= 0:
            raise ValueError(f"force strength k must be positive (attractive), got {k}")
        self.s = float(s)
        self.k = float(k)

    # ------------------------------------------------------------------ setup
    @classmethod
    def gravity(cls, mu: float = 1.0) -> "PowerLawCentralForce":
        r"""Model 2: :math:`\ddot r = -\mu r/\rho^3` with :math:`\mu = G(m_1+m_2)`."""
        return cls(s=1.0, k=mu)

    @classmethod
    def hooke(cls, k: float = 1.0) -> "PowerLawCentralForce":
        r"""Hooke's law :math:`\ddot r = -k r`, the other Bertrand exponent."""
        return cls(s=-2.0, k=k)

    @classmethod
    def from_potential(cls, alpha: float, s: float) -> "PowerLawCentralForce":
        r"""Build from the proposal's :math:`V(\rho) = -\alpha\rho^{-s}`.

        Requires :math:`\alpha s > 0` so that the force is attractive.
        """
        if s == 0:
            raise ValueError("s = 0 is the logarithmic potential; use PowerLawCentralForce(0, k)")
        if alpha * s <= 0:
            raise ValueError(
                f"V = -alpha*rho^-s is attractive only when alpha*s > 0 (got alpha={alpha}, s={s})"
            )
        return cls(s=s, k=alpha * s)

    @property
    def alpha(self) -> float:
        r"""The proposal's :math:`\alpha = k/s` (undefined for ``s = 0``)."""
        if self.s == 0:
            return float("nan")
        return self.k / self.s

    def __repr__(self) -> str:
        return f"PowerLawCentralForce(s={self.s:g}, k={self.k:g})"

    @property
    def name(self) -> str:
        special = {1.0: "gravity", -2.0: "Hooke", 0.0: "log", 2.0: "borderline"}
        tag = special.get(self.s)
        return f"s={self.s:g}" + (f" ({tag})" if tag else "")

    # ------------------------------------------------------------- dynamics
    def accel(self, t: float, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        rho = np.linalg.norm(x, axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return -self.k * x / rho ** (self.s + 2.0)

    # --------------------------------------------------- conserved quantities
    def potential(self, rho: Array | float) -> Array | float:
        r""":math:`V(\rho)`."""
        rho = np.asarray(rho, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if self.s == 0.0:
                out = self.k * np.log(rho)
            else:
                out = -(self.k / self.s) * rho ** (-self.s)
        return float(out) if out.ndim == 0 else out

    def potential_at_infinity(self) -> float:
        r""":math:`V(\infty)`: finite (``0``) only for ``s > 0``; escape is
        impossible otherwise because the well never levels off."""
        return 0.0 if self.s > 0 else float("inf")

    def energy(self, y: Array) -> float | Array:
        r"""Specific energy :math:`E = \tfrac12\|\dot r\|^2 + V(\rho)`."""
        x, v = split_state(y)
        rho = np.linalg.norm(x, axis=-1)
        E = 0.5 * np.sum(v * v, axis=-1) + self.potential(rho)
        return float(E) if np.ndim(E) == 0 else E

    def v_eff(self, rho: Array | float, L: float) -> Array | float:
        r"""Effective potential :math:`V_{\rm eff}(\rho) = V(\rho) + L^2/(2\rho^2)`.

        The second term is the centrifugal barrier: it always diverges as
        :math:`+\rho^{-2}`, so whether collision is possible at :math:`L \neq 0`
        is decided by whether :math:`V` diverges faster, i.e. by ``s`` vs ``2``.
        """
        rho = np.asarray(rho, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = self.potential(rho) + L**2 / (2.0 * rho**2)
        return float(out) if np.ndim(out) == 0 else out

    def dv_eff(self, rho: Array | float, L: float) -> Array | float:
        r""":math:`V_{\rm eff}'(\rho) = k\rho^{-(s+1)} - L^2/\rho^3`."""
        rho = np.asarray(rho, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = self.k * rho ** (-(self.s + 1.0)) - L**2 / rho**3
        return float(out) if np.ndim(out) == 0 else out

    def barrier_radius(self, L: float) -> float:
        r"""Stationary radius of :math:`V_{\rm eff}`, :math:`\rho_* = (L^2/k)^{1/(2-s)}`.

        It is the *minimum* (stable circular orbit) for ``s < 2`` and the
        *maximum* (top of the barrier the pair must clear to collide) for
        ``s > 2``.  Undefined at ``s = 2``, where the two terms scale alike.
        """
        if self.s == 2.0:
            return float("nan")
        if L == 0.0:
            return 0.0 if self.s < 2.0 else float("inf")
        return float((L**2 / self.k) ** (1.0 / (2.0 - self.s)))

    def barrier_height(self, L: float) -> float:
        r""":math:`V_{\rm eff}(\rho_*) = k\rho_*^{-s}(s-2)/(2s)`; positive for ``s > 2``."""
        rho_star = self.barrier_radius(L)
        if not np.isfinite(rho_star) or rho_star == 0.0:
            return float("nan")
        return float(self.v_eff(rho_star, L))

    # ------------------------------------------------------ initial conditions
    def circular_speed(self, rho: float) -> float:
        r"""Speed of a circular orbit of radius ``rho``: :math:`v^2 = k\rho^{-s}`."""
        return float(np.sqrt(self.k * rho ** (-self.s)))

    def circular_state(self, rho: float = 1.0, retrograde: bool = False) -> Array:
        """State of a circular orbit of radius ``rho``, starting on the ``+x`` axis."""
        v = self.circular_speed(rho)
        sign = -1.0 if retrograde else 1.0
        return make_state([rho, 0.0], [0.0, sign * v])

    def state_from_apsides(self, rho_min: float, rho_max: float) -> Array:
        if not 0 < rho_min < rho_max:
            raise ValueError("require 0 < rho_min < rho_max")
        num = 2.0 * (self.potential(rho_max) - self.potential(rho_min))
        den = rho_min**-2 - rho_max**-2
        L = float(np.sqrt(num / den))
        return make_state([rho_min, 0.0], [0.0, L / rho_min])

    def energy_and_L(self, y: Array) -> tuple[float, float]:
        """Convenience: ``(E, L)`` for a single state."""
        return float(self.energy(y)), float(self.angular_momentum(0.0, y))