from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from .integrators import Solution, apsis_event, reference_solve
from .systems import PowerLawCentralForce

__all__ = [
    "ConservationReport",
    "conservation_report",
    "convergence_order",
    "trajectory_error",
    "PolarAngleAugmented",
    "apsidal_angle_quadrature",
    "radial_period_quadrature",
    "apsidal_angle_near_circular",
    "measure_apsidal_angles",
    "turning_points",
]

Array = np.ndarray


# --------------------------------------------------------------- conservation
@dataclass
class ConservationReport:

    t: Array
    energy: Array
    angular_momentum: Array
    max_energy_drift: float
    max_L_drift: float
    secular_energy_drift: float
    secular_L_drift: float
    method: str

    def summary(self) -> str:
        return (
            f"{self.method:>12s}  max|dE/E| = {self.max_energy_drift:9.3e}   "
            f"max|dL/L| = {self.max_L_drift:9.3e}   "
            f"secular dE/E per unit t = {self.secular_energy_drift:9.3e}"
        )


def conservation_report(system, sol: Solution) -> ConservationReport:
    """Energy and angular-momentum drift along ``sol``."""
    E = np.asarray(system.energy(sol.y), dtype=float)
    L = np.asarray(system.angular_momentum(sol.t, sol.y), dtype=float)
    E0, L0 = E[0], L[0]
    dE = (E - E0) / abs(E0) if E0 != 0 else E - E0
    dL = (L - L0) / abs(L0) if L0 != 0 else L - L0
    return ConservationReport(
        t=sol.t,
        energy=E,
        angular_momentum=L,
        max_energy_drift=float(np.max(np.abs(dE))),
        max_L_drift=float(np.max(np.abs(dL))),
        secular_energy_drift=_slope(sol.t, dE),
        secular_L_drift=_slope(sol.t, dL),
        method=sol.method,
    )


def _slope(t: Array, q: Array) -> float:
    """Least-squares slope of ``q`` against ``t`` (0 if there is nothing to fit)."""
    if len(t) < 2 or np.ptp(t) == 0:
        return 0.0
    return float(np.polyfit(t, q, 1)[0])


def convergence_order(step_sizes: Array, errors: Array) -> float:
    h = np.asarray(step_sizes, dtype=float)
    e = np.asarray(errors, dtype=float)
    ok = np.isfinite(e) & (e > 0) & np.isfinite(h) & (h > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(h[ok]), np.log(e[ok]), 1)[0])


def trajectory_error(sol: Solution, exact_state, norm: str = "max") -> float:
    exact = np.asarray(exact_state(sol.t), dtype=float)
    err = np.linalg.norm(sol.y[:, :2] - exact[:, :2], axis=1)
    if norm == "max":
        return float(np.max(err))
    if norm == "final":
        return float(err[-1])
    raise ValueError("norm must be 'max' or 'final'")


# ------------------------------------------------------------- polar tracking
class PolarAngleAugmented:

    order = 2
    state_dim = 5

    def __init__(self, system):
        self.system = system

    def rhs(self, t: float, y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        x, v = y[:2], y[2:4]
        a = np.asarray(self.system.accel(t, x), dtype=float)
        rho2 = float(np.dot(x, x))
        phi_dot = (x[0] * v[1] - x[1] * v[0]) / rho2
        return np.array([v[0], v[1], a[0], a[1], phi_dot])

    def accel(self, t: float, x: Array) -> Array:
        return self.system.accel(t, x)

    def energy(self, y: Array) -> float | Array:
        return self.system.energy(np.asarray(y, dtype=float)[..., :4])

    def angular_momentum(self, t: float, y: Array) -> float | Array:
        return self.system.angular_momentum(t, np.asarray(y, dtype=float)[..., :4])

    @staticmethod
    def initial_state(y0: Array) -> Array:
        r"""Append :math:`\varphi_0 = \arctan2(y, x)` to a 4-vector state."""
        y0 = np.asarray(y0, dtype=float)
        return np.append(y0, np.arctan2(y0[1], y0[0]))


# ------------------------------------------------------------------ quadrature
def _apsides_to_EL(system: PowerLawCentralForce, rho_min: float, rho_max: float) -> tuple[float, float]:
    if not 0 < rho_min < rho_max:
        raise ValueError("require 0 < rho_min < rho_max")
    s, k = system.s, system.k
    lam = float(np.log1p((rho_max - rho_min) / rho_min))
    if s == 0.0:
        L2 = -2.0 * k * lam * rho_min**2 / np.expm1(-2.0 * lam)
    else:
        L2 = 2.0 * (k / s) * rho_min ** (2.0 - s) * np.expm1(-s * lam) / np.expm1(-2.0 * lam)
    L = float(np.sqrt(L2))
    E = float(system.potential(rho_min) + L2 / (2.0 * rho_min**2))
    return E, L


def _radial_quadrature(
    system: PowerLawCentralForce,
    rho_min: float,
    rho_max: float,
    kind: str,
    epsabs: float = 1e-13,
) -> float:
    if kind not in ("angle", "time"):
        raise ValueError("kind must be 'angle' or 'time'")
    E, L = _apsides_to_EL(system, rho_min, rho_max)
    s, k = system.s, system.k
    d = 0.5 * (rho_max - rho_min)
    A = (k / s) * rho_min ** (-s) if s != 0.0 else 0.0
    B = 0.5 * L**2 * rho_min**-2

    slope_min = -float(system.dv_eff(rho_min, L))  # > 0: V_eff decreasing at rho_-
    slope_max = float(system.dv_eff(rho_max, L))  # > 0: V_eff increasing at rho_+
    limit_min = np.sqrt(d / slope_min) if slope_min > 0 else 0.0
    limit_max = np.sqrt(d / slope_max) if slope_max > 0 else 0.0
    if kind == "angle":
        limit_min *= L / rho_min**2
        limit_max *= L / rho_max**2

    def integrand(theta: Array) -> Array:
        offset = 2.0 * d * np.sin(0.5 * theta) ** 2  # = rho - rho_min, exactly
        rho = rho_min + offset
        ell = -np.log1p(offset / rho_min)  # = ln(rho_min / rho) <= 0
        if s == 0.0:
            f = k * ell - B * np.expm1(2.0 * ell)
        else:
            f = A * np.expm1(s * ell) - B * np.expm1(2.0 * ell)
        w = L / rho**2 if kind == "angle" else np.ones_like(rho)
        good = f > 0.0
        value = w * d * np.sin(theta) / np.sqrt(2.0 * np.where(good, f, 1.0))
        # Round-off can push f below zero within a few epsilon of a turning point;
        # there the closed-form endpoint limit is the correct value.
        return np.where(good, value, np.where(theta < 0.5 * np.pi, limit_min, limit_max))

    return _gauss_legendre(integrand, 0.0, np.pi, tol=epsabs)


@lru_cache(maxsize=16)
def _leggauss(n: int) -> tuple[Array, Array]:
    return np.polynomial.legendre.leggauss(n)


def _gauss_legendre(f, a: float, b: float, tol: float = 1e-12, n_max: int = 1024) -> float:
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    previous = best = np.nan
    best_diff = np.inf
    n = 16
    while n <= n_max:
        nodes, weights = _leggauss(n)
        value = half * float(np.dot(weights, f(mid + half * nodes)))
        if np.isfinite(previous):
            diff = abs(value - previous)
            if diff <= tol * (1.0 + abs(value)):
                return value
            if diff > best_diff:
                return best
            best, best_diff = value, diff
        previous = value
        n *= 2
    return previous


def apsidal_angle_quadrature(system: PowerLawCentralForce, rho_min: float, rho_max: float) -> float:
    return _radial_quadrature(system, rho_min, rho_max, kind="angle")


def radial_period_quadrature(system: PowerLawCentralForce, rho_min: float, rho_max: float) -> float:
    """Radial (pericentre-to-pericentre) period of the same orbit."""
    return 2.0 * _radial_quadrature(system, rho_min, rho_max, kind="time")


def apsidal_angle_near_circular(s: float) -> float:
    if s >= 2.0:
        return float("inf")
    return float(np.pi / np.sqrt(2.0 - s))


def turning_points(system: PowerLawCentralForce, E: float, L: float, bracket=(1e-8, 1e8)) -> tuple[float, float]:
    r"""Radii where :math:`V_{\rm eff}(\rho) = E`, i.e. where :math:`\dot\rho = 0`."""
    rho_star = system.barrier_radius(L)
    if not np.isfinite(rho_star) or rho_star <= 0:
        raise ValueError("no effective-potential minimum: turning points are not bracketed")
    f = lambda rho: float(system.v_eff(rho, L)) - E
    lo, hi = bracket
    if f(rho_star) > 0:
        raise ValueError("E is below the minimum of V_eff: no motion is possible")
    rho_min = brentq(f, lo, rho_star, xtol=1e-14, rtol=1e-14)
    rho_max = brentq(f, rho_star, hi, xtol=1e-14, rtol=1e-14)
    return float(rho_min), float(rho_max)


# ------------------------------------------------------------------ ODE-based
def measure_apsidal_angles(
    system: PowerLawCentralForce,
    rho_min: float,
    rho_max: float,
    n_apsides: int = 6,
    rtol: float = 1e-13,
    atol: float = 1e-14,
) -> dict:
    y0 = system.state_from_apsides(rho_min, rho_max)
    T_r = radial_period_quadrature(system, rho_min, rho_max)
    t_end = 0.75 * T_r * (n_apsides + 1)

    aug = PolarAngleAugmented(system)
    sol = reference_solve(
        aug,
        PolarAngleAugmented.initial_state(y0),
        (0.0, t_end),
        events=[apsis_event()],
        rtol=rtol,
        atol=atol,
    )
    states = sol.event_states("apsis")
    times = sol.event_times("apsis")
    phi = states[:, 4]
    radii = np.linalg.norm(states[:, :2], axis=1)
    angles = np.abs(np.diff(phi))

    return {
        "measured": angles,
        "mean": float(np.mean(angles)) if len(angles) else float("nan"),
        "spread": float(np.ptp(angles)) if len(angles) > 1 else 0.0,
        "apsis_times": times,
        "apsis_radii": radii,
        "radial_period": T_r,
        "quadrature": apsidal_angle_quadrature(system, rho_min, rho_max),
        "near_circular": apsidal_angle_near_circular(system.s),
        "solution": sol,
    }