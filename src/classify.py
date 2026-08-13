from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .integrators import Solution, apsis_event, collision_event, escape_event, integrate, reference_solve
from .systems import PowerLawCentralForce, split_state

__all__ = [
    "COLLISION",
    "BOUNDED",
    "ESCAPE",
    "UNDETERMINED",
    "OUTCOMES",
    "Prediction",
    "predict_outcome",
    "NumericalOutcome",
    "classify_numerically",
    "characteristic_time",
    "outer_turning_point",
    "inner_turning_point",
]

Array = np.ndarray

COLLISION = "collision"
BOUNDED = "bounded"
ESCAPE = "escape"
UNDETERMINED = "undetermined"
OUTCOMES = (COLLISION, BOUNDED, ESCAPE)


def characteristic_time(system: PowerLawCentralForce, rho0: float) -> float:
    r"""Period of the circular orbit of radius ``rho0``: :math:`2\pi\rho_0^{(s+2)/2}/\sqrt{k}`.

    Used as the natural time unit so that a sweep over ``s`` integrates for a
    comparable number of dynamical times at every exponent.
    """
    return float(2.0 * np.pi * rho0 ** (0.5 * (system.s + 2.0)) / np.sqrt(system.k))


# ------------------------------------------------------------------ prediction
@dataclass
class Prediction:
    """Analytic outcome for one initial condition."""

    outcome: str
    reason: str
    E: float
    L: float
    rho0: float
    rho_dot0: float
    barrier_radius: float = float("nan")
    barrier_height: float = float("nan")
    marginal: bool = False

    def __str__(self) -> str:
        return f"{self.outcome} ({self.reason})"


def predict_outcome(
    system: PowerLawCentralForce,
    y0: Array,
    tol: float = 1e-9,
) -> Prediction:
    x, v = split_state(np.asarray(y0, dtype=float))
    rho0 = float(np.linalg.norm(x))
    if rho0 == 0.0:
        raise ValueError("initial separation must be nonzero")
    rho_dot0 = float(np.dot(x, v) / rho0)
    L = float(system.angular_momentum(0.0, y0))
    E = float(system.energy(y0))
    s, k = system.s, system.k

    scale = abs(E) + abs(system.v_eff(rho0, L)) + 1e-300
    near_zero_E = abs(E) <= tol * scale

    def result(outcome: str, reason: str, marginal: bool = False, **kw) -> Prediction:
        return Prediction(
            outcome=outcome,
            reason=reason,
            E=E,
            L=L,
            rho0=rho0,
            rho_dot0=rho_dot0,
            marginal=marginal,
            **kw,
        )

    # --- radial infall: no centrifugal barrier at all -----------------------
    if abs(L) <= tol * rho0 * max(np.linalg.norm(v), 1e-300):
        if s > 0 and E >= 0.0 and rho_dot0 > 0.0:
            return result(ESCAPE, "L=0, outward with E>=0", marginal=near_zero_E)
        return result(COLLISION, "L=0: no barrier, radial infall", marginal=abs(L) > 0)

    # --- s < 2: barrier always wins near the origin -------------------------
    if s < 2.0:
        if s <= 0.0:
            return result(BOUNDED, f"s={s:g}<=0: V(inf)=inf, every orbit is confined")
        if E >= 0.0:
            return result(ESCAPE, "s<2, L!=0, E>=0: unbounded", marginal=near_zero_E)
        return result(BOUNDED, "s<2, L!=0, E<0: barrier blocks collision", marginal=near_zero_E)

    # --- s = 2: barrier and attraction scale alike --------------------------
    if s == 2.0:
        margin = L**2 - k
        marginal = abs(margin) <= tol * k
        if margin > 0.0:
            return result(ESCAPE, "s=2, L^2>k: effective barrier repels, E>0", marginal=marginal)
        if margin < 0.0:
            if rho_dot0 < 0.0:
                return result(COLLISION, "s=2, L^2<k, inward: fall to the centre", marginal=marginal)
            if E >= 0.0:
                return result(ESCAPE, "s=2, L^2<k, outward with E>=0", marginal=marginal or near_zero_E)
            return result(COLLISION, "s=2, L^2<k, outward but E<0: turns and falls in", marginal=marginal)
        if rho_dot0 < 0.0:
            return result(COLLISION, "s=2, L^2=k: V_eff=0, drifts inward", marginal=True)
        if rho_dot0 > 0.0:
            return result(ESCAPE, "s=2, L^2=k: V_eff=0, drifts outward", marginal=True)
        return result(BOUNDED, "s=2, L^2=k, rho_dot=0: neutral circular orbit", marginal=True)

    # --- s > 2: a barrier that can be cleared, and no bound orbits at all ---
    #
    # For s > 2 the effective potential runs from -inf at the origin up to a single
    # maximum at rho_*, then back down to 0 from *above* (rho^-2 outlives rho^-s).
    # Having no minimum, it supports no stable circular orbit and no radial
    # oscillation: every trajectory either falls to the centre or leaves for good.
    # The existence of bound motion is thus itself decided by s < 2.
    rho_star = system.barrier_radius(L)
    v_max = system.barrier_height(L)
    kw = dict(barrier_radius=rho_star, barrier_height=v_max)
    near_barrier = abs(E - v_max) <= tol * (abs(v_max) + abs(E) + 1e-300)

    if near_barrier:
        return result(UNDETERMINED, "s>2, E at the top of the barrier (unstable circular orbit)",
                      marginal=True, **kw)
    if E > v_max:
        if rho_dot0 < 0.0:
            return result(COLLISION, "s>2, E clears the barrier, moving inward", **kw)
        return result(ESCAPE, "s>2, E clears the barrier, moving outward", **kw)
    if rho0 < rho_star:
        return result(COLLISION, "s>2, trapped inside the barrier: no bound orbits exist", **kw)
    return result(ESCAPE, "s>2, held outside the barrier, which forces E>0", **kw)


# --------------------------------------------------------------- numerical run
@dataclass
class NumericalOutcome:
    """Outcome read off an integrated trajectory."""

    outcome: str
    solution: Solution
    rho_min: float
    rho_max: float
    n_apsides: int
    t_end: float
    confirmed: bool = True

    def __str__(self) -> str:
        return f"{self.outcome} (rho in [{self.rho_min:.3g}, {self.rho_max:.3g}])"


def classify_numerically(
    system: PowerLawCentralForce,
    y0: Array,
    n_periods: float = 60.0,
    rho_collision: float = 1e-4,
    escape_factor: float = 100.0,
    method: str = "reference",
    dt: float | None = None,
    rtol: float = 1e-11,
    atol: float = 1e-12,
) -> NumericalOutcome:
    y0 = np.asarray(y0, dtype=float)
    rho0 = float(np.linalg.norm(y0[:2]))
    rho_escape = escape_factor * rho0
    tau = characteristic_time(system, rho0)
    t_end = n_periods * tau

    events = [collision_event(rho_collision), escape_event(rho_escape), apsis_event()]
    sol = _run(system, y0, (0.0, t_end), events, method, dt, rtol, atol)

    radius = sol.radius
    rho_min_seen = float(np.min(radius))
    rho_max_seen = float(np.max(radius))
    n_apsides = len(sol.events.get("apsis", []))

    if sol.terminated_by == "collision":
        return NumericalOutcome(COLLISION, sol, rho_min_seen, rho_max_seen, n_apsides, sol.final_time)

    if sol.terminated_by == "escape":
        confirmed = _confirm_escape(system, sol, rho_escape, method, dt, rtol, atol)
        outcome = ESCAPE if confirmed else BOUNDED
        return NumericalOutcome(outcome, sol, rho_min_seen, rho_max_seen, n_apsides, sol.final_time, confirmed)

    if sol.status in ("diverged", "failed", "max_steps"):
        return NumericalOutcome(UNDETERMINED, sol, rho_min_seen, rho_max_seen, n_apsides, sol.final_time, False)

    if n_apsides >= 2:
        return NumericalOutcome(BOUNDED, sol, rho_min_seen, rho_max_seen, n_apsides, sol.final_time)

    return NumericalOutcome(UNDETERMINED, sol, rho_min_seen, rho_max_seen, n_apsides, sol.final_time, False)


def outer_turning_point(system: PowerLawCentralForce, y0: Array, rho_huge: float = 1e12) -> float:
    E = float(system.energy(y0))
    L = float(system.angular_momentum(0.0, y0))
    rho0 = float(np.linalg.norm(np.asarray(y0, dtype=float)[:2]))
    if system.s > 0 and E >= 0.0:
        return float("inf")

    f = lambda rho: float(system.v_eff(rho, L)) - E
    # Start just outside rho0: when the initial state is itself a turning point,
    # f(rho0) = 0 and a bracket anchored there would return rho0 for the apocentre.
    lo = rho0 * (1.0 + 1e-10)
    if f(lo) >= 0.0:
        return rho0  # motion is blocked immediately outward: rho0 is the apocentre
    hi = 2.0 * lo
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > rho_huge:
            return float("inf")
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=8.9e-16))


def inner_turning_point(system: PowerLawCentralForce, y0: Array, rho_tiny: float = 1e-300) -> float:
    E = float(system.energy(y0))
    L = float(system.angular_momentum(0.0, y0))
    rho0 = float(np.linalg.norm(np.asarray(y0, dtype=float)[:2]))
    s, k = system.s, system.k
    if L == 0.0:
        return 0.0
    if s > 2.0:
        return 0.0  # V_eff -> -inf: nothing turns the pair around
    if s == 2.0:
        margin = L**2 - k
        if margin <= 0.0 or E <= 0.0:
            return 0.0
        return float(np.sqrt(margin / (2.0 * E)))

    # Solved in log-radius, on the sign-equivalent function
    #     h(u) = e^{2u} (V_eff(e^u) - E) = -(k/s)e^{(2-s)u} - E e^{2u} + L^2/2,
    # which for s < 2 tends to L^2/2 > 0 as u -> -inf instead of overflowing.  Both
    # rescalings matter: the pericentre is exponentially small as s -> 2^-, so a
    # bracket has to be measured in decades, and V_eff itself is unrepresentable
    # there while its sign is perfectly well defined.
    if s == 0.0:
        h = lambda u: k * u * np.exp(2.0 * u) - E * np.exp(2.0 * u) + 0.5 * L**2
    else:
        h = lambda u: -(k / s) * np.exp((2.0 - s) * u) - E * np.exp(2.0 * u) + 0.5 * L**2

    u_hi = np.log(rho0) + np.log1p(-1e-10)
    if h(u_hi) >= 0.0:
        return rho0  # motion is blocked immediately inward: rho0 is the pericentre
    u_lo, u_floor = u_hi - 1.0, np.log(rho_tiny)
    while h(u_lo) < 0.0:
        u_lo -= 1.0
        if u_lo < u_floor:
            return 0.0
    return float(np.exp(brentq(h, u_lo, u_hi, xtol=1e-13, rtol=8.9e-16)))


def _run(system, y0, t_span, events, method, dt, rtol, atol) -> Solution:
    if method == "reference":
        return reference_solve(system, y0, t_span, events=events, rtol=rtol, atol=atol)
    if dt is None:
        raise ValueError(f"method={method!r} is a fixed-step method and needs dt")
    return integrate(system, y0, t_span, dt, method=method, events=events, store_every=20)


def _confirm_escape(system, sol, rho_escape, method, dt, rtol, atol) -> bool:
    """Continue past the escape radius and require the separation to keep growing.

    A bound orbit that happens to reach the escape radius must turn around; an
    escaping one does not.  Continuing for the time it would take to double the
    separation at the current radial speed settles it without appealing to the
    energy.
    """
    y_ev = sol.y[-1]
    t_ev = sol.t[-1]
    x, v = y_ev[:2], y_ev[2:4]
    rho = float(np.linalg.norm(x))
    rho_dot = float(np.dot(x, v) / rho)
    if rho_dot <= 0.0:
        return False
    t_extra = 2.0 * rho / rho_dot
    events = [collision_event(1e-6 * rho), escape_event(1e6 * rho_escape)]
    cont = _run(system, y_ev, (t_ev, t_ev + t_extra), events, method, dt, rtol, atol)
    return bool(cont.radius[-1] > 1.5 * rho)