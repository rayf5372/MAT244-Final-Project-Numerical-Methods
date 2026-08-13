from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "Event",
    "Solution",
    "integrate",
    "reference_solve",
    "collision_event",
    "escape_event",
    "apsis_event",
    "METHODS",
    "METHOD_ORDERS",
]

Array = np.ndarray
EventFn = Callable[[float, Array], float]


# ---------------------------------------------------------------------- events
@dataclass
class Event:

    fn: EventFn
    direction: float = 0.0
    terminal: bool = True
    name: str = "event"

    def __call__(self, t: float, y: Array) -> float:
        return float(self.fn(t, y))


def collision_event(rho_collision: float = 1e-3, name: str = "collision") -> Event:


    def g(t: float, y: Array) -> float:
        return float(np.linalg.norm(np.asarray(y, dtype=float)[..., :2]) - rho_collision)

    return Event(g, direction=-1.0, terminal=True, name=name)


def escape_event(rho_escape: float = 1e3, name: str = "escape") -> Event:

    def g(t: float, y: Array) -> float:
        return float(np.linalg.norm(np.asarray(y, dtype=float)[..., :2]) - rho_escape)

    return Event(g, direction=1.0, terminal=True, name=name)


def apsis_event(name: str = "apsis") -> Event:

    def g(t: float, y: Array) -> float:
        y = np.asarray(y, dtype=float)
        return float(np.dot(y[:2], y[2:4]))

    return Event(g, direction=0.0, terminal=False, name=name)


# -------------------------------------------------------------------- solution
@dataclass
class Solution:
    """Result of an integration."""

    t: Array
    y: Array
    method: str
    status: str = "completed"
    events: dict[str, list[tuple[float, Array]]] = field(default_factory=dict)
    terminated_by: str | None = None
    n_steps: int = 0
    n_rhs_evals: int = 0
    dt: float | None = None

    @property
    def position(self) -> Array:
        return self.y[:, :2]

    @property
    def velocity(self) -> Array:
        if self.y.shape[1] < 4:
            raise AttributeError("first-order system: velocity is not part of the state")
        return self.y[:, 2:4]

    @property
    def radius(self) -> Array:
        r""":math:`\rho(t) = \|r(t)\|`."""
        return np.linalg.norm(self.y[:, :2], axis=1)

    @property
    def angle(self) -> Array:
        r"""Continuous (unwrapped) polar angle :math:`\varphi(t)`."""
        return np.unwrap(np.arctan2(self.y[:, 1], self.y[:, 0]))

    @property
    def final_time(self) -> float:
        return float(self.t[-1])

    def event_times(self, name: str) -> Array:
        return np.array([t for t, _ in self.events.get(name, [])])

    def event_states(self, name: str) -> Array:
        states = [y for _, y in self.events.get(name, [])]
        return np.array(states) if states else np.empty((0, self.y.shape[1]))

    def __repr__(self) -> str:
        return (
            f"Solution(method={self.method!r}, status={self.status!r}, "
            f"t=[{self.t[0]:g}, {self.t[-1]:g}], n_steps={self.n_steps}, "
            f"n_rhs_evals={self.n_rhs_evals})"
        )


# -------------------------------------------------------------------- steppers
class _Stepper:
    """One-step map ``(t, y, h) -> y_next``, with a force-evaluation counter."""

    name = "stepper"
    order = 0

    def __init__(self, system):
        self.system = system
        self.n_rhs_evals = 0

    def step(self, t: float, y: Array, h: float) -> Array:  # pragma: no cover
        raise NotImplementedError


class _Euler(_Stepper):

    name = "euler"
    order = 1

    def step(self, t: float, y: Array, h: float) -> Array:
        self.n_rhs_evals += 1
        return y + h * np.asarray(self.system.rhs(t, y), dtype=float)


class _RK4(_Stepper):

    name = "rk4"
    order = 4

    def step(self, t: float, y: Array, h: float) -> Array:
        f = self.system.rhs
        self.n_rhs_evals += 4
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
        k3 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class _VelocityVerlet(_Stepper):

    name = "verlet"
    order = 2

    def __init__(self, system):
        if not hasattr(system, "accel"):
            raise TypeError(
                f"{type(system).__name__} is a first-order (velocity-level) law, so it has "
                "no acceleration field and cannot be integrated symplectically; "
                "use method='euler' or 'rk4'."
            )
        super().__init__(system)
        self._cache: tuple[float, bytes, Array] | None = None

    def _accel(self, t: float, x: Array) -> Array:
        key = (t, x.tobytes())
        if self._cache is not None and self._cache[0] == key[0] and self._cache[1] == key[1]:
            return self._cache[2]
        self.n_rhs_evals += 1
        a = np.asarray(self.system.accel(t, x), dtype=float)
        self._cache = (key[0], key[1], a)
        return a

    def step(self, t: float, y: Array, h: float) -> Array:
        x, v = y[:2], y[2:]
        a0 = self._accel(t, x)
        x1 = x + h * v + 0.5 * h * h * a0
        a1 = self._accel(t + h, x1)
        v1 = v + 0.5 * h * (a0 + a1)
        return np.concatenate([x1, v1])


METHODS: dict[str, type[_Stepper]] = {
    "euler": _Euler,
    "rk4": _RK4,
    "verlet": _VelocityVerlet,
}
METHOD_ORDERS: dict[str, int] = {name: cls.order for name, cls in METHODS.items()}


# --------------------------------------------------------------------- driver
def integrate(
    system,
    y0: Array,
    t_span: tuple[float, float],
    dt: float,
    method: str = "rk4",
    events: Sequence[Event] = (),
    store_every: int = 1,
    max_steps: int = 20_000_000,
) -> Solution:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    if dt <= 0:
        raise ValueError("dt must be positive")

    t0, t1 = float(t_span[0]), float(t_span[1])
    if t1 <= t0:
        raise ValueError("require t_span[1] > t_span[0]")

    stepper = METHODS[method](system)
    y = np.asarray(y0, dtype=float).copy()
    if y.size != getattr(system, "state_dim", y.size):
        raise ValueError(
            f"{type(system).__name__} expects a state of size {system.state_dim}, got {y.size}"
        )

    t = t0
    ts: list[float] = [t0]
    ys: list[Array] = [y.copy()]
    events = list(events)
    g_prev = [ev(t, y) for ev in events]
    records: dict[str, list[tuple[float, Array]]] = {ev.name: [] for ev in events}

    status = "completed"
    terminated_by: str | None = None
    n_steps = 0
    eps = 1e-12 * max(1.0, abs(t1))

    while t < t1 - eps:
        if n_steps >= max_steps:
            status = "max_steps"
            break
        h = min(dt, t1 - t)
        y_new = stepper.step(t, y, h)

        if not np.all(np.isfinite(y_new)):
            status = "diverged"
            break

        t_new = t + h
        stop = False
        for i, ev in enumerate(events):
            g_new = ev(t_new, y_new)
            if _crossed(g_prev[i], g_new, ev.direction):
                t_ev, y_ev = _refine_event(stepper, ev, t, y, h, g_prev[i])
                records[ev.name].append((t_ev, y_ev))
                if ev.terminal:
                    t_new, y_new = t_ev, y_ev
                    status, terminated_by, stop = ev.name, ev.name, True
                    break
            g_prev[i] = g_new

        t, y = t_new, np.asarray(y_new, dtype=float)
        n_steps += 1

        if stop or n_steps % store_every == 0:
            ts.append(t)
            ys.append(y.copy())
        if stop:
            break
        g_prev = [ev(t, y) for ev in events]

    if ts[-1] != t:
        ts.append(t)
        ys.append(y.copy())

    return Solution(
        t=np.array(ts),
        y=np.array(ys),
        method=method,
        status=status,
        events=records,
        terminated_by=terminated_by,
        n_steps=n_steps,
        n_rhs_evals=stepper.n_rhs_evals,
        dt=dt,
    )


def _crossed(g_old: float, g_new: float, direction: float) -> bool:
    """Whether a zero crossing of the requested direction happened."""
    if g_old == 0.0:
        return False
    if g_new == 0.0:
        return direction == 0.0 or np.sign(g_new - g_old) == np.sign(direction)
    if np.sign(g_old) == np.sign(g_new):
        return False
    if direction == 0.0:
        return True
    return np.sign(g_new - g_old) == np.sign(direction)


def _refine_event(
    stepper: _Stepper,
    event: Event,
    t: float,
    y: Array,
    h: float,
    g_left: float,
    tol: float = 1e-14,
    max_iter: int = 100,
) -> tuple[float, Array]:
    lo, hi = 0.0, h
    y_hi = stepper.step(t, y, h)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        y_mid = stepper.step(t, y, mid)
        g_mid = event(t + mid, y_mid)
        if g_mid == 0.0 or hi - lo < tol * max(h, 1.0):
            return t + mid, y_mid
        if np.sign(g_mid) == np.sign(g_left):
            lo = mid
        else:
            hi, y_hi = mid, y_mid
    return t + hi, y_hi


# ------------------------------------------------------------------- reference
def reference_solve(
    system,
    y0: Array,
    t_span: tuple[float, float],
    events: Sequence[Event] = (),
    method: str = "DOP853",
    rtol: float = 1e-12,
    atol: float = 1e-12,
    t_eval: Array | None = None,
    n_out: int | None = None,
    max_step: float = np.inf,
) -> Solution:
    t0, t1 = float(t_span[0]), float(t_span[1])
    if t_eval is None and n_out is not None:
        t_eval = np.linspace(t0, t1, n_out)

    scipy_events = []
    for ev in events:
        fn = _as_scipy_event(ev)
        scipy_events.append(fn)

    sol = solve_ivp(
        system.rhs,
        (t0, t1),
        np.asarray(y0, dtype=float),
        method=method,
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        events=scipy_events if scipy_events else None,
        max_step=max_step,
        dense_output=False,
    )

    records: dict[str, list[tuple[float, Array]]] = {}
    terminated_by = None
    for ev, t_ev, y_ev in zip(events, sol.t_events or [], sol.y_events or []):
        records[ev.name] = [(float(tt), np.asarray(yy, dtype=float)) for tt, yy in zip(t_ev, y_ev)]
        if ev.terminal and len(t_ev) > 0 and sol.status == 1:
            if terminated_by is None or t_ev[0] < records[terminated_by][0][0]:
                terminated_by = ev.name

    if sol.status == 1:
        status = terminated_by or "event"
    elif sol.status == 0:
        status = "completed"
    else:
        status = "failed"

    t_out, y_out = sol.t, sol.y.T
    if terminated_by is not None and len(t_out) and records[terminated_by]:
        t_ev, y_ev = records[terminated_by][0]
        if t_out[-1] < t_ev:
            t_out = np.append(t_out, t_ev)
            y_out = np.vstack([y_out, y_ev])

    return Solution(
        t=t_out,
        y=y_out,
        method=f"scipy:{method}",
        status=status,
        events=records,
        terminated_by=terminated_by,
        n_steps=len(sol.t) - 1,
        n_rhs_evals=int(sol.nfev),
    )


def _as_scipy_event(ev: Event) -> EventFn:
    """Adapt an :class:`Event` to the attribute-tagged callable SciPy expects."""

    def fn(t: float, y: Array) -> float:
        return ev(t, y)

    fn.terminal = bool(ev.terminal)  # type: ignore[attr-defined]
    fn.direction = float(ev.direction)  # type: ignore[attr-defined]
    return fn