from __future__ import annotations

import numpy as np

from _common import Report, plt, save_figure, save_table, save_text

from twobody import (
    KeplerOrbit,
    MovingTargetPursuit,
    PowerLawCentralForce,
    PurePursuit,
    capture_event,
    collision_event,
    integrate,
    make_state,
    reference_solve,
)

RHO0 = 1.0
SPEED = 1.0
MU = 1.0


def run_pursuit(report: Report):
    report()
    report("A. Pure pursuit: dr/dt = -v r_hat")
    pursuit = PurePursuit(speed=SPEED)
    y0 = np.array([RHO0, 0.0])
    threshold = 1e-3
    dt = 1e-5  # must satisfy v*dt << threshold, or the step jumps over the target
    exact_capture = pursuit.capture_time(y0)

    # Integrate to just short of capture for the trajectory, then time the capture
    # separately with a step fine enough to resolve the threshold.
    sol = integrate(pursuit, y0, (0.0, 0.999 * exact_capture), 1e-4, method="rk4")
    timed = integrate(pursuit, y0, (0.0, 2.0 * exact_capture), dt, method="rk4",
                      events=[capture_event(pursuit, threshold)])
    assert timed.terminated_by == "capture", f"capture not detected: status {timed.status}"
    exact_at_threshold = (RHO0 - threshold) / SPEED

    L = np.array([pursuit.angular_momentum(t, y) for t, y in zip(sol.t, sol.y)])
    report(f"   exact capture time rho_0/v          = {exact_capture:.12f}")
    report(f"   measured time to reach rho = {threshold:.0e}  = {timed.final_time:.12f}")
    report(f"   exact time to reach the same radius = {exact_at_threshold:.12f}")
    report(f"   error                               = {abs(timed.final_time - exact_at_threshold):.2e}")
    report(f"   max|L| along the trajectory         = {np.max(np.abs(L)):.2e}   (identically zero)")
    report(f"   max|rho(t) - (rho_0 - vt)|          = "
           f"{np.max(np.abs(sol.radius - pursuit.exact_radius(sol.t, y0))):.2e}")
    return pursuit, sol, timed, exact_capture


def run_gravity(report: Report):
    report()
    report("B. Gravity, same rho_0 and same initial speed, two directions")
    gravity = PowerLawCentralForce.gravity(mu=MU)
    circular_speed = gravity.circular_speed(RHO0)
    report(f"   circular speed at rho_0 = {circular_speed:.6f}, so |v| = {SPEED} is the")
    report("   circular orbit when the velocity is tangential.")

    tangential = make_state([RHO0, 0.0], [0.0, SPEED])
    radial = make_state([RHO0, 0.0], [-SPEED, 0.0])

    period = KeplerOrbit(mu=MU, a=RHO0, e=0.0).period
    orbit = reference_solve(gravity, tangential, (0.0, 3.0 * period), n_out=1200)
    E_t, L_t = gravity.energy_and_L(tangential)
    report(f"   tangential: E = {E_t:+.6f}, L = {L_t:+.6f}  -> bounded, period {period:.6f}")
    report(f"     rho stays within [{orbit.radius.min():.9f}, {orbit.radius.max():.9f}]")
    L_series = gravity.angular_momentum(orbit.t, orbit.y)
    report(f"     max|L(t) - L(0)| = {np.max(np.abs(L_series - L_t)):.2e}  (conserved)")

    E_r, L_r = gravity.energy_and_L(radial)
    infall = reference_solve(gravity, radial, (0.0, 2.0), events=[collision_event(1e-9)],
                            n_out=1500)
    # This case has an exact answer too.  With E = -1/2 the radial orbit is the
    # degenerate ellipse of semi-major axis a = 1, so its apocentre is 2a = 2, and the
    # infall time from rho_0 = 1 is the free-fall time from 2 less the part already
    # travelled: pi - (pi/2 + 1) = pi/2 - 1.
    exact_infall = np.pi / 2.0 - 1.0
    report(f"   radial:     E = {E_r:+.6f}, L = {L_r:+.6f}  -> collision at "
           f"t = {infall.final_time:.9f}")
    report(f"     exact collision time pi/2 - 1     = {exact_infall:.9f} "
           f"(error {abs(infall.final_time - exact_infall):.1e})")
    report("   Same law, same speed, same separation: the outcome is decided by L, which")
    report("   gravity is free to choose.  Pursuit is not: its velocity is radial by")
    report("   construction, so L = 0 is imposed and every initial condition collides.")
    return gravity, orbit, infall, period


def run_moving_target(report: Report):
    report()
    report("C. Classical moving-target pursuit curve (independent solver check)")
    v, w = 1.0, 0.6
    mt = MovingTargetPursuit(speed=v, target_velocity=(0.0, w), target_position=(0.0, 0.0))
    y0 = np.array([2.0, 0.0])
    exact = mt.capture_time(y0)
    sol = integrate(mt, y0, (0.0, 1.2 * exact), 1e-5, method="rk4",
                    events=[capture_event(mt, 1e-3)])
    invariant = np.array([mt.invariant(t, y) for t, y in zip(sol.t, sol.y)])
    predicted = mt.invariant(0.0, y0) + (w**2 - v**2) * sol.t
    extrapolated = sol.final_time + invariant[-1] / (v**2 - w**2)
    report(f"   exact capture time v*rho_0/(v^2 - w^2) = {exact:.12f}")
    report(f"   max|I(t) - I(0) - (w^2 - v^2)t|        = {np.max(np.abs(invariant - predicted)):.2e}")
    report(f"   capture time from extrapolating I -> 0 = {extrapolated:.12f}"
           f"   (error {abs(extrapolated - exact):.1e})")
    report("   The invariant is linear to 1e-11 along a curved trajectory, which tests the")
    report("   solver far more sharply than a single closed-form radius does.")
    return mt, sol, invariant, predicted, exact


def make_figure(pursuit, pursuit_sol, gravity, orbit, infall, mt, mt_sol,
                invariant, predicted, period, exact_capture):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))

    ax = axes[0, 0]
    ax.plot(pursuit_sol.y[:, 0], pursuit_sol.y[:, 1], color="#c1272d",
            label="pursuit (L = 0 forced)")
    ax.plot(orbit.y[:, 0], orbit.y[:, 1], color="#0b6e4f",
            label="gravity, tangential (orbits)")
    ax.plot(infall.y[:, 0], infall.y[:, 1], color="#2b5d9e", ls="--",
            label="gravity, radial (L = 0)")
    ax.plot(0, 0, "k+", ms=8)
    ax.plot(RHO0, 0, "ko", ms=3)
    ax.annotate("start", (RHO0, 0), textcoords="offset points", xytext=(4, 5), fontsize=7)
    ax.set(xlabel="x", ylabel="y", title="Same separation, same speed, three outcomes")
    ax.set_aspect("equal")
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    ax.plot(pursuit_sol.t, pursuit_sol.radius, color="#c1272d", label="pursuit")
    ax.plot([0, exact_capture], [RHO0, 0.0], "k:", lw=1.0,
            label=r"exact $\rho_0 - vt$")
    keep = orbit.t <= 2.0
    ax.plot(orbit.t[keep], orbit.radius[keep], color="#0b6e4f", label="gravity, tangential")
    ax.plot(infall.t, infall.radius, color="#2b5d9e", ls="--", label="gravity, radial")
    ax.axvline(exact_capture, color="#c1272d", ls=":", lw=0.8)
    ax.annotate(r"capture at $T = \rho_0/v$", (exact_capture, 0.55), rotation=90,
                fontsize=7, ha="right")
    ax.set(xlabel="t", ylabel=r"$\rho(t)$", xlim=(0, 2.0), ylim=(0, 1.15),
           title="Separation: linear to zero, or oscillating")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    L_pursuit = np.array([pursuit.angular_momentum(t, y)
                          for t, y in zip(pursuit_sol.t, pursuit_sol.y)])
    ax.plot(pursuit_sol.t, np.abs(L_pursuit) + 1e-19, color="#c1272d", label="pursuit")
    L_orbit = np.abs(np.asarray(gravity.angular_momentum(orbit.t, orbit.y)))
    ax.plot(orbit.t[keep], L_orbit[keep], color="#0b6e4f", label="gravity, tangential")
    L_infall = np.abs(np.asarray(gravity.angular_momentum(infall.t, infall.y))) + 1e-19
    ax.plot(infall.t, L_infall, color="#2b5d9e", ls="--", label="gravity, radial")
    ax.set(xlabel="t", ylabel=r"$|L(t)|$", yscale="log", xlim=(0, 2.0), ylim=(1e-19, 5),
           title="Angular momentum: forced to zero, or conserved")
    ax.annotate("both are exactly zero;\nshifted onto the axis floor to be visible",
                (0.05, 3e-19), fontsize=7, color="#555555")
    ax.legend(loc="center right")

    ax = axes[1, 1]
    target = np.array([mt.target(t) for t in mt_sol.t])
    ax.plot(target[:, 0], target[:, 1], color="#7a7a7a", ls="--", label="target path")
    ax.plot(mt_sol.y[:, 0], mt_sol.y[:, 1], color="#c1272d", label="pursuit curve")
    ax.plot(*mt.capture_point(mt_sol.y[0]), "k*", ms=9, label="exact capture point")
    ax.set(xlabel="x", ylabel="y", title="Moving-target pursuit curve")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    inset = ax.inset_axes([0.14, 0.12, 0.42, 0.3])
    inset.plot(mt_sol.t, invariant, color="#c1272d", lw=1.0)
    inset.plot(mt_sol.t, predicted, "k:", lw=1.0)
    inset.set_title(r"$I = v\rho + wq$", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(alpha=0.2)

    save_figure(fig, "exp1_dichotomy.png")


def main():
    report = Report("Experiment 1: velocity-level pursuit versus force-level gravity")
    pursuit, pursuit_sol, timed, exact_capture = run_pursuit(report)
    gravity, orbit, infall, period = run_gravity(report)
    mt, mt_sol, invariant, predicted, mt_exact = run_moving_target(report)

    report()
    report("Summary of the dichotomy")
    report("   pursuit : first order, velocity radial by construction, L == 0 always,")
    report(f"             capture in finite time T = rho_0/v = {exact_capture:g}, no orbit possible")
    report("   gravity : second order, velocity a free initial condition, L conserved,")
    report("             collision only in the measure-zero case L = 0")

    save_table(
        [
            {"case": "pursuit", "E": "n/a", "L": 0.0,
             "outcome": "capture", "time": timed.final_time},
            {"case": "gravity_tangential", "E": gravity.energy(orbit.y[0]),
             "L": gravity.angular_momentum(0.0, orbit.y[0]), "outcome": "bounded",
             "time": period},
            {"case": "gravity_radial", "E": gravity.energy(infall.y[0]),
             "L": gravity.angular_momentum(0.0, infall.y[0]), "outcome": "collision",
             "time": infall.final_time},
            {"case": "moving_target_pursuit", "E": "n/a", "L": "n/a",
             "outcome": "capture", "time": mt_exact},
        ],
        "exp1_dichotomy.csv",
    )
    make_figure(pursuit, pursuit_sol, gravity, orbit, infall, mt, mt_sol,
                invariant, predicted, period, exact_capture)
    save_text(report.text(), "exp1_dichotomy.txt")


if __name__ == "__main__":
    main()