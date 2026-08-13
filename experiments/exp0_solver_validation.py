from __future__ import annotations

import numpy as np

from _common import Report, plt, save_figure, save_table, save_text

from twobody import (
    METHOD_ORDERS,
    KeplerOrbit,
    PowerLawCentralForce,
    conservation_report,
    convergence_order,
    integrate,
    reference_solve,
    trajectory_error,
)

GRAVITY = PowerLawCentralForce.gravity(mu=1.0)
ORBIT = KeplerOrbit(mu=1.0, a=1.0, e=0.5)
METHODS = ("euler", "rk4", "verlet")
LABELS = {"euler": "forward Euler", "rk4": "classical RK4", "verlet": "velocity-Verlet"}
STYLES = {"euler": "#c1272d", "rk4": "#2b5d9e", "verlet": "#0b6e4f"}

STEP_RANGES = {
    "euler": np.array([2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5]),
    "rk4": np.array([6.4e-2, 3.2e-2, 1.6e-2, 8e-3, 4e-3, 2e-3]),
    "verlet": np.array([1.6e-2, 8e-3, 4e-3, 2e-3, 1e-3, 5e-4]),
}


def convergence_study(report: Report):
    """Global error against the exact solution over half an orbit."""
    report()
    report("1. Order of accuracy (global position error vs the exact Kepler ellipse)")
    report(f"   orbit: a = {ORBIT.a}, e = {ORBIT.e}, period = {ORBIT.period:.6f}")
    y0 = ORBIT.initial_state()
    t_end = 0.5 * ORBIT.period

    errors, costs, rows = {}, {}, []
    for method in METHODS:
        step_sizes = STEP_RANGES[method]
        errs, cost = [], []
        for dt in step_sizes:
            sol = integrate(GRAVITY, y0, (0.0, t_end), dt, method=method)
            errs.append(trajectory_error(sol, ORBIT.state))
            cost.append(sol.n_rhs_evals)
        errors[method], costs[method] = np.array(errs), np.array(cost)
        order = convergence_order(step_sizes, errs)
        report(
            f"   {LABELS[method]:>16s}: measured order {order:5.2f}  (theory "
            f"{METHOD_ORDERS[method]})   errors {errs[0]:.2e} -> {errs[-1]:.2e} "
            f"over dt {step_sizes[0]:.1e} -> {step_sizes[-1]:.1e}"
        )
        for dt, err, c in zip(step_sizes, errs, cost):
            rows.append(
                {"method": method, "dt": dt, "position_error": err, "force_evals": c,
                 "measured_order": order, "theoretical_order": METHOD_ORDERS[method]}
            )
    save_table(rows, "exp0_convergence.csv")
    return errors, costs


def equal_cost_comparison(report: Report, errors, costs):
    report()
    report("2. Accuracy per force evaluation (equal-cost comparison)")
    budget = 5e4
    for method in METHODS:
        c, e = costs[method], errors[method]
        interpolated = np.exp(np.interp(np.log(budget), np.log(c), np.log(e)))
        report(f"   {LABELS[method]:>16s}: error {interpolated:.2e} at {budget:.0e} evaluations")
    report("   RK4 buys by far the most accuracy per evaluation over a short run, so it is")
    report("   the better choice when the goal is an accurate trajectory.  The case for")
    report("   Verlet is not accuracy at all; it is the structure preserved indefinitely.")


def long_run_conservation(report: Report):
    report()
    report("3-4. Conservation over 200 orbits (dt = 5e-3, ~1257 steps per orbit)")
    y0 = ORBIT.initial_state()
    duration = 200.0 * ORBIT.period
    reports, solutions, rows = {}, {}, []
    for method in METHODS:
        sol = integrate(GRAVITY, y0, (0.0, duration), 5e-3, method=method, store_every=20)
        rep = conservation_report(GRAVITY, sol)
        reports[method], solutions[method] = rep, sol
        secular_share = abs(rep.secular_energy_drift) * duration / rep.max_energy_drift
        report(
            f"   {LABELS[method]:>16s}: max|dE/E| = {rep.max_energy_drift:.3e}   "
            f"max|dL/L| = {rep.max_L_drift:.3e}   "
            f"secular share of energy error = {secular_share:6.1%}"
        )
        rows.append(
            {
                "method": method,
                "max_energy_drift": rep.max_energy_drift,
                "max_L_drift": rep.max_L_drift,
                "secular_energy_drift_per_time": rep.secular_energy_drift,
                "secular_share": secular_share,
                "final_radius": float(sol.radius[-1]),
                "exact_max_radius": ORBIT.rho_max,
            }
        )
    save_table(rows, "exp0_conservation.csv")
    return reports, solutions


def make_figure(errors, costs, reports, solutions):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))

    ax = axes[0, 0]
    for method in METHODS:
        steps = STEP_RANGES[method]
        ax.loglog(steps, errors[method], "o-", color=STYLES[method], label=LABELS[method])
        p = METHOD_ORDERS[method]
        reference = errors[method][0] * (steps / steps[0]) ** p
        ax.loglog(steps, 0.3 * reference, "k:", lw=0.8, alpha=0.6)
        ax.annotate(rf"$\Delta t^{p}$", (steps[-1], 0.3 * reference[-1]), fontsize=7, alpha=0.8)
    ax.set(xlabel=r"step size $\Delta t$", ylabel="max position error",
           title="Order of accuracy vs exact Kepler")
    ax.legend()

    ax = axes[0, 1]
    for method in METHODS:
        ax.loglog(costs[method], errors[method], "o-", color=STYLES[method], label=LABELS[method])
    ax.set(xlabel="force evaluations", ylabel="max position error",
           title="Accuracy per unit cost")
    ax.legend()

    ax = axes[0, 2]
    for method in METHODS:
        rep = reports[method]
        E0 = rep.energy[0]
        ax.semilogy(rep.t / ORBIT.period, np.abs((rep.energy - E0) / E0) + 1e-18,
                    color=STYLES[method], label=LABELS[method], lw=0.9)
    ax.set(xlabel="orbits", ylabel=r"$|\Delta E / E_0|$", title="Energy error over 200 orbits")
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    for method in METHODS:
        rep = reports[method]
        L0 = rep.angular_momentum[0]
        ax.semilogy(rep.t / ORBIT.period, np.abs((rep.angular_momentum - L0) / L0) + 1e-18,
                    color=STYLES[method], label=LABELS[method], lw=0.9)
    ax.axhline(2.2e-16, color="k", ls=":", lw=0.8)
    ax.annotate("machine epsilon", (1.0, 3e-16), fontsize=7)
    ax.set(xlabel="orbits", ylabel=r"$|\Delta L / L_0|$",
           title="Angular momentum: Verlet is exact")
    ax.legend(loc="center right")

    ax = axes[1, 1]
    # Euler is shown over its first 15 orbits only: by orbit 200 it has spiralled out
    # to rho ~ 26 and would compress everything else onto a single pixel.
    euler = solutions["euler"]
    keep = euler.t <= 15.0 * ORBIT.period
    ax.plot(euler.y[keep, 0], euler.y[keep, 1], color=STYLES["euler"], lw=0.5,
            label="Euler (15 orbits)")
    verlet = solutions["verlet"]
    ax.plot(verlet.y[:, 0], verlet.y[:, 1], color=STYLES["verlet"], lw=0.5,
            label="Verlet (200 orbits)")
    exact = ORBIT.state(np.linspace(0, ORBIT.period, 400))
    ax.plot(exact[:, 0], exact[:, 1], "k--", lw=1.0, label="exact ellipse")
    ax.plot(0, 0, "k+", ms=6)
    ax.set(xlabel="x", ylabel="y", xlim=(-3.6, 1.4), ylim=(-2.5, 2.5),
           title="Euler unwinds; Verlet retraces one ellipse")
    ax.set_aspect("equal")
    ax.legend(loc="lower left")

    ax = axes[1, 2]
    for method in METHODS:
        sol = solutions[method]
        ax.plot(sol.t / ORBIT.period, sol.radius, color=STYLES[method], lw=0.4,
                label=LABELS[method])
    ax.axhline(ORBIT.rho_max, color="k", ls="--", lw=0.8)
    ax.axhline(ORBIT.rho_min, color="k", ls="--", lw=0.8)
    ax.set(xlabel="orbits", ylabel=r"$\rho$", title="Separation stays in its band only for Verlet")
    ax.legend(loc="upper left")

    save_figure(fig, "exp0_solver_validation.png")



def main():
    report = Report("Experiment 0: solver validation against the exact Kepler solution")
    errors, costs = convergence_study(report)
    equal_cost_comparison(report, errors, costs)
    reports, solutions = long_run_conservation(report)
    make_figure(errors, costs, reports, solutions)
    save_text(report.text(), "exp0_solver_validation.txt")


if __name__ == "__main__":
    main()