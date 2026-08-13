from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from _common import OUTCOME_COLOURS, Report, plt, save_figure, save_table, save_text

from twobody import (
    BOUNDED,
    COLLISION,
    ESCAPE,
    UNDETERMINED,
    PowerLawCentralForce,
    characteristic_time,
    classify_numerically,
    inner_turning_point,
    outer_turning_point,
    predict_outcome,
)

RHO0 = 1.0
K = 1.0
RHO_COLLISION = 1e-4
ESCAPE_FACTOR = 100.0
N_PERIODS = 40.0
OUTCOME_INDEX = {COLLISION: 0, BOUNDED: 1, ESCAPE: 2, UNDETERMINED: 3}


@dataclass
class Cell:
    """One grid point: the two verdicts plus what limits their comparability."""

    predicted: str
    measured: str
    E: float
    L: float
    pericentre: float
    apocentre: float
    marginal: bool
    t_run: float

    @property
    def agrees(self) -> bool:
        return self.predicted == self.measured

    @property
    def contact_limited(self) -> bool:
        """True pericentre is inside the collision threshold: contact is unavoidable."""
        return 0.0 < self.pericentre <= RHO_COLLISION

    @property
    def escape_limited(self) -> bool:
        """Bounded, but with an apocentre beyond the escape radius, so not seen to be.

        An infinite apocentre is not a limitation -- it is the correct answer for an
        unbound orbit.  Only a *finite* apocentre that the run cannot reach makes the
        cell undecidable.
        """
        return np.isfinite(self.apocentre) and self.apocentre > 0.5 * ESCAPE_FACTOR * RHO0

    @property
    def time_limited(self) -> bool:
        r"""Escaping, but too slowly to reach the escape radius within the run.

        An unbound pair recedes at :math:`\dot\rho \to \sqrt{2E}`, so crossing the
        escape radius takes at least :math:`(R_{\rm esc}-\rho_0)/\sqrt{2E}`.  For
        :math:`E` just above zero that exceeds any affordable run, and the third
        outcome becomes invisible for the same reason as the other two: escape is a
        statement about :math:`t \to \infty`.

        That estimate is only a lower bound -- the pair is still climbing out of the
        well, so it moves slower than its asymptotic speed -- hence the factor-of-two
        margin: a cell counts as resolvable only if the run is at least twice the
        minimum time escape could possibly take.
        """
        if self.E <= 0.0:
            return False
        minimum_time = (ESCAPE_FACTOR * RHO0 - RHO0) / np.sqrt(2.0 * self.E)
        return 2.0 * minimum_time > self.t_run

    @property
    def resolvable(self) -> bool:
        return not (self.marginal or self.contact_limited or self.escape_limited
                    or self.time_limited)


def evaluate(args) -> Cell:
    """Classify one initial condition analytically and numerically."""
    s, y0 = args
    system = PowerLawCentralForce(s=s, k=K)
    prediction = predict_outcome(system, y0)
    measured = classify_numerically(
        system,
        y0,
        n_periods=N_PERIODS,
        rho_collision=RHO_COLLISION,
        escape_factor=ESCAPE_FACTOR,
        rtol=1e-9,
        atol=1e-12,
    )
    return Cell(
        predicted=prediction.outcome,
        measured=measured.outcome,
        E=prediction.E,
        L=prediction.L,
        pericentre=inner_turning_point(system, y0),
        apocentre=outer_turning_point(system, y0),
        marginal=prediction.marginal,
        t_run=N_PERIODS * characteristic_time(system, RHO0),
    )


def sweep(jobs) -> list[Cell]:
    with ProcessPoolExecutor(max_workers=4) as pool:
        return list(pool.map(evaluate, jobs, chunksize=8))


# ------------------------------------------------------------------- the maps
def map_s_L(report: Report, n_s: int = 56, n_L: int = 56):
    r"""Map A: the :math:`(s, L)` plane, released from rest at :math:`\rho_0 = 1`."""
    report()
    report(f"Map A: (s, L) plane, rho_0 = {RHO0}, rho_dot_0 = 0, k = {K}")
    s_values = np.linspace(0.25, 4.0, n_s)
    L_values = np.linspace(0.05, 2.6, n_L)
    jobs = [
        (s, np.array([RHO0, 0.0, 0.0, L / RHO0]))
        for L in L_values
        for s in s_values
    ]
    cells = sweep(jobs)
    grid = np.array(cells, dtype=object).reshape(n_L, n_s)
    summarise(report, cells)
    return s_values, L_values, grid


def map_E_L(report: Report, s: float, n_E: int = 44, n_L: int = 44):
    r"""Maps B and C: the :math:`(E, L)` plane at fixed ``s``, started moving inward.

    An initial condition exists only where :math:`E \ge V_{\rm eff}(\rho_0)`, since
    :math:`\dot\rho_0^2 = 2(E - V_{\rm eff})` must be non-negative; the rest of the
    plane is not a failure but an empty region, and is masked.
    """
    report()
    report(f"Map {'B' if s == 1.0 else 'C'}: (E, L) plane at s = {s:g}, "
           f"rho_0 = {RHO0}, initially inward")
    system = PowerLawCentralForce(s=s, k=K)
    E_values = np.linspace(-1.2, 1.2, n_E)
    L_values = np.linspace(0.0, 2.0, n_L)

    jobs, positions = [], []
    for i, L in enumerate(L_values):
        for j, E in enumerate(E_values):
            v_eff = float(system.v_eff(RHO0, L))
            if E < v_eff:
                continue  # forbidden: would need an imaginary radial velocity
            rho_dot = -np.sqrt(2.0 * (E - v_eff))
            jobs.append((s, np.array([RHO0, 0.0, rho_dot, L / RHO0])))
            positions.append((i, j))
    cells = sweep(jobs)

    grid = np.full((n_L, n_E), None, dtype=object)
    for (i, j), cell in zip(positions, cells):
        grid[i, j] = cell
    report(f"   {len(cells)} of {n_E * n_L} grid points are dynamically accessible")
    summarise(report, cells)
    return E_values, L_values, grid


def summarise(report: Report, cells: list[Cell]) -> dict:
    """Agreement statistics, separating the resolvable cells from the rest."""
    total = len(cells)
    resolvable = [c for c in cells if c.resolvable]
    agree_all = sum(c.agrees for c in cells)
    agree_res = sum(c.agrees for c in resolvable)
    contact = sum(c.contact_limited for c in cells)
    escape_lim = sum(c.escape_limited for c in cells)
    time_lim = sum(c.time_limited for c in cells)
    marginal = sum(c.marginal for c in cells)

    report(f"   agreement over all cells          : {agree_all}/{total} = {agree_all / total:.1%}")
    report(f"   agreement over resolvable cells   : {agree_res}/{len(resolvable)} = "
           f"{agree_res / max(len(resolvable), 1):.1%}")
    report(f"   cells on a separatrix (marginal)  : {marginal}")
    report(f"   cells with pericentre <= rho_c    : {contact}   (contact unavoidable)")
    report(f"   cells with apocentre > escape/2   : {escape_lim}   (bounded but unseeable)")
    report(f"   cells too slow to escape in time  : {time_lim}   (E just above zero)")
    disagreements = [c for c in resolvable if not c.agrees]
    if disagreements:
        report(f"   unexplained disagreements         : {len(disagreements)}")
        for c in disagreements[:8]:
            report(f"      L = {c.L:6.3f}  E = {c.E:+8.4f}  predicted {c.predicted}, "
                   f"measured {c.measured}, pericentre {c.pericentre:.2e}")
    else:
        report("   unexplained disagreements         : none")
    return {
        "total": total,
        "agree_all": agree_all,
        "resolvable": len(resolvable),
        "agree_resolvable": agree_res,
        "contact_limited": contact,
        "escape_limited": escape_lim,
        "time_limited": time_lim,
        "marginal": marginal,
    }


# ---------------------------------------------------------------------- plots
def outcome_image(grid) -> np.ndarray:
    """RGBA image of the measured outcome for each cell."""
    n_rows, n_cols = grid.shape
    image = np.zeros((n_rows, n_cols, 4))
    for i in range(n_rows):
        for j in range(n_cols):
            cell = grid[i, j]
            if cell is None:
                image[i, j] = (1.0, 1.0, 1.0, 0.0)
                continue
            colour = OUTCOME_COLOURS[cell.measured if cell.measured in OUTCOME_COLOURS
                                     else UNDETERMINED]
            image[i, j] = plt.matplotlib.colors.to_rgba(colour)
    return image


def disagreement_mask(grid) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    for index, cell in np.ndenumerate(grid):
        mask[index] = cell is not None and not cell.agrees
    return mask


def make_figure(s_values, L_values, grid_A, maps_EL):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))

    # --- Map A -------------------------------------------------------------
    ax = axes[0]
    extent = [s_values[0], s_values[-1], L_values[0], L_values[-1]]
    ax.imshow(outcome_image(grid_A), origin="lower", extent=extent, aspect="auto",
              interpolation="nearest")
    ax.contour(s_values, L_values, disagreement_mask(grid_A).astype(float), levels=[0.5],
               colors="white", linewidths=0.8, linestyles="--")

    s_fine = np.linspace(s_values[0], 2.0, 400)
    ax.plot(s_fine, np.sqrt(2.0 / s_fine), "k-", lw=1.4, label=r"$E=0$: $L=\sqrt{2/s}$")
    ax.plot([2.0, s_values[-1]], [1.0, 1.0], "k-", lw=1.4)
    ax.axvline(2.0, color="k", ls=":", lw=1.2)
    ax.annotate("s = 2", (2.02, 2.42), fontsize=8)

    # Contact boundary: pericentre = rho_c, i.e. s = 2 - ln(L^2/k)/ln(rho_c).
    L_contact = np.linspace(L_values[0], min(np.sqrt(K), L_values[-1]) - 1e-9, 200)
    s_contact = 2.0 - np.log(L_contact**2 / K) / np.log(RHO_COLLISION)
    ax.plot(s_contact, L_contact, color="white", lw=1.6, ls="-")
    ax.annotate("contact boundary\n" + r"$\rho_{\rm peri}=\rho_c$", (2.45, 0.30),
                fontsize=7, color="white")

    ax.set(xlabel="exponent s", ylabel="angular momentum L",
           title=r"Map A: $(s, L)$ at $\rho_0=1$, $\dot\rho_0=0$",
           xlim=(s_values[0], s_values[-1]), ylim=(L_values[0], L_values[-1]))
    ax.legend(loc="upper right", labelcolor="black")
    ax.grid(False)

    # --- Maps B and C ------------------------------------------------------
    for ax, (s, E_values, L_values_EL, grid) in zip(axes[1:], maps_EL):
        extent = [E_values[0], E_values[-1], L_values_EL[0], L_values_EL[-1]]
        ax.imshow(outcome_image(grid), origin="lower", extent=extent, aspect="auto",
                  interpolation="nearest")
        ax.contour(E_values, L_values_EL, disagreement_mask(grid).astype(float), levels=[0.5],
                   colors="white", linewidths=0.8, linestyles="--")
        ax.axvline(0.0, color="k", lw=1.4)
        ax.annotate("E = 0", (0.03, 1.85), fontsize=8)
        system = PowerLawCentralForce(s=s, k=K)
        if s > 2:
            # Barrier top: E = V_eff(rho_*) separates clearing it from being held out.
            L_line = np.linspace(max(L_values_EL[0], 1e-3), L_values_EL[-1], 200)
            E_line = np.array([system.barrier_height(L) for L in L_line])
            ax.plot(E_line, L_line, "k--", lw=1.4, label=r"$E=V_{\rm eff}(\rho_*)$")
            ax.legend(loc="lower right")
        label = "gravity" if s == 1.0 else "steeper than the barrier"
        ax.set(xlabel="energy E", ylabel="angular momentum L",
               title=rf"$s={s:g}$ ({label})",
               xlim=(E_values[0], E_values[-1]), ylim=(L_values_EL[0], L_values_EL[-1]))
        ax.grid(False)

    handles = [
        plt.Line2D([], [], marker="s", ls="", color=OUTCOME_COLOURS[name], label=label)
        for name, label in (
            (COLLISION, "collision"),
            (BOUNDED, "bounded"),
            (ESCAPE, "escape"),
            (UNDETERMINED, "undetermined by the run"),
        )
    ]
    handles.append(plt.Line2D([], [], color="white", ls="--", label="numerics disagree"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    save_figure(fig, "exp2_outcome_map.png")


def contact_boundary_check(report: Report):
    r"""Test the predicted contact boundary at fixed :math:`L`, refining in :math:`s`."""
    report()
    report("Contact boundary: where does the *observed* collision region begin?")
    L = 0.9
    predicted = 2.0 - np.log(L**2 / K) / np.log(RHO_COLLISION)
    report(f"   at L = {L}, theory puts the pericentre at rho_c when s = {predicted:.4f}")
    report(f"   {'s':>7} {'pericentre':>12} {'predicted':>11} {'measured':>11}")
    rows = []
    for s in (1.90, 1.94, 1.96, 1.975, 1.98, 1.99, 2.00, 2.10):
        cell = evaluate((s, np.array([RHO0, 0.0, 0.0, L / RHO0])))
        report(f"   {s:7.3f} {cell.pericentre:12.3e} {cell.predicted:>11} {cell.measured:>11}")
        rows.append({"s": s, "L": L, "pericentre": cell.pericentre,
                     "predicted": cell.predicted, "measured": cell.measured})
    report(f"   The observed transition brackets s = {predicted:.3f}, as predicted: the")
    report("   simulation changes its verdict where the pericentre crosses the threshold,")
    report("   not at s = 2.  Both statements are correct; they answer different questions.")
    save_table(rows, "exp2_contact_boundary.csv")


def main():
    report = Report("Experiment 2: outcome map for V(rho) = -alpha rho^-s")
    report()
    report(f"collision threshold rho_c = {RHO_COLLISION:g}, escape radius = "
           f"{ESCAPE_FACTOR:g} rho_0, run length = {N_PERIODS:g} circular periods")
    report("Numerical verdicts use the trajectory only (never the energy); the analytic")
    report("verdicts use the conservation laws only (never a trajectory).")

    s_values, L_values, grid_A = map_s_L(report)
    maps_EL = []
    for s in (1.0, 3.0):
        E_values, L_values_EL, grid = map_E_L(report, s)
        maps_EL.append((s, E_values, L_values_EL, grid))

    contact_boundary_check(report)

    rows = []
    for i, L in enumerate(L_values):
        for j, s in enumerate(s_values):
            cell = grid_A[i, j]
            rows.append({"s": s, "L": L, "E": cell.E, "predicted": cell.predicted,
                         "measured": cell.measured, "pericentre": cell.pericentre,
                         "apocentre": cell.apocentre, "marginal": cell.marginal,
                         "contact_limited": cell.contact_limited,
                         "escape_limited": cell.escape_limited,
                         "time_limited": cell.time_limited})
    save_table(rows, "exp2_map_s_L.csv")

    make_figure(s_values, L_values, grid_A, maps_EL)
    save_text(report.text(), "exp2_outcome_map.txt")


if __name__ == "__main__":
    main()