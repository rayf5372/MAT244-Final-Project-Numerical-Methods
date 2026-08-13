from __future__ import annotations

import numpy as np

from _common import Report, plt, save_figure, save_table, save_text

from twobody import (
    PowerLawCentralForce,
    apsidal_angle_near_circular,
    apsidal_angle_quadrature,
    measure_apsidal_angles,
    radial_period_quadrature,
    reference_solve,
)

# Turning-point pairs of increasing radial amplitude, all with the same geometric
# mean radius so that the comparison across amplitudes is not confounded by scale.
AMPLITUDE_CASES = [(0.99, 1.01), (0.9, 1.1), (0.7, 1.3), (0.5, 1.5), (0.25, 1.75)]
K = 1.0


def amplitude_of(rho_min: float, rho_max: float) -> float:
    return (rho_max - rho_min) / (rho_max + rho_min)


def three_routes(report: Report):
    """Cross-check the near-circular formula, the quadrature and the ODE."""
    report()
    report("1. Three independent routes to the apsidal angle Phi")
    report(f"   {'s':>6} {'near-circular':>14} {'quadrature':>12} {'ODE-measured':>13} "
           f"{'spread':>10}")
    rows, mismatch, tiny_mismatch = [], 0.0, 0.0
    for s in (-2.0, -1.0, 0.0, 0.5, 1.0, 1.5):
        system = PowerLawCentralForce(s=s, k=K)
        tiny = apsidal_angle_quadrature(system, 1.0 - 1e-4, 1.0 + 1e-4)
        result = measure_apsidal_angles(system, 0.7, 1.3, n_apsides=8)
        mismatch = max(mismatch, abs(result["mean"] - result["quadrature"]) / result["quadrature"])
        near = apsidal_angle_near_circular(s)
        tiny_mismatch = max(tiny_mismatch, abs(tiny - near) / near)
        report(
            f"   {s:6.2f} {near:14.9f} "
            f"{result['quadrature']:12.9f} {result['mean']:13.9f} {result['spread']:10.2e}"
        )
        rows.append(
            {
                "s": s,
                "phi_near_circular": near,
                "phi_quadrature_tiny_amplitude": tiny,
                "phi_quadrature": result["quadrature"],
                "phi_ode_measured": result["mean"],
                "ode_spread": result["spread"],
                "radial_period": result["radial_period"],
            }
        )
    report(f"   quadrature vs integrated trajectory, worst relative difference: {mismatch:.2e}")
    report(f"   near-circular formula vs quadrature at amplitude 1e-4:          "
           f"{tiny_mismatch:.2e}")
    report("   The first two columns are for a vanishing amplitude and the last two for")
    report("   turning points (0.7, 1.3), so they should agree only where Phi does not")
    report("   depend on amplitude -- at s = 1 and s = -2, and nowhere else in the table.")
    save_table(rows, "exp3_three_routes.csv")


def amplitude_dependence(report: Report):
    """The heart of the matter: is Phi independent of the size of the orbit?"""
    report()
    report("2. Amplitude dependence of Phi (this is the content of Bertrand's theorem)")
    exponents = np.concatenate([
        np.linspace(-2.5, 1.5, 41),
        np.array([-2.0, 1.0]),  # make sure the two special exponents are sampled exactly
        np.linspace(1.55, 1.95, 9),
    ])
    exponents = np.unique(np.round(exponents, 6))

    curves = {case: [] for case in AMPLITUDE_CASES}
    for s in exponents:
        system = PowerLawCentralForce(s=s, k=K)
        for case in AMPLITUDE_CASES:
            curves[case].append(apsidal_angle_quadrature(system, *case))
    curves = {case: np.array(values) for case, values in curves.items()}

    smallest, largest = AMPLITUDE_CASES[0], AMPLITUDE_CASES[-1]
    variation = np.abs(curves[largest] - curves[smallest]) / curves[smallest]

    # The threshold has to sit above the quadrature's own noise floor (~1e-11
    # relative at the largest amplitude) and far below any genuine variation.  The
    # measured values leave a gap of seven orders of magnitude, so the choice is not
    # delicate; the gap itself is reported below rather than assumed.
    threshold = 1e-9
    report(f"   relative change in Phi between amplitude {amplitude_of(*smallest):.4f} "
           f"and {amplitude_of(*largest):.4f}:")
    report(f"   {'s':>7} {'|dPhi|/Phi':>12}")
    for s, value in zip(exponents, variation):
        if s in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5) or value < threshold:
            flag = "  <- independent of amplitude" if value < threshold else ""
            report(f"   {s:7.2f} {value:12.3e}{flag}")

    invariant_exponents = exponents[variation < threshold]
    largest_invariant = variation[variation < threshold].max() if invariant_exponents.size else 0.0
    smallest_varying = variation[variation >= threshold].min()
    report()
    listed = ", ".join(f"{s:g}" for s in invariant_exponents)
    report(f"   exponents with no amplitude dependence: {listed}")
    report(f"   largest variation among those          : {largest_invariant:.2e}")
    report(f"   smallest variation among all others    : {smallest_varying:.2e}")
    report(f"   separation between the two groups      : a factor of "
           f"{smallest_varying / max(largest_invariant, 1e-300):.1e}")
    report("   The split is unambiguous, and the two exponents that come out of it are the")
    report("   inverse-square law (s = 1) and Hooke's law (s = -2) -- precisely the two")
    report("   Bertrand's theorem names.  At every other exponent Phi moves with amplitude,")
    report("   so no orbit shape can close for all bound initial conditions there.")

    rows = [
        {"s": s, **{f"phi_amp_{amplitude_of(*case):.4f}": curves[case][i]
                    for case in AMPLITUDE_CASES},
         "relative_variation": variation[i]}
        for i, s in enumerate(exponents)
    ]
    save_table(rows, "exp3_amplitude_dependence.csv")
    return exponents, curves, variation


def precession_rate(report: Report):
    r"""Report precession as the angle by which the apsides advance per radial period."""
    report()
    report("3. Apsidal precession per radial period, Delta = 2*Phi - 2*pi")
    report(f"   {'s':>6} {'Phi':>11} {'Delta (deg)':>13} {'closes?':>9}")
    rows = []
    for s in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.25, 1.5, 1.75):
        system = PowerLawCentralForce(s=s, k=K)
        phi = apsidal_angle_quadrature(system, 0.7, 1.3)
        delta = np.degrees(2.0 * phi - 2.0 * np.pi)
        ratio = phi / np.pi
        closes = _is_simple_rational(ratio)
        report(f"   {s:6.2f} {phi:11.7f} {delta:+13.5f} {str(closes):>9}")
        rows.append({"s": s, "phi": phi, "phi_over_pi": ratio,
                     "precession_deg_per_radial_period": delta, "closes": closes})
    report("   A negative Delta is regression of the apsides, a positive one advance.")
    report("   Gravity alone gives Delta = 0 exactly, which is why the Kepler ellipse is")
    report("   fixed in space rather than slowly turning.")
    save_table(rows, "exp3_precession.csv")


def _is_simple_rational(x: float, max_denominator: int = 12, tol: float = 1e-9) -> bool:
    """Whether ``x`` is a rational with a small denominator, i.e. the orbit closes soon."""
    for q in range(1, max_denominator + 1):
        if abs(x * q - round(x * q)) < tol:
            return True
    return False


def make_figure(exponents, curves, variation):
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8))

    ax = axes[0, 0]
    for case in AMPLITUDE_CASES:
        ax.plot(exponents, curves[case], lw=1.2,
                label=rf"amplitude {amplitude_of(*case):.3f}")
    fine = np.linspace(exponents[0], 1.97, 400)
    ax.plot(fine, [apsidal_angle_near_circular(s) for s in fine], "k:", lw=1.2,
            label=r"$\pi/\sqrt{2-s}$")
    for s, name in ((1.0, "gravity"), (-2.0, "Hooke")):
        ax.axvline(s, color="#999999", lw=0.8, ls="--")
        ax.annotate(name, (s + 0.05, 5.6), fontsize=7, rotation=90)
    ax.set(xlabel="exponent s", ylabel=r"apsidal angle $\Phi$", ylim=(1.2, 6.5),
           title=r"$\Phi$ vs $s$ at five amplitudes")
    ax.legend(loc="lower right", fontsize=7, ncol=2)

    ax = axes[0, 1]
    ax.semilogy(exponents, np.maximum(variation, 1e-17), "o-", ms=2.5, color="#2b5d9e")
    ax.axhline(1e-12, color="k", ls=":", lw=0.8)
    for s in (-2.0, 1.0):
        ax.axvline(s, color="#c1272d", lw=0.9, ls="--")
    for s, name in ((-2.0, "Hooke"), (1.0, "gravity")):
        ax.annotate(name, (s + 0.09, 2e-11), fontsize=7, rotation=90,
                    color="#c1272d", va="bottom", ha="left")
    ax.set(xlabel="exponent s", ylabel=r"$|\Delta\Phi| / \Phi$ across amplitudes",
           title="Amplitude dependence vanishes at exactly two exponents")

    ax = axes[0, 2]
    ax.plot(exponents, np.degrees(2.0 * curves[(0.7, 1.3)] - 2.0 * np.pi),
            color="#0b6e4f")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(1.0, color="#999999", ls="--", lw=0.8)
    ax.set(xlabel="exponent s", ylabel="precession per radial period (deg)",
           title="Apsides advance or regress except at s = 1")

    # Trajectories: two closed, two precessing.
    for ax, s in zip(axes[1], (1.0, -2.0, 0.5)):
        system = PowerLawCentralForce(s=s, k=K)
        rho_min, rho_max = 0.6, 1.4
        y0 = system.state_from_apsides(rho_min, rho_max)
        period = radial_period_quadrature(system, rho_min, rho_max)
        sol = reference_solve(system, y0, (0.0, 6.0 * period), n_out=4000)
        phi = apsidal_angle_quadrature(system, rho_min, rho_max)
        closed = _is_simple_rational(phi / np.pi)
        ax.plot(sol.y[:, 0], sol.y[:, 1], lw=0.7,
                color="#0b6e4f" if closed else "#c1272d")
        ax.plot(0, 0, "k+", ms=7)
        label = {1.0: "s = 1 (gravity)", -2.0: "s = -2 (Hooke)", 0.5: "s = 0.5"}[s]
        ax.set(xlabel="x", ylabel="y",
               title=f"{label}: " + (r"closed, $\Phi/\pi$ = "
                                     f"{phi / np.pi:.3f}" if closed else "precessing"))
        ax.set_aspect("equal")

    save_figure(fig, "exp3_bertrand.png")


def main():
    report = Report("Experiment 3: Bertrand's theorem via the apsidal angle")
    three_routes(report)
    exponents, curves, variation = amplitude_dependence(report)
    precession_rate(report)
    make_figure(exponents, curves, variation)
    save_text(report.text(), "exp3_bertrand.txt")


if __name__ == "__main__":
    main()