r"""Beam–plasma regimes.

A weak electron beam in a plasma excites Langmuir waves in two ways. A cold beam drives a
reactive (fluid) instability: the beam's slow space-charge wave couples to the plasma
oscillation at $k u \approx \omega_{pe}$ and grows at
$\gamma \approx (\sqrt 3/2)(n_b/2n)^{1/3}\,\omega_{pe}$, independent of the beam's spread.
A warm beam, with a thermal spread larger than about $(n_b/n)^{1/3}\,u$, drives a kinetic
instability instead, the bump-on-tail: inverse Landau damping on the rising slope of $f(v)$,
with a rate proportional to $n_b$ and falling as the beam spreads out. The transition between
the two was worked out by O'Neil and Malmberg.

The study runs picnic's 1D `VlasovPoisson` with a beam of 5 % of the electrons at
$u = 10^6$ m/s through a cool plasma (thermal speed $0.05\,u$, neutralizing background), at
five beam spreads from $0.02\,u$ to $0.45\,u$, each in a periodic box that holds the fastest
mode three times. The modes grow from the marker noise; picnic's `instability_report` fits
each mode's growth rate without a hand-picked window and names the fastest. The growth rate of
the theory's fastest mode is compared with the most unstable root of the kinetic dielectric
of the two drifting Maxwellians (`electrostatic_growth` in plasma-plots) and with the cold
limit (`beam_plasma_cold`).

Compared: $\gamma_{max}$ against the beam spread, across the reactive–kinetic transition. Not
compared: the real frequencies and the saturation (trapping for a cold beam, a quasilinear
plateau for a warm one).

Reference: T. M. O'Neil and J. H. Malmberg, Phys. Fluids 11, 1754 (1968).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
from plasma_plots.theory.kinetic import (
    Maxwellian,
    beam_plasma_cold,
    electrostatic_growth,
    maximum_growth,
)
from scipy import constants as c

DENSITY = 1e15  # all electrons [m^-3]
SPEED = 1e6  # the beam speed u [m/s]
BEAM = 0.05  # n_b/n
BULK = 0.05  # thermal speed of the plasma electrons / u
MODES = 8
PRESETS = {
    "smoke": {"spreads": [0.1], "markers": 2**11, "cells": 64, "time": 20.0},
    "full": {
        "spreads": [0.02, 0.1, 0.2, 0.3, 0.45],
        "markers": 2**16,
        "cells": 256,
        "time": None,
    },
}
OMEGA_PE = np.sqrt(DENSITY * c.e**2 / (c.epsilon_0 * c.m_e))


def theory(spread: float, k) -> np.ndarray:
    """The most unstable root ω/ω_pe at k in units of ω_pe/u, beam spread `spread` u."""
    species = [
        Maxwellian(1 - BEAM, thermal_speed=BULK),
        Maxwellian(BEAM, thermal_speed=spread, drift=1.0),
    ]
    return electrostatic_growth(np.asarray(k, dtype=float), species)


def fastest(spread: float) -> tuple[float, complex]:
    """The wavenumber of maximum growth (in ω_pe/u) and the root there."""
    return maximum_growth(lambda k: theory(spread, k), (0.3, 2.0))


def cold_limit() -> float:
    """γ_max/ω_pe of the cold beam in the cold plasma."""
    return maximum_growth(
        lambda k: beam_plasma_cold(k, 1.0, beam_density=BEAM, plasma_density=1 - BEAM),
        (0.3, 2.0),
    )[1].imag


def run(spread: float, markers: int, cells: int, time, seed: int) -> dict:
    """One picnic run; return the fastest mode's growth and history, and a beam sample."""
    from picnic import Simulation
    from picnic.models.vlasov_poisson import VlasovPoisson
    from picnic.parameters import (
        FieldBCs,
        FieldBoundaryCondition,
        GridParams,
        OutputParams,
        ParticleBCs,
        ParticleBoundaryCondition,
        SolverOptions,
        SolverParams,
        VlasovPoissonConfig,
    )
    from picnic.particles.element import Element
    from picnic.particles.sampling import quiet_start
    from picnic.post_processing import InstabilityMonitor, instability_report

    k_max, root = fastest(spread)
    length = 3 * 2 * np.pi / k_max * SPEED / OMEGA_PE
    time = 20.0 / root.imag if time is None else time
    dt = 0.1
    periodic_field = FieldBoundaryCondition.periodic()
    periodic = ParticleBoundaryCondition("periodic")
    populations = {"plasma": (1 - BEAM, 0.0, BULK), "beam": (BEAM, 1.0, spread)}
    config = VlasovPoissonConfig(
        grid=GridParams(
            ndim=1, num_gridpoints=cells, upper_bound=length, domain_decomposition=False
        ),
        bcs_field=FieldBCs(x_lower=periodic_field, x_upper=periodic_field),
        bcs_particles=ParticleBCs(x_lower=periodic, x_upper=periodic),
        solver=SolverParams(
            options=SolverOptions(type="spectral"), neutralize_charge_density=True
        ),
        species={
            name: Element(
                "e-",
                name=name,
                injection_method="manual",
                macroparticle_weight=n * DENSITY * length / markers,
            )
            for name, (n, _, _) in populations.items()
        },
        dt=dt / OMEGA_PE,
    )
    steps = int(np.ceil(time / dt))
    sim = Simulation(
        VlasovPoisson(config),
        t_end=steps * dt / OMEGA_PE,
        output=OutputParams(every=10**9),
    )
    rng = np.random.default_rng(seed)
    for i, (name, (_, mean, width)) in enumerate(populations.items()):
        x, v = quiet_start(markers, [0.0], [length], 1, velocity_bases=(3 + 2 * i,))
        x = (x + 1e-6 * length * rng.standard_normal(x.shape)) % length
        sim.model.set_species_markers(name, x, SPEED * (mean + width * v))
    sim.model.fields.deposit()
    sim.model.fields.solve()
    monitor = InstabilityMonitor(samples=4000)
    monitor.record(sim.model)
    sim.add_callback(monitor, every=max(1, steps // 400))
    sim.run()
    data = monitor.dataset()
    report = instability_report(data, max_mode=MODES)
    mode = next((m for m in report.modes if m.mode == 3), None)
    first = report.modes[0] if report.modes else None
    amplitude = np.abs(np.fft.rfft(data["potential_x"].values, axis=1))
    beam = sim.model.species_registry.get("beam").markers
    v = np.asarray(beam.get_active_velocities())[0]
    return {
        "spread": spread,
        "k_max": k_max,
        "gamma_theory": root.imag,
        "gamma": mode.growth_rate / OMEGA_PE if mode else np.nan,
        "name": first.name if first else "none",
        "mode": first.mode if first else 0,
        "tau": data.t.values * OMEGA_PE,
        "amplitude": amplitude[:, 3],
        "beam_velocity": np.histogram(
            v / SPEED, bins=80, range=(-0.5, 2.5), density=True
        ),
    }


def simulate(preset: str = "full") -> dict:
    """Run the preset's beam spreads, and the theory's γ_max against the spread."""
    sizes = PRESETS[preset]
    runs = [
        run(spread, sizes["markers"], sizes["cells"], sizes["time"], seed=i)
        for i, spread in enumerate(sizes["spreads"])
    ]
    spreads = np.linspace(0.01, 0.6, 30 if preset == "full" else 3)
    gamma = np.array([fastest(s)[1].imag for s in spreads])
    return {"runs": runs, "spreads": spreads, "gamma": gamma}


def plot(result: dict) -> None:
    """γ_max against the beam spread, the mode histories and the beams' final f(v)."""
    spreads, gamma = result["spreads"], result["gamma"]
    runs = result["runs"]
    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The growth rate of the fastest mode against the beam's thermal spread: "
        "picnic (markers) on the kinetic theory (line). Flat at the cold-beam rate "
        "(dashed) for a cold beam, falling once the spread exceeds about "
        "$(n_b/n)^{1/3}\\,u$ (dotted), where the instability becomes kinetic.",
    ) as fig:
        ax = fig[0]
        ax.plot(spreads, gamma, color="0.4", label="kinetic theory")
        ax.axhline(cold_limit(), color="k", ls="--", lw=1, label="cold beam")
        ax.axvline(BEAM ** (1 / 3), color="k", ls=":", lw=1)
        ax.plot(
            [r["spread"] for r in runs],
            [r["gamma"] for r in runs],
            "o",
            color="C0",
            label="picnic",
        )
        ax.set_xlabel(r"beam thermal spread $v_b/u$")
        ax.set_ylabel(r"$\gamma_{max}/\omega_{pe}$")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The amplitude of the fastest mode against time for each beam spread.",
    ) as fig:
        ax = fig[0]
        for i, r in enumerate(runs):
            ax.semilogy(
                r["tau"],
                r["amplitude"],
                color=f"C{i}",
                label=f"$v_b = {r['spread']}\\,u$",
            )
        ax.set_xlabel(r"$\omega_{pe} t$")
        ax.set_ylabel(r"$|\phi_k|$ [V]")
        ax.legend(fontsize=8)

    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The beam electrons' velocity distribution at the end of each run: a cold "
        "beam is broken up by trapping and slowed, a warm beam spreads down towards the "
        "plasma into a plateau.",
    ) as fig:
        ax = fig[0]
        for i, r in enumerate(runs):
            density, edges = r["beam_velocity"]
            ax.plot(
                0.5 * (edges[1:] + edges[:-1]),
                density,
                color=f"C{i}",
                label=f"$v_b = {r['spread']}\\,u$",
            )
        ax.set_xlabel("$v/u$")
        ax.set_ylabel("$f_b(v)$ [1/u]")
        ax.legend(fontsize=8)


def table(result: dict) -> str:
    """Each beam spread: the theory against picnic, and the report's name."""
    lines = [
        "| $v_b/u$ | Theory $\\gamma_{max}/\\omega_{pe}$ | picnic | Named |",
        "|---|---|---|---|",
    ]
    for r in result["runs"]:
        lines.append(
            f"| {r['spread']} | {r['gamma_theory']:.4f} | {r['gamma']:.4f} | "
            f"{r['name']} (mode {r['mode']}) |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    result = simulate(args.preset)
    print(table(result))
    plot(result)
    plt.show()
