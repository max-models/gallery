r"""Multi-species streaming.

Two electrostatic instabilities of plasmas with more than two populations, run with picnic's
1D `VlasovPoisson` in a periodic box that holds eight Fourier modes across the unstable range:
a cool electron beam through hot electrons (the electron acoustic instability, a slow wave
carried by the cool population, excited when the beam's drift puts the wave on the slope of
the distribution), and the three-species plasma of a negative-ion source's extraction region,
where H⁻ ions are drawn one way and H⁺ ions the other through hot electrons (an ion–ion
streaming instability). The ions have $m_i/m_e = 100$ and $T_i = T_e/20$.

As in the beam study, the modes grow from the marker noise, picnic's `instability_report`
fits every mode without a hand-picked window, names the fastest and marks the modes driven by
two growing ones, and the linear modes are compared with the most unstable root of the
electrostatic dielectric of the same drifting Maxwellians (`electrostatic_growth` in
plasma-plots). Speeds are in units of the electron thermal speed $v_{te}$, wavenumbers in
$1/\lambda_{De}$ (of the hot electrons, density 1), times in $1/\omega_{pe}$.

Compared: $\gamma(k)$ of each mode with linear kinetic theory. Not compared: the real
frequencies, the saturation, and the real mass ratio of hydrogen. Slow ion waves in hot
electrons (ion–ion and ion acoustic) are left out: at this number of markers their growth
stays below the electrons' noise.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
from plasma_plots.theory.kinetic import Maxwellian, electrostatic_growth, maximum_growth
from scipy import constants as c

DENSITY = 1e15  # the reference (hot) electron density [m^-3]
SPEED = 1e6  # the electron thermal speed v_te [m/s]
MODES = 8  # Fourier modes followed
ION = np.sqrt(1 / 20 / 100)  # the ions' thermal speed: m_i = 100 m_e, T_i = T_e/20
# Each case: its species as (density, charge, mass/m_e, thermal speed, drift), in units of
# the reference density and v_te; and how often the box holds the fastest mode.
CASES = {
    "electron acoustic": {
        "species": [(0.8, -1, 1, 1.0, 0.0), (0.2, -1, 1, 0.1, 1.5)],
        "at": 3,
        "min_r2": 0.9,  # the hot electrons' noise makes the growth ragged
    },
    "H⁻ extraction": {
        "species": [
            (1.0, -1, 1, 1.0, 0.0),
            (2.0, -1, 100, ION, 0.1),
            (3.0, 1, 100, ION, -0.1 * 2 / 3),
        ],
        "at": 3,
    },
}
PRESETS = {
    "smoke": {
        "cases": ["electron acoustic"],
        "markers": 2**12,
        "cells": 64,
        "time": 30,
    },
    "full": {"cases": list(CASES), "markers": 2**16, "cells": 256, "time": None},
}


def theory(species: list, k: np.ndarray) -> np.ndarray:
    """The most unstable root of the dielectric, ω/ω_pe, at k in units of 1/λ_De."""
    maxwellians = [
        Maxwellian(n, charge=q, mass=m, thermal_speed=vt, drift=u)
        for n, q, m, vt, u in species
    ]
    return electrostatic_growth(np.asarray(k, dtype=float), maxwellians)


def run(case: dict, markers: int, cells: int, time: float | None, seed: int) -> dict:
    """One picnic run; return the mode amplitudes, the report and a phase-space sample."""
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
    from picnic.post_processing.instabilities import COUPLED

    omega_pe = np.sqrt(DENSITY * c.e**2 / (c.epsilon_0 * c.m_e))
    k_max, root = maximum_growth(lambda k: theory(case["species"], k), (0.05, 3.0))
    length = case["at"] * 2 * np.pi / k_max * SPEED / omega_pe
    if time is None:  # about 12 e-foldings of the fastest mode, then saturation
        time = 20.0 / root.imag
    dt = min(0.1, 0.05 / root.imag)
    periodic_field = FieldBoundaryCondition.periodic()
    periodic = ParticleBoundaryCondition("periodic")
    species = {}
    for i, (n, q, m, _, _) in enumerate(case["species"]):
        species[f"s{i}"] = Element(
            "e-" if q < 0 else "H+",
            name=f"s{i}",
            mass=m * c.m_e,
            injection_method="manual",
            macroparticle_weight=n * DENSITY * length / markers,
        )
    config = VlasovPoissonConfig(
        grid=GridParams(
            ndim=1, num_gridpoints=cells, upper_bound=length, domain_decomposition=False
        ),
        bcs_field=FieldBCs(x_lower=periodic_field, x_upper=periodic_field),
        bcs_particles=ParticleBCs(x_lower=periodic, x_upper=periodic),
        solver=SolverParams(
            options=SolverOptions(type="spectral"), neutralize_charge_density=True
        ),
        species=species,
        dt=dt / omega_pe,
    )
    steps = int(np.ceil(time / dt))
    sim = Simulation(
        VlasovPoisson(config),
        t_end=steps * dt / omega_pe,
        output=OutputParams(every=10**9),
    )
    rng = np.random.default_rng(seed)
    for i, (_, _, _, vt, u) in enumerate(case["species"]):
        x, v = quiet_start(markers, [0.0], [length], 1, velocity_bases=(3 + 2 * i,))
        x = (x + 1e-6 * length * rng.standard_normal(x.shape)) % length
        sim.model.set_species_markers(f"s{i}", x, SPEED * (u + vt * v))
    sim.model.fields.deposit()
    sim.model.fields.solve()
    monitor = InstabilityMonitor(samples=4000)
    monitor.record(sim.model)
    sim.add_callback(monitor, every=max(1, steps // 400))
    sim.run()
    data = monitor.dataset()
    report = instability_report(data, min_r2=case.get("min_r2", 0.95))

    tau = data.t.values * omega_pe
    amplitude = np.abs(np.fft.rfft(data["potential_x"].values, axis=1))[
        :, 1 : MODES + 1
    ]
    k = 2 * np.pi * np.arange(1, MODES + 1) / length * SPEED / omega_pe
    rates, coupled = np.full(MODES, np.nan), np.zeros(MODES, dtype=bool)
    for mode in report.modes:
        if mode.axis == "x" and mode.mode <= MODES:
            rates[mode.mode - 1] = mode.growth_rate / omega_pe
            coupled[mode.mode - 1] = mode.name == COUPLED
    phase_space = []
    for i in range(len(case["species"])):
        collection = sim.model.species_registry.get(f"s{i}")
        x = np.asarray(collection.markers.get_active_positions())[0]
        v = np.asarray(collection.markers.get_active_velocities())[0]
        pick = rng.choice(x.size, min(3000, x.size), replace=False)
        phase_space.append((x[pick] / length, v[pick] / SPEED))
    fastest = report.modes[0] if report.modes else None
    return {
        "k": k,
        "rates": rates,
        "coupled": coupled,
        "tau": tau,
        "amplitude": amplitude,
        "k_max": k_max,
        "gamma_max": root.imag,
        "name": fastest.name if fastest else "none",
        "mode": fastest.mode if fastest else 0,
        "phase_space": phase_space,
    }


def simulate(preset: str = "full") -> dict:
    """Run the cases of the preset; return their mode growth rates and histories."""
    sizes = PRESETS[preset]
    return {
        name: run(CASES[name], sizes["markers"], sizes["cells"], sizes["time"], seed=i)
        for i, name in enumerate(sizes["cases"])
    }


def plot(result: dict) -> None:
    """γ(k) against theory, the fastest mode's history and the final phase space."""
    names = list(result)
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    with plasma_plots.figure(
        rows,
        cols,
        figsize=(3.2 * cols, 2.8 * rows),
        caption="Growth rate against wavenumber: picnic's box modes (filled) on the "
        "most unstable root of the kinetic dielectric of the same drifting Maxwellians "
        "(line). Hollow: modes driven by two growing modes (nonlinear coupling, at the "
        "sum of their rates), which linear theory calls stable.",
    ) as fig:
        for i, name in enumerate(names):
            r, ax = result[name], fig[i]
            k = np.linspace(0.02, 1.3 * r["k"][-1], 300)
            gamma = theory(CASES[name]["species"], k).imag
            ax.plot(k, np.where(gamma > 0, gamma, np.nan), color="0.4", label="theory")
            linear = np.where(r["coupled"], np.nan, r["rates"])
            driven = np.where(r["coupled"], r["rates"], np.nan)
            ax.plot(r["k"], linear, "o", ms=5, color="C0", label="picnic")
            ax.plot(r["k"], driven, "o", ms=5, mfc="none", color="C0", label="coupled")
            ax.set_title(name)
            ax.set_xlabel(r"$k\lambda_{De}$")
            ax.set_ylabel(r"$\gamma/\omega_{pe}$")
            ax.set_ylim(bottom=0)
        fig[0].legend(fontsize=8)

    with plasma_plots.figure(
        rows,
        cols,
        figsize=(3.2 * cols, 2.8 * rows),
        caption="The amplitude of the potential's Fourier modes against time: growth "
        "from the marker noise, exponential over several e-foldings, then saturation by "
        "trapping. Dashed: the theoretical growth rate of the fastest mode.",
    ) as fig:
        for i, name in enumerate(names):
            r, ax = result[name], fig[i]
            fastest = CASES[name]["at"] - 1
            for m in range(r["amplitude"].shape[1]):
                ax.semilogy(
                    r["tau"],
                    r["amplitude"][:, m],
                    lw=1.6 if m == fastest else 0.6,
                    color=None if m == fastest else "0.7",
                )
            gamma = theory(CASES[name]["species"], r["k"][fastest]).imag
            a = r["amplitude"][:, fastest]
            j = int(np.argmin(np.abs(a - np.sqrt(a[0] * a.max()))))
            ax.semilogy(
                r["tau"],
                a[j] * np.exp(gamma * (r["tau"] - r["tau"][j])),
                "k--",
                lw=1,
            )
            ax.set_ylim(0.3 * a[: len(a) // 10 + 1].min(), 3 * r["amplitude"].max())
            ax.set_title(name)
            ax.set_xlabel(r"$\omega_{pe} t$")
            ax.set_ylabel(r"$|\phi_k|$ [V]")

    with plasma_plots.figure(
        rows,
        cols,
        figsize=(3.2 * cols, 2.8 * rows),
        caption="The phase space at the end of each run, a sample of the markers of "
        "each species (velocities of the ions magnified tenfold): the slow populations "
        "are trapped by the waves they excite.",
    ) as fig:
        for i, name in enumerate(names):
            ax = fig[i]
            for (x, v), (_, _, m, _, _) in zip(
                result[name]["phase_space"], CASES[name]["species"], strict=True
            ):
                ax.plot(x, v * (10 if m > 1 else 1), ".", ms=1)
            ax.set_title(name)
            ax.set_xlabel("$x/L$")
            ax.set_ylabel("$v/v_{te}$")


def table(result: dict) -> str:
    """The fastest mode of each case against theory, and its name in the report."""
    lines = [
        "| Case | Theory $k\\lambda_{De}$ | Theory $\\gamma_{max}/\\omega_{pe}$ "
        "| picnic $\\gamma/\\omega_{pe}$ (same mode) | Named |",
        "|---|---|---|---|---|",
    ]
    for name, r in result.items():
        m = CASES[name]["at"] - 1
        lines.append(
            f"| {name} | {r['k_max']:.3f} | {r['gamma_max']:.4f} | "
            f"{r['rates'][m]:.4f} | {r['name']} (mode {r['mode']}) |"
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
