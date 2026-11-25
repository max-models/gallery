r"""Buneman mass scaling.

Electrons drifting at $u$ through cold ions drive the Buneman instability, a reactive (fluid)
instability between the electrons' slow beam mode and the ion plasma oscillation. Its growth
rate scales with the cube root of the mass ratio: for $m_e/m_i \to 0$ the cold dispersion
relation $1 = \omega_{pi}^2/\omega^2 + \omega_{pe}^2/(\omega - k u)^2$ gives
$\gamma_{max} = (\sqrt 3/2)\,(m_e/2m_i)^{1/3}\,\omega_{pe}$ at $k u \approx \omega_{pe}$, so the
lighter the ions the faster the growth, and simulations with a reduced mass ratio overestimate
it.

The study runs picnic's 1D `VlasovPoisson` with electrons drifting at $u = 10^6$ m/s (thermal
spread $0.05\,u$) through ions at rest, for $m_i/m_e$ = 25, 100, 400 and 1836 (hydrogen), in a
periodic box that holds the fastest mode three times. The modes grow from the marker noise;
picnic's `instability_report` fits each mode's growth rate without a hand-picked window and
names the fastest. The growth rate of the fastest mode is compared with the cold quartic
(`beam_plasma_cold` in plasma-plots) and with its asymptotic cube-root law.

Compared: $\gamma_{max}$ against the mass ratio, with the cold quartic and the cube-root law.
Not compared: the real frequency, the saturation by electron trapping and the ion heating that
follows, and the effect of the electrons' thermal spread on the rate.

Reference: O. Buneman, Phys. Rev. 115, 503 (1959).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
from plasma_plots.theory.kinetic import beam_plasma_cold, maximum_growth
from scipy import constants as c

DENSITY = 1e15  # electrons and ions [m^-3]
DRIFT = 1e6  # u [m/s]
SPREAD = 0.05  # electron thermal speed / u
MODES = 8
PRESETS = {
    "smoke": {"ratios": [100], "markers": 2**11, "cells": 64, "time": 40.0},
    "full": {
        "ratios": [25, 100, 400, 1836],
        "markers": 2**16,
        "cells": 256,
        "time": None,
    },
}
OMEGA_PE = np.sqrt(DENSITY * c.e**2 / (c.epsilon_0 * c.m_e))


def cold(ratio: float, k) -> np.ndarray:
    """The cold Buneman root ω/ω_pe at k in units of ω_pe/u."""
    return beam_plasma_cold(k, 1.0, beam_density=1.0, plasma_density=1.0 / ratio)


def asymptotic(ratio) -> np.ndarray:
    """The growth rate γ_max/ω_pe of the cold relation for m_e/m_i → 0."""
    return np.sqrt(3) / 2 * (0.5 / np.asarray(ratio, dtype=float)) ** (1 / 3)


def run(ratio: float, markers: int, cells: int, time, seed: int) -> dict:
    """One picnic run; return the fastest mode's growth and the mode histories."""
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

    k_max, root = maximum_growth(lambda k: cold(ratio, k), (0.3, 1.5))
    length = 3 * 2 * np.pi / k_max * DRIFT / OMEGA_PE
    time = 20.0 / root.imag if time is None else time
    dt = 0.1
    weight = DENSITY * length / markers
    periodic_field = FieldBoundaryCondition.periodic()
    periodic = ParticleBoundaryCondition("periodic")
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
            "electrons": Element(
                "e-",
                name="electrons",
                injection_method="manual",
                macroparticle_weight=weight,
            ),
            "ions": Element(
                "H+",
                name="ions",
                mass=ratio * c.m_e,
                injection_method="manual",
                macroparticle_weight=weight,
            ),
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
    for i, (name, mean, spread) in enumerate(
        (("electrons", 1.0, SPREAD), ("ions", 0.0, SPREAD / np.sqrt(ratio)))
    ):
        x, v = quiet_start(markers, [0.0], [length], 1, velocity_bases=(3 + 2 * i,))
        x = (x + 1e-6 * length * rng.standard_normal(x.shape)) % length
        sim.model.set_species_markers(name, x, DRIFT * (mean + spread * v))
    sim.model.fields.deposit()
    sim.model.fields.solve()
    monitor = InstabilityMonitor(samples=4000)
    monitor.record(sim.model)
    sim.add_callback(monitor, every=max(1, steps // 400))
    sim.run()
    data = monitor.dataset()
    report = instability_report(data, max_mode=MODES)
    fastest = next((m for m in report.modes if m.mode == 3), None)
    first = report.modes[0] if report.modes else None
    amplitude = np.abs(np.fft.rfft(data["potential_x"].values, axis=1))
    return {
        "ratio": ratio,
        "k_max": k_max,
        "gamma_cold": root.imag,
        "gamma": fastest.growth_rate / OMEGA_PE if fastest else np.nan,
        "name": first.name if first else "none",
        "mode": first.mode if first else 0,
        "tau": data.t.values * OMEGA_PE,
        "amplitude": amplitude[:, 3],
    }


def simulate(preset: str = "full") -> dict:
    """Run the preset's mass ratios."""
    sizes = PRESETS[preset]
    runs = [
        run(ratio, sizes["markers"], sizes["cells"], sizes["time"], seed=i)
        for i, ratio in enumerate(sizes["ratios"])
    ]
    return {"runs": runs}


def plot(result: dict) -> None:
    """γ_max against the mass ratio, and the fastest mode's growth in each run."""
    ratios = np.geomspace(10, 5000, 60)
    cold_max = np.array(
        [maximum_growth(lambda k, r=r: cold(r, k), (0.3, 1.5))[1].imag for r in ratios]
    )
    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The growth rate of the fastest mode against the mass ratio: picnic "
        "(markers) on the cold quartic (line), which approaches the cube-root law "
        "$(\\sqrt 3/2)(m_e/2m_i)^{1/3}$ (dashed) for heavy ions.",
    ) as fig:
        ax = fig[0]
        ax.loglog(ratios, cold_max, color="0.4", label="cold quartic")
        ax.loglog(ratios, asymptotic(ratios), "k--", lw=1, label="cube-root law")
        runs = result["runs"]
        ax.loglog(
            [r["ratio"] for r in runs],
            [r["gamma"] for r in runs],
            "o",
            color="C0",
            label="picnic",
        )
        ax.set_xlabel(r"mass ratio $m_i/m_e$")
        ax.set_ylabel(r"$\gamma_{max}/\omega_{pe}$")
        ax.legend(fontsize=8)

    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The amplitude of the fastest mode against time for each mass ratio: "
        "the heavier the ions, the slower the growth and the later the saturation.",
    ) as fig:
        ax = fig[0]
        for i, r in enumerate(result["runs"]):
            ax.semilogy(
                r["tau"],
                r["amplitude"],
                color=f"C{i}",
                label=f"$m_i/m_e = {r['ratio']}$",
            )
        ax.set_xlabel(r"$\omega_{pe} t$")
        ax.set_ylabel(r"$|\phi_k|$ [V]")
        ax.legend(fontsize=8)


def table(result: dict) -> str:
    """Each mass ratio: the cold quartic against picnic, and the report's name."""
    lines = [
        "| $m_i/m_e$ | Cold quartic $\\gamma_{max}/\\omega_{pe}$ | Cube-root law "
        "| picnic | Named |",
        "|---|---|---|---|---|",
    ]
    for r in result["runs"]:
        lines.append(
            f"| {r['ratio']} | {r['gamma_cold']:.4f} | {asymptotic(r['ratio']):.4f} | "
            f"{r['gamma']:.4f} | {r['name']} (mode {r['mode']}) |"
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
