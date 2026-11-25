r"""Ion acoustic instability.

Electrons drifting through ions at a fraction of their thermal speed excite ion acoustic
waves when the electrons are much hotter than the ions: the waves' phase speed, near the
sound speed $c_s = \sqrt{T_e/m_i}$, falls on the electrons' rising slope, and inverse Landau
damping on the electrons beats the Landau damping on the ions. Cold ions damp little, so the
threshold drift falls as $T_e/T_i$ rises.

The study runs picnic's 1D `VlasovPoisson` with drifting Maxwellian electrons and ions
($m_i/m_e = 100$, so $c_s = 0.1\,v_{te}$) at three temperature ratios with the electrons
drifting at $v_{te}$, and a case below threshold, in a periodic box that holds the fastest mode three times, with $2^{18}$
markers per species. The modes grow from the marker noise; the potential is averaged over
$40/\omega_{pe}$ to take out the electrons' Langmuir noise, far faster than the ion waves,
and picnic's `instability_report` fits every mode's growth rate without a hand-picked window
and names the fastest. Speeds are in units of the electron thermal speed
$v_{te}$, wavenumbers in $1/\lambda_{De}$, times in $1/\omega_{pe}$.

The first figure is the threshold: the fastest growth rate of the kinetic dielectric of the
two drifting Maxwellians (`electrostatic_growth` in plasma-plots) against the drift, with the
picnic runs on it. The second is $\gamma(k)$ of one run, the third the ions' phase space at the
end, mildly heated by the saturated waves.

Compared: the growth rate of every box mode with linear kinetic theory, and that the case
below threshold does not grow. Not compared: the real frequencies against $k c_s$, the
saturation (anomalous resistivity), and the real mass ratio.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
from plasma_plots.theory.kinetic import Maxwellian, electrostatic_growth, maximum_growth
from scipy import constants as c

DENSITY = 1e15  # electrons and ions [m^-3]
THERMAL = 1e6  # the electron thermal speed v_te [m/s]
MASS = 100  # m_i/m_e
MODES = 8  # Fourier modes followed
SMOOTH = 40.0  # running mean of the potential [1/ω_pe], over the plasma oscillations
STABLE = 1e-3  # γ/ω_pe below which a case counts as stable
RATIOS = [5, 10, 20, 50]  # T_e/T_i of the threshold curves
# (T_e/T_i, drift/v_te) of the runs; the first is below threshold
RUNS = [(5, 0.2), (10, 1.0), (20, 1.0), (50, 1.0)]
SHOWN = (20, 1.0)  # the run of the second and third figures
PRESETS = {
    "smoke": {
        "runs": [(20, 1.0)],
        "markers": 2**11,
        "cells": 32,
        "time": 60.0,
        "ratios": [20],
        "drifts": 3,
    },
    "full": {
        "runs": RUNS,
        "markers": 2**18,
        "cells": 64,
        "time": None,
        "ratios": RATIOS,
        "drifts": 23,
    },
}


def species(ratio: float, drift: float) -> list:
    """Drifting Maxwellian electrons and ions at rest, in units of n, v_te and m_e."""
    return [
        Maxwellian(1.0, charge=-1, mass=1, thermal_speed=1.0, drift=drift),
        Maxwellian(1.0, charge=1, mass=MASS, thermal_speed=np.sqrt(1 / ratio / MASS)),
    ]


def theory(ratio: float, drift: float, k) -> np.ndarray:
    """The most unstable root of the dielectric, ω/ω_pe, at k in units of 1/λ_De."""
    return electrostatic_growth(np.asarray(k, dtype=float), species(ratio, drift))


def fastest(ratio: float, drift: float) -> tuple[float, complex]:
    """The wavenumber of maximum growth and the root there."""
    return maximum_growth(lambda k: theory(ratio, drift, k), (0.05, 2.0))


def run(ratio: float, drift: float, markers: int, cells: int, time, seed: int) -> dict:
    """One picnic run; return the mode growth rates, histories and an ion sample."""
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
    debye = THERMAL / omega_pe
    k_max, root = fastest(ratio, drift)
    if root.imag < STABLE:  # below threshold: a box for the usual wavelengths
        k_max, root = 0.6, complex(root.real, 0.0)
    length = 3 * 2 * np.pi / k_max * debye
    if time is None:  # about 20 / γ, or 1000 / ω_pe below threshold
        time = 20.0 / root.imag if root.imag >= STABLE else 1000.0
    dt = 0.2
    periodic_field = FieldBoundaryCondition.periodic()
    periodic = ParticleBoundaryCondition("periodic")
    weight = DENSITY * length / markers
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
                mass=MASS * c.m_e,
                injection_method="manual",
                macroparticle_weight=weight,
            ),
        },
        dt=dt / omega_pe,
    )
    steps = int(np.ceil(time / dt))
    sim = Simulation(
        VlasovPoisson(config),
        t_end=steps * dt / omega_pe,
        output=OutputParams(every=10**9),
    )
    rng = np.random.default_rng(seed)
    thermal = {"electrons": 1.0, "ions": np.sqrt(1 / ratio / MASS)}
    for i, (name, mean) in enumerate((("electrons", drift), ("ions", 0.0))):
        x, v = quiet_start(markers, [0.0], [length], 1, velocity_bases=(3 + 2 * i,))
        x = (x + 1e-6 * length * rng.standard_normal(x.shape)) % length
        sim.model.set_species_markers(name, x, THERMAL * (mean + thermal[name] * v))
    sim.model.fields.deposit()
    sim.model.fields.solve()
    monitor = InstabilityMonitor(samples=4000)
    monitor.record(sim.model)
    sim.add_callback(monitor, every=max(1, steps // 400))
    sim.run()
    data = monitor.dataset()
    # the ion waves are slow (ω ≈ 0.06 ω_pe) under the electrons' Langmuir noise at ω_pe
    window = max(1, round(SMOOTH / omega_pe / float(np.diff(data.t)[0])))
    data["potential_x"] = (
        data["potential_x"].rolling(t=window, center=True, min_periods=1).mean()
    )
    report = instability_report(data)

    k = 2 * np.pi * np.arange(1, MODES + 1) / length * debye
    rates, coupled = np.full(MODES, np.nan), np.zeros(MODES, dtype=bool)
    for mode in report.modes:
        if mode.axis == "x" and mode.mode <= MODES:
            rates[mode.mode - 1] = mode.growth_rate / omega_pe
            coupled[mode.mode - 1] = mode.name == COUPLED
    ions = sim.model.species_registry.get("ions").markers
    x = np.asarray(ions.get_active_positions())[0]
    v = np.asarray(ions.get_active_velocities())[0]
    pick = rng.choice(x.size, min(4000, x.size), replace=False)
    first = report.modes[0] if report.modes else None
    return {
        "ratio": ratio,
        "drift": drift,
        "k": k,
        "rates": rates,
        "coupled": coupled,
        "k_max": k_max,
        "gamma_max": root.imag,
        "name": first.name if first else "none",
        "mode": first.mode if first else 0,
        "ions": (x[pick] / length, v[pick] / THERMAL),
    }


def simulate(preset: str = "full") -> dict:
    """Run the preset's cases; return their results and the threshold curves."""
    sizes = PRESETS[preset]
    drifts = np.linspace(0.1, 1.2, sizes["drifts"])
    curves = {
        ratio: np.array([fastest(ratio, u)[1].imag for u in drifts])
        for ratio in sizes["ratios"]
    }
    runs = [
        run(ratio, drift, sizes["markers"], sizes["cells"], sizes["time"], seed=i)
        for i, (ratio, drift) in enumerate(sizes["runs"])
    ]
    return {"drifts": drifts, "curves": curves, "runs": runs}


def measured(r: dict) -> float:
    """The growth rate of the box mode at the theory's fastest wavenumber (mode 3)."""
    return r["rates"][2] if r["gamma_max"] > 0 else np.nanmax(np.append(r["rates"], 0))


def plot(result: dict) -> None:
    """The threshold, γ(k) of one run and its ions' phase space."""
    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The fastest growth rate against the electron drift for four temperature "
        "ratios (lines, kinetic theory) and picnic's runs at the same ratios (markers): "
        "the hotter the electrons against the ions, the lower the threshold drift.",
    ) as fig:
        ax = fig[0]
        for i, (ratio, gamma) in enumerate(result["curves"].items()):
            ax.plot(
                result["drifts"],
                np.where(gamma > 0, gamma, np.nan),
                color=f"C{i}",
                label=f"$T_e/T_i = {ratio}$",
            )
            for r in result["runs"]:
                if r["ratio"] == ratio:
                    ax.plot(r["drift"], measured(r), "o", color=f"C{i}", ms=6)
        ax.set_xlabel(r"electron drift $u/v_{te}$")
        ax.set_ylabel(r"$\gamma_{max}/\omega_{pe}$")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    shown = next(
        (r for r in result["runs"] if (r["ratio"], r["drift"]) == SHOWN),
        result["runs"][-1],
    )
    with plasma_plots.figure(
        figsize=(6, 4),
        caption=f"Growth rate against wavenumber at $T_e/T_i = {shown['ratio']}$ and "
        f"$u = {shown['drift']}\\,v_{{te}}$: picnic's box modes (filled; hollow if driven "
        "by two growing modes) on the kinetic theory (line).",
    ) as fig:
        ax = fig[0]
        k = np.linspace(0.05, 1.2 * shown["k"][-1], 300)
        gamma = theory(shown["ratio"], shown["drift"], k).imag
        ax.plot(k, np.where(gamma > 0, gamma, np.nan), color="0.4", label="theory")
        linear = np.where(shown["coupled"], np.nan, shown["rates"])
        driven = np.where(shown["coupled"], shown["rates"], np.nan)
        ax.plot(shown["k"], linear, "o", color="C0", label="picnic")
        ax.plot(shown["k"], driven, "o", mfc="none", color="C0", label="coupled")
        ax.set_xlabel(r"$k\lambda_{De}$")
        ax.set_ylabel(r"$\gamma/\omega_{pe}$")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The ions' phase space at the end of the same run: a mild heating and a "
        "short tail towards the sound speed; at this saturation level few ions are "
        "trapped.",
    ) as fig:
        ax = fig[0]
        x, v = shown["ions"]
        ax.plot(x, v / np.sqrt(1 / MASS), ".", ms=1.5, color="C1")
        ax.set_xlabel("$x/L$")
        ax.set_ylabel("$v/c_s$")


def table(result: dict) -> str:
    """Each run: the theory's fastest mode against picnic, and the report's name."""
    lines = [
        "| $T_e/T_i$ | $u/v_{te}$ | Theory $k\\lambda_{De}$ | Theory $\\gamma_{max}/\\omega_{pe}$ "
        "| picnic $\\gamma/\\omega_{pe}$ | Named |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["runs"]:
        gamma = f"{r['gamma_max']:.4f}" if r["gamma_max"] > 0 else "stable"
        lines.append(
            f"| {r['ratio']} | {r['drift']} | {r['k_max']:.3f} | {gamma} | "
            f"{measured(r):.4f} | {r['name']} (mode {r['mode']}) |"
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
