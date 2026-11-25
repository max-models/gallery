r"""Two-stream threshold.

Two equal electron beams streaming through each other at $\pm u$ are unstable when they are
cold: the two-stream instability grows at up to $\omega_{pe}/(2\sqrt 2)$. Warming the beams
shrinks the unstable band and the growth rate, and beyond a thermal speed of about $0.77\,u$
the two beams have merged into a single flat-topped distribution, and the plasma is stable
(Penrose: no dip in $f(v)$ deep enough).

The study runs picnic's 1D `VlasovPoisson` with two Maxwellian electron beams at
$\pm 10^6$ m/s over a neutralizing background, at five thermal speeds from $0.05\,u$ to
$0.85\,u$, each in a periodic box that holds the fastest mode three times (the last, stable
case in a box for $k u = 0.5\,\omega_{pe}$). The modes grow from the marker noise; picnic's
`instability_report` fits each mode's growth rate without a hand-picked window and names the
fastest. The growth rate of the theory's fastest mode is compared with the purely growing root
of the kinetic dispersion relation of the two beams (`two_stream` in plasma-plots).

Compared: $\gamma_{max}$ against the beams' thermal speed, the threshold, and that the case
beyond it does not grow. Not compared: the saturation and the electron holes it leaves.

Reference: T. H. Stix, Waves in Plasmas (AIP, 1992), sec. 8.11.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
from plasma_plots.theory.kinetic import maximum_growth, two_stream
from scipy import constants as c

DENSITY = 1e15  # all electrons [m^-3]
DRIFT = 1e6  # u [m/s]
MODES = 8
STABLE = 1e-3  # γ/ω_pe below which a case counts as stable
PRESETS = {
    "smoke": {"spreads": [0.3], "markers": 2**11, "cells": 64, "time": 20.0},
    "full": {
        "spreads": [0.05, 0.3, 0.5, 0.65, 0.85],
        "markers": 2**17,
        "cells": 256,
        "time": None,
    },
}
OMEGA_PE = np.sqrt(DENSITY * c.e**2 / (c.epsilon_0 * c.m_e))


def theory(spread: float, k) -> np.ndarray:
    """The two-stream root ω/ω_pe at k in units of ω_pe/u, thermal speed `spread` u."""
    # two_stream takes k in 1/λ_De and speeds in v_th: here λ_De = spread u/ω_pe
    return two_stream(np.asarray(k, dtype=float) * spread, 1.0 / spread, 1.0)


def fastest(spread: float) -> tuple[float, complex]:
    """The wavenumber of maximum growth (in ω_pe/u) and the root there."""
    return maximum_growth(lambda k: theory(spread, k), (0.01, 1.5))


def run(spread: float, markers: int, cells: int, time, seed: int) -> dict:
    """One picnic run; return the fastest mode's growth and history, and a sample."""
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
    if root.imag < STABLE:
        k_max, root = 0.5, complex(root.real, 0.0)
    length = 3 * 2 * np.pi / k_max * DRIFT / OMEGA_PE
    if time is None:
        time = 20.0 / root.imag if root.imag >= STABLE else 300.0
    dt = 0.1
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
            name: Element(
                "e-",
                name=name,
                injection_method="manual",
                macroparticle_weight=0.5 * DENSITY * length / markers,
            )
            for name in ("forward", "backward")
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
    for i, (name, mean) in enumerate((("forward", 1.0), ("backward", -1.0))):
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
    mode = next((m for m in report.modes if m.mode == 3), None)
    first = report.modes[0] if report.modes else None
    amplitude = np.abs(np.fft.rfft(data["potential_x"].values, axis=1))
    sample = []
    for name in ("forward", "backward"):
        markers_ = sim.model.species_registry.get(name).markers
        x = np.asarray(markers_.get_active_positions())[0]
        v = np.asarray(markers_.get_active_velocities())[0]
        pick = rng.choice(x.size, min(3000, x.size), replace=False)
        sample.append((x[pick] / length, v[pick] / DRIFT))
    return {
        "spread": spread,
        "k_max": k_max,
        "gamma_theory": root.imag,
        "gamma": mode.growth_rate / OMEGA_PE if mode else 0.0,
        "name": first.name if first else "none",
        "mode": first.mode if first else 0,
        "tau": data.t.values * OMEGA_PE,
        "amplitude": amplitude[:, 3],
        "phase_space": sample,
    }


def simulate(preset: str = "full") -> dict:
    """Run the preset's thermal speeds, and the theory's γ_max against the spread."""
    sizes = PRESETS[preset]
    runs = [
        run(spread, sizes["markers"], sizes["cells"], sizes["time"], seed=i)
        for i, spread in enumerate(sizes["spreads"])
    ]
    spreads = np.linspace(0.02, 0.9, 30 if preset == "full" else 3)
    gamma = np.array([fastest(s)[1].imag for s in spreads])
    return {"runs": runs, "spreads": spreads, "gamma": gamma}


def plot(result: dict) -> None:
    """The threshold, the mode histories and the phase space of the warmest unstable run."""
    spreads, gamma = result["spreads"], result["gamma"]
    runs = result["runs"]
    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The growth rate of the fastest mode against the beams' thermal speed: "
        "picnic (markers) on the kinetic theory (line), which reaches zero near "
        "$0.77\\,u$; the run beyond it does not grow.",
    ) as fig:
        ax = fig[0]
        ax.plot(spreads, np.where(gamma > 0, gamma, 0.0), color="0.4", label="theory")
        ax.plot(
            [r["spread"] for r in runs],
            [r["gamma"] for r in runs],
            "o",
            color="C0",
            label="picnic",
        )
        ax.set_xlabel(r"thermal speed of each beam $v_{th}/u$")
        ax.set_ylabel(r"$\gamma_{max}/\omega_{pe}$")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The amplitude of the fastest mode against time at each thermal speed: "
        "the warmer the beams, the slower the growth and the lower the saturation; at "
        "$0.85\\,u$ only the noise remains.",
    ) as fig:
        ax = fig[0]
        for i, r in enumerate(runs):
            ax.semilogy(
                r["tau"],
                r["amplitude"],
                color=f"C{i}",
                label=f"$v_{{th}} = {r['spread']}\\,u$",
            )
        ax.set_xlabel(r"$\omega_{pe} t$")
        ax.set_ylabel(r"$|\phi_k|$ [V]")
        ax.legend(fontsize=8)

    unstable = [r for r in runs if r["gamma_theory"] > 0]
    shown = unstable[-1] if unstable else runs[-1]
    with plasma_plots.figure(
        figsize=(6, 4),
        caption=f"The phase space at the end of the run at $v_{{th}} = {shown['spread']}"
        "\\,u$, a sample of the markers of each beam: the beams are trapped into "
        "vortices, electron holes, that mix them.",
    ) as fig:
        ax = fig[0]
        for x, v in shown["phase_space"]:
            ax.plot(x, v, ".", ms=1.5)
        ax.set_xlabel("$x/L$")
        ax.set_ylabel("$v/u$")


def table(result: dict) -> str:
    """Each thermal speed: the theory against picnic, and the report's name."""
    lines = [
        "| $v_{th}/u$ | Theory $\\gamma_{max}/\\omega_{pe}$ | picnic | Named |",
        "|---|---|---|---|",
    ]
    for r in result["runs"]:
        gamma = f"{r['gamma_theory']:.4f}" if r["gamma_theory"] > 0 else "stable"
        lines.append(
            f"| {r['spread']} | {gamma} | {r['gamma']:.4f} | {r['name']} "
            f"(mode {r['mode']}) |"
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
