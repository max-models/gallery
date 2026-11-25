r"""Electron cyclotron drift instability.

Electrons drifting at $E/B$ across a magnetic field, through ions too heavy to feel it, drive
the electron cyclotron drift instability (ECDI) of Hall thrusters and magnetrons: waves along
the drift that grow where the Doppler-shifted electron Bernstein modes meet the ion acoustic
wave, at the resonances $k v_d \approx m\,\Omega_{ce}$.

The study runs picnic's `VlasovPoisson` in 1D3V in the frame of the electrons, where there is no
electric field: Maxwellian electrons gyrate in $B_z = 10$ mT and the ions stream past at
$-v_d$, with $n = 10^{17}$ m$^{-3}$, $T_e = 10$ eV, $T_i = T_e/50$, $E = 20$ kV/m and a reduced
ion mass $m_i = 100\,m_e$. The periodic box is one resonant wavelength $2\pi v_d/\Omega_{ce}$
long, so every box mode $m$ is a resonance. The modes grow from the marker noise; picnic's
`instability_report` fits each mode's growth rate without a hand-picked window and names the
fastest. They are compared with the ECDI root of the kinetic dispersion relation with the
Bessel sum over the cyclotron harmonics (`electron_cyclotron_drift` in plasma-plots).

Compared: the growth rate of each resonance with linear kinetic theory. Not compared: the real
frequencies, the saturation and the anomalous electron transport it drives, the real mass
ratio, and $k_\parallel \neq 0$, to which the instability is sensitive (2D and 3D).

Reference: Janhunen et al., Phys. Plasmas 25, 011608 (2018); Lafleur, Baalrud and Chabert,
Plasma Sources Sci. Technol. 25, 065013 (2016).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
from plasma_plots.theory.kinetic import electron_cyclotron_drift
from scipy import constants as c

DENSITY = 1e17  # [m^-3]
TEMPERATURE = 10.0  # T_e [eV]
FIELD = 0.01  # B_z [T]
ELECTRIC = 2e4  # E [V/m], the drift E/B
MASS = 100.0  # m_i/m_e
RATIO = 50.0  # T_e/T_i
MODES = 20  # box modes followed (each a resonance m)
PRESETS = {
    "smoke": {"markers": 2**11, "cells": 64, "time": 40.0},
    "full": {"markers": 2**18, "cells": 192, "time": 400.0},
}

OMEGA_PE = np.sqrt(DENSITY * c.e**2 / (c.epsilon_0 * c.m_e))
OMEGA_CE = c.e * FIELD / c.m_e
THERMAL = np.sqrt(TEMPERATURE * c.e / c.m_e)
DRIFT = ELECTRIC / FIELD
LENGTH = 2 * np.pi * DRIFT / OMEGA_CE  # one resonant wavelength


def theory(modes: np.ndarray) -> np.ndarray:
    """The ECDI root ω/ω_pe of the box modes (ion frame), nan where it does not grow."""
    k = 2 * np.pi * np.asarray(modes) / LENGTH * THERMAL / OMEGA_PE
    omega = electron_cyclotron_drift(
        k,
        DRIFT / THERMAL,
        OMEGA_CE / OMEGA_PE,
        temperature_ratio=RATIO,
        mass_ratio=MASS,
    )["ion-acoustic"]
    return np.where(omega.imag > 0, omega, np.nan)


def simulate(preset: str = "full") -> dict:
    """Run picnic in the electron frame; return the mode growth rates and histories."""
    from picnic import Simulation
    from picnic.models.vlasov_poisson import VlasovPoisson
    from picnic.parameters import (
        FieldBCs,
        FieldBoundaryCondition,
        GridParams,
        MagneticFieldParams,
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

    sizes = PRESETS[preset]
    markers, dt = sizes["markers"], 0.1
    weight = DENSITY * LENGTH / markers
    periodic_field = FieldBoundaryCondition.periodic()
    periodic = ParticleBoundaryCondition("periodic")
    config = VlasovPoissonConfig(
        grid=GridParams(
            ndim=1,
            nvel=3,
            num_gridpoints=sizes["cells"],
            upper_bound=LENGTH,
            domain_decomposition=False,
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
        magnetic_field=MagneticFieldParams(bz=FIELD),
        dt=dt / OMEGA_PE,
    )
    steps = int(np.ceil(sizes["time"] / dt))
    sim = Simulation(
        VlasovPoisson(config),
        t_end=steps * dt / OMEGA_PE,
        output=OutputParams(every=10**9),
    )
    rng = np.random.default_rng(0)
    x, v = quiet_start(markers, [0.0], [LENGTH], 3, velocity_bases=(3, 5, 7))
    x = (x + 1e-6 * LENGTH * rng.standard_normal(x.shape)) % LENGTH
    sim.model.set_species_markers("electrons", x, THERMAL * v)
    x, v = quiet_start(markers, [0.0], [LENGTH], 3, velocity_bases=(11, 13, 17))
    ion_thermal = THERMAL / np.sqrt(RATIO * MASS)
    v = ion_thermal * v
    v[0] -= DRIFT  # the ions stream past the electrons
    sim.model.set_species_markers("ions", x, v)
    sim.model.fields.deposit()
    sim.model.fields.solve()
    monitor = InstabilityMonitor(samples=4000)
    monitor.record(sim.model)
    sim.add_callback(monitor, every=max(1, steps // 500))
    sim.run()
    data = monitor.dataset()
    report = instability_report(data, max_mode=MODES)

    modes = np.arange(1, MODES + 1)
    rates, coupled = np.full(MODES, np.nan), np.zeros(MODES, dtype=bool)
    for mode in report.modes:
        if mode.axis == "x" and mode.mode <= MODES:
            rates[mode.mode - 1] = mode.growth_rate / OMEGA_PE
            coupled[mode.mode - 1] = mode.name == COUPLED
    amplitude = np.abs(np.fft.rfft(data["potential_x"].values, axis=1))
    first = report.modes[0] if report.modes else None
    return {
        "modes": modes,
        "rates": rates,
        "coupled": coupled,
        "theory": theory(modes),
        "tau": data.t.values * OMEGA_PE,
        "amplitude": amplitude[:, 1 : MODES + 1],
        "name": first.name if first else "none",
        "mode": first.mode if first else 0,
    }


def plot(result: dict) -> None:
    """γ of every resonance against theory, and the mode histories."""
    omega_pi = 1 / np.sqrt(MASS)  # in units of ω_pe
    with plasma_plots.figure(
        figsize=(6, 4),
        caption="Growth rate of every box mode, each a resonance $k v_d = m\\Omega_{ce}$: "
        "picnic (filled; hollow if driven by two growing modes) on the ECDI root of the "
        "kinetic dispersion relation (squares).",
    ) as fig:
        ax = fig[0]
        ax.plot(
            result["modes"],
            result["theory"].imag / omega_pi,
            "s",
            mfc="none",
            color="0.4",
            label="theory",
        )
        linear = np.where(result["coupled"], np.nan, result["rates"])
        driven = np.where(result["coupled"], result["rates"], np.nan)
        ax.plot(result["modes"], linear / omega_pi, "o", color="C0", label="picnic")
        ax.plot(
            result["modes"],
            driven / omega_pi,
            "o",
            mfc="none",
            color="C0",
            label="coupled",
        )
        ax.set_xlabel(r"resonance $m = k v_d/\Omega_{ce}$")
        ax.set_ylabel(r"$\gamma/\omega_{pi}$")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    fastest = int(np.nanargmax(result["theory"].imag))
    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The amplitude of the potential's resonances against time: growth from "
        f"the marker noise, then saturation. Blue: the fastest resonance in theory, "
        f"$m = {fastest + 1}$; dashed: its theoretical growth rate.",
    ) as fig:
        ax = fig[0]
        tau, amplitude = result["tau"], result["amplitude"]
        for m in range(amplitude.shape[1]):
            if m != fastest:
                ax.semilogy(tau, amplitude[:, m], color="0.75", lw=0.6)
        a = amplitude[:, fastest]
        ax.semilogy(tau, a, color="C0", lw=1.6)
        j = int(np.argmin(np.abs(a - np.sqrt(a[0] * a.max()))))
        gamma = result["theory"][fastest].imag
        ax.semilogy(tau, a[j] * np.exp(gamma * (tau - tau[j])), "k--", lw=1)
        ax.set_ylim(0.3 * a[: len(a) // 10 + 1].min(), 3 * amplitude.max())
        ax.set_xlabel(r"$\omega_{pe} t$")
        ax.set_ylabel(r"$|\phi_m|$ [V]")


def table(result: dict) -> str:
    """The theory's fastest resonances against picnic, and the report's name."""
    omega_pi = 1 / np.sqrt(MASS)
    order = np.argsort(-np.nan_to_num(result["theory"].imag))[:5]
    lines = [
        "| Resonance $m$ | Theory $\\gamma/\\omega_{pi}$ | picnic $\\gamma/\\omega_{pi}$ |",
        "|---|---|---|",
    ]
    for i in sorted(order):
        lines.append(
            f"| {result['modes'][i]} | {result['theory'][i].imag / omega_pi:.3f} | "
            f"{result['rates'][i] / omega_pi:.3f} |"
        )
    lines.append(
        f"\nFastest growing mode in picnic: {result['name']} (m = {result['mode']})."
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
