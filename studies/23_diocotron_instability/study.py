r"""Diocotron instability.

A layer of electrons in a magnetic field drifts with its own electric field at
$\mathbf E\times\mathbf B/B^2$, faster on one edge than on the other. That shear makes ripples of
the edges grow, the diocotron instability of non-neutral plasmas (Penning traps, magnetron and
gyrotron electron beams), the electrostatic counterpart of the Kelvin–Helmholtz instability; the
layer rolls up into vortices. Its rate is set by the diocotron frequency
$\omega_D = \omega_{pe}^2/\omega_{ce}$, and only a band of wavelengths around the layer's
width grows.

The study runs picnic's 2D `DriftKineticVlasov` (guiding-centre electrons, steps of
$0.05/\omega_D$ instead of resolving $\omega_{ce} = 100\,\omega_D$): an electron layer 2 mm wide
in the middle of a 1 cm gap between grounded walls, $B_z = 0.1$ T, periodic along $y$ over one
wavelength of the mode, seeded with a ripple of $10^{-3}$ of the half width $a$. The growth
rate of the potential's Fourier mode, at six wavenumbers from $ka = 0.2$ to $1.2$, is compared
with the eigenvalue of the linearized drift–Poisson equation for the same profile and walls,
$(\omega - k v_0)(\phi'' - k^2\phi) = -(e k n_0'/\varepsilon_0 B)\,\phi$. The setup, the loading
and the eigenvalue solver are those of picnic's diocotron example.

Compared: $\gamma(ka)$ with the linear eigenvalue. Not compared: the real frequency, full PIC
(the example compares both models at $ka = 0.4$), and the vortex merging after saturation.

Reference: R. H. Levy, Phys. Fluids 8, 1288 (1965); R. C. Davidson, Physics of Nonneutral
Plasmas (Imperial College Press, 2001), ch. 6.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots

PRESETS = {
    "smoke": {"modes": [0.4], "ny": 8, "per_cell": 4, "duration": 4.0, "curve": 2},
    "full": {
        "modes": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        "ny": 48,
        "per_cell": 16,
        "duration": 90.0,
        "curve": 25,
    },
}
SHOWN = 0.6  # ka of the layer drawn at the end


def run(ka: float, sizes: dict) -> dict:
    """One guiding-centre run at ka; return the mode history, the theory and the layer."""
    from dataclasses import replace

    from maybempi import MPI
    from picnic.examples.diocotron_2d import (
        SETUP,
        create_config,
        growth_rate,
        linear_growth,
        load_layer,
    )
    from picnic.models.drift_kinetic_vlasov import DriftKineticVlasov

    setup = replace(
        SETUP,
        ka=ka,
        ny=sizes["ny"],
        per_cell=sizes["per_cell"],
        duration=sizes["duration"],
    )
    omega, equilibrium = linear_growth(setup)
    model = DriftKineticVlasov(create_config(setup), comm=MPI.COMM_SELF)
    model.set_species_markers("electrons", *load_layer(setup, equilibrium))
    steps = int(round(setup.duration / setup.dt_omega_d))
    every = max(1, int(round(0.25 / setup.dt_omega_d)))
    times, amplitude = [], []
    for step in range(steps):
        model.step()
        if step % every == 0:
            phi = np.asarray(model.fields.potential.to_ndarray())
            phi = phi.reshape(setup.nx + 1, setup.ny)
            times.append(model.time * setup.omega_d)
            amplitude.append(np.abs(np.fft.rfft(phi[setup.nx // 2]))[1])
    times, amplitude = np.array(times), np.array(amplitude)
    markers = model.species_registry.get("electrons").markers
    positions = np.asarray(markers.get_active_positions())
    pick = np.random.default_rng(0).choice(
        positions.shape[1], min(6000, positions.shape[1]), replace=False
    )
    return {
        "ka": ka,
        "theory": omega.imag / setup.omega_d,
        "gamma": growth_rate(times, amplitude),
        "tau": times,
        "amplitude": amplitude,
        "layer": (positions[0, pick] / setup.length, positions[1, pick] / setup.width),
    }


def simulate(preset: str = "full") -> dict:
    """Run the preset's wavenumbers and the linear theory on a finer list."""
    from dataclasses import replace

    from picnic.examples.diocotron_2d import SETUP, linear_growth

    sizes = PRESETS[preset]
    curve = np.linspace(0.1, 1.6, sizes["curve"])
    theory = [
        linear_growth(replace(SETUP, ka=ka))[0].imag / SETUP.omega_d for ka in curve
    ]
    return {
        "curve": curve,
        "theory": np.array(theory),
        "runs": [run(ka, sizes) for ka in sizes["modes"]],
    }


def plot(result: dict) -> None:
    """γ(ka) against theory, the mode histories, and the rolled-up layer."""
    runs = result["runs"]
    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The growth rate against the wavenumber times the layer's half width: "
        "picnic's guiding-centre electrons (markers) on the eigenvalue of the linearized "
        "drift–Poisson equation for the same layer and walls (line).",
    ) as fig:
        ax = fig[0]
        ax.plot(
            result["curve"],
            np.maximum(result["theory"], 0.0),
            color="0.4",
            label="linear theory",
        )
        ax.plot(
            [r["ka"] for r in runs],
            [r["gamma"] for r in runs],
            "o",
            color="C0",
            label="picnic",
        )
        ax.set_xlabel(r"$k a$")
        ax.set_ylabel(r"$\gamma/\omega_D$")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The amplitude of the mode at the layer's centre against time for each "
        "wavenumber: exponential growth from the seed, then saturation as the layer "
        "rolls up.",
    ) as fig:
        ax = fig[0]
        for i, r in enumerate(runs):
            ax.semilogy(
                r["tau"], r["amplitude"], color=f"C{i}", label=f"$ka = {r['ka']}$"
            )
        ax.set_xlabel(r"$\omega_D t$")
        ax.set_ylabel(r"$|\hat\phi_1|$ [V]")
        ax.legend(fontsize=8)

    shown = min(runs, key=lambda r: abs(r["ka"] - SHOWN))
    with plasma_plots.figure(
        figsize=(6, 4),
        caption=f"The electrons at the end of the run at $ka = {shown['ka']}$, a sample "
        "of the guiding centres over one wavelength: the layer has rolled up into a "
        "vortex.",
    ) as fig:
        ax = fig[0]
        x, y = shown["layer"]
        ax.plot(y, x, ".", ms=1, color="C0")
        ax.set_xlabel("$y/\\lambda$")
        ax.set_ylabel("$x/L$ (wall to wall)")


def table(result: dict) -> str:
    """Each wavenumber: the linear theory against picnic."""
    lines = [
        "| $ka$ | Theory $\\gamma/\\omega_D$ | picnic $\\gamma/\\omega_D$ |",
        "|---|---|---|",
    ]
    for r in result["runs"]:
        lines.append(f"| {r['ka']} | {r['theory']:.4f} | {r['gamma']:.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    result = simulate(args.preset)
    print(table(result))
    plot(result)
    plt.show()
