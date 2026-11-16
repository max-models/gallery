r"""ITER-like injection.

One 16.5 MW beam of 1 MeV deuterium atoms, tangent to $R = 5.3$ m, enters a Solov'ev
equilibrium of ITER's size ($R_0 = 6.2$ m, outboard minor radius 1.8 m, elongation 1.7, 5.3 T)
filled with deuterium and 2% helium, with parabolic profiles and 20 keV in the centre. Each of
2000 markers is a straight line attenuated with Suzuki's stopping cross sections; the fast ions
it leaves slow down on their flux surface (Stix), heating mostly the electrons at this energy and
driving a current along the field.

The density scan shows why a beam may only be switched on above a minimum density: the
shine-through onto the inboard wall falls from about 40% of the power at a central density of
$1.5\times10^{19}\,\mathrm{m^{-3}}$ to 4% at $5\times10^{19}$ and below 0.1% at $10^{20}$. At
$10^{20}\,\mathrm{m^{-3}}$ the beam gives about 10 MW to the electrons and 6 MW to the ions and
drives about 0.6 MA. The Solov'ev shape is wider inboard than ITER's, and the trapped-electron
correction of the current is a fit still to be checked, so the numbers show what the model
computes rather than ITER's.

Compared: nothing; an example of `freeofcharge.deposit` and `freeofcharge.slow_down`. The
deposition and slowing down are checked against ASCOT5 in the study "ASCOT5 benchmark".

Reference: freeofcharge's example `examples/iter_like_injection.py`; ITER's heating neutral
beams (1 MeV D, 16.5 MW each) after R. Hemsworth et al., Nucl. Fusion 49, 045006 (2009).
"""

import argparse

import freeofcharge as foc
import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
import xarray as xr

#: markers: beam markers; births: ions born per marker in the slowing down; step: the step
#: along each line [m]; densities: the central densities of the shine-through scan [1/m^3].
PRESETS = {
    "smoke": {"markers": 200, "births": 1, "step": 0.05, "densities": [2e19, 1e20]},
    "full": {
        "markers": 2000,
        "births": 4,
        "step": 0.02,
        "densities": [1.5e19, 2e19, 3e19, 5e19, 7e19, 1e20],
    },
}

IONS = {"D": (2, 1, 0.96), "He": (4, 2, 0.02)}
TEMPERATURE = 2.0e4  # central [eV]
DENSITY = 1e20  # central, of the heating and current profiles [1/m^3]


def simulate(preset: str = "full") -> dict:
    """The shine-through scan, and the heating and current at the highest density."""
    options = PRESETS[preset]
    equilibrium = foc.Equilibrium.solovev(6.2, 1.8, 1.7, 5.3, wall_psi_n=1.3)
    origin, direction = foc.tangential_injection(5.3, 12.0)
    beam = foc.NeutralBeam.gaussian(
        origin,
        direction,
        1.0e6,
        16.5e6,
        width=0.3,
        divergence=0.005,
        markers=options["markers"],
        seed=1,
    )
    rho_edges = np.linspace(0.0, 1.0, 26)
    surfaces = equilibrium.flux_surfaces(rho_edges)

    densities = np.asarray(options["densities"])
    shine = []
    for ne0 in densities:
        plasma = foc.Plasma.parabolic(ne0, TEMPERATURE, ions=IONS)
        result = foc.deposit(
            beam,
            equilibrium,
            plasma,
            step=options["step"],
            births_per_marker=0,
            rho_edges=rho_edges,
        )
        shine.append(100 * result.shine_through_fraction)
    shine = xr.DataArray(
        shine,
        coords={
            "ne0": (
                "ne0",
                densities,
                {"long_name": "central density", "units": "1/m^3"},
            )
        },
        dims="ne0",
        attrs={"long_name": "shine-through", "units": "%"},
    )

    plasma = foc.Plasma.parabolic(DENSITY, TEMPERATURE, ions=IONS)
    deposition = foc.deposit(
        beam,
        equilibrium,
        plasma,
        step=options["step"],
        births_per_marker=options["births"],
        rho_edges=rho_edges,
        seed=2,
    )
    steady = foc.slow_down(deposition, plasma, surfaces)
    fast = steady.to_dataset()
    volume = xr.DataArray(surfaces.volume, dims="rho")
    power = deposition.to_dataset(surfaces).power_density / 1e3
    power.attrs.update(long_name="deposited power density", units="kW/m^3")
    heating = fast.heating / volume / 1e3
    heating.attrs.update(long_name="heating power density", units="kW/m^3")
    current = xr.concat(
        [fast.fast_ion_current_density, fast.current_density], dim="current"
    ).assign_coords(current=["fast ions", "driven, with shielding"])
    current = current / 1e3
    current.attrs.update(long_name="current density", units="kA/m^2")
    return {
        "preset": preset,
        "shine": shine,
        "power": power,
        "heating": heating,
        "current": current,
        "totals": {
            "shine_through": 100 * deposition.shine_through_fraction,
            "electron_heating": float(steady.electron_heating.sum()),
            "ion_heating": float(steady.ion_heating.sum()),
            "driven_current": float(steady.driven_current),
            "stored_energy": float(steady.stored_energy),
        },
    }


def plot(result: dict) -> None:
    """The shine-through scan, the heating and the current."""
    with plasma_plots.figure(
        figsize=(7, 4),
        caption="Shine-through of the 1 MeV beam against the central electron density "
        "(logarithmic scale).",
    ) as fig:
        result["shine"].plasma.plot.lineout(x="ne0", logy=True, ax=fig[0])
    with plasma_plots.figure(
        2,
        1,
        sharex=True,
        figsize=(7, 7),
        caption="At $n_{e0} = 10^{20}\\,\\mathrm{m^{-3}}$: the power deposited by the beam (top) "
        "and the heating of the electrons and the ions by the fast ions (bottom), against "
        "$\\rho_\\mathrm{pol}$.",
    ) as fig:
        result["power"].plasma.plot.lineout(x="rho", ax=fig[0])
        result["heating"].plasma.plot.profiles(x="rho", over="species", ax=fig[1])
    with plasma_plots.figure(
        figsize=(7, 4),
        caption="At $n_{e0} = 10^{20}\\,\\mathrm{m^{-3}}$: the current density of the fast ions "
        "and the driven current density after the electrons' shielding.",
    ) as fig:
        result["current"].plasma.plot.profiles(x="rho", over="current", ax=fig[0])


def table(result: dict) -> str:
    """The totals at the highest density."""
    totals = result["totals"]
    return "\n".join(
        [
            "| At $n_{e0} = 10^{20}$ m$^{-3}$ | |",
            "|---|---|",
            f"| Shine-through | {totals['shine_through']:.3f} % |",
            f"| Electron heating | {totals['electron_heating'] / 1e6:.2f} MW |",
            f"| Ion heating | {totals['ion_heating'] / 1e6:.2f} MW |",
            f"| Driven current | {totals['driven_current'] / 1e6:.2f} MA |",
            f"| Fast-ion energy | {totals['stored_energy'] / 1e6:.2f} MJ |",
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    plot(simulate(args.preset))
    plt.show()
