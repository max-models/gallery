r"""Injector chain.

A negative-ion neutral-beam injector of ITER's size from source to plasma, one stage per
code (`freeofcharge.chain`), scanned over the RF power and the neutralizer gas. The
RF source is a carafe global model; the extracted D$^-$ current density is
30 % of its positive-ion Bohm flux, a normalized perveance of 0.09 to 0.47 against the
Child-Langmuir limit of a 9 kV, 6 mm first gap. A slit extraction gap solved with
tree-d-3d (boundary-integral electrodes, meshless space charge) gives the V-curve of the
divergence against the perveance; the acceleration to 1 MeV shrinks the angles by
$\sqrt{9\,\mathrm{kV}/1\,\mathrm{MV}}$, and 70 % of the ions survive the accelerator.
plasmacoll's charge-state solution gives the neutral share of the gas neutralizer, a 23 m
duct of 0.5 m passes a share of the Gaussian beam that falls with its divergence, and
freeofcharge deposits the beam in an ITER-like plasma and drives current.

The two knobs pull against each other: more RF power gives more current but, past the
optimum perveance, a more divergent beam that the duct scrapes; more gas neutralizes more
of the beam up to the maximum (57.5 % near $1.4\times10^{20}\,\mathrm{m^{-2}}$), then
re-ionizes it. The most current, about 2 MA from 23 MW, comes at 950 kW and a line density
of $1.5\times10^{20}\,\mathrm{m^{-2}}$, next to the neutralizer's optimum.

Not modelled: caesium surface production, co-extracted electrons, the meniscus, round
apertures, the accelerator's optics and stripping, and halos.

Compared: plasmacoll's neutral share with its closed form.

Reference: freeofcharge's example `examples/injector_chain.py`; the injector after
Hemsworth et al., Nucl. Fusion 49, 045006 (2009).
"""

import argparse
import time

import freeofcharge as foc
import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
import xarray as xr
from freeofcharge import chain

#: rf_power: RF powers of the scan [kW]; line_density: neutralizer line densities
#: [1e20 m^-2]; perveance: points of the V-curve; optics: the resolution of the slit
#: optics (the full one is the library's default);
#: markers: beam markers in the tokamak.
PRESETS = {
    "smoke": {
        "rf_power": 3,
        "line_density": 4,
        "perveance": [0.05, 0.5],
        "optics": {"rays": 8, "iterations": 1, "panel": 0.1, "dt": 0.05},
        "markers": 100,
    },
    "full": {
        "rf_power": 7,
        "line_density": 12,
        "perveance": [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        "optics": {"rays": 24, "iterations": 8, "panel": 0.05, "dt": 0.02},
        "markers": 1000,
    },
}

EXTRACTION_VOLTAGE, GAP = 9.0e3, 6.0e-3  # V, m: the first (extraction) gap
APERTURE_AREA = 0.2  # m^2: 1280 apertures of 14 mm
BEAM_ENERGY = 1.0e6  # eV
ACCELERATOR_TRANSMISSION = 0.7  # share of the extracted ions that leave the accelerator
SOURCE_WIDTH, DUCT_LENGTH, DUCT_HALF_APERTURE = 0.1, 23.0, 0.25  # m


def tokamak(divergence, markers):
    """Shine-through fraction and driven current per watt for each beam divergence."""
    equilibrium = foc.Equilibrium.solovev(6.2, 1.8, 1.7, 5.3, wall_psi_n=1.3)
    origin, direction = foc.tangential_injection(5.3, 12.0)
    plasma = foc.Plasma.parabolic(
        5e19, 2.0e4, ions={"D": (2, 1, 0.96), "He": (4, 2, 0.02)}
    )
    rho_edges = np.linspace(0.0, 1.0, 21)
    surfaces = equilibrium.flux_surfaces(rho_edges)
    shine, per_watt = [], []
    for theta in divergence:
        width = np.hypot(SOURCE_WIDTH, DUCT_LENGTH * theta)
        beam = foc.NeutralBeam.gaussian(
            origin, direction, BEAM_ENERGY, 1.0, width=width, divergence=theta,
            markers=markers, seed=1,
        )  # fmt: skip
        deposition = foc.deposit(
            beam, equilibrium, plasma, step=0.03, births_per_marker=2,
            rho_edges=rho_edges, seed=2,
        )  # fmt: skip
        steady = foc.slow_down(deposition, plasma, surfaces)
        shine.append(deposition.shine_through_fraction)
        per_watt.append(steady.driven_current / deposition.injected_power)
    return np.array(shine), np.array(per_watt)


def simulate(preset: str = "full") -> dict:
    """Run the chain stage by stage and scan RF power against neutralizer gas."""
    p = PRESETS[preset]
    start = time.time()
    rf_power = np.linspace(200e3, 1100e3, p["rf_power"])  # W
    line_density = np.linspace(0.25e20, 3.0e20, p["line_density"])  # m^-2
    grid = np.array(p["perveance"])

    # 1. source: the current density against the RF power
    sources = [chain.deuterium_source(power) for power in rf_power]
    current_density = np.array([s.current_density for s in sources])
    perveance = current_density / chain.child_langmuir(EXTRACTION_VOLTAGE, GAP)
    # the slit optics converge only up to a perveance of 0.5: stay inside the scan
    assert perveance.max() <= grid.max(), "RF power beyond the optics scan"

    # 2. optics: the V-curve, then the angles after the acceleration
    curve, extracted = chain.slit_divergence(grid, **p["optics"])
    shrink = np.sqrt(EXTRACTION_VOLTAGE / BEAM_ENERGY)
    divergence = np.interp(perveance, grid, curve) * shrink  # rad
    transmission = np.array(
        [
            chain.duct_transmission(SOURCE_WIDTH, t, DUCT_LENGTH, DUCT_HALF_APERTURE)
            for t in divergence
        ]
    )

    # 3. neutralizer: the neutral share against the gas line density
    pi_fine = np.linspace(0.0, 4.0e20, 401)
    neutral_fine = chain.neutral_fraction(pi_fine)
    closed_form = chain.neutral_fraction_closed_form(pi_fine)
    neutral = chain.neutral_fraction(line_density)

    # 4. tokamak, and the scan of the two knobs
    shine, per_watt = tokamak(divergence, p["markers"])
    accelerated = current_density * APERTURE_AREA * BEAM_ENERGY
    accelerated *= ACCELERATOR_TRANSMISSION  # W
    delivered = (accelerated * transmission)[:, None] * neutral[None, :]
    dims = ("rf_power", "line_density")
    scan = xr.Dataset(
        {
            "delivered": (
                dims,
                delivered / 1e6,
                {"long_name": "neutral power delivered", "units": "MW"},
            ),
            "driven": (
                dims,
                delivered * per_watt[:, None] / 1e6,
                {"long_name": "driven current", "units": "MA"},
            ),
            "shone": (
                dims,
                delivered * shine[:, None] / 1e6,
                {"long_name": "shine-through onto the wall", "units": "MW"},
            ),
        },
        coords={
            "rf_power": (
                "rf_power",
                rf_power / 1e3,
                {"long_name": "RF power", "units": "kW"},
            ),
            "line_density": (
                "line_density",
                line_density / 1e20,
                {"long_name": "neutralizer line density", "units": "10^20 m^-2"},
            ),
        },
    )
    stages = xr.Dataset(
        {
            "electron_density": ("rf_power", [s.electron_density for s in sources]),
            "electron_temperature": (
                "rf_power",
                [s.electron_temperature for s in sources],
            ),
            "current_density": ("rf_power", current_density),
            "perveance": ("rf_power", perveance),
            "divergence": ("rf_power", divergence),
            "transmission": ("rf_power", transmission),
            "shine_through": ("rf_power", shine),
            "current_per_watt": ("rf_power", per_watt),
        },
        coords={"rf_power": scan.rf_power},
    )
    return {
        "preset": preset,
        "stages": stages,
        "v_curve": {"perveance": grid, "divergence": curve, "extracted": extracted},
        "neutralizer": {
            "line_density": pi_fine,
            "neutral": neutral_fine,
            "closed_form": closed_form,
        },
        "scan": scan,
        "shrink": shrink,
        "runtime": time.time() - start,
    }


def table(result: dict) -> str:
    """The stages at each RF power, and the best point of the scan."""
    st, scan = result["stages"], result["scan"]
    lines = [
        "| RF power (kW) | n_e (m^-3) | j (A/m^2) | perveance | divergence (mrad) "
        "| duct (%) | shine-through (%) | current per power (A/W) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in range(st.sizes["rf_power"]):
        r = st.isel(rf_power=i)
        lines.append(
            f"| {float(r.rf_power):.0f} | {float(r.electron_density):.2e} | "
            f"{float(r.current_density):.0f} | {float(r.perveance):.2f} | "
            f"{1e3 * float(r.divergence):.1f} | {100 * float(r.transmission):.0f} | "
            f"{100 * float(r.shine_through):.1f} | {float(r.current_per_watt):.3f} |"
        )
    neutral = result["neutralizer"]
    k = int(np.argmax(neutral["neutral"]))
    error = np.abs(neutral["neutral"] - neutral["closed_form"]).max()
    best = scan.driven.where(scan.driven == scan.driven.max(), drop=True)
    i, j = float(best.rf_power[0]), float(best.line_density[0])
    point = scan.sel(rf_power=i, line_density=j)
    lines += [
        "",
        f"Neutralizer: at most {100 * neutral['neutral'][k]:.1f} % neutral at "
        f"{neutral['line_density'][k]:.2e} m^-2; the closed form agrees to {error:.0e}.",
        "",
        f"Most current: {float(point.driven):.2f} MA from {float(point.delivered):.1f} MW "
        f"at {i:.0f} kW and {j:.2f}e20 m^-2, with {float(point.shone):.2f} MW shone "
        "through onto the wall.",
    ]
    return "\n".join(lines)


def plot(result: dict) -> None:
    """The four stages, then the maps of the scan."""
    st, curve = result["stages"], result["v_curve"]
    power = st.rf_power.values

    # 1. source
    with plasma_plots.figure(
        2,
        1,
        sharex=True,
        figsize=(7, 6),
        caption="Source (carafe): the extracted D⁻ current density (top) and the "
        "electron density (bottom) against the RF power; the electron temperature is "
        f"{float(st.electron_temperature[0]):.1f} eV at every power.",
    ) as fig:
        fig[0].plot(power, st.current_density, "o-")
        fig[0].set_ylabel("D$^-$ current density [A/m$^2$]")
        fig[1].plot(power, st.electron_density, "o-")
        fig[1].set_ylabel("electron density [m$^{-3}$]")
        fig[1].set_xlabel("RF power [kW]")

    # 2. the V-curve
    with plasma_plots.figure(
        figsize=(7, 4),
        caption="Extraction optics (tree-d-3d): the rms divergence of the slit beamlet at "
        "extraction against the normalized perveance (line), and the operating points "
        "of the RF powers of the scan (squares).",
    ) as fig:
        ax = fig[0]
        ax.plot(curve["perveance"], 1e3 * curve["divergence"], "o-", label="slit gap")
        ax.plot(
            st.perveance,
            1e3 * st.divergence / result["shrink"],
            "s",
            color="C1",
            label="operating points",
        )
        ax.set_xlabel(r"normalized perveance $j / j_\mathrm{CL}$")
        ax.set_ylabel("rms divergence at extraction [mrad]")
        ax.legend()

    # 3. neutralizer
    neutral = result["neutralizer"]
    pi = neutral["line_density"] / 1e20
    with plasma_plots.figure(
        figsize=(7, 4),
        caption="Gas neutralizer (plasmacoll): the neutral share of the beam against the "
        "line density, the rate equations (line) and the closed form (circles); the "
        "dotted line marks the maximum.",
    ) as fig:
        ax = fig[0]
        ax.plot(pi, 100 * neutral["neutral"], label="plasmacoll")
        ax.plot(
            pi[::25],
            100 * neutral["closed_form"][::25],
            "o",
            mfc="none",
            label="closed form",
        )
        ax.axvline(pi[np.argmax(neutral["neutral"])], color="0.6", ls=":")
        ax.set_xlabel("neutralizer line density [$10^{20}$ m$^{-2}$]")
        ax.set_ylabel("neutral share [%]")
        ax.legend()

    # 4. beam line
    with plasma_plots.figure(
        2,
        1,
        sharex=True,
        figsize=(7, 6),
        caption="Beam line: the divergence after the acceleration to 1 MeV (top) and "
        "the share of the beam the 23 m duct passes (bottom), against the RF power.",
    ) as fig:
        fig[0].plot(power, 1e3 * st.divergence, "o-")
        fig[0].set_ylabel("beam divergence [mrad]")
        fig[1].plot(power, 100 * st.transmission, "s-", color="C1")
        fig[1].set_ylabel("duct transmission [%]")
        fig[1].set_xlabel("RF power [kW]")

    # 5. the scan
    scan = result["scan"]
    best = scan.driven.where(scan.driven == scan.driven.max(), drop=True)
    star = float(best.line_density[0]), float(best.rf_power[0])
    with plasma_plots.figure(
        3,
        1,
        sharex=True,
        figsize=(7, 10),
        caption="The scan over RF power and neutralizer line density: from top to "
        "bottom the driven current, the neutral power delivered to the torus and the "
        "power shone through onto the inboard wall; the star marks the most current.",
    ) as fig:
        for ax, name in zip(
            (fig[0], fig[1], fig[2]), ("driven", "delivered", "shone"), strict=True
        ):
            scan[name].plasma.plot.slice(x="line_density", y="rf_power", ax=ax)
            ax.plot(*star, "w*", ms=12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    output = simulate(args.preset)
    print(table(output))
    plot(output)
    plt.show()
