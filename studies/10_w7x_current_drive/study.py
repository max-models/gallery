"""W7-X current drive.

The current that two neutral beams drive in W7-X, with the electrons' response on 3D flux
surfaces (freeofcharge). The electrons shield the fast-ion current $j_f$,
$j = j_f\\,[1 - Z_b/Z_\\mathrm{eff}\\,(1 - G)]$, and the trapped electrons, which cannot follow,
reduce the shielding by $G$. `slow_down` gives the current with $G = 0$ or with $G$ = Sauter's
$L_{31}$ of the **effective trapped fraction** $f_t$ of each 3D surface, the collisionality
taking $q = 1/|\\iota|$ and the mirror ratio of $|B|$ (`field_epsilon`) for the inverse aspect
ratio.

The equilibrium is GVEC's W7-X example at reduced resolution (slowcooker's), sampled on a
$64 \\times 32 \\times 64$ grid per field period, with ten shells. The two stand-in NI21 sources
deposit 55 keV hydrogen in a made-up plasma ($Z_\\mathrm{eff} = 1.3$ from 1 % carbon,
$T_0 = 2$ keV); the ions slow down on their birth surface with their local pitch.

**Finding.** $f_t$ is 0.48 to 0.62, far above the tokamak estimate: the mirror variation of
$|B|$ traps electrons on every surface. The correction raises the net current by 1.8 to 2.5 per
shell ($-13$ to $-32$ kA at $n_{e0} = 4\\times10^{19}$ m$^{-3}$, 3.5 MW absorbed); with it the
current falls from $-102$ kA at $10^{19}$ to $-10$ kA at $10^{20}$ m$^{-3}$.

Compared: $f_t$ with the tokamak estimate $1.46\\sqrt{\\epsilon}$; the current with and without
the correction. Not compared: another code or a measurement; $L_{31}(f_t)$ lacks the stellarator
geometric factor, and orbits and losses are left out.

Reference: freeofcharge's tutorial "Stellarator current drive" (there on DESC's W7-X); Sauter,
Angioni and Lin-Liu, Phys. Plasmas 6, 2834 (1999).
"""

from __future__ import annotations

import argparse
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

#: Grid of the equilibrium, markers per source and energy component, central densities.
PRESETS = {
    "smoke": {"grid": (32, 16, 32), "markers": 20, "densities": [4e19]},
    "quick": {
        "grid": (64, 32, 64),
        "markers": 500,
        "densities": [1e19, 2e19, 4e19, 7e19, 1e20],
    },
}

REFERENCE_DENSITY = 4e19  # [m^-3], the case of the profiles
IONS = {"H": (1, 1, 0.94), "C": (12, 6, 0.01)}  # Z_eff = 1.3


def simulate(preset: str = "quick") -> dict:
    """Sample GVEC's W7-X, then the driven current at each central density."""
    import freeofcharge as foc
    from freeofcharge.injectors import w7x_injectors
    from freeofcharge.slowing_down import sauter_trapped_electron_correction, slow_down
    from slowcooker.data import gvec_equilibrium

    settings = PRESETS[preset]
    start = time.perf_counter()
    w7x = foc.GriddedEquilibrium(gvec_equilibrium("w7x"), settings["grid"])
    rho_edges = np.linspace(0.0, 1.0, 11)
    surfaces = w7x.flux_surfaces(rho_edges, field_points=(24, 24), radial_points=4)
    shells = surfaces.to_dataset()
    sources = w7x_injectors(sources={"NI21": [1, 2]})

    def drive(ne0: float):
        plasma = foc.Plasma.parabolic(ne0, 2e3, ions=IONS, radial_coordinate="toroidal")
        births = foc.deposit_sources(
            sources,
            w7x,
            plasma,
            markers=settings["markers"],
            seed=1,
            step=0.02,
            rho_edges=rho_edges,
        ).total()
        steady = {
            name: slow_down(
                births,
                plasma,
                surfaces,
                orbit_average=False,
                trapped_electron_correction=name,
            )
            for name in ("none", "sauter")
        }
        return plasma, births, steady

    plasma, births, steady = drive(REFERENCE_DENSITY)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = slow_down(births, plasma, surfaces, orbit_average=False)
    none_set = steady["none"].to_dataset(surfaces)
    sauter_set = steady["sauter"].to_dataset(surfaces)

    scan = {"none": [], "sauter": [], "absorbed": []}
    for ne0 in settings["densities"]:
        _, deposited, result = drive(ne0)
        scan["absorbed"].append(deposited.absorbed_power)
        for name in ("none", "sauter"):
            scan[name].append(result[name].driven_current)

    return {
        "preset": preset,
        "rho": np.asarray(shells.rho.values),
        "trapped_fraction": np.asarray(surfaces.trapped_fraction),
        "inverse_aspect_ratio": np.asarray(shells.inverse_aspect_ratio.values),
        "field_epsilon": np.asarray(surfaces.field_epsilon),
        "G": np.asarray(sauter_trapped_electron_correction(surfaces, plasma)),
        "j_fast": none_set.fast_ion_current_density.values,
        "j_none": none_set.current_density.values,
        "j_sauter": sauter_set.current_density.values,
        "default_current": float(default.driven_current),
        "current_none": float(steady["none"].driven_current),
        "current_sauter": float(steady["sauter"].driven_current),
        "absorbed": float(births.absorbed_power),
        "densities": np.asarray(settings["densities"]),
        "scan_none": np.asarray(scan["none"]),
        "scan_sauter": np.asarray(scan["sauter"]),
        "scan_absorbed": np.asarray(scan["absorbed"]),
        "wall_time": time.perf_counter() - start,
    }


def plot(result: dict) -> None:
    """The trapped fraction, the correction, the current profiles and the density scan."""
    rho = result["rho"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.caption = (
        "The effective trapped fraction of each shell of GVEC's W7-X against the tokamak "
        "estimate from the inverse aspect ratio, and the mirror ratio used in the collisionality."
    )
    ax.plot(rho, result["trapped_fraction"], marker="o", label="$f_t$, 3D surface")
    ax.plot(
        rho,
        1.46 * np.sqrt(result["inverse_aspect_ratio"]),
        marker="s",
        ls="--",
        label=r"tokamak $1.46\sqrt{\epsilon}$",
    )
    ax.plot(
        rho,
        result["field_epsilon"],
        marker="^",
        ls=":",
        label=r"mirror $\epsilon$ (field_epsilon)",
    )
    ax.set_xlabel(r"$\rho = \sqrt{\Phi/\Phi_b}$")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, None)
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.caption = (
        "The trapped-electron correction G = Sauter's L31 of each shell at "
        "n_e0 = 4e19 m^-3 and T_0 = 2 keV."
    )
    ax.plot(rho, result["G"], marker="o")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$G = L_{31}$")
    ax.set_ylim(0, None)
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.caption = (
        "Parallel current density at n_e0 = 4e19 m^-3: the fast ions' current and the net "
        "current without (G = 0) and with the trapped-electron correction; negative is against B."
    )
    ax.plot(rho, result["j_fast"] / 1e3, marker="o", label="fast ions")
    ax.plot(rho, result["j_none"] / 1e3, marker="s", label="net, $G = 0$")
    ax.plot(rho, result["j_sauter"] / 1e3, marker="^", label="net, Sauter with $f_t$")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"current density [kA/m$^2$]")
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.caption = (
        "Total beam-driven current against the central density, without and with the "
        "trapped-electron correction."
    )
    ax.plot(result["densities"], result["scan_none"] / 1e3, marker="s", label="$G = 0$")
    ax.plot(
        result["densities"],
        result["scan_sauter"] / 1e3,
        marker="^",
        label="Sauter with $f_t$",
    )
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$n_{e0}$ [m$^{-3}$]")
    ax.set_ylabel("driven current [kA]")
    ax.legend()
    fig.tight_layout()


def table(result: dict) -> str:
    """The shells, the currents at the reference density and the density scan."""
    rows = [
        "| ρ | f_t | 1.46√ε | mirror ε | G | j_net(Sauter) / j_net(G=0) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    ratio = result["j_sauter"] / result["j_none"]
    for k, rho in enumerate(result["rho"]):
        rows.append(
            f"| {rho:.2f} | {result['trapped_fraction'][k]:.3f} | "
            f"{1.46 * np.sqrt(result['inverse_aspect_ratio'][k]):.3f} | "
            f"{result['field_epsilon'][k]:.3f} | {result['G'][k]:.3f} | {ratio[k]:.2f} |"
        )
    rows += [
        "",
        f"At $n_{{e0}} = 4\\times10^{{19}}$ m$^{{-3}}$: absorbed {result['absorbed'] / 1e6:.2f} MW; "
        f"driven current {result['current_none'] / 1e3:+.1f} kA with $G = 0$, "
        f"{result['current_sauter'] / 1e3:+.1f} kA with Sauter's $L_{{31}}(f_t)$; "
        f"`slow_down`'s default in 3D gives {result['default_current']} (no correction "
        "chosen, with a warning).",
        "",
        "| n_e0 [m⁻³] | absorbed [MW] | I, G = 0 [kA] | I, Sauter [kA] |",
        "|---:|---:|---:|---:|",
    ]
    for ne0, p, i0, i1 in zip(
        result["densities"],
        result["scan_absorbed"],
        result["scan_none"],
        result["scan_sauter"],
        strict=True,
    ):
        rows.append(
            f"| {ne0:.0e} | {p / 1e6:.2f} | {i0 / 1e3:+.1f} | {i1 / 1e3:+.1f} |"
        )
    rows.append(f"\nWall time {result['wall_time']:.0f} s.")
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    args = parser.parse_args()
    outcome = simulate(args.preset)
    print(table(outcome))
    plot(outcome)
    plt.show()
