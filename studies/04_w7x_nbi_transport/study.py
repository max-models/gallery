"""W7-X NBI transport.

How much of the heat that two neutral beams put into a W7-X plasma leaves by neoclassical
transport, and how does that depend on the anomalous (turbulent) transport that comes on top?
This study runs the whole slowcooker chain on the W7-X standard configuration: TORAX evolves the
electron and ion temperatures on GVEC's flux-surface geometry; before each 0.2 s segment
freeofcharge recomputes the beams from the plasma (two sources of the NI21 box, hydrogen at
55 keV, about 3.6 MW, freeofcharge's stand-in geometry) and GVEC re-solves the equilibrium with
TORAX's pressure; the transport is slowcooker's stellarator neoclassical model (the 1/ν and √ν
regimes with the ambipolar radial electric field) plus a constant anomalous diffusivity
$\\chi_\\mathrm{an}$, the same for electrons and ions, which is scanned.

**Set-up.** GVEC's W7-X example (five field periods, nearly no toroidal current) at reduced
resolution, as slowcooker ships it; the effective ripple $\\epsilon_\\mathrm{eff} = 0.01$, flat
(an assumption, not computed from the equilibrium); a flat density of $5\\times10^{19}$ m$^{-3}$
held fixed (the beams' fuelling is passed to TORAX, but the density is not evolved); hydrogen
beams in a deuterium plasma (TORAX's default main ion); edge temperatures 0.1 keV; 0.6 s, which
is several confinement times for every $\\chi_\\mathrm{an}$ of the scan.

**Measured.** At the end of the run: the temperature profiles, the neoclassical diffusivities
of each species, the **neoclassical share** of the heat flux $\\chi_\\mathrm{neo} /
(\\chi_\\mathrm{neo} + \\chi_\\mathrm{an})$ (both act on the same temperature gradient), the
ambipolar $E_r$ (recomputed from the final profiles with the model's own root finder), the
absorbed beam power and the energy confinement time $W/P$.

**Finding (quick).** The neoclassical share depends steeply on the temperature, as the
$1/\nu$ regime's $D \\propto T^{7/2}$ makes it. With $\\chi_\\mathrm{an} = 0.1$ m$^2$/s the
3.6 MW of beam power (0.3 % shine-through at this density) heat the core to 2.2 keV
($\tau_E$ = 250 ms) and the neoclassical channel carries 30 to 50 % of the heat inside
$\rho \approx 0.5$, falling to zero at the edge; with $\\chi_\\mathrm{an}$ = 0.3 and 1 m$^2$/s
the plasma stays at 1 keV and 0.4 keV and the neoclassical share is a few percent and below.
$E_r$ is negative (the ion root, $T_e \approx T_i$) everywhere, a few kV/m, up to 14 kV/m near
the edge of the hottest case. W7-X experiments put the neoclassical share of the losses at about
30 % of the heating at high temperatures (Beidler et al. 2021) and below about 40 % in the core
(Pablant et al.): the hottest case of the scan is in that range, the others are far colder than
those discharges. Nothing more can be said with a model whose prefactors are not yet verified.

**Not verified, so indicative only.** The neoclassical model's prefactors are written from the
standard forms and not yet checked against Nemov et al. (1999), Beidler et al. (2011) or a
monoenergetic database (DKES, MONKES, NTX); $\\epsilon_\\mathrm{eff}$ is assumed; the beam
geometry is a stand-in for IPP's; the anomalous transport is a constant. Compared: the
published neoclassical shares, from abstracts and summaries only. Not compared: any measured
W7-X profile.

References: C. D. Beidler et al., Demonstration of reduced neoclassical energy transport in
Wendelstein 7-X, Nature 596, 221 (2021), doi:10.1038/s41586-021-03687-w (abstract and press
summary); N. A. Pablant et al., IAEA FEC 2018 contribution on W7-X core $E_r$ and neoclassical
transport (abstract).
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

#: Anomalous diffusivities [m^2/s], segment length [s], end time [s], markers per source.
PRESETS = {
    "smoke": {
        "chi": [0.3],
        "segment": 0.1,
        "t_final": 0.2,
        "markers": 100,
        "n_rho": 12,
    },
    "quick": {
        "chi": [0.1, 0.3, 1.0],
        "segment": 0.2,
        "t_final": 0.6,
        "markers": 500,
        "n_rho": 20,
    },
}

DENSITY = 5.0e19  # [m^-3]
EPSILON_EFF = 0.01
KEV = 1.602176634e-16  # [J]


def ambipolar_profile(state, geometry_b0: float, major_radius: float, rho_b: float):
    """The model's ambipolar E_r [V/m] from the final profiles, on the radius ``rho``."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from slowcooker.neoclassical import M_E, M_P, ambipolar_field

    rho = state["rho"].values
    radius = np.maximum(rho * rho_b, 1e-3)
    n_e, n_i = state["n_e"].values, state["n_i"].values
    t_e, t_i = state["T_e"].values * KEV, state["T_i"].values * KEV
    dr = np.gradient(radius)
    electrons = (M_E, -1.0, n_e, t_e)
    ions = (float(state["A_i"]) * M_P, 1.0, n_i, t_i)
    ge = (np.gradient(n_e) / dr, np.gradient(t_e) / dr)
    gi = (np.gradient(n_i) / dr, np.gradient(t_i) / dr)
    field = ambipolar_field(
        electrons, ions, ge, gi, EPSILON_EFF, geometry_b0, major_radius, radius
    )
    return np.asarray(field)


def run_one(chi: float, preset: dict) -> dict:
    """One run of the chain with the anomalous diffusivity ``chi``."""
    import freeofcharge as foc
    from freeofcharge.injectors import w7x_injectors
    from gvec.util.params import read_parameters
    from slowcooker import base_config, gvec_geometry, merge, profiles, run_coupled
    from slowcooker.data import equilibrium_files, gvec_equilibrium
    from slowcooker.equilibrium import EquilibriumCoupling
    from slowcooker.nbi import NBICoupling
    from slowcooker.neoclassical import register

    register()
    equilibrium = gvec_equilibrium("w7x")
    rho = np.linspace(0.02, 1.0, 30)
    geometry = gvec_geometry(equilibrium, rho=rho, angles=(32, 32))
    config = merge(
        base_config(
            geometry,
            density=DENSITY,
            temperature=(1.0, 0.1),
            chi=chi,
            t_final=preset["t_final"],
            dt=0.05,
            n_rho=preset["n_rho"],
        ),
        {
            "sources": {"generic_heat": None},
            "transport": {
                "core_transport_models": {
                    "neo": {
                        "model_name": "stellarator_neoclassical",
                        "epsilon_eff": EPSILON_EFF,
                    }
                }
            },
        },
    )
    beams = NBICoupling(
        foc.GriddedEquilibrium(equilibrium, (48, 32, 48)),
        w7x_injectors(sources={"NI21": [1, 2]}),
        rho_edges=np.linspace(0.0, 1.0, 16),
        options={"markers": preset["markers"], "seed": 1},
    )
    pressure = EquilibriumCoupling(
        read_parameters(str(equilibrium_files("w7x")[0])),
        equilibrium.state,
        rho=rho,
        angles=(32, 32),
        B_0=config["geometry"]["B_0"],
    )
    times = np.arange(0.0, preset["t_final"] + 1e-9, preset["segment"])
    output = run_coupled(config, [beams, pressure], times)
    final = profiles(output).isel(t=-1)
    neo = output["auxiliary/turbulent_transport/core/neo"].to_dataset().isel(time=-1)
    face = neo["rho_face_norm"].values
    chi_neo_i = neo["chi_turb_i"].values
    chi_neo_e = neo["chi_turb_e"].values
    torax_geometry = output["geometry"].to_dataset() if "geometry" in output else None
    major = float(config["geometry"]["R_major"])
    b0 = float(config["geometry"]["B_0"])
    phi_b = 2 * np.pi * float(geometry.attrs["toroidal_flux_edge"])
    rho_b = float(np.sqrt(phi_b / (np.pi * b0)))
    del torax_geometry
    last = beams.results[max(beams.results)]
    absorbed = float(last["absorbed_power"])
    stored = float(final["W_thermal_total"])
    return {
        "chi": chi,
        "rho": final["rho"].values,
        "T_e": final["T_e"].values,
        "T_i": final["T_i"].values,
        "face": face,
        "chi_neo_i": chi_neo_i,
        "chi_neo_e": chi_neo_e,
        "E_r": ambipolar_profile(final, b0, major, rho_b),
        "absorbed": absorbed,
        "shine_through": float(last["shine_through_fraction"]),
        "stored": stored,
        "tau_e": stored / absorbed,
    }


def simulate(preset: str = "quick") -> dict:
    """Run the scan of the anomalous diffusivity; return one record per value."""
    start = time.perf_counter()
    settings = PRESETS[preset]
    runs = [run_one(chi, settings) for chi in settings["chi"]]
    return {"preset": preset, "runs": runs, "wall_time": time.perf_counter() - start}


def share(run: dict, species: str) -> np.ndarray:
    """chi_neo / (chi_neo + chi_an) on the faces."""
    neo = run[f"chi_neo_{species}"]
    return neo / (neo + run["chi"])


def plot(result: dict) -> None:
    """The temperatures, the neoclassical shares and E_r of every run."""
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(result["runs"])))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for run, color in zip(result["runs"], colors, strict=True):
        label = rf"$\chi_\mathrm{{an}}$ = {run['chi']:g} m$^2$/s"
        ax.plot(run["rho"], run["T_e"], color=color, label=label)
        ax.plot(run["rho"], run["T_i"], color=color, ls="--")
    ax.set_xlabel(r"$\rho = \sqrt{\Phi/\Phi_b}$")
    ax.set_ylabel("temperature [keV]")
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhspan(0.3, 0.4, color="0.85", label="W7-X, reported (abstracts)")
    for run, color in zip(result["runs"], colors, strict=True):
        label = rf"$\chi_\mathrm{{an}}$ = {run['chi']:g} m$^2$/s"
        ax.plot(run["face"][1:], share(run, "i")[1:], color=color, label=label)
        ax.plot(run["face"][1:], share(run, "e")[1:], color=color, ls="--")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\chi_\mathrm{neo} / (\chi_\mathrm{neo} + \chi_\mathrm{an})$")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for run, color in zip(result["runs"], colors, strict=True):
        label = rf"$\chi_\mathrm{{an}}$ = {run['chi']:g} m$^2$/s"
        ax.plot(run["rho"][1:-1], run["E_r"][1:-1] / 1e3, color=color, label=label)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$E_r$ [kV/m]")
    ax.legend()
    fig.tight_layout()


def table(result: dict) -> str:
    """The central temperatures, the shares at mid-radius, E_r and the confinement time."""
    rows = [
        "| χ_an [m²/s] | T_e(0) [keV] | T_i(0) [keV] | neo. share of ion heat at ρ=0.5 | of electron heat | E_r at ρ=0.5 [kV/m] | absorbed [MW] | τ_E = W/P [ms] |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        mid = int(np.argmin(np.abs(run["face"] - 0.5)))
        e_r = np.interp(0.5, run["rho"], run["E_r"]) / 1e3
        rows.append(
            f"| {run['chi']:g} | {run['T_e'][0]:.2f} | {run['T_i'][0]:.2f} | "
            f"{100 * share(run, 'i')[mid]:.0f} % | {100 * share(run, 'e')[mid]:.0f} % | "
            f"{e_r:.1f} | {run['absorbed'] / 1e6:.2f} | {1e3 * run['tau_e']:.0f} |"
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
