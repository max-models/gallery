r"""ASCOT5 benchmark.

freeofcharge against ASCOT5 on one tokamak case (`freeofcharge.ascot5.benchmark_case`): an
ASDEX Upgrade-sized Solov'ev equilibrium ($R_0 = 1.65$ m, $a = 0.5$ m, elongation 1.6, 2.5 T)
of deuterium with 1% carbon, $5\times10^{19}\,\mathrm{m^{-3}}$ and 2.5 keV in the centre, and a
2.5 MW beam of 60 keV deuterium tangent to $R = 1$ m. freeofcharge attenuates 2000 straight
markers with Suzuki's stopping cross sections and slows the ions down on their birth surfaces
(Stix) until $\max(2T_i, 1\,\mathrm{keV})$. BBNBI5 flies along the same lines with the same cross
sections; ascot5_main follows 1000 of its ions as guiding centres with Monte Carlo collisions,
once along their orbits and once held at their birth points. freeofcharge's current is that of
the orbit-averaged pitch (44% of the ions are born trapped), or of the local pitch against the
run without orbits.

The shine-through and the deposition agree within BBNBI5's Monte Carlo noise. Without orbits,
the heating, the fast-ion content and the fast-ion current agree to 2-5%, about the 4% by which
ASCOT5's moments overshoot its birth power. With orbits, freeofcharge's fast-ion current is 18%
above ASCOT5's, most likely from the orbit width it leaves out, and its electron heating 9%
below.

Compared: the shine-through at three densities; profiles and totals of the deposited power,
heating, fast-ion density and current. Not compared: the driven current with shielding, whose
trapped-electron corrections differ (freeofcharge's documentation page "ASCOT5 benchmark").

Reference: ASCOT5 (commit 24be846) results in `data/ascot5_results.json`, made by
`benchmarks/ascot5/run_ascot5.py` of the freeofcharge repository; J. Varje et al.,
arXiv:1908.02482 (2019).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
import xarray as xr
from freeofcharge import ascot5

#: markers: beam markers of freeofcharge; step: the step along each line [m].
PRESETS = {
    "smoke": {"markers": 200, "step": 0.02},
    "full": {"markers": 2000, "step": 0.005},
}

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "data" / "ascot5_results.json"


def simulate(preset: str = "full") -> dict:
    """Run freeofcharge on the benchmark case and compare it with the stored ASCOT5 results."""
    options = PRESETS[preset]
    reference = ascot5.load_results(REFERENCE)
    case = ascot5.benchmark_case(markers=options["markers"])
    ours = ascot5.run_freeofcharge(case, step=options["step"])
    return {
        "preset": preset,
        "ours": ours,
        "reference": reference,
        "comparison": ascot5.compare(ours, reference),
        "comparison_local": ascot5.compare(
            ascot5.without_orbits(ours), ascot5.without_orbits(reference)
        ),
    }


def profile(
    runs: dict, name: str, scale: float, long_name: str, units: str
) -> xr.DataArray:
    """A profile of several runs, with a ``code`` axis."""
    edges = np.asarray(next(iter(runs.values()))["rho_edges"])
    rho = 0.5 * (edges[1:] + edges[:-1])
    return xr.DataArray(
        np.stack([np.asarray(run[name]) for run in runs.values()]) * scale,
        coords={
            "code": list(runs),
            "rho": ("rho", rho, {"long_name": r"$\rho_\mathrm{pol}$"}),
        },
        dims=("code", "rho"),
        attrs={"long_name": long_name, "units": units},
    )


def plot(result: dict) -> None:
    """The shine-through scan and the profiles of both codes."""
    ours, reference = result["ours"], result["reference"]
    deposition = {"freeofcharge": ours, "ASCOT5": reference}
    # freeofcharge's heating and density are the same with either current
    slowing_down = {**deposition, "ASCOT5, no orbits": ascot5.without_orbits(reference)}
    currents = {
        **deposition,
        "freeofcharge, local pitch": ascot5.without_orbits(ours),
        "ASCOT5, no orbits": slowing_down["ASCOT5, no orbits"],
    }
    shine = xr.DataArray(
        100
        * np.stack(
            [
                ours["scan_shine_through_fraction"],
                reference["scan_shine_through_fraction"],
            ]
        ),
        coords={
            "code": list(deposition),
            "ne0": (
                "ne0",
                np.asarray(ours["scan_ne0"]),
                {"long_name": "central density", "units": "1/m^3"},
            ),
        },
        dims=("code", "ne0"),
        attrs={"long_name": "shine-through", "units": "%"},
    )
    with plasma_plots.figure(
        figsize=(7, 4),
        caption="Shine-through against the central electron density, freeofcharge and BBNBI5 "
        "(ASCOT5) along the same lines.",
    ) as fig:
        shine.plasma.plot.profiles(x="ne0", over="code", logy=True, ax=fig[0])
    with plasma_plots.figure(
        figsize=(7, 4),
        caption="Deposited power per shell of $\\rho_\\mathrm{pol}$ at $n_{e0} = 5\\times10^{19}"
        "\\,\\mathrm{m^{-3}}$, freeofcharge and BBNBI5.",
    ) as fig:
        profile(
            deposition, "deposited_power", 1e-3, "deposited power per shell", "kW"
        ).plasma.plot.profiles(x="rho", over="code", ax=fig[0])
    with plasma_plots.figure(
        figsize=(7, 4),
        caption="Fast-ion density: freeofcharge (its slowing down does not depend on the orbits) "
        "and ASCOT5 with and without orbits.",
    ) as fig:
        profile(
            slowing_down, "fast_ion_density", 1e-17, "fast-ion density", "10^17/m^3"
        ).plasma.plot.profiles(x="rho", over="code", ax=fig[0])
    with plasma_plots.figure(
        2,
        1,
        sharex=True,
        figsize=(7, 7),
        caption="Heating per shell of the electrons (top) and the ions (bottom): freeofcharge and "
        "ASCOT5 with and without orbits.",
    ) as fig:
        profile(
            slowing_down, "electron_heating", 1e-3, "electron heating per shell", "kW"
        ).plasma.plot.profiles(x="rho", over="code", ax=fig[0])
        profile(
            slowing_down, "ion_heating", 1e-3, "ion heating per shell", "kW"
        ).plasma.plot.profiles(x="rho", over="code", ax=fig[1])
    with plasma_plots.figure(
        figsize=(7, 4),
        caption="Fast-ion current density: freeofcharge with the orbit-averaged pitch against ASCOT5 "
        "with orbits, and with the local pitch against ASCOT5 without orbits.",
    ) as fig:
        profile(
            currents,
            "fast_ion_current_density",
            1e-3,
            "fast-ion current density",
            "kA/m^2",
        ).plasma.plot.profiles(x="rho", over="code", ax=fig[0])


def table(result: dict) -> str:
    """The totals of both codes, with orbits and without, and their ratios."""
    orbits, local = result["comparison"]["totals"], result["comparison_local"]["totals"]
    profiles = result["comparison"]["profiles"]
    profiles_local = result["comparison_local"]["profiles"]
    rows = [
        ("Electron heating [MW]", "electron_heating", 1e-6, "{:.3f}", True),
        ("Ion heating [MW]", "ion_heating", 1e-6, "{:.3f}", True),
        ("Thermalised power [kW]", "thermalised_power", 1e-3, "{:.0f}", False),
        ("Fast ions [$10^{19}$]", "fast_ions", 1e-19, "{:.3f}", False),
        ("Fast-ion energy [kJ]", "stored_energy", 1e-3, "{:.1f}", False),
        ("Fast-ion current [kA]", "fast_ion_current", 1e-3, "{:.1f}", False),
        (
            "Driven current, with shielding [kA]",
            "driven_current",
            1e-3,
            "{:.1f}",
            False,
        ),
    ]
    lines = [
        "| Quantity | freeofcharge | ASCOT5 | ratio | freeofcharge, local pitch | "
        "ASCOT5, no orbits | ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["comparison"]["scan"]:
        mantissa, exponent = f"{row['ne0']:.0e}".split("e")
        lines.append(
            f"| Shine-through, $n_{{e0}} = {mantissa}\\times10^{{{int(exponent)}}}$ m$^{{-3}}$ | "
            f"{100 * row['ours']:.3f} % | {100 * row['reference']:.3f} ± "
            f"{100 * row['reference_error']:.3f} % | {row['ours'] / row['reference']:.3f} | | | |"
        )
    for label, name, scale, fmt, is_profile in rows:
        a, b = (profiles, profiles_local) if is_profile else (orbits, local)
        cells = [label]
        for entry in (a[name], b[name]):
            cells += [
                fmt.format(entry["ours"] * scale),
                fmt.format(entry["reference"] * scale),
                f"{entry['ratio']:.3f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    plot(simulate(args.preset))
    plt.show()
