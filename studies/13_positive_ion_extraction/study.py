r"""Positive-ion extraction.

The plasma of a deuterium RF source feeds one beamlet of a positive-ion injector, like
those of ASDEX Upgrade, JET or W7-X. freeofcharge's `deuterium_source` (a wellbalanced
global model) gives the electron density and temperature and the D$_2^+$ Bohm flux to the
plasma grid: $j_+ = $ 130, 260 and 520 A/m$^2$ at 100, 200 and 400 kW, with $T_e = 5.8$ eV.
beamtracer's `BeamOptics` with an `ExtractionPlasma` solves the beamlet in its own field:
the ions leave the plasma at the Bohm speed and are traced, the plasma electrons follow a
Boltzmann distribution at the source's $T_e$, and a Vlasov iteration repeats the solve until
field and trajectories agree, which places the meniscus. As in IBSimu, the plasma boundary
is held at the plasma potential 0.5 mm in front of the plasma grid: the meniscus moves
freely downstream of it, but not upstream.

The round beamlet ($x$-$r$, illustrative geometry) has a plasma grid (PG, 0 V) with a 6 mm
aperture, an extraction grid (EG) 5 mm behind it at $-U_\mathrm{ex}$ and a grounded grid at
$-0.9\,U_\mathrm{ex}$. Scanning $U_\mathrm{ex}$ gives the perveance V-curve of the rms
divergence: over-focused at high voltage, blown up by space charge and scraped by the EG at
low voltage. The minimum, 32 mrad, lies at $j_+/j_\mathrm{CL} \approx 0.22$ for every power,
so the optimum voltage rises as $j_+^{2/3}$: 6.6, 10.6 and 16.8 kV. Transmission stays
above 99 % up to $j_+/j_\mathrm{CL} = 0.36$ and falls to about half at 0.75.

Compared: nothing outside the models. Not compared: a measurement, or a PIC model of the
sheath.

Reference: beamtracer's example `examples/negative_ion_extraction_optics.py` (the beamlet
set-up) and freeofcharge's `chain.deuterium_source`.
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    ExtractionPlasma,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    ObjectParams,
)
from freeofcharge import chain
from scipy import constants

#: rf_power: RF powers of the beamlet scans [W]; source_scan: RF powers of the source curve
#: [W]; perveance: the normalized perveances j+ / j_CL of each scan; cell: mesh spacing [m];
#: particles: trajectories; iterations: the largest number of Vlasov iterations.
PRESETS = {
    "smoke": {
        "rf_power": [200e3],
        "source_scan": [100e3, 400e3],
        "perveance": [0.3],
        "cell": 0.25e-3,
        "particles": 20,
        "iterations": 3,
    },
    "full": {
        "rf_power": [100e3, 200e3, 400e3],
        "source_scan": list(np.linspace(50e3, 500e3, 10)),
        "perveance": [0.12, 0.17, 0.22, 0.28, 0.36, 0.5, 0.75],
        "cell": 0.05e-3,
        "particles": 300,
        "iterations": 80,
    },
}

ION_MASS = 2 * chain.DEUTERIUM_MASS  # D2+, the ion of the source model [kg]
#: Aperture radius and (front, back) along the axis of each grid [m]. The plasma boundary is
#: the left face of the mesh, 0.5 mm in front of the PG, held at the plasma potential: the
#: convention of IBSimu; a longer plasma region is not a steady solution of this model (the
#: ions enter at the Bohm speed, with no presheath).
GRIDS = {
    "PG": (3.0e-3, (0.5e-3, 1.5e-3)),
    "EG": (2.5e-3, (6.5e-3, 8.0e-3)),
    "GG": (3.0e-3, (10.5e-3, 12.0e-3)),
}
GAP = GRIDS["EG"][1][0] - GRIDS["PG"][1][1]  # the PG-EG gap of j_CL [m]
GG_SHARE = 0.9  # the GG at -0.9 U_ex: the beam leaves with 0.9 e U_ex
PLASMA_POTENTIAL = 20.0  # V above the PG, about 3.5 T_e
ION_TEMPERATURE = 0.5  # transverse, eV
LENGTH, RADIUS = 15.5e-3, 5.0e-3
EXIT = LENGTH - 0.5e-3  # the diagnostic plane behind the GG


def grid_vertices(name: str) -> tuple:
    """A grid's cross section above the axis; the PG has a chamfer towards the gap [m]."""
    radius, (x0, x1) = GRIDS[name]
    outer = RADIUS + 1e-3  # past the outer face of the mesh
    inner = radius + (x1 - x0) if name == "PG" else radius
    return ((x0, radius), (x0, outer), (x1, outer), (x1, inner))


def config(
    voltage: float, current_density: float, te: float, p: dict
) -> BeamOpticsConfig:
    """The beamlet at extraction voltage ``voltage``, D2+ current density and T_e [eV]."""
    cell = p["cell"]
    ions = BeamSpecies(
        "D2+",
        current_density,
        (0.0, 0.0),
        (0.0, GRIDS["PG"][0] - cell),  # the ions leave over the PG aperture
        charge=1.0,
        mass=ION_MASS / constants.atomic_mass,
        energy=te / 2,  # the Bohm speed
        transverse_temperature=ION_TEMPERATURE,
        num_particles=p["particles"],
    )
    voltages = {"PG": 0.0, "EG": -voltage, "GG": -GG_SHARE * voltage}
    neumann = FieldBoundaryCondition.neumann(0.0)
    return BeamOpticsConfig(
        grid=GridParams(
            ndim=2,
            num_gridpoints=(round(LENGTH / cell) + 1, round(RADIUS / cell) + 1),
            lower_bound=(0.0, 0.0),
            upper_bound=(LENGTH, RADIUS),
            coordinates="cylindrical",
        ),
        bcs_field=FieldBCs(
            x_lower=FieldBoundaryCondition.dirichlet(PLASMA_POTENTIAL),
            x_upper=FieldBoundaryCondition.dirichlet(voltages["GG"]),
            y_lower=neumann,
            y_upper=neumann,
        ),
        objects={
            name.lower(): ObjectParams(
                type="aperture2d",
                axis=0,
                position=0.0,
                vertices=grid_vertices(name),
                bc_type="dirichlet",
                bc_value=voltages[name],
            )
            for name in GRIDS
        },
        beams=[ions],
        # the electrons' density at U_p is that of the ions at their start (sum J / v)
        plasma=ExtractionPlasma(te, PLASMA_POTENTIAL, GRIDS["PG"][1][0]),
        iterations=p["iterations"],
        tolerance=1e-2,
        relaxation=0.5,
        diagnostic_planes=[EXIT],
        save_trajectories=10,
    )


def solve(voltage: float, current_density: float, te: float, p: dict) -> dict:
    """Solve the beamlet; the beam behind the GG, the field and the paths."""
    model = BeamOptics(config(voltage, current_density, te, p))
    start = time.perf_counter()
    model.run()
    seconds = time.perf_counter() - start
    moments = model.beam_moments(EXIT, "D2+")
    emission = GRIDS["PG"][0] - p["cell"]
    return {
        "extraction_voltage": voltage,
        "perveance": current_density
        / chain.child_langmuir(voltage, GAP, mass=ION_MASS),
        "converged": bool(model.stats["converged"]),
        "iterations": int(model.stats["iterations"]),
        "seconds": seconds,
        "transmitted": moments.current / (np.pi * emission**2 * current_density),
        "divergence": float(moments.rms_angle),
        "emittance": float(moments.emittance),
        "energy": float(moments.mean_energy),
        "potential": model.potential_field.evaluate(name="potential", units="V"),
        "trajectories": [np.asarray(t) for t in model.trajectories],
    }


def optimum(runs: list) -> tuple[float, float]:
    """The perveance and divergence at the minimum of a parabola in log(perveance).

    The parabola goes through the smallest sampled divergence and its two neighbours.
    """
    divergence = np.array([run["divergence"] for run in runs])
    if len(runs) < 3:
        k = int(np.argmin(divergence))
        return runs[k]["perveance"], float(divergence[k])
    k = int(np.clip(np.argmin(divergence), 1, len(runs) - 2))
    x = np.log([runs[i]["perveance"] for i in (k - 1, k, k + 1)])
    a, b, c = np.polyfit(x, divergence[k - 1 : k + 2], 2)
    x0 = np.clip(-b / (2 * a), x.min(), x.max()) if a > 0 else x[1]
    return float(np.exp(x0)), float(np.polyval((a, b, c), x0))


def simulate(preset: str = "full") -> dict:
    """The source against the RF power, then a perveance scan of the beamlet per power."""
    p = PRESETS[preset]
    start = time.time()
    curve = [chain.deuterium_source(power) for power in p["source_scan"]]
    scans = []
    for power in p["rf_power"]:
        source = chain.deuterium_source(power)
        j = constants.e * source.positive_flux  # the D2+ current density [A/m^2]
        j_cl_per_volt = chain.child_langmuir(1.0, GAP, mass=ION_MASS)
        voltages = [(j / (pv * j_cl_per_volt)) ** (2 / 3) for pv in p["perveance"]]
        runs = [solve(v, j, source.electron_temperature, p) for v in voltages]
        best_perveance, best_divergence = optimum(runs)
        scans.append(
            {
                "rf_power": power,
                "electron_density": source.electron_density,
                "electron_temperature": source.electron_temperature,
                "current_density": j,
                "optimum_perveance": best_perveance,
                "optimum_voltage": (j / (best_perveance * j_cl_per_volt)) ** (2 / 3),
                "minimum_divergence": best_divergence,
                "runs": runs,
            }
        )
    return {
        "preset": preset,
        "source": {
            "rf_power": np.array([s.rf_power for s in curve]),
            "electron_density": np.array([s.electron_density for s in curve]),
            "electron_temperature": np.array([s.electron_temperature for s in curve]),
            "current_density": constants.e * np.array([s.positive_flux for s in curve]),
        },
        "scans": scans,
        "runtime": time.time() - start,
    }


def table(result: dict) -> str:
    """The source and the optimum of each scan, and whether every solve converged."""
    lines = [
        "| RF power (kW) | n_e (m⁻³) | T_e (eV) | j+ (A/m²) | optimum j+/j_CL "
        "| optimum U_ex (kV) | min. rms divergence (mrad) | transmitted at optimum "
        "| converged | iterations |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for scan in result["scans"]:
        runs = scan["runs"]
        best = min(runs, key=lambda run: run["divergence"])
        iterations = [run["iterations"] for run in runs]
        lines.append(
            f"| {scan['rf_power'] / 1e3:.0f} | {scan['electron_density']:.2e} | "
            f"{scan['electron_temperature']:.2f} | {scan['current_density']:.0f} | "
            f"{scan['optimum_perveance']:.2f} | {scan['optimum_voltage'] / 1e3:.1f} | "
            f"{scan['minimum_divergence'] * 1e3:.1f} | {best['transmitted']:.1%} | "
            f"{sum(run['converged'] for run in runs)} of {len(runs)} | "
            f"{min(iterations)} to {max(iterations)} |"
        )
    return "\n".join(lines)


def joined(paths: list) -> tuple[np.ndarray, np.ndarray]:
    """The paths as one NaN-separated line, in mm."""
    if not paths:
        return np.zeros(0), np.zeros(0)
    pieces = []
    for path in paths:
        pieces += [path * 1e3, np.full((1, 2), np.nan)]
    line = np.vstack(pieces[:-1])
    return line[:, 0], line[:, 1]


def plot(result: dict) -> None:
    """The V-curves, the source, and the beamlet at its optimum."""
    scans = result["scans"]
    labels = [f"{scan['rf_power'] / 1e3:.0f} kW" for scan in scans]

    caption = (
        "The perveance V-curve: the rms divergence of the D₂⁺ beamlet behind the GG (top) "
        "and the share of the extracted current that reaches it (bottom) against the "
        "extraction voltage, for three RF powers (black crosses: the iteration did not converge)."
    )
    with plasma_plots.figure(2, 1, sharex=True, figsize=(7, 6), caption=caption) as fig:
        for scan, label in zip(scans, labels, strict=True):
            runs = scan["runs"]
            voltage = np.array([run["extraction_voltage"] for run in runs]) / 1e3
            divergence = 1e3 * np.array([run["divergence"] for run in runs])
            ok = np.array([run["converged"] for run in runs])
            (line,) = fig[0].plot(voltage, divergence, marker="o", label=label)
            if not ok.all():
                fig[0].plot(
                    voltage[~ok],
                    divergence[~ok],
                    "x",
                    ms=10,
                    mew=2,
                    color="black",
                )
            transmitted = 100 * np.array([run["transmitted"] for run in runs])
            fig[1].plot(voltage, transmitted, marker="o", label=label)
        fig[0].set_ylabel("rms divergence [mrad]")
        fig[0].legend(fontsize=8)
        fig[1].set_ylabel("transmitted [%]")
        fig[1].set_xlabel("extraction voltage $U_\\mathrm{ex}$ [kV]")

    caption = (
        "The same divergences against the normalized perveance j₊/j_CL (Child-Langmuir "
        "current of D₂⁺ over the 5 mm PG-EG gap): the three RF powers agree within a few "
        "mrad; the squares mark the optimum of each scan (a parabola through the "
        "lowest three points)."
    )
    with plasma_plots.figure(figsize=(7, 4), caption=caption) as fig:
        ax = fig[0]
        for scan, label in zip(scans, labels, strict=True):
            runs = scan["runs"]
            perveance = [run["perveance"] for run in runs]
            (line,) = ax.plot(
                perveance,
                [1e3 * run["divergence"] for run in runs],
                marker="o",
                label=label,
            )
            ax.scatter(
                [scan["optimum_perveance"]],
                [1e3 * scan["minimum_divergence"]],
                marker="s",
                s=60,
                color=line.get_color(),
            )
        ax.set_xscale("log")
        ax.set_xlabel("normalized perveance $j_+ / j_\\mathrm{CL}$")
        ax.set_ylabel("rms divergence [mrad]")
        ax.legend(fontsize=8)

    source = result["source"]
    caption = (
        "The source (wellbalanced global model in freeofcharge): the D₂⁺ current density "
        "to the plasma grid (top) and the electron density (bottom) against the RF power; "
        "the circles mark the powers of the beamlet scans. The electron temperature is "
        f"{source['electron_temperature'][0]:.2f} eV at every power."
    )
    with plasma_plots.figure(
        2, 1, sharex=True, figsize=(7, 5.5), caption=caption
    ) as fig:
        power = source["rf_power"] / 1e3
        fig[0].plot(power, source["current_density"], color="tab:gray")
        fig[1].plot(power, source["electron_density"], color="tab:gray")
        for scan, label in zip(scans, labels, strict=True):
            kw = scan["rf_power"] / 1e3
            fig[0].scatter([kw], [scan["current_density"]], s=50, label=label)
            fig[1].scatter([kw], [scan["electron_density"]], s=50)
        fig[0].set_ylabel("$j_+$ [A/m$^2$]")
        fig[0].legend(fontsize=8)
        fig[1].set_ylabel("$n_e$ [m$^{-3}$]")
        fig[1].set_xlabel("RF power [kW]")

    scan = scans[len(scans) // 2]
    best = min(scan["runs"], key=lambda run: run["divergence"])
    potential = best["potential"]
    te = scan["electron_temperature"]
    meniscus = PLASMA_POTENTIAL - 2 * te  # the electron density down to e^-2
    zoom = potential.sel(x=slice(0.0, 4e-3))
    zoom = zoom.assign_coords(x=zoom.x * 1e3, y=zoom.y * 1e3)
    zoom.attrs.update(long_name="potential", units="V")
    zoom.x.attrs.update(long_name="x", units="mm")
    zoom.y.attrs.update(long_name="r", units="mm")
    potential = potential / 1e3
    potential = potential.assign_coords(x=potential.x * 1e3, y=potential.y * 1e3)
    potential.attrs.update(long_name="potential", units="kV")
    potential.x.attrs.update(long_name="x", units="mm")
    potential.y.attrs.update(long_name="r", units="mm")
    caption = (
        f"The beamlet with the lowest divergence at {scan['rf_power'] / 1e3:.0f} kW "
        f"(U_ex = {best['extraction_voltage'] / 1e3:.1f} kV, "
        f"j₊/j_CL = {best['perveance']:.2f}): the potential, the PG, EG and GG (grey), "
        "D₂⁺ trajectories (black) and the meniscus (white), where the Boltzmann electron "
        "density has fallen to e⁻² of the plasma's."
    )
    with plasma_plots.figure(2, 1, figsize=(7, 6.5), caption=caption) as fig:
        x, y = joined(best["trajectories"])
        for ax, field, xmax in ((fig[0], potential, LENGTH), (fig[1], zoom, 4e-3)):
            field.plasma.plot.slice(x="x", y="y", cmap="viridis", ax=ax)
            for name in GRIDS:
                v = np.asarray(grid_vertices(name) + grid_vertices(name)[:1]) * 1e3
                v[:, 1] = np.minimum(v[:, 1], RADIUS * 1e3)
                ax.plot(v[:, 0], v[:, 1], color="tab:gray", lw=2)
            ax.plot(x, y, color="black", lw=0.6)
            level = meniscus / 1e3 if field is potential else meniscus
            ax.contour(field.x, field.y, field.values.T, levels=[level], colors="white")
            ax.set_xlim(0, xmax * 1e3)
            ax.set_ylim(0, RADIUS * 1e3)
        fig[1].set_title("the plasma side, enlarged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    output = simulate(args.preset)
    print(table(output))
    plot(output)
    plt.show()
