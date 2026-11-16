r"""Negative-ion extraction.

An H$^-$ beamlet and the electrons extracted with it, solved in their own field in one steady
calculation with beamtracer's `BeamOptics` and `NegativeExtractionPlasma`: both species are
traced from the plasma and deposit their space charge, the plasma's positive ions, repelled by
the extraction field, follow a Boltzmann distribution $\rho_p \exp(-(\phi - U_p)/T_p)$ that
forms the meniscus, and a Vlasov iteration repeats the solve until field and trajectories
agree. A solve takes seconds, where a time-dependent PIC run of a beamlet takes tens of
minutes: it is the evaluation an optimisation of the grids would repeat many times.

The beamlet is round ($r$-$z$): a plasma grid (PG, 0 V) with an 8 mm chamfered aperture, an
extraction grid (EG) at $U_\mathrm{ex}$, a grounded grid (GG) at 20 kV, and 300 A/m$^2$ each of
H$^-$ and electrons; the geometry is illustrative. From 10 to 4 kV (0.14 to 0.54 of the
Child-Langmuir current of the PG-EG gap) the rms divergence behind the GG falls from 41 to 16
mrad: at high voltage the concave meniscus over-focuses the beam. Below 8 kV the beam grazes
the EG (0.7 to 5.7 % of the H$^-$ lost) and the iteration does not settle within 40
iterations; an optimisation has to treat such points as unconverged.

Without a magnetic field the electrons take nearly the paths of the H$^-$: in a static electric
field a trajectory depends on charge and energy, not mass. Dumping them needs deflection
magnets, which this axisymmetric model cannot hold, and the electron current is an input set by
the source plasma.

Compared: nothing outside the model. Not compared: a measurement or a PIC model of the source.

Reference: beamtracer's example `examples/negative_ion_extraction_optics.py`.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    NegativeExtractionPlasma,
    ObjectParams,
)
from scipy import constants

#: voltages: extraction voltages [V]; cell: mesh spacing [m]; particles: trajectories per
#: species; iterations: the largest number of Vlasov iterations.
PRESETS = {
    "smoke": {"voltages": [8e3], "cell": 1e-3, "particles": 20, "iterations": 2},
    "quick": {
        "voltages": [4e3, 5e3, 6e3, 8e3, 10e3],
        "cell": 0.1e-3,
        "particles": 300,
        "iterations": 40,
    },
}

H_MINUS_MASS = 1.00849  # u
#: Aperture radius and (front, back) along the axis of each grid [m].
GRIDS = {
    "PG": (4.0e-3, (1.0e-3, 2.0e-3)),
    "EG": (3.0e-3, (6.0e-3, 8.0e-3)),
    "GG": (3.5e-3, (16.0e-3, 18.0e-3)),
}
GG_VOLTAGE = 20.0e3
PLASMA_POTENTIAL = -1.0
ION_TEMPERATURE = 1.0  # eV
CURRENT_DENSITY = 300.0  # A/m^2 of H-, and as much of electrons
LENGTH, RADIUS = 24.0e-3, 6.0e-3
EXIT = LENGTH - 0.5e-3  # the diagnostic plane behind the GG


def grid_vertices(name: str) -> tuple:
    """A grid's cross section above the axis; the PG has a chamfer towards the gap [m]."""
    radius, (x0, x1) = GRIDS[name]
    outer = RADIUS + 1e-3  # past the outer face of the mesh
    inner = radius + (x1 - x0) if name == "PG" else radius
    return ((x0, radius), (x0, outer), (x1, outer), (x1, inner))


def child_langmuir(voltage: float) -> float:
    """The Child-Langmuir current density of H- over the PG-EG gap [A/m^2]."""
    gap = GRIDS["EG"][1][0] - GRIDS["PG"][1][0]
    charge_to_mass = constants.e / (H_MINUS_MASS * constants.atomic_mass)
    return (
        4
        * constants.epsilon_0
        / 9
        * np.sqrt(2 * charge_to_mass)
        * voltage**1.5
        / gap**2
    )


def config(voltage: float, p: dict) -> BeamOpticsConfig:
    """The beamlet at the extraction voltage ``voltage`` with the numerics of preset ``p``."""
    cell = p["cell"]
    emission = GRIDS["PG"][0] - cell  # the beams leave over the PG aperture
    h_minus = BeamSpecies(
        "H-",
        CURRENT_DENSITY,
        (0.0, 0.0),
        (0.0, emission),
        charge=-1.0,
        mass=H_MINUS_MASS,
        energy=1.0,
        transverse_temperature=1.0,
        num_particles=p["particles"],
    )
    electrons = replace(
        h_minus,
        name="e",
        mass=constants.m_e / constants.atomic_mass,
        transverse_temperature=2.0,
    )
    voltages = {"PG": 0.0, "EG": voltage, "GG": GG_VOLTAGE}
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
            x_upper=FieldBoundaryCondition.dirichlet(GG_VOLTAGE),
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
        beams=[h_minus, electrons],
        plasma=NegativeExtractionPlasma(
            ION_TEMPERATURE, PLASMA_POTENTIAL, GRIDS["PG"][1][0]
        ),
        iterations=p["iterations"],
        tolerance=1e-2,
        relaxation=0.5,
        diagnostic_planes=[EXIT],
        save_trajectories=10,
    )


def solve(voltage: float, p: dict) -> dict:
    """Solve the beamlet; the H- beam behind the GG, its losses, the field and the paths."""
    cfg = config(voltage, p)
    model = BeamOptics(cfg)
    start = time.perf_counter()
    model.run()
    seconds = time.perf_counter() - start
    h_minus = model.beam_moments(EXIT, "H-")
    emission = GRIDS["PG"][0] - p["cell"]
    status = model.state.trace.status
    species = np.repeat(np.arange(len(model.species_names)), p["particles"])
    return {
        "extraction_voltage": voltage,
        "perveance_ratio": CURRENT_DENSITY / child_langmuir(voltage),
        "converged": bool(model.stats["converged"]),
        "iterations": int(model.stats["iterations"]),
        "seconds": seconds,
        "transmitted": h_minus.current / (np.pi * emission**2 * CURRENT_DENSITY),
        "divergence": float(h_minus.rms_angle),
        "emittance": float(h_minus.emittance),
        "energy": float(h_minus.mean_energy),
        "lost_h_minus": int(np.sum((status != 0) & (species == 0))),
        "history": np.array(
            [h["max_potential_change"] for h in model.convergence_history[1:]]
        ),
        "potential": model.potential_field.evaluate(name="potential", units="V"),
        "trajectories": [np.asarray(t) for t in model.trajectories],
        "trajectory_species": np.asarray(model.trajectory_species),
    }


def simulate(preset: str = "quick") -> dict:
    """Solve the beamlet at every extraction voltage of the preset."""
    p = PRESETS[preset]
    start = time.time()
    runs = [solve(voltage, p) for voltage in p["voltages"]]
    return {"preset": preset, "runs": runs, "runtime": time.time() - start}


def table(result: dict) -> str:
    """Each solve: the H- beam behind the GG and the cost."""
    lines = [
        "| U_ex (kV) | J / J_CL | converged | iterations | solve (s) | H- transmitted "
        "| H- lost | rms divergence (mrad) | emittance (mm mrad) | energy (keV) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in result["runs"]:
        lines.append(
            f"| {run['extraction_voltage'] / 1e3:.0f} | {run['perveance_ratio']:.2f} | "
            f"{'yes' if run['converged'] else 'no'} | {run['iterations']} | "
            f"{run['seconds']:.0f} | {run['transmitted']:.1%} | {run['lost_h_minus']} | "
            f"{run['divergence'] * 1e3:.1f} | {run['emittance'] * 1e6:.2f} | "
            f"{run['energy'] / 1e3:.1f} |"
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
    """The beamlet with the lowest divergence, the scan, the convergence."""
    runs = result["runs"]
    converged = [run for run in runs if run["converged"]] or runs
    best = min(converged, key=lambda run: run["divergence"])

    potential = best["potential"] / 1e3
    potential = potential.assign_coords(x=potential.x * 1e3, y=potential.y * 1e3)
    potential.attrs.update(long_name="potential", units="kV")
    potential.x.attrs.update(long_name="x", units="mm")
    potential.y.attrs.update(long_name="r", units="mm")
    caption = (
        f"The potential of the beamlet with the lowest converged divergence "
        f"(U_ex = {best['extraction_voltage'] / 1e3:.0f} kV), the PG, EG and GG, and "
        "trajectories of the H- (black) and of the electrons (red), on nearly the same paths."
    )
    with plasma_plots.figure(figsize=(7, 3.4), caption=caption) as fig:
        ax = fig[0]
        potential.plasma.plot.slice(x="x", y="y", cmap="viridis", ax=ax)
        for name in GRIDS:
            v = np.asarray(grid_vertices(name) + grid_vertices(name)[:1]) * 1e3
            ax.plot(v[:, 0], np.minimum(v[:, 1], RADIUS * 1e3), color="tab:gray", lw=2)
        species, paths = best["trajectory_species"], best["trajectories"]
        for k, color, name in ((1, "red", "electrons"), (0, "black", "H-")):
            x, y = joined([q for q, s in zip(paths, species, strict=True) if s == k])
            ax.plot(x, y, color=color, lw=0.7, label=name)
        ax.set_ylim(0, RADIUS * 1e3)
        ax.legend(loc="upper right", fontsize=8)

    voltages = np.array([run["extraction_voltage"] for run in runs]) / 1e3
    ok = np.array([run["converged"] for run in runs])
    caption = (
        "The H- beam behind the GG against the extraction voltage at 300 A/m²: top, its "
        "rms divergence (crosses: the iteration did not converge); bottom, the share "
        "of the emitted H- current transmitted."
    )
    with plasma_plots.figure(
        2, 1, sharex=True, figsize=(7, 5.5), caption=caption
    ) as fig:
        divergence = np.array([run["divergence"] for run in runs]) * 1e3
        fig[0].plot(voltages, divergence, color="tab:blue")
        fig[0].scatter(
            voltages[ok], divergence[ok], color="tab:blue", label="converged"
        )
        fig[0].scatter(
            voltages[~ok],
            divergence[~ok],
            color="tab:red",
            marker="x",
            s=60,
            label="not converged",
        )
        fig[0].set_ylabel("rms divergence [mrad]")
        fig[0].legend(fontsize=8)
        transmitted = 100 * np.array([run["transmitted"] for run in runs])
        fig[1].plot(voltages, transmitted, marker="o")
        fig[1].set_ylabel("H- transmitted [%]")
        fig[1].set_xlabel("extraction voltage [kV]")

    caption = (
        "Convergence of the Vlasov iteration: the largest change of the potential between "
        "iterations, for each extraction voltage."
    )
    with plasma_plots.figure(figsize=(7, 3.6), caption=caption) as fig:
        ax = fig[0]
        for run in runs:
            history = run["history"]
            ax.plot(
                np.arange(1, history.size + 1),
                history,
                marker="o",
                ms=3,
                label=f"{run['extraction_voltage'] / 1e3:.0f} kV",
            )
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("largest change of the potential [V]")
        ax.legend(fontsize=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    args = parser.parse_args()
    output = simulate(args.preset)
    print(table(output))
    plot(output)
    plt.show()
