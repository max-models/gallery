"""Beamline power balance.

A neutral-beam line from the grid to the port, built from the pieces of `beamtracer.beamline`:
5 by 16 aimed beamlets of 1 MeV deuterium (5 mrad each, focused at 7 m), the neutralizer's
charge states (60 % D$^0$, 20 % each of D$^-$ and D$^+$), the residual-ion dump (two plates
0.12 m apart from 0.5 to 2.3 m, 20 kV between them), the duct (simgeometry walls from 2.3 to
6 m round a 0.12 by 0.24 m bore, $10^{18}$ m$^{-3}$ of D$_2$ that reionizes neutrals through
`plasmacoll.ChargeStateTransitions`) and a vertical stray field rising from 5 to 50 mT towards
the port. The ions are pushed through the fields (`FieldRegion`), the neutrals followed on
straight lines (`Duct`), and `beamline_summary` adds up where the power goes.

Of the 16 MW, the dump takes 40 %, the beam that left the neutralizer charged; the duct walls
take 4.5 %, the tails of the neutrals; the gas reionizes 5 % of the neutrals, and the stray
field turns most of them (2.5 % of the beam) onto the walls in the last 3 m. 53 % reaches the
port, 23 % of that more than 5 cm from the beam's centre. The neutrals through the port are
then placed in a tokamak's frame (port at $R = 9$ m, tangency radius 5.3 m) as a dataset with
their weight in atoms per second, which freeofcharge's `NeutralBeam.from_dataset` reads.

The numbers are illustrative; the reionization cross section ($1.5 \\times 10^{-20}$ m$^2$) is
a placeholder, as plasmacoll holds no beam-stripping data.

Compared: the power balance, which `beamline_summary` checks to round-off (port and walls
against the beam plus the fields' work). Not compared: an experiment or another code.

Reference: beamtracer's tutorial `tutorials/beamline.ipynb`.
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
from beamtracer.beamline import (
    ApertureGrid,
    Duct,
    FieldRegion,
    Markers,
    beamline_summary,
    box_field,
    gaussian_beam,
    profile_field,
)
from plasmacoll import ChargeStateTransitions
from scipy import constants
from simgeometry import Cuboid

#: markers: markers per beamlet; bins: of the footprint at the port.
PRESETS = {"smoke": {"markers": 10, "bins": 20}, "full": {"markers": 1500, "bins": 50}}

MASS = 2.014 * constants.atomic_mass
ENERGY = 1e6  # eV
#: The dump: start, end and gap of its plates [m], its voltage [V].
DUMP = {"start": 0.5, "end": 2.3, "gap": 0.12, "voltage": 20e3}
#: The duct: the port [m], the half bore in y and z and the wall thickness [m].
PORT, HALF_Y, HALF_Z, WALL = 6.0, 0.06, 0.12, 0.05
#: The tokamak: the port at R = START, the beam aimed at R = TANGENCY [m].
TANGENCY, START = 5.3, 9.0
HALO = 0.05  # m


def dump_plates():
    """The dump's two plates across the narrow side of the group."""
    start, end, gap = DUMP["start"], DUMP["end"], DUMP["gap"]
    return [
        Cuboid(((start + end) / 2, s * (gap / 2 + 0.005), 0), (end - start, 0.01, 0.5))
        for s in (1, -1)
    ]


def duct_walls():
    """The duct's four walls round the bore."""
    middle, length = (DUMP["end"] + PORT) / 2, PORT - DUMP["end"]
    sides = [
        Cuboid(
            (middle, s * (HALF_Y + WALL / 2), 0), (length, WALL, 2 * (HALF_Z + WALL))
        )
        for s in (1, -1)
    ]
    tops = [
        Cuboid((middle, 0, s * (HALF_Z + WALL / 2)), (length, 2 * HALF_Y, WALL))
        for s in (1, -1)
    ]
    return sides + tops


def simulate(preset: str = "full") -> dict:
    """Trace the group through the dump, the duct and the stray field; the power balance."""
    p = PRESETS[preset]
    start_time = time.time()
    speed = np.sqrt(2 * ENERGY * constants.e / MASS)
    beamlet = gaussian_beam(
        p["markers"], power=2e5, speed=speed, width=2e-3, divergence=5e-3
    )
    grid = ApertureGrid.regular(16, 5, (0.020, 0.022), focal_length=(7.0, 7.0))
    beam = grid.markers(beamlet)
    # the neutralizer: 60 % neutrals, 20 % each of D- and D+
    beam.state = np.random.default_rng(0).choice(3, size=len(beam), p=[0.2, 0.6, 0.2])

    start, end, gap = DUMP["start"], DUMP["end"], DUMP["gap"]
    field = box_field(
        (0, DUMP["voltage"] / gap, 0), (start, -gap / 2, -0.25), (end, gap / 2, 0.25)
    )
    dump = FieldRegion(
        dump_plates(), exit=end, mass=MASS, electric=field, step=2e-3
    ).trace(beam)

    walls = duct_walls()
    transitions = ChargeStateTransitions({"D2": {("0", "+"): 1.5e-20}})
    duct = Duct(walls, port=PORT, transitions=transitions, gas={"D2": 1e18}).trace(
        dump.exit
    )
    stray = profile_field([end, PORT], [[0, 0], [0, 0], [0.005, 0.05]])
    bent = FieldRegion(walls, exit=PORT, mass=MASS, magnetic=stray).trace(
        duct.reionized
    )

    through = Markers.concatenate([duct.port, bent.exit])
    on_walls = {"dump": dump.wall, "duct": duct.wall, "duct, reionized": bent.wall}
    summary = beamline_summary(
        through,
        on_walls,
        power=beam.total_power,
        work=dump.work + bent.work,  # what the dump field gave the ions
        halo_radius=HALO,
        footprint_extent=(-0.1, 0.1, -0.15, 0.15),
        bins=p["bins"],
    )
    edges = np.linspace(0.0, PORT, 61)
    along = {name: m.along_x(edges) for name, m in on_walls.items()}

    at_port = through.drift_to(PORT)
    neutral = at_port.subset(at_port.state == at_port.states.index("0"))
    along_port = np.sqrt(START**2 - TANGENCY**2)
    neutrals = neutral.placed(
        (TANGENCY, -along_port, 0.0), (0.0, 1.0, 0.0), x=PORT
    ).to_dataset(MASS)
    return {
        "preset": preset,
        "markers": len(beam),
        "summary": summary,
        "along": along,
        "neutrals": neutrals,
        "reionized_fraction": duct.reionized_fraction,
        "runtime": time.time() - start_time,
    }


def table(result: dict) -> str:
    """Where the power goes, and the neutrals handed on."""
    s = result["summary"]
    total = float(s.power_in)
    rows = [("port", float(s.power_port))] + [
        (str(part), float(s.power_on.sel(part=part))) for part in s.part.values
    ]
    lines = ["| Part | power (MW) | share of the beam |", "|---|---|---|"]
    lines += [f"| {name} | {v / 1e6:.2f} | {v / total:.1%} |" for name, v in rows]
    lines.append(f"| beam (in) | {total / 1e6:.2f} | 100 % |")
    lines.append(
        f"| work of the dump and stray fields | {float(s.power_work) / 1e6:.3f} | - |"
    )
    n = result["neutrals"]
    lines += [
        "",
        f"Markers: {result['markers']}. Reionized in the duct: "
        f"{result['reionized_fraction']:.1%} of the neutrals entering it. "
        f"Beyond {100 * HALO:.0f} cm of the beam's centre at the port: "
        f"{float(s.power_halo) / float(s.power_port):.1%} of the port power. "
        f"Neutrals handed on: {n.sizes['marker']} markers, "
        f"{float(n.power.sum()) / 1e6:.2f} MW, "
        f"{float(n.weight.sum()):.3g} atoms/s.",
    ]
    return "\n".join(lines)


def plot(result: dict) -> None:
    """The power balance, the power on the walls along the beamline, the footprint."""
    s = result["summary"]
    parts = ["port"] + [str(p) for p in s.part.values]
    values = np.array([float(s.power_port)] + list(s.power_on.values)) / 1e6
    total = float(s.power_in) / 1e6
    caption = (
        "Where the power of the 1 MeV deuterium beam goes: through the port, onto the "
        "residual-ion dump, onto the duct walls (neutrals) and onto the duct walls after "
        "reionization in the gas, as a share of the beam."
    )
    with plasma_plots.figure(figsize=(7, 3.2), caption=caption) as fig:
        ax = fig[0]
        k = np.arange(len(parts))
        ax.plot(values, k, "o", ms=9)
        ax.hlines(k, 0, values, lw=6)
        for i, (part, v) in enumerate(zip(parts, values, strict=True)):
            ax.text(v, i, f"  {part}: {v:.1f} MW ({v / total:.0%})", va="center")
        ax.set_yticks([])  # the parts are named at their bars
        ax.set_ylim(len(parts) - 0.5, -0.5)
        ax.set_xlabel("power [MW]")
        ax.set_xlim(0, 1.7 * values.max())

    caption = (
        "Power per unit length on the walls along the beamline: the dump plates from 0.5 "
        "to 2.3 m, the duct walls from 2.3 m to the port at 6 m, hit by neutrals and by the "
        "reionized ones the stray field turns onto them."
    )
    with plasma_plots.figure(figsize=(7, 3.6), caption=caption) as fig:
        ax = fig[0]
        for name, line in result["along"].items():
            power = (line / 1e6).where(line > 0)  # gaps where a part takes nothing
            ax.plot(line.x, power, label=name, drawstyle="steps-mid")
        ax.set_yscale("log")
        ax.set_xlabel("distance from the grid [m]")
        ax.set_ylabel("power per length [MW/m]")
        ax.legend()

    footprint = (s.footprint / 1e6).assign_attrs(
        units="MW/m²", long_name="power density"
    )
    caption = (
        "Power density at the port over the cross-section (y across the narrow side, z "
        "up), clipped at y = ±0.06 m by the duct; the dashed circle has a radius of 5 cm "
        "round the beam's centre."
    )
    with plasma_plots.figure(figsize=(6, 6), caption=caption) as fig:
        footprint.plasma.plot.slice(
            x="y", y="z", equal_aspect=True, colorbar_label="MW/m²", ax=fig[0]
        )
        angle = np.linspace(0, 2 * np.pi, 200)
        fig[0].plot(HALO * np.cos(angle), HALO * np.sin(angle), color="white", ls="--")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    output = simulate(args.preset)
    print(table(output))
    plot(output)
    plt.show()
