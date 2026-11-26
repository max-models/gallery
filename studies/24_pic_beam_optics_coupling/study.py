r"""PIC–beam-optics coupling.

A particle-in-cell run of the plasma and the meniscus is coupled to steady beam optics
downstream, as ONIX and IBSimu were (Lindqvist et al., Nucl. Fusion 62, 126068, 2022), and
checked against particle-in-cell over the whole domain.

The case is picnic's 2D proton extraction: a hydrogen plasma ($n_0 = 10^{15}\,\mathrm{m^{-3}}$,
$T_e = 2\,\mathrm{eV}$), a grounded knife-edge plasma grid (PG) at $8$–$11\,\mathrm{mm}$ and an
extraction grid (EG) at $-300\,\mathrm{V}$ at $18$–$21\,\mathrm{mm}$, on $0.25\,\mathrm{mm}$ cells.
picnic's `VlasovPoisson` covers the plasma up to its face $x_e = 18\,\mathrm{mm}$; beamtracer's
`BeamOptics` traces the beam with its space charge from a hand-off plane
$x_h = 14\,\mathrm{mm}$, past the plasma, to the end. Each exchange
(`picnic.coupling.BeamOpticsCoupling`) samples the protons crossing $x_h$ and the potential
there over a window, starts `BeamOptics` with them, and gives the PIC the potential
`BeamOptics` finds on $x_e$ as its new face; the PIC keeps running. The first face is the
vacuum one; the paper stopped after one exchange, here they go on until the face settles.

The first three figures show the coupled run stitched at $x_h$; the others compare it with
picnic over the whole $30\,\mathrm{mm}$, run with the same windows.

Compared: the face potential, and the current and divergence behind the EG. Not compared:
negative ions, co-extracted electrons, cylindrical or 3D geometry, and a grid system long
enough for the coupling to save time.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
from matplotlib.colors import LogNorm
from scipy import constants as c

TRACE_EVERY = 10  # steps between the recorded positions of picnic's protons
TRACE_FROM = 8.5e-3  # protons are followed downstream of this x [m]
SHOWN_PATHS = 80  # picnic proton paths drawn
PRESETS = {
    "smoke": {
        "cells": (61, 24),
        "markers": 4.0,
        "settle": 800,
        "sample": 400,
        "exchanges": 2,
    },
    "full": {
        "cells": (121, 48),
        "markers": 10.0,
        "settle": 3000,
        "sample": 1500,
        "exchanges": 4,
    },
}


def coupling_case(sizes: dict):
    """picnic's coupling case at the preset's resolution and schedule."""
    from picnic.examples.beam_optics_coupling_2d import CouplingCase
    from picnic.examples.plasma_meniscus_2d import MeniscusCase

    return CouplingCase(
        meniscus=MeniscusCase(
            num_gridpoints=sizes["cells"], particles_per_cell=sizes["markers"]
        ),
        settle_steps=sizes["settle"],
        sample_steps=sizes["sample"],
        exchanges=sizes["exchanges"],
    )


def run_coupled(case) -> dict:
    """The coupled exchanges, with the last window recorded for the stitched figures."""
    import dataclasses

    from picnic import Simulation
    from picnic.coupling import BeamOpticsCoupling
    from picnic.examples import beam_optics_coupling_2d as example
    from picnic.examples.plasma_meniscus_2d import species_density
    from picnic.models.vlasov_poisson import VlasovPoisson
    from picnic.parameters import OutputParams

    vacuum = example.vacuum_face(case)
    sim = Simulation(
        VlasovPoisson(example.coupled_config(case, vacuum)),
        t_end=1.0,
        output=OutputParams(every=10**9),
    )
    record = {"on": False, "paths": {}, "potential": [], "ions": [], "electrons": []}

    def snapshot(simulation):
        if not record["on"]:
            return
        model = simulation.model
        ions = model.species_registry.get("ions").markers
        ids = np.asarray(ions.active_marker_ids())
        x, y = np.asarray(ions.get_active_positions())[:2]
        for i, xi, yi in zip(
            ids[x > TRACE_FROM], x[x > TRACE_FROM], y[x > TRACE_FROM], strict=True
        ):
            record["paths"].setdefault(int(i), []).append((xi, yi))
        phi = model.fields.potential.get_field_data(gathered=True)
        record["potential"].append(np.asarray(phi)[..., 0].copy())
        record["ions"].append(species_density(model, "ions").copy())
        record["electrons"].append(species_density(model, "electrons").copy())

    sim.add_callback(snapshot, every=TRACE_EVERY)
    coupling = BeamOpticsCoupling(
        sim,
        lambda beams, upstream: dataclasses.replace(
            example.beam_optics_config(case, beams, upstream), save_trajectories=3
        ),
        case.handoff,
        species=["ions"],
    )
    windows = []
    for k in range(case.exchanges):
        record["on"] = k == case.exchanges - 1
        step = coupling.exchange(case.settle_steps, case.sample_steps)
        hits = step.beam_optics.plane(case.exit)
        forward = hits.direction > 0
        windows.append(
            {
                "handoff_current": step.sample.species[0].total_current,
                "exit_current": float(np.sum(hits.current[forward])),
                "exit_angle": example.rms_angle(
                    hits.velocity[:, forward], hits.current[forward]
                ),
                "face": step.returned,
                "change": step.change,
            }
        )
    optics = coupling.steps[-1].beam_optics
    grid = optics.config.grid
    # the paths that start near the meniscus and reach the hand-off plane, cut there
    paths = [
        p[p[:, 0] <= case.handoff]
        for p in map(np.array, record["paths"].values())
        if p[:, 0].min() < 10.5e-3 and p[:, 0].max() >= case.handoff
    ]
    shown = np.random.default_rng(1).choice(
        len(paths), size=min(SHOWN_PATHS, len(paths)), replace=False
    )

    def small(array) -> np.ndarray:
        return np.asarray(array, dtype=np.float32)

    return {
        "vacuum": vacuum,
        "windows": windows,
        "potential": small(np.mean(record["potential"], axis=0)),
        "ions": small(np.mean(record["ions"], axis=0)),
        "electrons": small(np.mean(record["electrons"], axis=0)),
        "num_paths": len(paths),
        "paths": [small(paths[i]) for i in shown],
        "optics_x": np.linspace(
            grid.lower_bound[0], grid.upper_bound[0], grid.num_gridpoints[0]
        ),
        "optics_y": np.linspace(
            grid.lower_bound[1], grid.upper_bound[1], grid.num_gridpoints[1]
        ),
        "optics_potential": small(optics.potential),
        "optics_density": small(np.asarray(optics.state.charge_density) / c.e),
        "trajectories": [small(t) for t in optics.trajectories],
    }


def simulate(preset: str = "full") -> dict:
    """Run the reference and the coupled run; return their results as plain arrays."""
    from picnic.examples import beam_optics_coupling_2d as example
    from picnic.examples.plasma_meniscus_2d import meniscus_config

    case = coupling_case(PRESETS[preset])
    reference = example.run_reference(case)
    coupled = run_coupled(case)
    config = meniscus_config(case.meniscus)
    return {
        "spacing": case.spacing,
        "edge": case.edge,
        "handoff": case.handoff,
        "length": (case.meniscus.length_x, case.meniscus.length_y),
        "face_points": case.face_points,
        "electrodes": {name: config.objects[name].vertices for name in ("pg", "eg")},
        "reference": [
            {k: w[k] for k in ("handoff_current", "exit_current", "exit_angle", "face")}
            for w in reference
        ],
        **coupled,
    }


def _domain(result: dict, ax, labels: bool = True, planes: str = "k") -> None:
    """The electrodes, the two planes and the names of the two models."""
    mm = 1e3
    length_x, length_y = result["length"]
    for vertices in result["electrodes"].values():
        xs = [v[0] * mm for v in vertices]
        for side in (1, -1):
            ys = [(length_y / 2 + side * v[1]) * mm for v in vertices]
            ax.fill(xs, ys, color="0.55", zorder=5)
    for x, name, style in (
        (result["handoff"], "x_h", ":"),
        (result["edge"], "x_e", "-."),
    ):
        ax.axvline(x * mm, color=planes, ls=style, lw=1, zorder=6)
        ax.text(x * mm, length_y * mm + 0.15, f"${name}$", ha="center", va="bottom")
    if labels:
        ax.text(0.3, 0.4, "picnic", fontsize=8, zorder=7)
        ax.text(
            result["handoff"] * mm + 0.3,
            0.4,
            "BeamOptics",
            fontsize=8,
            color="w",
            zorder=7,
        )
    ax.set_xlim(0, length_x * mm)
    ax.set_ylim(0, length_y * mm)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")


def _stitched(result: dict, field: str, optics_field: str):
    """picnic's field up to the hand-off plane and BeamOptics's beyond, on mm axes."""
    mm = 1e3
    data = result[field]
    data = np.concatenate(
        [data, data[:, :1]], axis=1
    )  # periodic y: repeat y = 0 at the top
    nx, ny = data.shape
    x = np.arange(nx) * result["spacing"]
    y = np.arange(ny) * result["spacing"]
    upstream = x <= result["handoff"] + 1e-12
    downstream = result["optics_x"] >= result["handoff"] - 1e-12
    return (
        (x[upstream] * mm, y * mm, data[upstream].T),
        (
            result["optics_x"][downstream] * mm,
            result["optics_y"] * mm,
            result[optics_field][downstream].T,
        ),
    )


def plot(result: dict) -> None:
    """The stitched density, trajectories and potential, then the comparison."""
    mm = 1e3
    with plasma_plots.figure(
        figsize=(8, 3.6),
        caption="Proton density after the last exchange: picnic, averaged over the window, up "
        "to $x_h$ and BeamOptics beyond; the cyan line, $n_e = 10^{14}\\,\\mathrm{m^{-3}}$, "
        "is the plasma edge with the meniscus in the PG aperture.",
    ) as fig:
        ax = fig[0]
        norm = LogNorm(1e12, 3e15)
        (xp, yp, ni), (xb, yb, nb) = _stitched(result, "ions", "optics_density")
        image = ax.pcolormesh(
            xp, yp, np.clip(ni, 1e10, None), norm=norm, cmap="magma", shading="gouraud"
        )
        ax.pcolormesh(
            xb, yb, np.clip(nb, 1e10, None), norm=norm, cmap="magma", shading="gouraud"
        )
        (xe, ye, ne), _ = _stitched(result, "electrons", "optics_density")
        ax.contour(xe, ye, ne, levels=[1e14], colors="c", linewidths=0.8)
        plt.colorbar(image, ax=ax, label=r"$n_{H^+}$ [m$^{-3}$]", shrink=0.9)
        _domain(result, ax, planes="w")

    with plasma_plots.figure(
        figsize=(8, 3.6),
        caption=f"Proton trajectories: {len(result['paths'])} of the "
        f"{result['num_paths']} picnic markers that went from the meniscus to $x_h$ in the "
        "last window (orange, followed by their id), then BeamOptics's trajectories from the "
        "crossings handed over at $x_h$ (red, every third).",
    ) as fig:
        ax = fig[0]
        for p in result["paths"]:
            ax.plot(p[:, 0] * mm, p[:, 1] * mm, color="C1", lw=0.5, alpha=0.8)
        for t in result["trajectories"]:
            ax.plot(t[:, 0] * mm, t[:, 1] * mm, color="C3", lw=0.4, alpha=0.6)
        _domain(result, ax, labels=False)

    with plasma_plots.figure(
        figsize=(8, 3.6),
        caption="Potential of the coupled run: picnic, averaged over the sampling window, up "
        "to $x_h$, BeamOptics beyond. The contours run on across $x_h$; the beam's space "
        "charge bends them out through the EG aperture.",
    ) as fig:
        ax = fig[0]
        levels = np.linspace(-300, 10, 63)
        (xp, yp, pp), (xb, yb, pb) = _stitched(result, "potential", "optics_potential")
        image = ax.contourf(xp, yp, pp, levels=levels, cmap="viridis")
        ax.contourf(xb, yb, pb, levels=levels, cmap="viridis")
        ax.contour(xp, yp, pp, levels=levels[::3], colors="w", linewidths=0.4)
        ax.contour(xb, yb, pb, levels=levels[::3], colors="w", linewidths=0.4)
        plt.colorbar(image, ax=ax, label="potential [V]", shrink=0.9)
        _domain(result, ax)

    y = result["face_points"] * mm
    with plasma_plots.figure(
        figsize=(6, 4),
        caption="Potential on picnic's downstream face $x_e$: the vacuum start, what "
        "BeamOptics returns at each exchange, and picnic over the whole domain in the last "
        "window. The beam's positive space charge lifts it by 7 V on the axis.",
    ) as fig:
        ax = fig[0]
        ax.plot(y, result["vacuum"], color="0.5", label="vacuum")
        for k, window in enumerate(result["windows"]):
            ax.plot(y, window["face"], color=f"C{k}", label=f"exchange {k + 1}")
        ax.plot(y, result["reference"][-1]["face"], "k--", label="picnic, whole domain")
        ax.set_xlabel("y [mm]")
        ax.set_ylabel(r"$\phi(x_e, y)$ [V]")
        ax.legend(fontsize=8)

    exchanges = np.arange(1, len(result["windows"]) + 1)
    with plasma_plots.figure(
        figsize=(6, 4),
        caption="The largest change of the face potential from one exchange to the next: one "
        "exchange does most of it, as in the paper; after the second the changes are the noise "
        "of the sampling window.",
    ) as fig:
        ax = fig[0]
        ax.semilogy(exchanges, [w["change"] for w in result["windows"]], "o-")
        ax.set_xlabel("exchange")
        ax.set_ylabel("largest change [V]")

    for key, label, scale, caption in (
        (
            "exit_current",
            "proton current [mA/m]",
            1e3,
            "The proton current behind the EG",
        ),
        (
            "exit_angle",
            "rms divergence [mrad]",
            1e3,
            "The rms divergence behind the EG",
        ),
    ):
        with plasma_plots.figure(
            figsize=(6, 4),
            caption=f"{caption} in each window, coupled and picnic over the whole domain; "
            "they differ by about as much as the windows scatter.",
        ) as fig:
            ax = fig[0]
            ax.plot(
                exchanges,
                [scale * w[key] for w in result["windows"]],
                "o-",
                label="coupled",
            )
            ax.plot(
                exchanges,
                [scale * w[key] for w in result["reference"]],
                "s--",
                label="picnic, whole domain",
            )
            ax.set_xlabel("exchange")
            ax.set_ylabel(label)
            ax.legend(fontsize=8)


def table(result: dict) -> str:
    """Each exchange: the face change and agreement, and the beam against the reference."""
    middle = len(result["face_points"]) // 2
    inside = np.abs(result["face_points"] - result["length"][1] / 2) < 5e-3
    lines = [
        (
            "| Exchange | Face change [V] | Face on axis [V] | Face vs whole domain [V] "
            "| Current [mA/m] coupled / whole | Divergence [mrad] coupled / whole |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for k, (w, r) in enumerate(
        zip(result["windows"], result["reference"], strict=True)
    ):
        difference = np.max(np.abs(w["face"] - r["face"])[inside])
        lines.append(
            f"| {k + 1} | {w['change']:.2f} | {w['face'][middle]:.1f} | {difference:.2f} | "
            f"{1e3 * w['exit_current']:.2f} / {1e3 * r['exit_current']:.2f} | "
            f"{1e3 * w['exit_angle']:.0f} / {1e3 * r['exit_angle']:.0f} |"
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
