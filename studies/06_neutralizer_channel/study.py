r"""Neutralizer channel.

A 1 MeV, 10 A deuterium negative-ion beam crosses one channel of a gas neutralizer, 3 m
long and 10 cm wide, the size of a channel of ITER's injectors. On the D$_2$ gas, and on
the weak plasma it makes there itself, the beam changes charge state; the plasma is a
wellbalanced global model of the channel, fed by the beam's stripped and secondary
electrons. Alone, that model treats the beam as a line integral along the axis. Here
beamtracer traces it instead (`beamtracer.neutralizer.NeutralizerChannel`):
`BeamOptics` trajectories through the channel, each carrying its charge-state fractions,
and the plasma takes its sources from the traced paths; the two are iterated to a fixed
point. The study scans the beam's divergence at the entrance.

With a parallel beam the traced result is the axis model's. As the beam diverges, more of
it hits the walls, about a fifth at 10 mrad, and the neutral fraction falls from 0.57 to
about 0.47. The neutral share of the beam that leaves stays at 0.57: the loss is all
geometry. The neutrals that leave diverge less than the beam entered.

The same channel with magnetic cusps on its walls is the beam-driven plasma neutralizer,
a DEMO concept, not ITER's design: the cusps confine the beam's plasma so that it strips
too. The table compares the two in the axis model: the cusps raise the plasma density
about thirteenfold, but at below 0.2 % ionization the neutral fraction does not move.

The beam's cross sections are illustrative placeholders, not data (the plasma rates are
OpenCR's).

Compared: the traced channel with the global model alone; the gas neutralizer with the
cusp-confined one.

Reference: beamtracer's example `examples/neutralizer_channel_2d.py`.
"""

import argparse
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
from beamtracer.neutralizer import NeutralizerChannel
from wellbalanced.cr import deuterium_neutralizer

#: divergences: rms divergences at the entrance [rad]; channel: options of the traced
#: channel (particles, mesh cell [m], time step per cell, fixed-point tolerance and
#: iterations). The full preset takes a step of a whole cell, twice the default, and a
#: tolerance of 1e-3: at 10 mrad the result agrees with the defaults to 1e-6.
PRESETS = {
    "smoke": {
        "divergences": [0.0, 0.01],
        "channel": {"num_particles": 10, "cell": 0.01, "max_iterations": 1},
    },
    "full": {
        "divergences": [0.0, 0.003, 0.006, 0.01, 0.015],
        "channel": {"num_particles": 200, "step_fraction": 1.0, "tolerance": 1e-3},
    },
}

BEAM_WIDTH = 0.08  # m: the beam at the entrance, in the 10 cm channel
GAS_INFLOW = 6.0e21  # molecules/s: near the gas's best neutral fraction
LENGTH, HALF_WIDTH = 3.0, 0.05  # m: the channel
SHOWN = 0.01  # rad: the divergence whose trajectories are drawn
STATES = ("-", "0", "+")


def neutralizer(cusp=None):
    """The channel with OpenCR's plasma rates (the beam data are placeholders, which warn).

    Without ``cusp`` it is ITER's gas neutralizer: nothing confines the beam's plasma.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return deuterium_neutralizer(gas_inflow=GAS_INFLOW, cusp=cusp)


def default_cusp():
    """The magnetic cusps of wellbalanced's default channel (the DEMO concept)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return deuterium_neutralizer().cusp


def compare_concepts() -> list[dict]:
    """The axis model of the gas neutralizer and of the cusp-confined channel."""
    rows = []
    concepts = (
        ("gas neutralizer (ITER)", None),
        ("cusp-confined (DEMO concept)", default_cusp()),
    )
    for name, cusp in concepts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = neutralizer(cusp).solve()
        gas = float(np.mean(result.gas_density))
        ne = float(result.plasma.electron_density)
        rows.append(
            {
                "concept": name,
                "electron_density": ne,
                "gas_density": gas,
                "ionization": ne / (ne + gas),
                "neutral": float(result.neutral_fraction),
            }
        )
    return rows


def simulate(preset: str = "full") -> dict:
    """Solve the traced channel at every divergence of the preset."""
    p = PRESETS[preset]
    start = time.time()
    runs = []
    for divergence in p["divergences"]:
        result = NeutralizerChannel(
            neutralizer(),
            beam_width=BEAM_WIDTH,
            divergence=divergence,
            **p["channel"],
        ).solve()
        runs.append(
            {
                "divergence": divergence,
                "neutral": result.neutral_fraction,
                "axis_neutral": result.axis.neutral_fraction,
                "transmission": result.transmission,
                "exit": dict(result.exit_fractions),
                "neutral_divergence": result.neutral_divergence,
                "electron_density": result.plasma.electron_density,
                "axis_electron_density": result.axis.plasma.electron_density,
                # every fifth point is enough to draw a straight-ish path
                "trajectories": [np.asarray(t)[::5] for t in result.model.trajectories],
                "axis_position": np.asarray(result.axis.position),
                "axis_fractions": np.asarray(result.axis.fractions),
                "converged": result.converged,
                "iterations": result.iterations,
            }
        )
    return {
        "preset": preset,
        "runs": runs,
        "concepts": compare_concepts(),
        "runtime": time.time() - start,
    }


def table(result: dict) -> str:
    """The traced channel against the axis model at each divergence."""
    lines = [
        "| divergence (mrad) | transmission | neutral fraction, traced | along the axis "
        "| neutrals in the transmitted beam | neutral divergence (mrad) | n_e (m^-3) "
        "| iterations |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in result["runs"]:
        lines.append(
            f"| {1e3 * run['divergence']:.0f} | {run['transmission']:.3f} | "
            f"{run['neutral']:.3f} | {run['axis_neutral']:.3f} | "
            f"{run['neutral'] / run['transmission']:.3f} | "
            f"{1e3 * run['neutral_divergence']:.1f} | {run['electron_density']:.2e} | "
            f"{run['iterations']}{'' if run['converged'] else ' (not converged)'} |"
        )
    lines += [
        "",
        "| channel (axis model) | n_e (m^-3) | gas (m^-3) | ionization | neutral fraction |",
        "|---|---|---|---|---|",
    ]
    for row in result.get("concepts", []):
        lines.append(
            f"| {row['concept']} | {row['electron_density']:.2e} | "
            f"{row['gas_density']:.2e} | {100 * row['ionization']:.2f} % | "
            f"{row['neutral']:.4f} |"
        )
    return "\n".join(lines)


def plot(result: dict) -> None:
    """The trajectories, the scan over the divergence and the charge states."""
    runs = result["runs"]
    shown = min(runs, key=lambda run: abs(run["divergence"] - SHOWN))

    # 1. the trajectories
    with plasma_plots.figure(
        figsize=(7, 3.5),
        caption=f"The traced beam through the channel at {1e3 * shown['divergence']:.0f} "
        "mrad rms divergence: every tenth trajectory (blue) between the channel's walls (black); "
        "the ones that reach a wall end there.",
    ) as fig:
        ax = fig[0]
        for y in (-HALF_WIDTH, HALF_WIDTH):
            ax.plot([0.0, LENGTH], [1e3 * y] * 2, color="black", lw=3)
        for path in shown["trajectories"]:
            ax.plot(path[:, 0], 1e3 * path[:, 1], color="#2a78d6", lw=0.5)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [mm]")

    # 2. the scan over the divergence
    divergence = [1e3 * run["divergence"] for run in runs]
    with plasma_plots.figure(
        2,
        1,
        sharex=True,
        figsize=(7, 7),
        caption="Against the beam's divergence at the entrance: top, the share of the "
        "incoming particles transmitted, neutral at the exit (traced and along the axis) "
        "and the neutral share of the transmitted beam; bottom, the rms divergence of the "
        "neutrals at the exit.",
    ) as fig:
        ax = fig[0]
        curves = (
            ("transmission (all states)", "transmission", ":"),
            ("neutral fraction, traced", "neutral", "-"),
            ("neutral fraction, along the axis", "axis_neutral", "--"),
        )
        for label, key, style in curves:
            ax.plot(divergence, [run[key] for run in runs], "o", ls=style, label=label)
        ax.plot(
            divergence,
            [run["neutral"] / run["transmission"] for run in runs],
            "o-.",
            label="neutrals in the transmitted beam",
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("share of the incoming particles")
        ax.legend(fontsize=8)
        ax = fig[1]
        ax.plot(divergence, divergence, color="0.6", ls=":", label="entrance")
        ax.plot(
            divergence,
            [1e3 * run["neutral_divergence"] for run in runs],
            "o-",
            label="neutrals at the exit",
        )
        ax.set_xlabel("rms divergence at the entrance [mrad]")
        ax.set_ylabel("rms divergence [mrad]")
        ax.legend(fontsize=8)

    # 3. the charge states
    first = runs[0]
    with plasma_plots.figure(
        figsize=(7, 4),
        caption="Charge-state fractions along the channel in the axis model (lines), and "
        "at the exit of the traced beam for every divergence of the scan (markers, lower "
        "for wider beams).",
    ) as fig:
        ax = fig[0]
        for k, state in enumerate(STATES):
            line = ax.plot(
                first["axis_position"],
                first["axis_fractions"][:, k],
                label=f"D{state}, along the axis",
            )[0]
            ax.plot(
                [LENGTH] * len(runs),
                [run["exit"][state] for run in runs],
                "o",
                color=line.get_color(),
                label=f"D{state} at the exit, traced",
            )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("fraction of the incoming particles")
        ax.legend(fontsize=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    output = simulate(args.preset)
    print(table(output))
    plot(output)
    plt.show()
