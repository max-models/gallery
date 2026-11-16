"""Fast-ion orbits.

Neutral-beam ions followed as guiding centres with freeofcharge's `follow_orbits`: RK4 on the
cubic-spline field of a `GriddedEquilibrium`, with Monte Carlo Coulomb collisions (plasmacoll's
drags and Coulomb logarithms) until each ion thermalises or crosses a simgeometry wall. In
Solov'ev tokamaks: five 55 keV protons traced without collisions, the errors of the energy and of
$p_\\varphi$ against the step, the width of passing orbits against the exact value from
$p_\\varphi$ and the estimate $2q\\rho_\\parallel$, and, at 10 T where the orbits are narrow, the
fast-ion moments of the orbits against the local slowing down (`slow_down`).

Then W7-X, on GVEC's W7-X example at reduced resolution (slowcooker's, sampled on a
$64 \\times 32 \\times 64$ grid per field period): the two stand-in NI21 sources deposit hydrogen
at 55 keV ($n_{e0} = 4\\times10^{19}$ m$^{-3}$, $T_0 = 2$ keV), and about 300 births are followed
without collisions for 1 ms, then to thermalisation or a wall 10 cm off GVEC's boundary.

**Finding.** The energy error falls as the fifth power of the step; the passing-orbit width
equals the $p_\\varphi$ value, and $2q\\rho_\\parallel$ fails towards the trapped-passing boundary.
At 10 T the orbits' heating and fast ions agree with the local model to about 1 %. In W7-X no
birth is lost promptly; with collisions 3 % of the markers (1 % of the power) reach the wall
after about 15 ms, too few to resolve the wall-load pattern.

Compared: the invariants and the orbit width with their exact values; the moments with
`slow_down`. Not compared: ASCOT5 or a measurement; the wall and injectors are stand-ins.

Reference: freeofcharge's tutorial "Fast-ion orbits" (there on DESC's W7-X).
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

#: Sizes: tokamak traces, orbit-width runs, markers of the collisional checks, W7-X grid.
PRESETS = {
    "smoke": {
        "trace_time": 2e-5,
        "samples": 101,
        "steps": [0.1],
        "width_pitches": [0.9],
        "width_time": 5e-5,
        "tokamak_markers": 40,
        "grid": (32, 16, 32),
        "w7x_markers": 5,
        "max_time": 2e-3,
    },
    "quick": {
        "trace_time": 2e-4,
        "samples": 1001,
        "steps": [0.2, 0.1, 0.05],
        "width_pitches": [0.95, 0.6, -0.9],
        "width_time": 1.5e-4,
        "tokamak_markers": 400,
        "grid": (64, 32, 64),
        "w7x_markers": 50,
        "max_time": 1.0,
    },
}

ENERGY = 55e3  # [eV]


def gvec_wall(gap: float):
    """GVEC's W7-X boundary as a simgeometry surface, and its offset by ``gap`` [m]."""
    from gvec.util.params import read_parameters
    from simgeometry.objects.fourier_surface import FourierSurface
    from slowcooker.data import equilibrium_files

    params = read_parameters(str(equilibrium_files("w7x")[0]))
    big_m, big_n = params["X1_mn_max"]
    r_cos = np.zeros((big_m + 1, 2 * big_n + 1))
    z_sin = np.zeros_like(r_cos)
    # GVEC's zeta runs against the cylindrical angle (x = R cos zeta, -R sin zeta): n -> -n
    for (m, n), value in params["X1_b_cos"].items():
        r_cos[m, big_n - n] += value
    for (m, n), value in params["X2_b_sin"].items():
        z_sin[m, big_n - n] += value
    boundary = FourierSurface(r_cos, z_sin, int(params["nfp"]))
    return boundary, boundary.offset(gap, resolution=(128, 64))


def tokamak_orbits(preset: dict) -> dict:
    """Collisionless orbits of five protons, the invariants and the passing-orbit widths."""
    import freeofcharge as foc
    from freeofcharge import orbits
    from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE

    tokamak = foc.Equilibrium.solovev(3.0, 0.6, 1.5, 3.0, 1.2, wall_psi_n=1.3)
    field = orbits.orbit_field(tokamak, (128, 8, 128), box=((2.0, 3.9), (-1.25, 1.25)))
    pitch = np.array([-0.9, -0.3, 0.1, 0.5, 0.95])
    start = np.tile([3.3, 0.0, 0.0], (5, 1))
    b = np.asarray(field.field_at(start))
    unit = b / np.linalg.norm(b, axis=1)[:, None]
    across = np.cross(unit, [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across, axis=1)[:, None]
    speed = np.sqrt(2 * ENERGY * ELEMENTARY_CHARGE / ATOMIC_MASS)
    velocity = speed * (pitch[:, None] * unit + np.sqrt(1 - pitch**2)[:, None] * across)
    markers = orbits.guiding_centres(start, velocity, np.ones(5), field, ATOMIC_MASS)
    traces = orbits.trace_orbits(
        markers, field, preset["trace_time"], samples=preset["samples"], step=0.1
    )

    psi_span = ELEMENTARY_CHARGE * (tokamak.psi_boundary - tokamak.psi_axis)
    invariants = []
    for step in preset["steps"]:
        run = orbits.trace_orbits(markers, field, 2e-3, samples=201, step=step)
        energy = run.energy.values
        r, z = run.R.values, run.z.values
        bb = np.asarray(field.field_at(np.stack([run.x.values, run.y.values, z], -1)))
        b_phi = (-run.y.values * bb[..., 0] + run.x.values * bb[..., 1]) / r
        p_phi = ATOMIC_MASS * r * run.v_parallel.values * b_phi / run.B.values
        p_phi += ELEMENTARY_CHARGE * tokamak.psi(r, z)
        invariants.append(
            {
                "step": step,
                "energy": float(np.max(np.abs(energy / energy[0] - 1))),
                "p_phi": float(np.max(np.abs(p_phi - p_phi[0])) / psi_span),
            }
        )

    # passing-orbit width on a large-aspect-ratio tokamak
    r0, b0, q0 = 10.0, 2.0, 1.5
    large = foc.Equilibrium.solovev(r0, 1.0, 1.0, b0, q0, wall_psi_n=None, points=257)
    large_field = orbits.orbit_field(large, (256, 4, 256))
    shells = large.flux_surfaces(np.linspace(0, 1, 41))
    widths = []
    samples = int(round(preset["width_time"] / 1e-7)) + 1
    for xi in preset["width_pitches"]:
        p0 = np.array([[r0 + 0.5, 0.0, 0.0]])
        u_b = np.asarray(large_field.field_at(p0))[0]
        u_b /= np.linalg.norm(u_b)
        e1 = np.cross(u_b, [0, 0, 1.0])
        e1 /= np.linalg.norm(e1)
        v = speed * (xi * u_b + np.sqrt(1 - xi**2) * e1)
        gc = orbits.guiding_centres(p0, v[None], np.ones(1), large_field, ATOMIC_MASS)
        run = orbits.trace_orbits(
            gc, large_field, preset["width_time"], samples=samples, step=0.02
        )
        r, z, vp = (run[k].values[:, 0] for k in ("R", "z", "v_parallel"))
        k = np.nonzero(np.sign(z[1:]) != np.sign(z[:-1]))[0]
        t = z[k] / (z[k] - z[k + 1])
        rc, vc = r[k] + t * (r[k + 1] - r[k]), vp[k] + t * (vp[k + 1] - vp[k])
        uc = (rc**2 - r0**2) / (2 * r0)
        bc = np.linalg.norm(
            np.asarray(large_field.field_at(np.stack([rc, 0 * rc, 0 * rc], -1))),
            axis=-1,
        )
        out, inn = uc > 0, uc < 0
        width = uc[out].mean() + uc[inn].mean()
        r_mid = 0.5 * (uc[out].mean() - uc[inn].mean())
        exact = (
            (ATOMIC_MASS * b0 * r0 / ELEMENTARY_CHARGE)
            * (vc[inn].mean() / bc[inn].mean() - vc[out].mean() / bc[out].mean())
            / (2 * b0 / (2 * r0**2 * q0) * r0**2 * r_mid)
        )
        q = np.interp(
            float(large.rho(r0 + r_mid, 0.0)), shells.rho, shells.safety_factor
        )
        estimate = 2 * q * ATOMIC_MASS * np.abs(vc).mean() / (ELEMENTARY_CHARGE * b0)
        widths.append(
            {"pitch": xi, "width": width, "exact": exact, "estimate": estimate}
        )

    return {
        "boundary": np.asarray(tokamak.boundary.vertices),
        "wall": np.asarray(tokamak.wall.vertices),
        "pitch": pitch,
        "time": preset["trace_time"],
        "R": traces.R.values,
        "z": traces.z.values,
        "invariants": invariants,
        "widths": widths,
    }


def tokamak_moments(preset: dict) -> dict:
    """Orbit following with collisions against the local slowing down at 10 T."""
    import freeofcharge as foc
    from freeofcharge import orbits

    high = foc.Equilibrium.solovev(3.0, 0.6, 1.5, 10.0, 1.2, wall_psi_n=None)
    rho_edges = np.linspace(0, 1, 11)
    shells = high.flux_surfaces(rho_edges)
    field = orbits.orbit_field(high, (128, 4, 128))
    origin, direction = foc.tangential_injection(2.9, 6.0)
    beam = foc.NeutralBeam.gaussian(
        origin,
        direction,
        ENERGY,
        1e6,
        mass_number=1,
        width=0.1,
        divergence=0.005,
        markers=preset["tokamak_markers"],
        seed=1,
    )
    plasma = foc.Plasma.parabolic(1e20, 1.5e3, ions={"H": (1, 1, 1.0)})
    births = foc.deposit(
        beam, high, plasma, step=0.01, births_per_marker=1, rho_edges=rho_edges, seed=2
    )
    local = foc.slow_down(
        births, plasma, shells, orbit_average=False, thermal_factor=2.0
    )
    followed = orbits.follow_orbits(
        births,
        field,
        plasma,
        surfaces=shells,
        step=0.2,
        thermal_factor=2.0,
        energy_diffusion=False,
        seed=3,
    )
    p = local.birth_power
    return {
        "rho": followed.to_dataset().rho.values,
        "density_orbits": followed.to_dataset().fast_ion_density.values,
        "density_local": local.to_dataset(shells).fast_ion_density.values,
        "fractions": {
            "electron heating": (
                followed.electron_heating.sum() / p,
                local.electron_heating.sum() / p,
            ),
            "ion heating": (
                followed.ion_heating.sum() / p,
                local.ion_heating.sum() / p,
            ),
            "thermalised": (
                followed.thermalised_power / p,
                local.thermalised_power / p,
            ),
        },
        "ratios": {
            "fast ions": followed.particles.sum()
            / np.sum(local.fast_ion_density * shells.volume),
            "fast-ion energy": followed.stored_energy / local.stored_energy,
        },
        "balance": float(followed.power_balance),
    }


def w7x_orbits(preset: dict) -> dict:
    """W7-X births followed without and with collisions to the wall."""
    import freeofcharge as foc
    from freeofcharge import orbits
    from freeofcharge.injectors import w7x_injectors
    from slowcooker.data import gvec_equilibrium

    w7x = foc.GriddedEquilibrium(gvec_equilibrium("w7x"), preset["grid"])
    _, wall = gvec_wall(0.10)
    plasma = foc.Plasma.parabolic(
        4e19, 2e3, ions={"H": (1, 1, 1.0)}, radial_coordinate="toroidal"
    )
    rho_edges = np.linspace(0, 1, 11)
    deposition = foc.deposit_sources(
        w7x_injectors(sources={"NI21": [1, 2]}),
        w7x,
        plasma,
        markers=preset["w7x_markers"],
        seed=1,
        step=0.01,
        rho_edges=rho_edges,
    ).total()

    prompt = orbits.follow_orbits(
        deposition, w7x, None, wall=wall, step=0.1, max_time=1e-3, interval=5e-5
    )
    error = np.abs(prompt.markers.end_energy / prompt.markers.birth_energy - 1).values

    surfaces = w7x.flux_surfaces(rho_edges)
    slowing = orbits.follow_orbits(
        deposition,
        w7x,
        plasma,
        surfaces=surfaces,
        wall=wall,
        step=0.1,
        max_time=preset["max_time"],
        seed=1,
    )
    local = foc.slow_down(
        deposition,
        plasma,
        surfaces,
        orbit_average=False,
        thermal_factor=2.0,
        trapped_electron_correction=lambda surfaces, plasma: np.zeros(
            len(rho_edges) - 1
        ),
    )
    orbit_set = slowing.to_dataset()
    local_set = local.to_dataset(surfaces)
    loads = slowing.wall_loads(theta_bins=16, phi_bins=60)
    lost = slowing.losses()
    load_map = loads.wall_load_map.fillna(0.0)
    return {
        "births": int(deposition.births.weights.shape[0]),
        "absorbed": float(deposition.absorbed_power),
        "pitch_range": (
            float(np.min(deposition.birth_pitch)),
            float(np.max(deposition.birth_pitch)),
        ),
        "prompt_fates": {k: float(v) for k, v in prompt.fate_fractions().items()},
        "prompt_error": (float(np.median(error)), float(error.max())),
        "fates": {k: float(v) for k, v in slowing.fate_fractions().items()},
        "powers": {k: float(v) for k, v in slowing.power_fractions().items()},
        "balance": float(slowing.power_balance),
        "longest": float(slowing.markers.time.max()),
        "rho": orbit_set.rho.values,
        "heating_orbits": orbit_set.heating_density.sum("species").values,
        "heating_local": local_set.heating_density.sum("species").values,
        "density_orbits": orbit_set.fast_ion_density.values,
        "density_local": local_set.fast_ion_density.values,
        "lost_power": float(loads.attrs["lost_power"]),
        "peak_load": float(loads.attrs["peak_wall_load"]),
        "lost_markers": int(lost.sizes["marker"]),
        "lost_energy": float(lost.end_energy.mean()) if lost.sizes["marker"] else 0.0,
        "lost_time": float(np.median(lost.time)) if lost.sizes["marker"] else 0.0,
        "map_theta": load_map.theta.values,
        "map_phi": load_map.phi.values,
        "map": load_map.transpose("theta", "phi").values,
    }


def simulate(preset: str = "quick") -> dict:
    """Run the tokamak checks and the W7-X case."""
    settings = PRESETS[preset]
    start = time.perf_counter()
    result = {
        "preset": preset,
        "tokamak": tokamak_orbits(settings),
        "moments": tokamak_moments(settings),
        "w7x": w7x_orbits(settings),
    }
    result["wall_time"] = time.perf_counter() - start
    return result


def plot(result: dict) -> None:
    """The tokamak orbits, the moments against the local slowing down, the wall loads."""
    tok = result["tokamak"]
    fig, ax = plt.subplots(figsize=(5, 6))
    fig.caption = (
        f"Guiding-centre orbits of five 55 keV protons over {1e3 * tok['time']:.2g} ms in a 3 T "
        "Solov'ev tokamak, "
        "projected on the poloidal plane, with the plasma boundary (black) and the wall (grey)."
    )
    ax.plot(tok["boundary"][:, 0], tok["boundary"][:, 1], color="black", lw=1)
    ax.plot(tok["wall"][:, 0], tok["wall"][:, 1], color="gray", lw=1)
    for k, p in enumerate(tok["pitch"]):
        ax.plot(tok["R"][:, k], tok["z"][:, k], lw=1, label=f"pitch {p:+.2f}")
    ax.set_aspect("equal")
    ax.set_xlabel("$R$ [m]")
    ax.set_ylabel("$Z$ [m]")
    ax.legend(fontsize=8)
    fig.tight_layout()

    mom = result["moments"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.caption = (
        "Fast-ion density against the radius in a 10 T Solov'ev tokamak: orbit following "
        "with collisions and the local slowing down of the same births."
    )
    ax.plot(mom["rho"], mom["density_orbits"], marker="o", label="orbits")
    ax.plot(mom["rho"], mom["density_local"], marker="s", ls="--", label="local")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"fast-ion density [m$^{-3}$]")
    ax.legend()
    fig.tight_layout()

    w7x = result["w7x"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True)
    fig.caption = (
        "W7-X (GVEC): beam heating (top) and fast-ion density (bottom) against the radius, "
        "orbit following to thermalisation or the wall against the local slowing down."
    )
    top.plot(w7x["rho"], w7x["heating_orbits"] / 1e3, marker="o", label="orbits")
    top.plot(w7x["rho"], w7x["heating_local"] / 1e3, marker="s", ls="--", label="local")
    top.set_ylabel(r"heating [kW/m$^3$]")
    top.legend()
    bottom.plot(w7x["rho"], w7x["density_orbits"], marker="o", label="orbits")
    bottom.plot(w7x["rho"], w7x["density_local"], marker="s", ls="--", label="local")
    bottom.set_xlabel(r"$\rho = \sqrt{\Phi/\Phi_b}$")
    bottom.set_ylabel(r"fast-ion density [m$^{-3}$]")
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fig.caption = (
        "Load of the lost beam ions on the stand-in W7-X wall (10 cm off GVEC's boundary), "
        "against the wall's toroidal and poloidal angles."
    )
    mesh = ax.pcolormesh(
        w7x["map_phi"], w7x["map_theta"], w7x["map"] / 1e3, cmap="inferno"
    )
    fig.colorbar(mesh, ax=ax, label=r"wall load [kW/m$^2$]")
    ax.set_xlabel(r"$\varphi$ [rad]")
    ax.set_ylabel(r"$\theta$ [rad]")
    fig.tight_layout()


def table(result: dict) -> str:
    """The checks in the tokamaks and the W7-X fates and powers."""
    tok, mom, w7x = result["tokamak"], result["moments"], result["w7x"]
    rows = [
        "| step [cm] | largest energy error | largest $p_\\varphi$ error [of $e\\Delta\\psi$] |",
        "|---:|---:|---:|",
    ]
    rows += [
        f"| {100 * r['step']:.0f} | {r['energy']:.1e} | {r['p_phi']:.1e} |"
        for r in tok["invariants"]
    ]
    rows += [
        "",
        "| pitch | width [cm] | from $p_\\varphi$ [cm] | $2q\\rho_\\parallel$ [cm] | width / estimate |",
        "|---:|---:|---:|---:|---:|",
    ]
    rows += [
        f"| {w['pitch']:+.2f} | {100 * w['width']:+.3f} | {100 * w['exact']:+.3f} | "
        f"{100 * w['estimate']:.3f} | {abs(w['width']) / w['estimate']:.3f} |"
        for w in tok["widths"]
    ]
    rows += [
        "",
        "| 10 T tokamak, share of the birth power | orbits | local |",
        "|---|---:|---:|",
    ]
    rows += [
        f"| {k} | {float(a):.4f} | {float(b):.4f} |"
        for k, (a, b) in mom["fractions"].items()
    ]
    rows += [
        f"| {k}, orbits / local | {float(v):.4f} | |" for k, v in mom["ratios"].items()
    ]
    fates = ", ".join(f"{k} {100 * v:.1f} %" for k, v in w7x["fates"].items() if v)
    powers = ", ".join(f"{k} {100 * v:.1f} %" for k, v in w7x["powers"].items() if v)
    prompt = ", ".join(
        f"{k} {100 * v:.1f} %" for k, v in w7x["prompt_fates"].items() if v
    )
    rows += [
        "",
        "| W7-X (GVEC) | |",
        "|---|---|",
        f"| births, absorbed | {w7x['births']}, {w7x['absorbed'] / 1e6:.2f} MW |",
        f"| birth pitch | {w7x['pitch_range'][0]:.2f} to {w7x['pitch_range'][1]:.2f} |",
        f"| collisionless 1 ms | {prompt} |",
        f"| energy error after 1 ms | median {w7x['prompt_error'][0]:.1e}, largest {w7x['prompt_error'][1]:.1e} |",
        f"| fates with collisions | {fates} |",
        f"| power | {powers} |",
        f"| lost | {w7x['lost_markers']} markers, {w7x['lost_power'] / 1e3:.0f} kW, mean energy {w7x['lost_energy'] / 1e3:.1f} keV, median time {1e3 * w7x['lost_time']:.2f} ms |",
        f"| peak wall load | {w7x['peak_load'] / 1e3:.1f} kW/m² |",
        "",
        f"Wall time {result['wall_time']:.0f} s.",
    ]
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    args = parser.parse_args()
    outcome = simulate(args.preset)
    print(table(outcome))
    plot(outcome)
    plt.show()
