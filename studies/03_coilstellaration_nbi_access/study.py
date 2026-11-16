"""CoilStellaration NBI access.

Whether a neutral beam fits between the coils limits tangential injection in a stellarator. The
[CoilStellaration](https://huggingface.co/datasets/proxima-fusion/coilstellaration) dataset of
Proxima Fusion (arXiv:2611.12689, MIT) has coilsets for the quasi-isodynamic plasma boundaries of
[ConStellaration](https://huggingface.co/datasets/proxima-fusion/constellaration), each designed
for randomly drawn engineering requirements: the smallest coil–coil and coil–plasma distance, the
largest curvature, and 4 to 8 coils per half period. This study scans, for each instance, which
horizontal tangential beam lines reach the plasma between the coils, and asks how the access
depends on those requirements.

**Instances.** The `eval` split of the benchmark's `variable_shape` track (2 to 5 field periods,
4 to 8 coils per half period) with a DESC-refined coilset and its metrics: 8058 instances on 709
boundaries. `quick` takes one instance at random (seed 0) of every combination of field periods
and coils per half period (17 instances on 17 boundaries; two combinations have no DESC metrics;
their coilsets and boundaries are kept in `data/coilstellaration_quick.json`, so it
runs without a download); `full` takes 2000 of the 8058 at random (seed 0) and downloads them once into `data/coilstellaration/`.

**Lengths in minor radii.** As the dataset's metrics, every length is divided by the plasma's
minor radius $a = \\sqrt{\\bar A / \\pi}$ ($\\bar A$ the mean cross-section area at constant
$\\varphi$, `plasma_plots.from_boundary`); the boundaries have $R_0 = 1$ m and $a$ from 0.08 to
0.25 m (aspect ratios 4 to 12), so a device of a given size follows by scaling.

**Beam lines.** `freeofcharge.coils.injection_access`: straight axes in the plane $Z = 0$, tangent
to the circle $R = R_\\mathrm{tan}$ at the toroidal angle $\\varphi$, with
$x = (R_\\mathrm{tan} - R_0)/a$ from $-1$ to $1$ (21 values) and $\\varphi$ over one field period
(48 values), all towards $+\\varphi$ (stellarator symmetry maps a line towards $-\\varphi$ at
$\\varphi$ on one towards $+\\varphi$ at $-\\varphi$, so one sense covers both). Each axis starts
outside the coils and is followed to where it enters the plasma (the ConStellaration boundary,
`PlasmaBoundary`). Its **clearance** $c$ is the smallest distance from the axis, up to the entry,
to a coil filament (the expanded coilset, 64 points per coil).

**Choices.** A line is **admissible** if it reaches the plasma and
$c \\ge r_\\mathrm{coil} + r_\\mathrm{beam}$: the beam a cylinder of radius $r_\\mathrm{beam}$
without divergence, the coil a tube of radius $r_\\mathrm{coil}$ around its filament. Only the sum
$g = r_\\mathrm{coil} + r_\\mathrm{beam}$ matters, so the result keeps $c$ and the admissible
fraction for every $g$ from 0 to $a$. The headline choice is $r_\\mathrm{coil} = d_{cc}/2$, half
the instance's coil–coil distance requirement (the largest winding pack the requirement leaves
room for: neighbouring tubes touch where the coils are closest), and $r_\\mathrm{beam} = 0.15\\,a$
(8 cm on W7-X's $a = 0.53$ m; an assumption, the real beam and port geometry are IPP data). The
admissible fraction is over the lines that reach the plasma.

**Finding (quick, 17 instances).** At the headline gap ($g$ = 0.30 to 0.47 $a$) a median 42 % of
the horizontal tangential lines that reach the plasma pass between the coils (2 % to 85 %); with
a gap of 0.2 $a$ it is 67 % (33 % to 91 %), with 0.5 $a$ 35 % (0 % to 77 %). Every instance has
some admissible line: the best clearance is 0.5 to 3 $a$. The number of field periods matters
most (median 61 % for 2, 58 % for 3, 22 % for 4, 28 % for 5; rank correlation $-0.78$), then the
coil–plasma distance the coilset actually reaches ($+0.56$, $+0.91$ for the 12 coilsets with a
mean field error below 1 %), while the coil–plasma distance *requirement* hardly does ($-0.33$):
the DESC coilsets often keep much further from the plasma than required (up to 1.6 $a$ for a
0.57 $a$ requirement). More coils per half period close the gaps ($-0.45$), and a larger
coil–coil requirement does too ($-0.60$), largely because it widens the coil tubes by
construction. Five of the 17 coilsets have a mean field error above 1 % and one (2 field periods,
5 coils) 16 %, with coils 0.08 $a$ from the plasma against a 0.85 $a$ requirement. Seventeen
instances on 17 boundaries cannot separate these effects (field periods, coil number and the
distances vary together); the full preset can, with 2000 instances: the access against each
requirement at fixed field periods and coil number, the gap at which the admissible fraction
falls to zero, and whether the requirements a coil designer sets predict beam access or only the
distances the coilset reaches do.

The horizontal plane $Z = 0$ is a choice: some boundaries move far off it (a 4-period boundary
of the `eval` split, met while setting up this study, misses $Z = 0$ over 70 % of the field period, by up to 1.1 $a$), so their lines
should follow the magnetic axis or be tilted; `injection_access` scans heights and vertical
angles, which the full run can add.

Compared: the coil–coil and coil–plasma distances of the expanded coilsets
(`plasma_plots.coils`) with the dataset's own metrics, a check of the reading and of $a$. Not
compared: a real port geometry, the vacuum vessel, the beam's divergence and its deposition.
Deposition on ConStellaration's DESC equilibria, the second half of this question, waits for
freeofcharge's 3D deposition on DESC equilibria.

Reference: J.-H. Ewers et al., CoilStellaration, NeurIPS 2026 (arXiv:2611.12689); S. A. Cadena
et al., ConStellaration, NeurIPS 2025 (arXiv:2506.19583). Both datasets: MIT, Copyright (c)
Proxima Fusion GmbH.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
import xarray as xr
from freeofcharge.coils import CoilSet, PlasmaBoundary, injection_access
from plasma_plots import coils as pp_coils

PRESETS = {
    "smoke": {
        "source": "quick",
        "instances": 2,
        "tangency": 5,
        "toroidal": 8,
        "check": 32,
    },
    "quick": {
        "source": "quick",
        "instances": None,
        "tangency": 21,
        "toroidal": 48,
        "check": 128,
    },
    "full": {
        "source": "full",
        "instances": 2000,
        "tangency": 21,
        "toroidal": 48,
        "check": 64,
    },
}

BEAM_RADIUS = 0.15  # in minor radii
FIELD_ERROR_LIMIT = 0.01  # mean |B.n|/|B| above which a coilset is marked as poor
POINTS_PER_COIL = 64
GAPS = np.round(np.arange(0.0, 1.0001, 0.05), 2)  # g = r_coil + r_beam, in minor radii
MAP_INSTANCE = {"nfp": 3, "coils_per_half_period": 6}

HERE = Path(__file__).resolve().parent
QUICK_FILE = HERE / "data" / "coilstellaration_quick.json"
FULL_FILE = HERE / "data" / "coilstellaration" / "full.json.gz"  # not committed
HF = "datasets/proxima-fusion"
RESULTS_FILE = f"{HF}/coilstellaration/results/eval-part-0.parquet"
REQUIREMENTS = {
    "coil_coil_distance": "reqs/normalized_min_coil_to_coil_distance",
    "coil_plasma_distance": "reqs/normalized_min_coil_plasma_distance",
    "curvature": "reqs/normalized_max_coil_curvature",
}
METRICS = {
    "coil_coil_distance": "desc_metrics/normalized_coil_to_coil_min_distances/min",
    "coil_plasma_distance": "desc_metrics/normalized_coil_to_plasma_min_distances/min",
    "curvature": "desc_metrics/normalized_coil_curvatures/max",
    "field_error": "desc_metrics/normalized_field_error/mean",
    "minor_radius": "desc_metrics/minor_radius",
}
LICENSE = (
    "CoilStellaration and ConStellaration, MIT License, Copyright (c) Proxima Fusion GmbH: "
    "https://huggingface.co/datasets/proxima-fusion/coilstellaration, "
    "https://huggingface.co/datasets/proxima-fusion/constellaration"
)


def _read_by_id(fs, path: str, id_column: str, columns: list[str], ids: set) -> list:
    """Rows of a Hugging Face parquet file whose ``id_column`` is in ``ids``.

    Reads the ID column, then only the row groups that hold a wanted row.
    """
    import pyarrow.parquet as pq

    with fs.open(path) as handle:
        parquet = pq.ParquetFile(handle)
        found = np.isin(
            parquet.read(columns=[id_column]).column(0).to_numpy(zero_copy_only=False),
            list(ids),
        )
        ends = np.cumsum(
            [
                parquet.metadata.row_group(i).num_rows
                for i in range(parquet.num_row_groups)
            ]
        )
        groups = sorted(
            {int(np.searchsorted(ends, i, side="right")) for i in np.nonzero(found)[0]}
        )
        if not groups:
            return []
        table = parquet.read_row_groups(groups, columns=[id_column, *columns])
    return [row for row in table.to_pylist() if row[id_column] in ids]


def fetch(source: str) -> dict:
    """Download the instances of ``source`` ("quick" or "full") from Hugging Face.

    Needs ``huggingface_hub``, ``pandas`` and ``pyarrow`` (the ``coilstellaration`` extra).
    Reads the results table (15 MB), the coilsets' ID map (11 MB), the row groups of the
    coilsets that hold the chosen ones (all 2.3 GB for ``full``) and the boundary column of
    ConStellaration's row groups that hold the chosen boundaries.
    """
    import pandas as pd
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    columns = [
        "constellaration_boundary_id",
        "desc_coilset_id",
        "boundary.n_field_periods",
        "reqs/n_coils_per_half_period",
        "benchmark/variable_shape_track",
        *REQUIREMENTS.values(),
        *METRICS.values(),
    ]
    with fs.open(RESULTS_FILE) as handle:
        results = pd.read_parquet(handle, columns=columns)
    pool = results[
        results["benchmark/variable_shape_track"]
        & results["desc_coilset_id"].notna()
        & results[METRICS["field_error"]].notna()
    ].sort_values(["constellaration_boundary_id", "desc_coilset_id"])
    if source == "quick":
        keys = ["boundary.n_field_periods", "reqs/n_coils_per_half_period"]
        chosen = pool.groupby(keys).sample(n=1, random_state=0).sort_values(keys)
    else:
        chosen = pool.sample(
            n=min(PRESETS["full"]["instances"], len(pool)), random_state=0
        )
    print(f"{len(chosen)} of {len(pool)} instances")

    coilset_ids = set(chosen["desc_coilset_id"])
    with fs.open(f"{HF}/coilstellaration/coilsets/id_to_file_map.parquet") as handle:
        where = pd.read_parquet(handle, filters=[("id", "in", sorted(coilset_ids))])
    coilsets = {}
    for path, group in where.groupby("path"):
        rows = _read_by_id(
            fs, f"{HF}/coilstellaration/{path}", "id", ["json"], set(group["id"])
        )
        coilsets.update({row["id"]: json.loads(row["json"]) for row in rows})
        print(f"{path}: {len(rows)} coilsets")

    boundary_ids = set(chosen["constellaration_boundary_id"])
    boundaries = {}
    for part in range(3):
        rows = _read_by_id(
            fs,
            f"{HF}/constellaration/data/train-{part:05d}-of-00003.parquet",
            "plasma_config_id",
            ["boundary.json"],
            boundary_ids,
        )
        boundaries.update(
            {row["plasma_config_id"]: json.loads(row["boundary.json"]) for row in rows}
        )
    print(f"{len(boundaries)} boundaries")

    instances = []
    for _, row in chosen.iterrows():
        instances.append(
            {
                "boundary_id": row["constellaration_boundary_id"],
                "coilset_id": row["desc_coilset_id"],
                "nfp": int(row["boundary.n_field_periods"]),
                "coils_per_half_period": int(row["reqs/n_coils_per_half_period"]),
                "requirements": {k: float(row[v]) for k, v in REQUIREMENTS.items()},
                "metrics": {k: float(row[v]) for k, v in METRICS.items()},
                "coilset": coilsets[row["desc_coilset_id"]],
            }
        )
    return {"license": LICENSE, "instances": instances, "boundaries": boundaries}


def load_instances(source: str) -> dict:
    """The instances of ``source``, downloaded once and then read from the data file."""
    path = QUICK_FILE if source == "quick" else FULL_FILE
    if not path.exists():
        data = fetch(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt") as handle:
            json.dump(data, handle, separators=(",", ":"))
            handle.write("\n")
        print(f"wrote {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def midplane_cut(surface: PlasmaBoundary, phi, n_theta: int = 512):
    """Where the plane Z = 0 cuts the cross-sections at ``phi``.

    Returns the inner and outer R (NaN where it misses), the number of crossings and the
    distance of the cross-section from the plane (0 where it cuts it).
    """
    theta = np.linspace(0.0, 2 * np.pi, n_theta + 1)
    r, z = surface.rz(theta[:, None], np.asarray(phi)[None, :])
    change = (z[:-1] > 0) != (z[1:] > 0)
    t = z[:-1] / np.where(change, z[:-1] - z[1:], 1.0)
    cut = r[:-1] + t * (r[1:] - r[:-1])
    inner = np.where(change, cut, np.inf).min(axis=0)
    outer = np.where(change, cut, -np.inf).max(axis=0)
    missed = ~change.any(axis=0)
    gap = np.where(missed, np.abs(z).min(axis=0), 0.0)
    inner[missed], outer[missed] = np.nan, np.nan
    return inner, outer, change.sum(axis=0), gap


class MidplaneBoundary:
    """A `PlasmaBoundary` with a fast test for points in the plane Z = 0.

    The beam lines of this study lie in that plane. Where it cuts the cross-section in one
    interval of R, tabulated at ``n_phi`` angles per field period and interpolated linearly, a
    point is inside if its R lies in the interval; where the plane misses the cross-section by
    more than ``margin``, it is outside. Everything else goes to the even–odd test of
    `PlasmaBoundary.inside_at`. The scan of the plasma entry gets tens of times faster, and the
    entry points agree with the even–odd test to 0.002 minor radii.
    """

    def __init__(
        self, boundary: PlasmaBoundary, margin: float, n_phi: int = 512
    ) -> None:
        self.boundary = boundary
        self.period = 2 * np.pi / boundary.nfp
        self.n_phi = n_phi
        self.inner, self.outer, crossings, gap = midplane_cut(
            boundary, np.linspace(0.0, self.period, n_phi + 1)
        )
        # each interval between two tabulated angles
        self.simple = (crossings[:-1] == 2) & (crossings[1:] == 2)
        self.missed = (
            (crossings[:-1] == 0)
            & (crossings[1:] == 0)
            & (np.minimum(gap[:-1], gap[1:]) > margin)
        )

    def inside_at(self, points):
        points = np.asarray(points, dtype=float)
        flat = points.reshape(-1, 3)
        r = np.hypot(flat[:, 0], flat[:, 1])
        phase = np.mod(np.arctan2(flat[:, 1], flat[:, 0]), self.period) / self.period
        k = np.minimum((phase * self.n_phi).astype(int), self.n_phi - 1)
        w = phase * self.n_phi - k
        plane = flat[:, 2] == 0.0
        simple = plane & self.simple[k]
        inner = (1 - w) * self.inner[k] + w * self.inner[np.minimum(k + 1, self.n_phi)]
        outer = (1 - w) * self.outer[k] + w * self.outer[np.minimum(k + 1, self.n_phi)]
        result = simple & (r > inner) & (r < outer)
        exact = ~simple & ~(plane & self.missed[k])
        if exact.any():
            result[exact] = np.asarray(self.boundary.inside_at(flat[exact]))
        return result.reshape(points.shape[:-1])


def access(
    instance: dict, boundary: dict, n_tangency: int, n_toroidal: int, n_check: int
) -> dict:
    """The clearance map of one instance and its distance checks, lengths in minor radii."""
    coil_points = plasma_plots.from_coilset(
        instance["coilset"], n_points=POINTS_PER_COIL
    )
    coils = CoilSet.from_dataset(coil_points)
    plasma = plasma_plots.from_boundary(boundary)
    a, major = plasma.attrs["minor_radius"], plasma.attrs["major_radius"]
    surface = MidplaneBoundary(PlasmaBoundary.from_boundary(boundary), 0.1 * a)
    x = np.linspace(-1.0, 1.0, n_tangency)
    period = 2 * np.pi / instance["nfp"]
    phi = np.linspace(0.0, period, n_toroidal, endpoint=False)
    coil_r = np.hypot(coil_points.X, coil_points.Y)
    scan = injection_access(
        coils,
        surface,
        major + a * x,
        phi,
        start_distance=float(coil_r.max()) + a,  # every start lies outside the coils
        coil_radius=0.0,
        beam_radius=0.0,
        step=0.1 * a,
    ).isel(vertical_angle=0)
    reaches = np.isfinite(scan.plasma_distance.values)
    fine = plasma_plots.from_coilset(instance["coilset"], n_points=n_check)
    return {
        "clearance": np.where(reaches, scan.clearance.values / a, np.nan),
        "x": x,
        "phi": phi / period,
        "minor_radius": a,
        "coil_coil": float(pp_coils.coil_to_coil_distance(fine).min()) / a,
        "coil_plasma": float(pp_coils.distance_to_surface(fine, plasma).min()) / a,
        "coil_r": (float(coil_r.min()) - major) / a,
    }


def top_view(instance: dict, boundary: dict) -> dict:
    """The coils and the plasma's cut by the plane Z = 0, in minor radii, for the figure."""
    plasma = plasma_plots.from_boundary(boundary)
    a = plasma.attrs["minor_radius"]
    coils = plasma_plots.from_coilset(instance["coilset"], n_points=POINTS_PER_COIL)
    phi = np.linspace(0.0, 2 * np.pi, 721)
    inner, outer, _, _ = midplane_cut(PlasmaBoundary.from_boundary(boundary), phi)
    closed = np.append(np.arange(coils.sizes["s"]), 0)
    coil_r = np.hypot(coils.X, coils.Y)
    return {
        "major_radius": plasma.attrs["major_radius"] / a,
        "start_distance": float(coil_r.max()) / a + 1.0,
        "coil_x": (coils.X.transpose("coil", "s").values[:, closed] / a).astype(
            np.float32
        ),
        "coil_y": (coils.Y.transpose("coil", "s").values[:, closed] / a).astype(
            np.float32
        ),
        "phi": phi,
        "plasma_inner": inner / a,
        "plasma_outer": outer / a,
    }


def _is_map_instance(instance: dict) -> bool:
    return all(instance[k] == v for k, v in MAP_INSTANCE.items())


def _as_dataset(rows: list[dict], maps: list, x, phi) -> dict:
    """Plain data for the cache: one Dataset of the instances, one of the maps."""
    scalar = [k for k in rows[0] if not isinstance(rows[0][k], (list, str))]
    instances = xr.Dataset(
        {
            k: ("instance", np.array([row[k] for row in rows], dtype=np.float32))
            for k in scalar
        },
        coords={"instance": np.arange(len(rows)), "gap": GAPS, "x": x},
    )
    instances["fraction_vs_gap"] = (
        ("instance", "gap"),
        np.array([row["fraction_vs_gap"] for row in rows], dtype=np.float16),
    )
    if (
        len(rows) <= 100
    ):  # the full run's instances are found by their order in its data file
        for k in ("boundary_id", "coilset_id"):
            instances[k] = (
                "instance",
                np.array([row[k] for row in rows], dtype=object),
            )
    instances["best_clearance"] = (
        ("instance", "x"),
        np.array([row["best_clearance"] for row in rows], dtype=np.float16),
    )
    clearance = xr.DataArray(
        np.array([m[1] for m in maps], dtype=np.float32),
        dims=("instance", "x", "phi"),
        coords={"instance": [m[0] for m in maps], "x": x, "phi": phi},
        name="clearance",
    )
    instances.x.attrs.update(label=r"$(R_\mathrm{tan} - R_0)/a$")
    instances.gap.attrs.update(
        label=r"gap $g = r_\mathrm{coil} + r_\mathrm{beam}$ [$a$]"
    )
    clearance.attrs.update(label=r"clearance $c/a$")
    clearance.phi.attrs.update(label=r"$\varphi$ [field periods]")
    clearance.x.attrs.update(label=r"$(R_\mathrm{tan} - R_0)/a$")
    return {"instances": instances, "maps": clearance, "license": LICENSE}


def simulate(preset: str = "quick") -> dict:
    """Scan every instance of the preset; return a Dataset over the instances and the maps."""
    options = PRESETS[preset]
    data = load_instances(options["source"])
    instances = data["instances"][: options["instances"]]
    start = time.time()
    rows, maps, view = [], [], None
    for i, instance in enumerate(instances):
        scan = access(
            instance,
            data["boundaries"][instance["boundary_id"]],
            options["tangency"],
            options["toroidal"],
            options["check"],
        )
        clearance = scan["clearance"]
        reaches = np.isfinite(clearance)
        n_reach = max(int(reaches.sum()), 1)
        fractions = [
            float(np.sum(np.nan_to_num(clearance, nan=-1.0) >= g)) / n_reach
            for g in GAPS
        ]
        gap = instance["requirements"]["coil_coil_distance"] / 2 + BEAM_RADIUS
        rows.append(
            {
                **{f"req_{k}": v for k, v in instance["requirements"].items()},
                **{f"metric_{k}": v for k, v in instance["metrics"].items()},
                "nfp": instance["nfp"],
                "coils_per_half_period": instance["coils_per_half_period"],
                "check_coil_coil": scan["coil_coil"],
                "check_coil_plasma": scan["coil_plasma"],
                "check_minor_radius": scan["minor_radius"],
                "headline_gap": gap,
                "admissible_fraction": float(
                    np.sum(np.nan_to_num(clearance, nan=-1.0) >= gap) / n_reach
                ),
                "reach_fraction": float(reaches.mean()),
                "fraction_vs_gap": fractions,
                "best_clearance": np.nanmax(
                    np.where(reaches, clearance, -np.inf), axis=1
                ).tolist(),
                "boundary_id": instance["boundary_id"],
                "coilset_id": instance["coilset_id"],
            }
        )
        if preset != "full" or (not maps and _is_map_instance(instance)):
            maps.append((i, clearance))
        if view is None and _is_map_instance(instance):
            view = top_view(instance, data["boundaries"][instance["boundary_id"]])
        if (i + 1) % 50 == 0 or i + 1 == len(instances):
            print(f"{i + 1}/{len(instances)} instances, {time.time() - start:.0f} s")
    if view is None:  # the smoke preset may not have the map instance
        view = top_view(instances[0], data["boundaries"][instances[0]["boundary_id"]])
    return {**_as_dataset(rows, maps, scan["x"], scan["phi"]), "top_view": view}


def _map_index(instances: xr.Dataset, maps: xr.DataArray) -> int:
    for i in maps.instance.values:
        if all(int(instances[k][i]) == v for k, v in MAP_INSTANCE.items()):
            return int(i)
    return int(maps.instance.values[0])


def _top_view(ax, view: dict, clearance: xr.DataArray, nfp: int) -> None:
    """The instance's coils, its plasma's cut by the plane Z = 0 and its best beam lines."""
    for x, y in zip(view["coil_x"], view["coil_y"], strict=True):
        ax.plot(x, y, color="0.45", lw=0.6)
    phi = view["phi"]
    for r in (view["plasma_inner"], view["plasma_outer"]):
        ax.plot(r * np.cos(phi), r * np.sin(phi), color="tab:orange", lw=1.2)
    # the line of largest clearance at three tangency radii, from its start to its tangency point
    for x, colour in zip(
        (-0.5, 0.0, 0.5), ("tab:blue", "tab:green", "tab:purple"), strict=True
    ):
        row = clearance.sel(x=x, method="nearest")
        if not np.isfinite(row).any():
            continue
        best = int(np.argmax(row.fillna(-1.0).values))
        angle = 2 * np.pi / nfp * float(row.phi[best])
        radius = view["major_radius"] + float(row.x)
        tangent = radius * np.array([np.cos(angle), np.sin(angle)])
        start = tangent - view["start_distance"] * np.array(
            [-np.sin(angle), np.cos(angle)]
        )
        ax.plot(
            [start[0], tangent[0]],
            [start[1], tangent[1]],
            color=colour,
            lw=1.5,
            label=f"x = {float(row.x):+.1f}: c = {float(row[best]):.2f} a",
        )
    ax.set_aspect("equal")
    ax.set_xlabel("$X/a$")
    ax.set_ylabel("$Y/a$")
    ax.legend(fontsize=7, loc="upper right", title="best lines", title_fontsize=7)
    ax.set_title("coils (grey), the plasma at Z = 0 (orange)", fontsize=9)


def _summary(instances: xr.Dataset) -> None:
    n = instances.sizes["instance"]
    frac = instances.admissible_fraction.values
    print(
        f"{n} instances; admissible fraction at the headline gap: median {np.median(frac):.3f}, range {frac.min():.3f} to {frac.max():.3f}"
    )
    for name in (
        "req_coil_plasma_distance",
        "metric_coil_plasma_distance",
        "req_coil_coil_distance",
        "metric_coil_coil_distance",
        "coils_per_half_period",
        "nfp",
        "metric_field_error",
    ):
        rank = np.corrcoef(
            np.argsort(np.argsort(instances[name].values)), np.argsort(np.argsort(frac))
        )[0, 1]
        print(f"rank correlation of the fraction with {name}: {rank:+.2f}")
    for k in ("coil_coil", "coil_plasma"):
        ratio = instances[f"check_{k}"] / instances[f"metric_{k}_distance"]
        print(
            f"{k} distance, expanded coils / dataset: {float(ratio.min()):.3f} to {float(ratio.max()):.3f}"
        )
    ratio = instances.check_minor_radius / instances.metric_minor_radius
    print(
        f"minor radius, plasma-plots / dataset: {float(ratio.min()):.4f} to {float(ratio.max()):.4f}"
    )


def plot(result: dict) -> None:
    """The access map, the top view, the best clearance, the fraction against the requirements and the gap."""
    instances, maps = result["instances"], result["maps"]
    i = _map_index(instances, maps)
    gap = float(instances.headline_gap[i])
    colours = plt.cm.viridis(np.linspace(0, 0.9, 5))  # 4 to 8 coils per half period

    # 1. the access map of one instance
    with plasma_plots.figure(1, 1, figsize=(7, 4.5)) as fig:
        clearance = maps.sel(instance=i)
        clearance.plasma.plot.slice(
            x="phi", y="x", cmap="magma", ax=fig[0], colorbar_label="clearance c/a"
        )
        fig[0].contourf(
            clearance.phi,
            clearance.x,
            clearance.fillna(-1.0),
            levels=[gap, np.inf],
            colors="none",
            hatches=["///"],
        )
        fig[0].contour(
            clearance.phi,
            clearance.x,
            clearance,
            levels=[gap],
            colors="white",
            linewidths=1.0,
        )
        fig[0].set_title(
            f"{int(instances.nfp[i])} field periods, {int(instances.coils_per_half_period[i])} coils "
            f"per half period: {100 * float(instances.admissible_fraction[i]):.0f} % admissible at "
            f"g = {gap:.2f} a",
            fontsize=9,
        )

    # 2. the coils and the best lines of that instance, seen from above
    with plasma_plots.figure(1, 1, figsize=(6, 6)) as fig:
        _top_view(fig[0], result["top_view"], clearance, int(instances.nfp[i]))

    # 3. the best clearance per tangency radius
    with plasma_plots.figure(1, 1, figsize=(7, 4)) as fig:
        ax = fig[0]
        for j in instances.instance.values:
            n = int(instances.coils_per_half_period[j])
            ax.plot(
                instances.x,
                instances.best_clearance[j],
                color=colours[n - 4],
                lw=1,
                alpha=0.8,
            )
        for n in range(4, 9):
            ax.plot([], [], color=colours[n - 4], label=f"{n} coils per half period")
        ax.set_xlabel(r"tangency radius $(R_\mathrm{tan} - R_0)/a$")
        ax.set_ylabel("best clearance c/a over the toroidal angle")
        ax.legend(fontsize=8)

    # 4. the admissible fraction against the requirements
    nfp_colours = {2: "tab:blue", 3: "tab:orange", 4: "tab:green", 5: "tab:red"}
    colour = np.array([nfp_colours[int(n)] for n in instances.nfp])
    good = instances.metric_field_error.values <= FIELD_ERROR_LIMIT
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, instances.sizes["instance"])
    with plasma_plots.figure(3, 1, figsize=(7, 10), sharey=True) as fig:
        for ax, x, label in (
            (
                fig[0],
                instances.req_coil_plasma_distance,
                "coil–plasma distance requirement [a]",
            ),
            (
                fig[1],
                instances.metric_coil_plasma_distance,
                "coil–plasma distance of the coilset [a]",
            ),
            (fig[2], instances.coils_per_half_period + jitter, "coils per half period"),
        ):
            size = 18 if good.size < 100 else 6
            ax.scatter(
                x[good], instances.admissible_fraction[good], c=colour[good], s=size
            )
            ax.scatter(
                x[~good],
                instances.admissible_fraction[~good],
                facecolors="none",
                edgecolors=colour[~good],
                s=size,
            )
            ax.set_xlabel(label)
            ax.set_ylabel(f"admissible fraction (g = d_cc/2 + {BEAM_RADIUS} a)")
        for nfp, c in nfp_colours.items():
            fig[2].scatter([], [], color=c, label=f"{nfp} field periods")
        fig[2].scatter(
            [],
            [],
            facecolors="none",
            edgecolors="0.4",
            label=f"field error > {FIELD_ERROR_LIMIT:.0%}",
        )
        fig[2].legend(fontsize=8)

    # 5. the admissible fraction against the gap a line needs
    with plasma_plots.figure(1, 1, figsize=(7, 4)) as fig:
        ax = fig[0]
        for j in instances.instance.values:
            n = int(instances.coils_per_half_period[j])
            ax.plot(
                instances.gap,
                instances.fraction_vs_gap[j],
                color=colours[n - 4],
                lw=1,
                alpha=0.7,
            )
            ax.scatter(
                instances.headline_gap[j],
                instances.admissible_fraction[j],
                color=colours[n - 4],
                s=12,
                zorder=3,
            )
        for n in range(4, 9):
            ax.plot([], [], color=colours[n - 4], label=f"{n} coils per half period")
        ax.set_xlabel(r"gap $g = r_\mathrm{coil} + r_\mathrm{beam}$ [a]")
        ax.set_ylabel("admissible fraction of the lines that reach the plasma")
        ax.legend(fontsize=8)

    _summary(instances)


def table(result: dict) -> str:
    """One row per instance (quick), or per field-period and coil-number combination (full)."""
    ds = result["instances"]
    head = "| nfp | coils per half period | d_cc req. [a] | d_cp req. [a] | curvature req. [1/a] | field error | lines reaching the plasma | gap g [a] | admissible | best clearance [a] |\n|---|---|---|---|---|---|---|---|---|---|\n"
    frame = ds[
        [
            "nfp",
            "coils_per_half_period",
            "req_coil_coil_distance",
            "req_coil_plasma_distance",
            "req_curvature",
            "metric_field_error",
            "reach_fraction",
            "headline_gap",
            "admissible_fraction",
        ]
    ].to_dataframe()
    frame["best"] = ds.best_clearance.max("x").values
    if len(frame) > 40:
        frame = frame.groupby(["nfp", "coils_per_half_period"]).median().reset_index()
    lines = [
        f"| {int(r.nfp)} | {int(r.coils_per_half_period)} | {r.req_coil_coil_distance:.2f} | "
        f"{r.req_coil_plasma_distance:.2f} | {r.req_curvature:.2f} | "
        f"{100 * r.metric_field_error:.2f} % | {100 * r.reach_fraction:.0f} % | {r.headline_gap:.2f} | "
        f"{100 * r.admissible_fraction:.0f} % | {r.best:.2f} |"
        for r in frame.itertuples()
    ]
    return head + "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    args = parser.parse_args()
    plot(simulate(args.preset))
    plt.show()
