# max-models gallery

Computational studies with the max-models codes: verification, method demonstrations and examples.

Site: <https://max-models.github.io/gallery/>

| Study | Kind | Codes |
|---|---|---|
| [Exponential decay](https://max-models.github.io/gallery/studies/exponential-decay/) | example | numpy |
| [Einzel lens optimisation](https://max-models.github.io/gallery/studies/einzel-lens-optimisation/) | verification | beamtracer, plasma-plots |
| [CoilStellaration NBI access](https://max-models.github.io/gallery/studies/coilstellaration-nbi-access/) | example | freeofcharge, plasma-plots |
| [W7-X NBI transport](https://max-models.github.io/gallery/studies/w7x-nbi-transport/) | example | slowcooker, freeofcharge |
| [Injector chain](https://max-models.github.io/gallery/studies/injector-chain/) | example | freeofcharge, wellbalanced, tree-d-3d, plasmacoll, plasma-plots |
| [Neutralizer channel](https://max-models.github.io/gallery/studies/neutralizer-channel/) | example | beamtracer, wellbalanced, opencr, plasmacoll, plasma-plots |
| [Beamline power balance](https://max-models.github.io/gallery/studies/beamline/) | example | beamtracer, plasmacoll, simgeometry, plasma-plots |
| [Negative-ion extraction](https://max-models.github.io/gallery/studies/negative-ion-extraction/) | example | beamtracer, plasma-plots |
| [Fast-ion orbits](https://max-models.github.io/gallery/studies/fast-ion-orbits/) | example | freeofcharge, plasmacoll, simgeometry, slowcooker |
| [W7-X current drive](https://max-models.github.io/gallery/studies/w7x-current-drive/) | example | freeofcharge, slowcooker |
| [ASCOT5 benchmark](https://max-models.github.io/gallery/studies/ascot5-benchmark/) | verification | freeofcharge, plasmacoll, plasma-plots |
| [ITER-like injection](https://max-models.github.io/gallery/studies/iter-like-injection/) | example | freeofcharge, plasmacoll, plasma-plots |
| [Positive-ion extraction](https://max-models.github.io/gallery/studies/positive-ion-extraction/) | example | freeofcharge, wellbalanced, beamtracer, plasma-plots |
| [Beam instabilities](https://max-models.github.io/gallery/studies/beam-instabilities/) | verification | picnic, plasma-plots |
| [Ion acoustic instability](https://max-models.github.io/gallery/studies/ion-acoustic-instability/) | verification | picnic, plasma-plots |
| [Multi-species streaming](https://max-models.github.io/gallery/studies/multispecies-streaming/) | verification | picnic, plasma-plots |

## Running a study

```bash
uv sync
uv run python studies/<study>/study.py --preset smoke
uv run gallery build --cached-only   # the pages of the site, from the cached results
```

This repository is generated from a private gallery with `gallery export` ([gallery-template](https://github.com/spokes3d/gallery-template)): changes come in as pull requests from the sync and are not made here by hand.
