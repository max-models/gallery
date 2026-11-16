# max-models gallery

Computational studies with the max-models codes: verification, method demonstrations and examples.

Site: <https://max-models.github.io/gallery/>

| Study | Kind | Codes |
|---|---|---|
| [Exponential decay](https://max-models.github.io/gallery/studies/exponential-decay/) | example | numpy |
| [Einzel lens optimisation](https://max-models.github.io/gallery/studies/einzel-lens-optimisation/) | verification | beamtracer, plasma-plots |
| [CoilStellaration NBI access](https://max-models.github.io/gallery/studies/coilstellaration-nbi-access/) | example | freeofcharge, plasma-plots |
| [W7-X NBI transport](https://max-models.github.io/gallery/studies/w7x-nbi-transport/) | example | slowcooker, freeofcharge |

## Running a study

```bash
uv sync
uv run python studies/<study>/study.py --preset smoke
uv run gallery build --cached-only   # the pages of the site, from the cached results
```

This repository is generated from a private gallery with `gallery export` ([gallery-template](https://github.com/spokes3d/gallery-template)): changes come in as pull requests from the sync and are not made here by hand.
