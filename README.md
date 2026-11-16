# max-models gallery

Computational studies with the max-models codes: verification, method demonstrations and examples.

Site: <https://max-models.github.io/gallery/>

| Study | Kind | Codes |
|---|---|---|
| [Exponential decay](https://max-models.github.io/gallery/studies/exponential-decay/) | example | numpy |

## Running a study

```bash
uv sync
uv run python studies/<study>/study.py --preset smoke
uv run gallery build --cached-only   # the pages of the site, from the cached results
```

This repository is generated from a private gallery with `gallery export` ([gallery-template](https://github.com/spokes3d/gallery-template)): changes come in as pull requests from the sync and are not made here by hand.
