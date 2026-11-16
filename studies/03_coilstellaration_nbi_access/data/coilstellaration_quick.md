# coilstellaration_quick.json

The instances of the `quick` preset of `studies/03_coilstellaration_nbi_access/study.py`, written by
its `fetch("quick")`: one instance at random (seed 0) of every combination of field periods and
coils per half period in the `eval` split of CoilStellaration's `variable_shape` track that has a
DESC coilset and its metrics (17 instances). For each: the boundary and coilset IDs, the
requirements and the dataset's metrics (`results` subset), the DESC coilset (the `json` of its
row in the `coilsets` subset) and, under `boundaries`, the ConStellaration `boundary.json` of
each plasma boundary. Delete the file to fetch it again (about 4 minutes).

Both datasets are MIT-licensed, Copyright (c) Proxima Fusion GmbH:
https://huggingface.co/datasets/proxima-fusion/coilstellaration and
https://huggingface.co/datasets/proxima-fusion/constellaration.
