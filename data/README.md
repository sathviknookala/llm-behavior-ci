# Data

AppWorld is installed and its data downloaded (`docs/DATA.md`). `processed/` holds, locally, the catalog and the resolved `train`, `dev`, and `train_smoke` task sets whose hashes are in `configs/tasks/`. No episode log, outcome, or public manifest exists.

- `raw/`: the local AppWorld root (`APPWORLD_ROOT`) after Sathvik accepts the terms in `docs/DATA.md`. Local only.
- `processed/`: resolved task sets, episode logs, per-task outcomes, and derived task streams. Local only.
- `manifests/`: tracked public items only: AppWorld version, split names, selection rules, seeds, hashes of the resolved local task sets, counts, and provenance.

Everything on the local-only list in `docs/DATA.md` stays out of git.
