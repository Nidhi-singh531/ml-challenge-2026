# Project checkpoint

This backup contains the pipeline source, README, dependencies, VS Code
settings, validator, experiment logs, alias lexicons, helper scripts, and
the current `work/dev/model40k.pkl` model.

Current measured validation macro F0.5 is 0.9629 with the prefilter, on
10,000 held-out entities. See `work/dev/train40k.log` and the project README
for details and remaining work.

## Restore

1. Clone this repository and open its root directory.
2. Restore the original challenge files under `dataset/train` and
   `dataset/test` from your separate dataset backup.
3. Create a Python 3.12 virtual environment and install dependencies using
   the setup commands in `code/business_entity_resolution/README.md`.
4. Create `output/` as needed. Regenerate token caches, candidate sidecars,
   and feature shards using the README pipeline commands before continuing
   training or prediction. Preserve the documented frozen blocking settings
   and use `work/aliases_full.tsv` for full-data runs.

Datasets, the virtual environment, generated submission files, large arrays,
candidate/feature caches, and AWS upload archives are intentionally excluded.
This checkpoint restores code and the saved development model; it does not
restore the expensive intermediate computations or any remote AWS state.

`work/dev/after_blocking.ps1` is preserved as a historical run script. It
contains a process ID from the original session; use the individual README
commands when restoring on another machine.
