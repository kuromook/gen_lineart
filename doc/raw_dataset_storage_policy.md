# Raw Dataset Storage Policy

## Canonical Location

Store uploaded raw dataset archives under:

```bash
dataset/raw_zips/
```

This keeps workspace root from accumulating large upload artifacts while
preserving the original archives next to the derived `dataset/` outputs.

Current archives:

- `dataset/raw_zips/dataset_ako5ver2.zip`
- `dataset/raw_zips/dataset_hamlabi.zip`
- `dataset/raw_zips/dataset_fitness_v4.zip`
- `dataset/raw_zips/dataset_housei.zip`
- `dataset/raw_zips/dataset_fighting.zip`

`fitness` and `fighting` were renamed from their originally uploaded names
(one of which was a person's username) once extraction work on them started;
see `doc/dataset_status.md` for what each source is.

## Compatibility Symlinks

Root-level symlinks are allowed, but only when an existing script or historical
command literally references the root-level filename:

- `dataset_ako5ver2.zip -> dataset/raw_zips/dataset_ako5ver2.zip`
- `dataset_ako5.zip -> dataset/raw_zips/dataset_ako5ver2.zip`
- `dataset_hamlabi.zip -> dataset/raw_zips/dataset_hamlabi.zip`

For sources with no hardcoded root-level reference (`housei`, `fighting`,
`fitness`), skip the root-level symlink entirely; every current tool for them
is invoked with an explicit `--zip dataset/raw_zips/...` path, so the alias
would add nothing.

Prefer the canonical `dataset/raw_zips/...` path in new commands and logs.
Keep symlinks only as compatibility aliases, never as the primary reference.

## Naming

Use the uploaded dataset name when it distinguishes source versions:

- keep `dataset_ako5ver2.zip`, not a lossy overwrite of `dataset_ako5.zip`
- add aliases only when needed for old scripts

If a new version arrives, add a new archive instead of replacing the old one:

```text
dataset/raw_zips/dataset_<source>_v<N>.zip
```

Record which archive was used in every extraction log and manifest.

## Git Policy

Raw zip archives are data artifacts, not source files. Do not commit them unless
explicitly requested. Commit only scripts, manifests, QC summaries, and docs
needed to reproduce extraction decisions.
