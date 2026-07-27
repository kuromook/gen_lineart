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
- `dataset/raw_zips/dataset_housei.zip` (superseded, kept for reference)
- `dataset/raw_zips/dataset_housei_v2.zip` (superseded by `_v3`; adds per-page
  `*_koma.jpg` panel-border layer and `koma_manifest.json`; zip root changed
  from flat to `dataset_housei/`, so tools need `--zip-root dataset_housei`)
- `dataset/raw_zips/dataset_housei_v3.zip` (current housei source; only
  `housei_004_sketch.jpg` differs from v2, replacing a ラフ layout sketch
  with the correct 下絵 — confirmed by hashing every archive member)
- `dataset/raw_zips/dataset_ako5_koma.zip` (ako5ver2 panel-border layer,
  arrived 2026-07-26; integrity-checked; panel-detection pass in progress
  2026-07-27)
- `dataset/raw_zips/dataset_hamlabi_koma.zip` (hamlabi panel-border layer,
  arrived 2026-07-26; integrity-checked; panel detection completed
  2026-07-27, 64 panels across all 13 pages, pending chamfer-gate review)
- `dataset/raw_zips/dataset_fitness_koma.zip` (fitness panel-border layer;
  arrived 2026-07-26 as `dataset_kurip.zip`, renamed 2026-07-27 — do not use
  the `kurip` name going forward, including in any new results/output
  filenames for this source. Reconciled against the existing
  `dataset_fitness_v4.zip`: same 38 pages, all 76 line/sketch JPEGs
  byte-identical by md5 to `dataset_fitness_v4.zip` (internal zip root
  `dataset_kurip_v4`), so this archive is exactly that source plus a
  per-page `*_koma.jpg` panel-border layer and `koma_manifest.json`, nothing
  else changed. Internal zip root is still literally `dataset_kurip` (baked
  into the archive's own file paths, not repackaged) — pass
  `--zip-root dataset_kurip` when reading it; this is an internal parameter
  only, not a visible name, same asymmetry already accepted elsewhere in
  this project (see `fitness` section below and
  `doc/raw_dataset_extraction_knowledge.md`))
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
