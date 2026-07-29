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
- `dataset/raw_zips/dataset_housei_v3.zip` (superseded by `_v4`; only
  `housei_004_sketch.jpg` differed from v2, replacing a ラフ layout sketch
  with the correct 下絵)
- `dataset/raw_zips/dataset_housei_v4.zip` (current housei source; arrived
  2026-07-28 fixing a page-extraction bug on the user's side, same bug that
  affected gakuen/fitness below. Hashed against v3: 10 sketch files
  (`housei_001/002/003/004/006/007/008/010/011/012_sketch.jpg`) differ, not
  just `housei_004` this time — notably includes `010`/`011`/`012`, the
  page cluster previously flagged as a distinct high-residual-misalignment
  group (see `doc/raw_dataset_extraction_knowledge.md`, "housei"); worth
  re-checking whether that finding still holds once re-run against this
  archive. Same 58 entries, backslash-separated internal paths
  (`dataset_housei\housei_001_koma.jpg`, already handled by the existing
  `read_zip_member` fallback), `--zip-root dataset_housei`)
- `dataset/raw_zips/dataset_ako5_koma_v2.zip` (ako5ver2 panel-border layer;
  original `dataset_ako5_koma.zip` arrived 2026-07-26, but was destroyed by
  the same SCP-through-root-symlink incident as gakuen (below) on 2026-07-28
  — a page-extraction bug on the user's side affected ako5ver2, hamlabi, and
  gakuen, and corrected replacements were re-delivered to the same root
  symlink paths for all three. This `_v2` archive is the only copy; there is
  no recoverable v1. Flat zip layout (changed from the old `dataset_ako5/`
  prefix), use `--zip-root ""`. 150 entries (up from 147) — page count/content
  changes not yet reconciled against the panel-detection results already
  produced from the destroyed v1 (`results/ako5ver2_koma_panels_20260728_clean.csv`,
  179 panels); treat that CSV as based on stale/superseded source data until
  re-run against this archive)
- `dataset/raw_zips/dataset_hamlabi_koma_v2.zip` (hamlabi panel-border layer;
  same incident and same-day replacement as ako5ver2 above. Flat zip layout
  (changed from `dataset_hamlabi/` prefix), use `--zip-root ""`. Same entry
  count as before (43) but restructured, so likely a targeted content fix
  rather than added pages; existing
  `results/hamlabi_koma_panels_20260728_clean.csv` (64 panels) should
  likewise be treated as stale until re-run against this archive)
- `dataset/raw_zips/dataset_fitness_koma.zip` (superseded by `_v2`; arrived
  2026-07-26 as `dataset_kurip.zip`, renamed 2026-07-27 — do not use the
  `kurip` name going forward, including in any new results/output filenames
  for this source)
- `dataset/raw_zips/dataset_fitness_koma_v2.zip` (current fitness panel-border
  source; arrived 2026-07-28 as `dataset_kurip_v4.zip` fixing the same
  page-extraction bug as housei/gakuen — hashed against `dataset_fitness_koma.zip`:
  many `line`/some `sketch` files differ across most pages (not an isolated
  fix), plus a new unused `review_montage.jpg`. Flat zip layout this time
  (changed from the `dataset_kurip/` prefix `_koma.zip` had) — pass
  `--zip-root ""`, not `--zip-root dataset_kurip`)
- `dataset/raw_zips/dataset_fighting.zip`
- `dataset/raw_zips/dataset_gakuen_v2.zip` (superseded by `_v3`; this was
  itself the recovery from the SCP-through-symlink incident below, fixing
  `page0001`/`page0011`/`page0012`'s ラフ-vs-下絵 issue, but the user asked
  to cancel processing on it because the underlying page-extraction bug
  (see `housei_v4`/`fitness_koma_v2` above) wasn't fully fixed yet)
- `dataset/raw_zips/dataset_gakuen_v3.zip` (current gakuen source; arrived
  2026-07-28 fixing the fuller page-extraction bug — hashed against `_v2`:
  22 files differ, mostly `line` across nearly every page plus a few more
  `sketch` files (`0004/0005/0007/0015/0016`) and `review_montage.jpg`. Flat
  zip layout unchanged, `--zip-root ""`)

Also arrived 2026-07-28: `dataset_kazenagare.zip`, a mislabeled duplicate
upload — byte-identical to `dataset_housei_v4.zip`'s 58 members despite the
different outer name and internal `housei_NNN_*` filenames (confirmed by
the user: a naming mistake, not a real distinct source). Deleted; do not
recreate a `kazenagare` source from it.

`fitness` and `fighting` were renamed from their originally uploaded names
(one of which was a person's username) once extraction work on them started;
see `doc/dataset_status.md` for what each source is.

## Compatibility Symlinks — Discontinued (2026-07-28)

**Do not create root-level symlinks into `dataset/raw_zips/`, for any
source, under any circumstance.** This supersedes the earlier rule that
allowed them for sources with a hardcoded legacy reference. Three of them
(`gakuen`, `dataset_ako5.zip`, `dataset_hamlabi.zip`) were each destroyed
the same day by an `scp` upload that followed the symlink and overwrote the
real archive in place instead of replacing the link (see the incident note
below) — the convenience was never worth this risk, per explicit user
decision, and the remaining ones (`dataset_housei.zip`,
`dataset_ako5ver2.zip`, a dangling `dataset_kurip.zip` left over from the
`fitness` rename) were removed at the same time even though they hadn't
been hit yet.

A handful of old, largely-superseded scripts still hardcode a bare
root-relative default path (`~/dataset_ako5.zip` in
`align_pairs.py`/`match_ako5_regions.py`/`extract_ako5_region_tiles.py`/
`match_ako5_pages.py`/`prepare_ako5.py`; `dataset_hamlabi.zip` in
`run_region_search_loop.py`/`vlm_review_hamlabi_regions.py`/
`materialize_hamlabi_review_regions.py`/`match_hamlabi_regions.py`/
`finalize_region_search_loop.py`) — these are the ako5ver2 heavyweight-route
and pre-koma hamlabi tools, not part of the current active pipeline. Their
defaults are now stale (no root file to find) and calling them without an
explicit `--zip dataset/raw_zips/...` argument will simply fail to find the
file; that is the intended, safe failure mode now, not a bug to silently
paper over with a new symlink. Fix the hardcoded default in the specific
script being revived if one of these is ever needed again, rather than
recreating a root-level alias.

Always use the canonical `dataset/raw_zips/...` path explicitly in new
commands, logs, and any script's own default.

### Incident: SCP Through Root Symlinks Destroyed Three v1 Archives (2026-07-28)

Happened three times the same day, same mechanism each time: `gakuen` first,
then `ako5_koma`/`hamlabi_koma` (the latter two already had pre-existing
root-level `dataset_ako5.zip -> dataset/raw_zips/dataset_ako5_koma.zip` /
`dataset_hamlabi.zip -> dataset/raw_zips/dataset_hamlabi_koma.zip` symlinks
from 2026-07-26, `gakuen` had one auto-created on its own upload). None of
these three sources have any hardcoded root-level reference in this
project's tooling, so per the rule above all three symlinks should have
been skipped/never created. When the user `scp`'d corrected replacement
archives (fixing an unrelated page-extraction bug on their side affecting
all three sources) to the workspace root paths, each transfer followed its
symlink and overwrote the real file under `dataset/raw_zips/` in place,
rather than replacing the symlink itself. The original (pre-correction)
bytes are gone for all three; there was no way to recover or diff against
them. Renamed each resulting (corrected) archive to a `_v2` filename and
deleted both the now-pointless root symlinks and the now-redundant,
identically-overwritten original filenames under `dataset/raw_zips/`.

Lesson: a root-level symlink into `dataset/raw_zips/` is not just a
navigation convenience — any tool (`scp`, `cp`, editors) that opens the
symlink path for writing will silently write through to the real file,
since `open()` follows symlinks by default. This is exactly why the rule
above already says to skip root-level symlinks for sources with no
hardcoded reference; this incident is a concrete example of the risk that
rule exists to avoid, not just a style preference. If a root-level
compatibility symlink must exist for a source, treat the target file as
unsafe to overwrite by any means other than this project's own established
versioned-rename flow (copy new upload in as `_v<N+1>.zip`, never write
directly to an existing filename).

## Naming

Use the uploaded dataset name when it distinguishes source versions:

- keep `dataset_ako5ver2.zip`, not a lossy overwrite of `dataset_ako5.zip`
- never add a root-level symlink alias — see "Compatibility Symlinks —
  Discontinued" above

If a new version arrives, add a new archive instead of replacing the old one:

```text
dataset/raw_zips/dataset_<source>_v<N>.zip
```

Record which archive was used in every extraction log and manifest.

## Git Policy

Raw zip archives are data artifacts, not source files. Do not commit them unless
explicitly requested. Commit only scripts, manifests, QC summaries, and docs
needed to reproduce extraction decisions.
