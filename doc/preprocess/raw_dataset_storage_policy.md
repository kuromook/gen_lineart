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

- `dataset/raw_zips/dataset_clip_pairs_v2.zip` (arrived 2026-08-21 at repo
  root as `dataset_clip_pairs.zip`, renamed `_v2` per this file's own
  versioning rule -- **the extraction tool's reply to this project's
  koma-layer request** (`doc/preprocess/clip_pairs_extraction_feedback_20260821.md`),
  delivered via a new `inbox/` convention (`inbox/reply_clip_pairs_20260821.md`;
  see `doc/work_log.md`'s 2026-08-21 "Extraction Tool Reply" entries for the
  full digest). 11.4GB, 5360 entries, SHA-256
  `ef087dff29d0fa86f75bc2b4b18ecde8e788a60f11a62385d521b4cd250885bc`
  (verified against the sender's own checksum). **This IS the final
  QC-enhanced version** -- 1644 koma-layer files (`_koma.jpg`), and every
  pair's `manifest.json` entry (not `koma_manifest.json`, a separate
  per-page koma-only file in the same folder -- an initial same-session
  mixup checking the wrong file briefly suggested QC was missing; corrected
  after the sender supplied a direct verification command and checksum) has
  `qc` (`pair_quality`/`content_fingerprint`/`grid_correlation`/coverage
  fractions/etc.), `source_path`, `source_size`. Cross-source QC table at
  `clip_pairs/clip_pairs_qc.csv` inside the zip. **1272 pairs** is the
  confirmed correct starting pool (`qc=ok` + unique/`is_primary` + has
  koma) -- the sender's own reply had a stale pre-fix number (1151) in one
  place that they corrected on follow-up.

  **Known caveat for next use**: as of this delivery, only 289 of 2644
  total pages were re-extracted with the tool's latest (bug-fixed) code;
  the remaining 1451 pages still reflect the older extraction. A full
  2644-page re-extraction with the fixed code is planned by the sender for
  the next machine-availability window (~Monday from 2026-08-21); some
  `ok`-flagged pages may still improve then (their own example:
  `066_2024_housei page0007` coverage 0.208 -> 0.561 under the newer code
  even though it was already `ok`). `content_fingerprint` is designed to
  let this project detect which specific pairs changed content between
  versions. **Safe to start pipeline work on the current 1272-pair pool
  now** per the sender's own explicit confirmation -- not blocked on
  Monday's re-extraction, just expect some pairs to need re-processing
  afterward.
- `dataset/raw_zips/dataset_clip_pairs.zip` (arrived 2026-08-17 at repo root,
  moved into place 2026-08-19 -- 8.0GB, 3590 entries. **This is real paired
  rough/line data**, the long-awaited item this project has been blocked on
  since the 2026-08-08 ControlNet-LoRA-drop decision (`doc/work_log.md`),
  originally estimated ~2026-08-31; arrived ~2 weeks early. Auto-extracted by
  the user's own tooling from CLIP STUDIO `.clip` files: `clip_pairs\<NNN>_
  <year>_<project>\<subproject>\<...>_page<NNNN>_{line,sketch}.jpg`, one
  `manifest.json` per subproject folder (74 top-level project folders, each
  possibly containing multiple subprojects) recording per-page `line`/
  `sketch` filenames plus rich per-page alignment diagnostics (`line_bbox`/
  `sketch_bbox`/centroids/`normalized_centroid_distance`/
  `block_density_correlation`/`aligned` bool) -- this alignment data is
  already computed by the extraction tool, unlike every prior source in this
  project which needed the region-matching pipeline built from scratch.
  Covers the artist's manuscript history 2015-2024, including subprojects
  that share names with several already-known sources (`ako5`/`ako6`/`ako7`/
  `ako3B`/`ako4`/`housei`/`fitness`/`hamlabi`). Not yet run through this
  project's own review/tiling pipeline. **Not a monolithic source** -- 74
  top-level project folders, each with its own subproject(s) and
  `manifest.json`; treat per-project, not as one blob.

  **2026-08-20 name-collision investigation** (dedup/overlap check on the
  four folders whose names matched existing sources, per user request):
  - `011_2017_ako5` -- **confirmed a different manuscript**, not the known
    `ako5ver2` source, despite the name match. Canvas 6543x7016 (near-square,
    single-illustration aspect) vs `ako5ver2`'s 4961x7016; direct visual
    check of the shared `page0001` filename shows a bikini pin-up
    illustration here vs. `ako5ver2`'s multi-panel train-scene manga page --
    unrelated content. 66 pages across 7 subfolders
    (`ako5`/`color`/`done`/`done2`/`done3`/`done4`/`done5`). Catalogue as new
    material, do not dedup against `ako5ver2`.
  - `066_2024_housei\kazenagare` -- same manuscript project as the known
    `housei` archive, but **confirmed degraded, not just auto-extracted**:
    only 10 of the known archive's 18 pages present (0003/0005/0013-0018
    missing), and `line` output file sizes are systematically ~1/3 the known
    archive's across nearly every page -- visual check of `page0001`'s
    `line` output shows an almost-blank layer fragment (a tiny window/rim
    doodle) instead of the known archive's full page content, i.e. the
    auto-selector picked the wrong/an incomplete layer. Also: this folder's
    `sketch`/`line` page-number filenames do **not** line up 1:1 with the
    known archive's page index (a page0001-numbered sketch here showed
    unrelated content to `page0001`'s line output) -- any future per-page
    comparison against the known archive needs re-indexing, not assumed
    alignment by filename. **Do not use this housei batch without full
    review; the existing `housei_koma_subregion_refined` extraction remains
    better and should stay the training source.**
  - `069_2024b_fitness\kurip` -- same source (folder is still internally
    named `kurip`, the pre-rename name, consistent with being the same
    manuscript), **quality comparable to the existing extraction, no
    degradation found**: several `sketch` files are byte-identical to the
    known `fitness_v4` archive (pages 0004/0005/0006/0009), `line` file
    sizes are close (not the housei-style 3x gap), and a direct visual check
    of `page0001` shows the same character/pose/composition. Worth
    inventorying as a possible supplemental/cross-check source later, not
    urgent.
  - `070_2024b_hamlabi\works` (18 pages) -- `page0002`'s `line` and `sketch`
    are **byte-identical (MD5)** to `dataset_hamlabi_koma_v2.zip`'s
    `hamlabi_page0002_{line,sketch}.jpg`. Same manuscript, same extraction
    result -- effectively a re-export, not independently degraded, but also
    not new content.
  - `063_2023b_hamlabi\works` (18 pages, a second hamlabi-named folder the
    user had not explicitly flagged) -- **byte-identical to
    `070_2024b_hamlabi\works`** (all 36 files match by CRC after
    normalizing filenames), i.e. the *same content duplicated under two
    different year-tags inside `clip_pairs` itself*, not two distinct
    hamlabi works. Treat as one source when inventorying, not two.
  - All four folders carry genuine per-page `line`+`sketch` pairing with an
    `alignment` diagnostic block (`aligned: true/false`, roughly half-and-half
    in the samples checked) -- real attempted pairing, unlike
    `psd_line`/`comicstudio_line`.

  **2026-08-21 inventory of the remaining `ako*`-family folders** (21 raw
  folders with no name match against any existing source). CRC-based
  cross-comparison first (cheap, exact-byte-match, no image content needed)
  found these collapse to **~12 distinct works**, several duplicated
  wholesale or near-wholesale across multiple year-tagged folder names
  inside `clip_pairs` itself (same pattern as the `hamlabi` 2023/2024
  duplicate found above) -- use only the most-complete representative of
  each cluster, not all of them:

  | work | representative folder/subproject | pairs (aligned=true) | duplicate/superseded entries (same content, skip) |
  |---|---|---:|---|
  | ako6 | `018_2018_ako6\ako6` (+9 more subfolders, 115 images total) | 24 (18) | -- |
  | ako7 | `026_2019_ako7\ako7` | 38 (35) | `031_2019b_ako7` (100% identical); `019_2018_ako7` (11 pairs, subset, 59% overlap) |
  | ako8 | `032_2019b_ako8\ako8` | 18 (14) | -- |
  | ako9 | `037_2020_ako9\ako9` | 23 (7) | -- |
  | ako10 | `039_2020b_ako10\aco10` | 12 (3) | `052_2022b_ako10` (100% identical) |
  | akogoods | `049_2022_akogoods\akogoods` | 41 (6) | `054_2022b_akogoods` (100% identical); `041_2021_akogoods` (98% overlap, near-dup) |
  | akokate | `042_2021_akokate\akokatei` | 30 (8) | -- |
  | akocult | `053_2022b_akocult\akocult` | 33 (17) | `048_2022_akocult` (26 pairs, 100% subset of the above) |
  | akocultB | `061_2023b_akocultB\akocultB` | 23 (15) | -- (confirmed a genuinely different work from `akocult`, not a near-dup: different cast/setting) |
  | ako3 | `064_2024_ako3\ako3` | 27 (13) | `060_2023b_ako3` (14 pairs, 93% subset of the above) |
  | ako3B | `065_2024_ako3B\ako3B` | 31 (15) | `067_2024b_ako3B` (98% identical) -- confirmed genuinely different work from `ako3`, not a near-dup |
  | ako4 | `072_2024c_ako\ako4` | 47 (25) | `068_2024b_ako4` (14 pairs, 86% subset of the above) |

  Also found (low-priority oddity, not investigated further): 8 files are
  byte-identical between `018_2018_ako6` and `019_2018_ako7`/`026_2019_ako7`
  (mostly `_sketch.jpg`, one `_line.jpg` pair) -- real content reuse across
  two nominally-different works, not a hash coincidence; cause not
  determined (possible cross-project filing overlap during production).

  **Visual spot-check** (1-3 sample pages per work) confirmed all 12 are
  genuine sequential multi-panel manga (not single illustrations like
  `011_2017_ako5` turned out to be -- `akokate` samples are even annotated
  with real published volume/page numbers). But extraction quality varies
  sharply and **the manifest's own `alignment.aligned` field is a reliable
  predictor of real per-page success, confirmed by direct visual check**:

  - **Good, `aligned:true` pairs are usable as-is**: `ako7`, `ako8`,
    `akokate` (aligned=true samples), `akocult`, `akocultB`, `ako3`, `ako3B`
    all gave clean, well-corresponding sketch/line pairs on `aligned:true`
    samples.
  - **`ako9`, `ako10`, `akogoods` have a much higher real failure rate than
    their low `aligned_true` fractions alone suggest** -- sampled `line`
    outputs were severely broken even on nominally-picked pages: near-blank
    line layers against fully-drawn sketches (`ako9`, `ako10`), or
    solid-black-fill/shadow fragments with no actual stroke linework
    instead of real line-art (`akogoods`, despite the "goods" name --
    content itself is normal sequential manga, not merchandise). These
    three sources need more than just `aligned:true` filtering before use;
    treat as needing manual review, similar to `housei`.
  - **`ako4` has a distinct third failure mode**: sampled page's `sketch`
    side is almost blank (a stray gesture line or two) while `line` is a
    fully finished, screentone/solid-black page -- looks like the
    auto-selector grabbed an early layout/thumbnail layer as "sketch"
    instead of an actual pencil rough. Needs manual review before use, same
    caution as `housei`.

  **Practical rule going forward for any `clip_pairs` use**: filter to
  `alignment.aligned == true` entries at minimum (this alone would have
  excluded nearly every broken sample found above), and still spot-check
  even `aligned:true` entries per-source before bulk training use -- per
  both the `housei` page-numbering mismatch found earlier and `akokate`'s
  own `aligned:false` sample showing a genuine partial-extraction failure
  (only a background panel extracted, not the page's actual content).

  **2026-08-21 (later): full inventory of the remaining 48 `clip_pairs`
  folders (everything with no name collision against a known source)**.
  Three sub-findings, then a full-archive close-out:

  - **The 3 `*_4th` folders (`047_2022_4th`/`051_2022b_4th`/`059_2023b_4th`)
    are the same manuscript as the already-processed
    `dataset/raw_zips/dataset_4th_koma.zip`, not new pages.** Per-page CRC
    check against the known 36-page archive found most page numbers match
    exactly, but several (21/22/26/27/31/32, plus most of `047`'s smaller
    set) have non-matching bytes at the *same* page number. Direct visual
    comparison of `page0021`'s `line` output confirmed this is the same
    `housei`-style degraded-extraction failure, not new content: the known
    archive's page0021 is a full page (hospital room, two characters,
    dialogue, background building); `clip_pairs`' version of the same page
    is almost entirely blank (only a bed frame outline and a prosthetic-leg
    object survive). **Do not use `clip_pairs`' `4th` folders; the existing
    `dataset_4th_koma.zip` extraction remains better.**
  - **`022_2018_skima10` (6 images) / `035_2019b_skima2` (4 images) are
    unrelated to the known `skima` unpaired-rough pool**
    (`dataset/unpaired_rough/skima`, 626 pages, rough-only by design, no
    line-art ever produced for it -- see `doc/work_log.md`, 2026-07-31).
    These two `clip_pairs` folders are a completely different, much smaller
    source that happens to share the "skima" nickname, and unlike the known
    pool they *do* have real sketch+line pairs. Negligible size either way
    (10 pages total combined) -- not worth further investigation.
  - **The remaining 43 folders collapse to ~29 distinct works** after the
    same CRC-based dedup pass (representative folder chosen per cluster):
    `gakusai3`(001)+`gakusai1-3DL`(013, superset, representative)`;
    `comics4`(021, itself a 3-work compilation: china/mars/underworld,
    the `underworld` sub-work 100%-duplicates standalone `025`);
    `toramusume1`(023)+`toramusume2`(024, superset, representative);
    `UNI`(036)/`UNI_ml`(038, partial-overlap sibling, *not* a pure
    duplicate -- confirmed same series/cast on sampled pages, but with at
    least one page, the title/cover page, duplicated verbatim across both
    exports); `nurse`(040/045/057, three near-identical exports, `045`
    representative). Full per-work manifest pair/aligned counts recorded in
    this investigation's session log (`doc/work_log.md`, 2026-08-21 later
    entry) rather than duplicated here.

  **Visual spot-check across all ~29 distinct works** (parallelized, one to
  two sample pages per work): confirmed genuine varied content -- roughly
  40% sequential multi-panel manga, the rest single illustrations,
  character reference/turnaround sheets, or (one case, `024_toramusume2`)
  a character design sheet. No systematic content-type surprise like
  `ako5`'s. Quality-wise, most `aligned:true` samples were clean, but this
  pass found **the first confirmed counterexamples to "`aligned:true` is a
  reliable success predictor"**:
  - `014_2017_kacho`: an `aligned:true` sample where both `sketch` and
    `line` show large unfinished white-void patches -- looks like an
    intermediate/incomplete production stage got selected on both sides,
    not a clean pair.
  - `029_2019_fringe3`: an `aligned:true` sample where `sketch` and `line`
    are **entirely different documents** -- sketch is a hospital/dialogue
    manga page, line is a *color instruction sheet* (character standing
    poses + text annotations). A new failure pattern beyond the ones
    already catalogued (blank layer / wrong-layer-type / partial-page):
    **completely unrelated document mispaired**, undetected by the
    alignment diagnostic.
  - `043_2021_hero`: an `aligned:true` sample (`page0006`) with sketch a
    full dialogue page but line almost entirely blank -- the same
    near-blank-line failure mode seen elsewhere, but this time on an
    `aligned:true`-flagged entry. Its own `aligned:false` sample
    (`page0002`) was even worse: sketch and line show *completely
    unrelated scenes* (a 3-character dialogue close-up vs. two children
    swimming in a cave) -- looks like a page-index mixup, not just a
    layer-selection failure.
  - `073_2024c_mayer`: `aligned:true` sample has loosely-stylized,
    debatable sketch/line correspondence (mountain-vista sketch vs. a
    city-silhouette line rendering); its `aligned:false` sample reproduces
    the familiar severe near-blank-line pattern.

  **Revised standing rule**: `alignment.aligned == true` remains a useful
  coarse pre-filter (it removes most, not all, broken pairs) but is **not
  sufficient on its own** -- confirmed false positives exist across
  multiple unrelated works (`kacho`/`fringe3`/`hero`/arguably `mayer`), not
  just isolated noise. Any bulk use of `clip_pairs` still needs per-page (or
  at least per-source-sampled) visual or automated content-correspondence
  verification before training, not just the manifest's own alignment flag.

  **This closes the full `clip_pairs` (74-folder) name-collision and
  inventory investigation.** Summary disposition: usable-as-is candidates
  (pending the per-page verification above) are `ako5`/`ako7`/`ako8`/
  `akokate`/`akocult`/`akocultB`/`ako3`/`ako3B`/`fitness`, plus most of the
  29 newly-inventoried works (content-type and rough quality now known per
  work, see the session log for the full table); needs-manual-review or
  higher-failure-rate sources are `housei`/`ako9`/`ako10`/`akogoods`/`ako4`
  and now also `kacho`/`fringe3`/`hero`/`mayer` from this later pass;
  no-new-content sources are both `hamlabi` entries and the 3 `4th`
  entries. External feedback for the extraction tool covering the new
  "unrelated document paired" failure mode should be added to
  `doc/preprocess/clip_pairs_extraction_feedback_20260821.md` if/when that
  report is revised.
- `dataset/raw_zips/dataset_comicstudio_line.zip` (arrived 2026-08-17/18 at
  repo root, moved into place 2026-08-19 -- 4.9GB, 3214 entries, 88 project
  folders spanning 2008-2014+pending). Line-only, **unpaired** (`"aligned_
  pair": false`, `"alignment": null` throughout the per-project
  `manifest.json` files) -- same extraction tool family as `dataset_clip_
  pairs.zip` but for CLIP STUDIO sources where only a line layer was
  classified/extracted (`"classification": "線画のみ"`), no sketch
  counterpart. Usable as more line-domain-LoRA training material (same role
  as `dataset_psd_line_v2.zip`/`_v3.zip` below), not for paired/ControlNet
  training. Not yet run through this project's pipeline.
- `dataset/raw_zips/dataset_psd_line_v3.zip` (arrived 2026-08-18/19 at repo
  root as a same-named `dataset_psd_line.zip`, renamed to `_v3` on arrival
  per this file's own naming rule to avoid colliding with the existing `_v2`
  below -- **not a re-encoding of `_v2`, a genuinely new/larger extraction
  run**: confirmed by hash and content diff, 735 PSD documents scanned (was
  731) yielding 317 line-only outputs (was 275), still line-only by design
  (`pairs_in_one_document: 0`, `cross_document_pairs: 0`). Same `format:
  photoshop` / `extract_line_and_sketch` tool as the original `_v1`/`_v2`.
  Not yet checked for filename-encoding issues (the `_v1`->`_v2` mojibake
  fix below was specific to that batch; verify before assuming `_v3`'s
  member names are clean). Not yet run through this project's pipeline.
- `dataset/raw_zips/dataset_psd_line.zip` (arrived 2026-08-08, line-only batch
  from a new source: 731 PSD documents auto-scanned by the user's own
  `extract_line_and_sketch` tool on another PC, 275 line layers extracted
  successfully (0 sketch/pairs -- line-only by design, per user; pairs are
  a separate planned follow-up extraction). Zip member filenames are
  mojibake (Shift-JIS/CP932 bytes misread as CP437 by the zip format's
  default non-UTF8-flag fallback) -- e.g. `0001_名称未設定 1_line.png`
  appeared as `0001_хРНчз░цЬкшинхоЪ 1_line.png`. Content (image bytes,
  `manifest.json`, `index.tsv`) is unaffected, only the zip directory
  entry names. Fixed by matching each garbled entry to `index.tsv`'s
  correctly-UTF-8-encoded filename column via the numeric prefix (e.g.
  `0001_`), which survives the corruption since digits are encoding-
  invariant -- all 275 entries matched cleanly, no ambiguity. Repackaged
  with correct names as `dataset/raw_zips/dataset_psd_line_v2.zip`
  (standard Python 3 `zipfile` write, proper UTF-8 flag). Kept the
  original `dataset_psd_line.zip` for audit; use `_v2` for any extraction
  work. `manifest.json` also records 415 skipped (no line layer name
  match), 26 ng, 4 errors (`aggdraw` package missing on the user's
  extraction tool for vector-shape PSDs -- a fix for their side, not
  ours), 11 excluded composite, 4 excluded tone. Not yet run through this
  project's own extraction/tiling pipeline.

Also arrived 2026-08-19 at repo root: `dataset_4th.zip` -- checked
per-file CRC against `dataset/raw_zips/dataset_4th_koma.zip`: 99/100 members
byte-identical (only `manifest.json` differs), confirming this was a
duplicate re-upload of already-archived, already-processed data (koma
panels + tiles saved 2026-08-01, `dataset/pairs_480/valid_train_4th_koma_
20260801.txt`), not new material. Deleted from the repo root the same day
per user confirmation; the real archive remains at
`dataset/raw_zips/dataset_4th_koma.zip`.

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
