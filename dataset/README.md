# Dataset Layout

Top-level dataset folders are grouped under this directory.

```text
dataset/
  raw/                  # Original rough/line source images (paired)
  raw_zips/              # Original paired-source zips
  pairs/                # Original 256px paired dataset
  pairs_256/            # 256px generated paired dataset
  pairs_480/            # 480px paired dataset used by current training/evaluation
  unpaired_rough_raw/   # Original zips for rough-only (no paired line art) sources
  unpaired_rough/       # Extracted rough-only sources, one subfolder per source
```

`unpaired_rough/` holds sources that have no paired line-art counterpart (see
memory `project_unpaired_data_pools.md`), kept separate from the paired
`raw/`/`raw_zips/` folders. Each source subfolder mirrors its zip's own layout
rather than a project-wide convention. First source: `unpaired_rough/skima/`
(from `unpaired_rough_raw/skima_text_removal.zip`, 626 pages) -- full manga
pages that are pencil-only (no inked line art was ever produced), with
dialogue/text auto-removed. Two subfolders: `cleaned/` (626 jpg, the
text-removed rough page -- use this for tiling) and `auto_mask/` (626 png,
the mask used for the text removal -- only needed if auto-removal turns out
to have over-erased genuine line content, not for normal tiling).

The dataset contents are ignored by git. Keep small, durable metadata or notes in
tracked files such as this README.
