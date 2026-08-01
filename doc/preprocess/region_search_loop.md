# Region Search Loop

CPU-only background loop for variable-aspect rough/line pair search.

## Purpose

Run deterministic region pair exploration while GPU training is running or
while the operator is away.

The loop does not promote candidates into a training dataset by itself. It
creates candidate CSV/JSON files and QC montages for review.

## Current Runner

Script:

- `tools/pair_extraction/run_region_search_loop.py`

Current supported dataset:

- `hamlabi`

The runner orchestrates:

- `tools/pair_extraction/match_hamlabi_regions.py`
- `tools/pair_extraction/refine_hamlabi_large_regions.py`

## CPU/GPU Separation

The runner sets:

- `CUDA_VISIBLE_DEVICES=`
- `OMP_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `MKL_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`

This keeps the pair search CPU-bound and avoids occupying the training GPU.

## State

Default state file:

- `results/region_search_loop/state.json`

Each round records:

- status
- parent profile
- parent candidate counts
- child candidate counts
- output CSV/QC paths

Completed rounds are skipped on restart unless `--force` is used.

## Outputs

Default output root:

- `results/region_search_loop/`

Example round directory:

- `results/region_search_loop/hamlabi/001_parent_balanced/`

Typical files:

- `parent_candidates.csv`
- `parent_candidates.json`
- `parent_candidates_qc.png`
- `child_balanced_candidates.csv`
- `child_balanced_qc.png`
- `child_fine_candidates.csv`
- `child_fine_qc.png`
- `run.log`

## Service Use

One-shot exploration:

```bash
systemd-run --user --unit=region-search-loop-hamlabi --collect \
  --property=WorkingDirectory=/home/sh1/deepl/lineart \
  --property=StandardOutput=append:/home/sh1/deepl/lineart/logs/region_search_loop_hamlabi.log \
  --property=StandardError=append:/home/sh1/deepl/lineart/logs/region_search_loop_hamlabi.log \
  /home/sh1/deepl/lineart/venv/bin/python \
  /home/sh1/deepl/lineart/tools/pair_extraction/run_region_search_loop.py \
  --dataset hamlabi \
  --zip /home/sh1/deepl/lineart/dataset_hamlabi.zip \
  --zip-root dataset_hamlabi \
  --out-root /home/sh1/deepl/lineart/results/region_search_loop \
  --state /home/sh1/deepl/lineart/results/region_search_loop/state.json \
  --cpu-threads 2 \
  --qc-count 80 \
  --output-long-side 768 \
  --stop-on-error
```

The runner also supports `--watch`, which sleeps and repeats. Current use should
prefer one-shot rounds until review-driven parameter updates are added.

## Review Rule

Candidate generation is not acceptance.

Human review, Codex VLM review, or another explicit review gate must approve
pairs before materialization into a training dataset.
