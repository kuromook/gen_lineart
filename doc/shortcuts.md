# shortcuts

Known limits, in the form the foundation's fourth notice asked for:
`shortcut: <limit>, <when it is raised>`. One file so a limit cannot hide in
a docstring. Added to as tools are written; nothing is removed, only marked
raised with the date.

## stage 2 (fill segmentation on koma panels)

- `koma_windows.py` — shortcut: a panel smaller than 480px in a dimension is
  padded with background white rather than being scored at its own scale,
  raised when a panel set with many sub-480 panels is used (clip_pairs_v3's
  median min side is 1235px, so this affects few panels; the count is printed
  by every run).
- `koma_windows.py` — shortcut: overlapping predictions are averaged and then
  thresholded at 0.5, raised if a run shows the seam lines in the stitched
  output, which the IC4 sheet would make visible.
- `region_metrics.py` — shortcut: the matcher is greedy in descending IoU, not
  optimal (Hungarian), raised if any arm's verdict sits within 0.01 region-F1
  of a threshold, since greedy and optimal can differ only when two GT regions
  compete for one prediction.

## stage 1 / chainer (carried over)

- `stroke_chain.py` — shortcut: `MERGE_PX = 6.0` absorbs junction stubs up to
  about twice the stroke width, raised if a material with a different stroke
  width is used, since the stub's length scales with the width.
- `stroke_curves.py` — shortcut: `np.polyfit` emits RankWarning on
  near-collinear short segments; the class is decided by `line_rms` there, so
  it is noted rather than silenced. Raised if the piecewise fit is ever
  promoted out of "exploratory".
