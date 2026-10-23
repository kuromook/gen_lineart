# Track M work log

## 2026-10-24 -- the tree was opened

Cut from `foundation` at `e9d680b` by user approval, after the proposal
(`doc/track_m_colour_manuscripts_proposal_20261024.md`) was written on the
foundation side. Nobody has worked in this tree yet; this entry records what it
was handed.

**The author answered three things on approval, and the first one changes the
order of the candidates**:

1. the structure is **a base layer that indicates the area, with 濃淡 layers on
   top of it** (with some exceptions), and **the base is what to go after**,
   because finding areas is the current target
2. **single illustrations go in** -- a colour-painted manga page is likely less
   tractable than a single illustration, which is made more carefully
3. open the tree

Consequence of (1): the area division is **already a layer**, not something to
recover from composited pixels. Candidate 1 of the proposal (quantise the
composite, take connected components) is handicapped by exactly the 濃淡 that
sits on top; candidate 2 (read the layer structure) becomes the main line. **The
reordering comes from the author's process knowledge and not from a measurement,
and the registration has to say so.** What must not follow from it is deciding
which layer is the base **by its name** -- that is the 2026-10-23 pattern (four
times in one day) and the condition has to be measured and registered.

Verified at birth, not assumed:

- `git check-ignore -q data/a.png` passes in this tree -- the rule arrived with
  `f7372e6` and this branch was cut after it. 10 of the 14 trees on the shelf
  did not have it on 2026-10-24
- `doc/CURRENT.md` was replaced with a pointer the same day, before any session
  read it (audit check 8; Track J found its copy 14 days and 3,682 lines behind)
