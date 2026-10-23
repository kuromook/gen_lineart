"""Stage 2 learned arms (M0 / M1) and NC4, on the 956-panel koma table.

Written 2026-10-23, **after** the 2026-10-19 slot finished and after the
trainer's own 40-panel holdout curve had been seen. That order is disclosed
in doc/work_log.md and is the reason the two anchors below are mandatory:
nothing here may be tuned, and the bars are the ones registered 2026-10-10.

Why this file exists: `score_fill_koma.py` scores the non-learned arms and
NC1/NC2/NC3/NC5 only -- it never loads a checkpoint -- and NC4 was written
down in both that file and `scripts/train_fill_seg.py` as living in the other
one, so it existed nowhere. C2 requires it, so C2 was unreadable.

Arms (the checkpoints are read, never written)
  M0                   window only, 1ch  input
  M1                   window + whole-koma context + crop position, 4ch input
  blank                paint nothing -- recomputed HERE as anchor 1, so that
                       the mask path of this tool is checked against a number
                       produced by a different tool

Negative controls, one named error each, all derived from the SAME prediction
mask so that only the named error differs
  NC2 offbyone   panel i's prediction against panel i+1's GT, on the common
                 centre crop, with the arm scored on that same crop
                 (`arm@nc2crop`) -- amendment H1's geometry, unchanged
  NC3 hflip      the prediction mirrored left-right
  NC4 ctxshuffle **M1 run with another panel's global context.** This is the
                 reading rule for C2: M1 - M0 >= 0.03 AND M1's NC4 must drop
                 by >= 0.03, both, or the difference did not come from the
                 context stream. The shuffle is a seeded derangement over the
                 held panels, so no panel keeps its own context and the
                 assignment is reproducible
  NC5 shift50    the prediction rolled 50px. Sensitivity only, no bar

The two anchors (discipline 2 of the 2026-10-23 notice; `--anchors` runs them
alone and nothing else is read if either fails)
  A1  `blank` region-F1 on the 956 normal-stratum panels must reproduce
      **0.0084** from results/fill_koma_20261010/report_held_nc2repaired.txt
      -> the GT mask / region path here is the one that table was read on
  A2  the trainer's own 40-panel holdout at step 40000 must reproduce
      `train_M0.jsonl` exactly -- and it is reproduced by CALLING the
      trainer's `score_holdout`, not by a re-implementation of it
      -> the checkpoint is being loaded the way the trainer wrote it

Reuse, not re-derivation: the fill mask, the region matcher, the window grid
and stitcher, the split and the model all come from the modules where they
were settled (IC-j9 / IC-j11 / IC-j12 closed `MERGE_PX` and
`FILL_HALF_WIDTH_PX`; nothing here touches them).

The panel, not the padded canvas, is the unit of judgement: predictions and
GT are cropped back to the panel's own (h, w) before scoring, which is what
`score_fill_koma.py` scores on and therefore what anchor A1 compares to. The
trainer's holdout keeps the pad, which is one reason A2 is run through the
trainer's own function instead of this loop.

Usage:
  score_fill_seg_models.py --anchors
  score_fill_seg_models.py --side held --limit 0 \
      --out results/fill_koma_20261010/per_panel_held_models.csv
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
from fill_mask import fill_mask, ink_of  # noqa: E402
from koma_split import ROOT, load as load_split  # noqa: E402
from koma_windows import SIZE, grid, pad_to, stitch  # noqa: E402
from region_metrics import score_panel  # noqa: E402
from train_fill_seg import (CTX, UNet, in_channels, make_input,  # noqa: E402
                            panel_tensors, score_holdout)

OUT = Path("results/fill_koma_20261010")
CKPT_DIR = Path("results/fill_seg_20261010")
NEAR_BLANK_INK = 0.002          # score_fill_koma.py's threshold, unchanged
NC2_MIN_OVERLAP = 0.25          # H1 guard, unchanged
SHUFFLE_SEED = 20261023
A1_BLANK_F1 = 0.0084            # report_held_nc2repaired.txt, 956 normal panels
THRESH = 0.5                    # the trainer's own threshold on the stitched prob


def load_model(path, arm, device):
    ck = torch.load(path, map_location=device)
    assert ck["arm"] == arm, f"{path} says arm {ck['arm']}, asked for {arm}"
    m = UNet(in_channels(arm)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, int(ck["step"])


def ctx_of(name, root=ROOT):
    """The context channel alone: the whole koma's rough, padded then
    downscaled to CTX. Identical to `panel_tensors(...)[2]` -- asserted in
    `instrument_checks` rather than assumed -- but without the distance
    transform, because NC4 needs another panel's context 969 times."""
    rough = cv2.imread(str(Path(root) / "rough" / name), cv2.IMREAD_GRAYSCALE)
    if rough is None:
        return None
    rough, _ = pad_to(rough, SIZE, value=255)
    ctx = cv2.resize(rough, (CTX, CTX), interpolation=cv2.INTER_AREA)
    return ctx.astype(np.float32) / 255.0


@torch.no_grad()
def predict(model, arm, rough, ctx, device):
    """One panel -> a bool mask at PADDED panel size (the caller crops).

    One window per forward pass, which is the trainer's own path, so anchor
    A2 covers this code and not a faster variant of it. Batching was tried
    and REJECTED on measurement, not on taste: 4 windows per pass ran 8.3s
    against 8.7s on an 88-window panel (a single 480x480 window already
    saturates the card, so there was nothing to win) and it moved the
    stitched probability by up to 1.2e-4, which flipped 8 pixels of 15M
    across the 0.5 threshold. A free change that perturbs the mask is not
    free."""
    h, w = rough.shape
    coords = grid(h, w)
    preds = []
    for (y, x) in coords:
        xin = make_input(arm, rough[y:y + SIZE, x:x + SIZE], ctx, y, x, h, w)
        p = torch.sigmoid(model(torch.from_numpy(xin)[None].to(device)))
        preds.append(p[0, 0].cpu().numpy())
    return stitch(preds, coords, (h, w), reduce="mean") >= THRESH


def derangement(names, seed=SHUFFLE_SEED):
    """name -> the name whose context it gets. No fixed point, reproducible."""
    rng = random.Random(seed)
    src = list(names)
    for _ in range(1000):
        dst = src[:]
        rng.shuffle(dst)
        if all(a != b for a, b in zip(src, dst)):
            return dict(zip(src, dst))
    raise RuntimeError("no derangement found")


def centre_crop(m, h, w):
    y = (m.shape[0] - h) // 2
    x = (m.shape[1] - w) // 2
    return m[y:y + h, x:x + w]


def instrument_checks(device):
    """Cheap, and they fail loudly: the ctx shortcut must equal the trainer's
    own context channel, and the derangement must have no fixed point."""
    split = load_split()
    held = [n for n, _, _ in split["held"]]
    for name in held[:3]:
        t = panel_tensors(name)
        c = ctx_of(name)
        assert t is not None and c is not None, name
        assert np.array_equal(t[2], c), f"ctx shortcut differs from the trainer: {name}"
    print("IC  ctx_of == panel_tensors()[2] on 3 panels: PASS")
    d = derangement(held)
    assert len(d) == len(held) and all(k != v for k, v in d.items())
    assert len(set(d.values())) == len(held), "derangement is not a permutation"
    print(f"IC  NC4 derangement over {len(held)} held panels, 0 fixed points: PASS")
    return True


def anchor_a2(device, step=40000):
    """The trainer's 40-panel holdout, reproduced by calling the trainer's own
    scorer. Compared against `train_<arm>.jsonl`, which is NOT a verdict
    quantity (notice section 6) -- it is used here only to prove the
    checkpoint loads as the trainer wrote it."""
    split = load_split()
    held = [n for n, _, _ in split["held"]]
    ok = True
    for arm in ("M0", "M1"):
        path = CKPT_DIR / f"{arm}_step{step}.pt"
        want = None
        for line in open(CKPT_DIR / f"train_{arm}.jsonl"):
            import json
            r = json.loads(line)
            if r["step"] == step:
                want = r
        model, got_step = load_model(path, arm, device)
        assert got_step == step, (got_step, step)
        agg, _ = score_holdout(model, held, arm, device, limit=40)
        good = all(abs(agg[k] - want[k]) < 1e-9 for k in
                   ("region_f1", "pixel_iou", "pred_area_frac", "gt_area_frac"))
        ok &= good
        print(f"A2  {arm}_step{step} through the trainer's own score_holdout, "
              f"40 panels: region-F1 {agg['region_f1']:.10f} "
              f"(jsonl {want['region_f1']:.10f})  pixel IoU "
              f"{agg['pixel_iou']:.10f} (jsonl {want['pixel_iou']:.10f})  "
              f"-> {'PASS' if good else 'FAIL'}")
        del model
        torch.cuda.empty_cache()
    return ok


def anchor_a1(limit=0):
    """`blank` on the 956 normal-stratum panels, through THIS tool's mask
    path, against the number the non-learned table was read on."""
    split = load_split()
    names = [n for n, _, _ in split["held"]]
    if limit:
        names = names[:limit]
    vals = []
    for k, name in enumerate(names):
        g = cv2.imread(str(Path(ROOT) / "line" / name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        gt_ink = ink_of(g)
        if gt_ink.mean() < NEAR_BLANK_INK:
            continue
        gt = fill_mask(gt_ink)
        vals.append(score_panel(np.zeros(gt.shape, bool), gt)["region_f1"])
        if (k + 1) % 200 == 0:
            print(f"  A1 {k + 1}/{len(names)}", flush=True)
    f1 = float(np.mean(vals))
    good = abs(round(f1, 4) - A1_BLANK_F1) < 5e-5
    print(f"A1  blank region-F1 on {len(vals)} normal-stratum panels: "
          f"{f1:.6f} (report_held_nc2repaired.txt: {A1_BLANK_F1}) "
          f"-> {'PASS' if good else 'FAIL'}")
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", default="held", choices=("held", "train"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--step", type=int, default=40000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--anchors", action="store_true",
                    help="run the instrument checks and both anchors, then stop")
    ap.add_argument("--anchor-limit", type=int, default=0,
                    help="A1 over the first N panels only (smoke; not the anchor)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}  step {args.step}")
    ok = instrument_checks(device)
    if args.anchors:
        ok &= anchor_a1(args.anchor_limit)
        ok &= anchor_a2(device, args.step)
        print(f"\nANCHORS {'PASS' if ok else 'FAIL'} -- "
              f"{'the table may be read' if ok else 'nothing else is read'}")
        sys.exit(0 if ok else 1)

    split = load_split()
    names = [n for n, _, _ in split[args.side]]
    fam = {n: f for n, _, f in split["train"] + split["held"]}
    if args.limit:
        names = names[:args.limit]
    shuffle = derangement([n for n, _, _ in split["held"]])

    models = {arm: load_model(CKPT_DIR / f"{arm}_step{args.step}.pt", arm, device)[0]
              for arm in ("M0", "M1")}
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out or OUT / f"per_panel_{args.side}_models.csv")
    rows, pending = [], None
    for k, name in enumerate(names):
        t = panel_tensors(name)
        if t is None:
            continue
        rough, fill, ctx, orig = t
        h0, w0 = orig
        gt_ink = ink_of(cv2.imread(str(Path(ROOT) / "line" / name),
                                   cv2.IMREAD_GRAYSCALE))
        r_ink = ink_of(cv2.imread(str(Path(ROOT) / "rough" / name),
                                  cv2.IMREAD_GRAYSCALE))
        gt = fill[:h0, :w0]
        preds = {
            "blank": np.zeros((h0, w0), bool),
            "M0": predict(models["M0"], "M0", rough, ctx, device)[:h0, :w0],
            "M1": predict(models["M1"], "M1", rough, ctx, device)[:h0, :w0],
        }
        other = ctx_of(shuffle.get(name, name))
        if other is not None:
            preds["nc4_ctxshuffle:M1"] = predict(
                models["M1"], "M1", rough, other, device)[:h0, :w0]
        meta = {"panel": name, "family": fam.get(name, "?"),
                "gt_ink_ratio": round(float(gt_ink.mean()), 6),
                "near_blank": int(gt_ink.mean() < NEAR_BLANK_INK),
                "nc4_ctx_from": shuffle.get(name, "")}
        for arm, p in preds.items():
            rows.append({**meta, "arm": arm,
                         **score_panel(p, gt, r_ink, gt_ink)})
        for arm in ("M0", "M1"):
            p = preds[arm]
            for nc, mutated in (("nc3_hflip", p[:, ::-1]),
                                ("nc5_shift50", np.roll(p, (50, 50), (0, 1)))):
                rows.append({**meta, "arm": f"{nc}:{arm}",
                             **score_panel(mutated, gt, r_ink, gt_ink)})
        if pending is not None:
            pmeta, ppreds, pshape = pending
            ch, cw = min(pshape[0], h0), min(pshape[1], w0)
            ov = (ch * cw) / float(min(pshape[0] * pshape[1], h0 * w0))
            if ov >= NC2_MIN_OVERLAP:
                gtc = centre_crop(gt, ch, cw)
                gic = centre_crop(gt_ink, ch, cw)
                ric = centre_crop(r_ink, ch, cw)
                for arm in ("M0", "M1"):
                    rows.append({**meta, "arm": f"{arm}@nc2crop",
                                 **score_panel(centre_crop(preds[arm], ch, cw),
                                               gtc, ric, gic)})
                    rows.append({**meta, "arm": f"nc2_offbyone:{arm}",
                                 **score_panel(centre_crop(ppreds[arm], ch, cw),
                                               gtc, ric, gic)})
        pending = (meta, {a: preds[a] for a in ("M0", "M1")}, (h0, w0))
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(names)}", flush=True)

    fields = sorted({key for r in rows for key in r})
    head = ["panel", "family", "arm", "near_blank", "gt_ink_ratio", "nc4_ctx_from"]
    fields = head + [f for f in fields if f not in head]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows for {len(names)} panels: {out_path}")


if __name__ == "__main__":
    main()
