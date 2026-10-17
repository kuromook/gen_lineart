"""IC-j10 C1, re-measured paired: cv2 and exact EDT nesting violations on the
SAME panels. The original run summed cv2 over all 969 panels against the
exact leg's 49, so its comparison sentence was unsupported (amendment F3).
Also measures the cv2 rim distance, which the original run never did."""
import sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "tools/evaluation")
from fill_mask import fill_mask, ink_of
from koma_split import ROOT, load as load_split
from sens_fill_radius import RADII, fill_mask_exact, nesting_violations

names = sorted(n for n, _, _ in load_split()["held"])[::20]
print(f"panels: {len(names)} (every 20th of the 969 held, the same rule the run used)")
t0 = time.time()
rows = []
for k, name in enumerate(names):
    gg = cv2.imread(str(Path(ROOT) / "line" / name), cv2.IMREAD_GRAYSCALE)
    if gg is None:
        print(f"  MISSING {name}"); continue
    ink = ink_of(gg)
    cvm = {r: fill_mask(ink, r) for r in RADII}
    exm = {r: fill_mask_exact(ink, r) for r in RADII}
    cv_v, cv_w = nesting_violations(cvm, rim=True)
    ex_v, ex_w = nesting_violations(exm, rim=True)
    rows.append((name, cv_v, cv_w, ex_v, ex_w, int(cvm[RADII[0]].sum())))
    if (k + 1) % 10 == 0:
        print(f"  {k+1}/{len(names)}  {time.time()-t0:.0f}s", flush=True)

cvn = sum(r[1] for r in rows); evn = sum(r[3] for r in rows)
print(f"\npaired on {len(rows)} panels:")
print(f"  cv2   maskSize-3 : {cvn} violating pixels, worst rim {max(r[2] for r in rows):.2f}px")
print(f"  exact EDT        : {evn} violating pixels, worst rim {max(r[4] for r in rows):.2f}px")
print(f"  lattice share of cv2's count: {evn / max(cvn,1):.1%}")
print(f"  cv2 violations as a share of fill(r=2) area, median panel: "
      f"{sorted(r[1]/max(r[5],1) for r in rows)[len(rows)//2]:.5f}")
