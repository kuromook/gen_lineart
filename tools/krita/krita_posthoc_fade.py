"""Track O stage 2, POST-HOC probe (not registered, no bar, changes no verdict).

Question it asks: does a rough whose contrast is merely lowered uniformly already give a
positive D (Canny-based deletion) of the size Sketchy/inv/x1 gives?  fade k: g' = 255-(255-g)*k.
Same decomposition, same matched-null partners as krita_stage2_score.py.
"""
import random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import krita_stage2_score as S
sd = S.sd
sd.GT_DIR = Path(sd.GT_DIR_DEFAULT)
out = S.ROOT / "results/krita_stage2_20261104/_posthoc_fade"
tiles = {p: [l.strip() for l in open(f) if l.strip()] for p, f in S.POOLS.items()}
cache = {}
for ts in tiles.values():
    for t in ts:
        cond = sd.load_gray(S.COND / t, invert=True); gt = sd.load_gray(sd.gt_tile_path(t))
        cache[t] = (cond, gt, sd.decompose(cond, gt))
rng = random.Random(S.SEED); partner = {}
for ts in tiles.values():
    for t in ts:
        partner[t] = rng.choice([u for u in ts if S.source(u) != S.source(t)])
for k in (0.7, 0.5, 0.35, 0.25):
    d = out / f"fade_{k}"; d.mkdir(parents=True, exist_ok=True)
    for pool, ts in tiles.items():
        D, Dn, a, c, nei, drops = [], [], [], [], [], defaultdict(int)
        for t in ts:
            g = np.asarray(Image.open(S.rough_path(t)).convert("L")).astype(float)
            p = d / f"{Path(t).stem}_out.png"
            Image.fromarray(np.clip(255 - (255 - g) * k, 0, 255).astype(np.uint8)).save(p)
            parts, jp = cache[t][2], cache[partner[t]][2]
            if not (parts[0].any() and parts[2].any() and jp[0].any() and jp[2].any()):
                drops["D_undefined"] += 1; continue
            r = sd.score_tile(p, cache[t][0], cache[t][1], parts)
            rn = sd.score_tile(p, *cache[partner[t]][:2], jp)
            D.append(r["a_recall"] - r["c_survival"]); Dn.append(rn["a_recall"] - rn["c_survival"])
            a.append(r["a_recall"]); c.append(r["c_survival"])
            if not np.isnan(r["neither_ink"]): nei.append(r["neither_ink"])
        print(f"fade k={k:<4} {pool:8s} n_D={len(D)} D={np.mean(D):.4f} D_null={np.mean(Dn):.4f} "
              f"D'={np.mean(D)-np.mean(Dn):.4f} a={np.mean(a):.4f} c={np.mean(c):.4f} neither={np.mean(nei):.4f} drops={dict(drops)}", flush=True)
