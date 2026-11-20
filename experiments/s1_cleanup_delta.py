"""Track S 登録 S1(doc/work_log.md 2026-11-20「登録 S1」)の器具。

対(厳しい門)の線画の骨格を 残/動/足、下絵の骨格の 捨 に分ける。腕A(位置合わせなし)・腕B(頁全体の平行移動)、
位置合わせは S1a(残の割合の全探索)。塗り(線幅 > 15px)は S1b で4分類から外す。帰無値 N1(512px 区画ごとの移動 = ずれの上界)・N2(同じ作品の別頁の下絵 = 偶然の床)、対照 C1(線画 対 線画)。

使い方: ../lineart/venv/bin/python experiments/s1_cleanup_delta.py --work 016_2017_running --pages 4 --out results/s1_smoke --crops
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/clip_pairs_v6/clip_pairs"
QC = P / "clip_pairs_qc_v6.csv"
NAME_RULE = ROOT / "inbox/s_v6_pairs_page_name_rule_20261120.csv"
EXCL = ROOT.parent / "lineart-census/data/census_full/author_exclusions.csv"

R1, R2 = 6, 40           # 残 / 動 の境(px、登録 S1)
WHITE = 243              # Otsu を当てる画素(明るさ < 243)
SPECK = 8                # 塵(連結成分の面積 < 8)
MAXSHIFT = 64            # 腕B の打ち切り
TILE = 512               # N1 の区画
N1_STEP = 4              # N1 の移動の刻み(実装の細目)
HOLD_MARGIN = 0.10       # 判定保留: 残B < 残N2 + 0.10
FILL_W = 15              # S1b: 線幅がこれを超える線画の墨は「塗り」
FILL_PAD = 3             # S1b: 塗りに掛かる骨格を外す余白


def population():
    nr = {r["page_path"]: r for r in csv.DictReader(open(NAME_RULE, encoding="utf-8"))}
    excl = {r["page_path"] for r in csv.DictReader(open(EXCL, encoding="utf-8"))}
    out = []
    for r in csv.DictReader(open(QC, encoding="utf-8")):
        x = nr[r["source_path"]]
        if (r["source_path"] not in excl and r["pair_quality"] == "ok"
                and x["line_rules"] == "清書" and x["sketch_rules"] == "下絵"):
            out.append(r)
    return out


def find(r, key):
    return next((P / r["work_id"]).glob(f"*/{r[key]}"))


def binarize(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    t, _ = cv2.threshold(g[g < WHITE].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = (g < t).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    small = np.where(st[:, cv2.CC_STAT_AREA] < SPECK)[0]
    small = small[small > 0]
    if len(small):
        b[np.isin(lab, small)] = 0
    return b.astype(bool), int(t), int(len(small))


def fill_mask(line_b):
    """S1b: 線幅 > FILL_W の線画の墨(芯 = 内側の距離 > FILL_W/2 を半径 8 で膨らませ、墨と重ねる)。"""
    inner = cv2.distanceTransform(line_b.astype(np.uint8), cv2.DIST_L2, 5)
    core = (inner > FILL_W / 2).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    return (cv2.dilate(core, k) > 0) & line_b


def shift(b, dx, dy):
    """b を (dx, dy) だけ動かす(はみ出しは墨なし)。"""
    out = np.zeros_like(b)
    h, w = b.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = b[ys, xs]
    return out


def estimate_shift(line_sk, sk_b):
    """S1a: 線画の骨格の点(2万点、種0)が下絵の墨から R1 以内に入る割合を最大にする (dx, dy)。
    ±MAXSHIFT・2px 刻みの全探索。移動0を含む。返す resp は最大の割合、clipped は端に張り付いたか。"""
    ly, lx = np.nonzero(line_sk)
    if len(ly) == 0:
        return 0, 0, float("nan"), False
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ly), size=min(20000, len(ly)), replace=False)
    ly, lx = ly[pick], lx[pick]
    dS = dist_to_ink(sk_b)
    h, w = sk_b.shape
    best, bdx, bdy = -1.0, 0, 0
    for dx in range(-MAXSHIFT, MAXSHIFT + 1, 2):
        for dy in range(-MAXSHIFT, MAXSHIFT + 1, 2):
            # 下絵を (dx, dy) 動かす = 線画の点 (y - dy, x - dx) で距離を引く(はみ出しは遠い扱い)
            y2, x2 = ly - dy, lx - dx
            ok = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
            f = (dS[y2[ok], x2[ok]] <= R1).sum() / len(ly)
            if f > best or (f == best and abs(dx) + abs(dy) < abs(bdx) + abs(bdy)):
                best, bdx, bdy = f, dx, dy
    clipped = max(abs(bdx), abs(bdy)) == MAXSHIFT
    return bdx, bdy, float(best), clipped


def dist_to_ink(b):
    return cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)


def classify(d_at_skel):
    keep = d_at_skel <= R1
    moved = (d_at_skel > R1) & (d_at_skel <= R2)
    added = d_at_skel > R2
    return keep, moved, added


def shares(d_at_skel):
    n = max(len(d_at_skel), 1)
    k, m, a = classify(d_at_skel)
    return k.sum() / n, m.sum() / n, a.sum() / n


def arm(line_b, line_sk, sk_b, sk_sk, align):
    if align:
        dx, dy, resp, clipped = estimate_shift(line_sk, sk_b)
        sk_b, sk_sk = shift(sk_b, dx, dy), shift(sk_sk, dx, dy)
    else:
        dx = dy = 0; resp = float("nan"); clipped = False
    dS = dist_to_ink(sk_b)
    ly, lx = np.nonzero(line_sk)
    d_line = dS[ly, lx]
    dL = dist_to_ink(line_b)
    sy, sx = np.nonzero(sk_sk)
    dropped = (dL[sy, sx] > R2).mean() if len(sy) else float("nan")
    return dict(dx=dx, dy=dy, resp=resp, clipped=clipped, d_line=d_line, ly=ly, lx=lx,
                sk_b=sk_b, sk_sk=sk_sk, dS=dS, dropped=dropped)


def n1_blockwise(dS, ly, lx, shape):
    """区画ごとに ±R2 の移動(刻み N1_STEP)で 残 を最大にし、分け直した距離を返す。"""
    h, w = shape
    d_new = np.empty(len(ly), dtype=np.float32)
    ty, tx = ly // TILE, lx // TILE
    tid = ty * ((w + TILE - 1) // TILE) + tx
    order = np.argsort(tid, kind="stable")
    bounds = np.flatnonzero(np.diff(tid[order])) + 1
    shifts = [(u, v) for u in range(-R2, R2 + 1, N1_STEP) for v in range(-R2, R2 + 1, N1_STEP)]
    for idx in np.split(order, bounds):
        yy, xx = ly[idx], lx[idx]
        best, best_d = -1, None
        for u, v in shifts:
            # 下絵を (u, v) 動かす = 線画の点で (y - v, x - u) の距離を引く
            y2 = np.clip(yy - v, 0, h - 1); x2 = np.clip(xx - u, 0, w - 1)
            d = dS[y2, x2]
            c = int((d <= R1).sum())
            if c > best:
                best, best_d = c, d
        d_new[idx] = best_d
    return d_new


def render_crop(r, line_b, line_sk, F, B, out_dir, tag, at=None):
    """原寸 1024 の切り出し: 下絵(合わせ後)| 線画 | 重ね(残=黒 動=青 足=赤 捨=橙)。"""
    h, w = line_b.shape
    C = 1024
    # 塗りでない線画の骨格が最も多い窓(S1b)
    ii = cv2.integral(line_sk.astype(np.uint8))
    best, by, bx = -1, 0, 0
    for y in range(0, max(h - C, 1), 256):
        for x in range(0, max(w - C, 1), 256):
            s = ii[y + C, x + C] - ii[y, x + C] - ii[y + C, x] + ii[y, x]
            if s > best:
                best, by, bx = s, y, x
    if at:
        bx, by = at
    win = (slice(by, by + C), slice(bx, bx + C))
    skA = np.where(B["sk_b"][win], 140, 255).astype(np.uint8)
    lnA = np.where(line_b[win], 0, 255).astype(np.uint8)
    ov = np.full((C, C, 3), 255, np.uint8)
    ov[B["sk_b"][win]] = (215, 215, 215)
    ov[F[win]] = (200, 160, 230)                      # 塗り = 薄い紫(4分類の外)
    k, m, a = classify(B["d_line"])
    sel = (B["ly"] >= by) & (B["ly"] < by + C) & (B["lx"] >= bx) & (B["lx"] < bx + C)
    dL = dist_to_ink(line_b)
    sy, sx = np.nonzero(B["sk_sk"][win])
    drop = dL[sy + by, sx + bx] > R2
    ov[sy[drop], sx[drop]] = (0, 140, 255)          # 捨 = 橙(BGR)
    for mask, col in ((k, (0, 0, 0)), (m, (255, 80, 0)), (a, (0, 0, 230))):
        s2 = sel & mask
        ov[B["ly"][s2] - by, B["lx"][s2] - bx] = col
    ov = cv2.erode(ov, np.ones((2, 2), np.uint8))  # 骨格を2px に(白地なので erode で太る)
    panel = np.concatenate([cv2.cvtColor(skA, cv2.COLOR_GRAY2BGR), cv2.cvtColor(lnA, cv2.COLOR_GRAY2BGR), ov], 1)
    bar = np.full((40, panel.shape[1], 3), 255, np.uint8)
    txt = (f"{r['work_id']} {r['page']} x{bx} y{by} 1:1 | sketch(aligned) | line | "
           f"overlay: keep=black moved=blue added=red dropped=orange fill=purple(excluded)  r1={R1} r2={R2} fill>{FILL_W}px")
    cv2.putText(bar, txt.encode("ascii", "replace").decode(), (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    out = out_dir / f"s1_smoke_crop_{tag}.png"
    cv2.imwrite(str(out), np.concatenate([bar, panel], 0))
    return out, (bx, by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pages", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crops", action="store_true")
    ap.add_argument("--crop-at", action="append", default=[], help="頁:x,y で窓を指定(例 running.clip:4608,4864)")
    a = ap.parse_args()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    pop = population()
    work = sorted([r for r in pop if r["work_id"] == a.work], key=lambda r: (r["slug"], r["page"]))
    print(f"厳しい門の母集団 {len(pop)} 頁 / 作品 {a.work}: {len(work)} 頁")
    todo = work[: a.pages] if a.pages else work
    rows = []
    for i, r in enumerate(todo):
        Lb, tL, spL = binarize(find(r, "line"))
        Sb, tS, spS = binarize(find(r, "sketch"))
        if Lb.shape != Sb.shape:
            print(f"  落とす(大きさが違う): {r['page']}"); continue
        Lsk_all, Ssk = skeletonize(Lb), skeletonize(Sb)
        F = fill_mask(Lb)
        Fpad = cv2.dilate(F.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FILL_PAD + 1,) * 2)) > 0
        Lsk = Lsk_all & ~Fpad
        fill_ink = F.sum() / max(Lb.sum(), 1)
        fill_skel = 1 - Lsk.sum() / max(Lsk_all.sum(), 1)
        A = arm(Lb, Lsk, Sb, Ssk, align=False)
        B = arm(Lb, Lsk, Sb, Ssk, align=True)
        kB, mB, aB = shares(B["d_line"])
        dN1 = n1_blockwise(B["dS"], B["ly"], B["lx"], Lb.shape)
        kN1, mN1, aN1 = shares(dN1)
        # N2: 同じ作品の別の頁(同じ大きさの次の頁、巡回)
        n2 = None
        for j in range(1, len(work)):
            o = work[(work.index(r) + j) % len(work)]
            Ob, _, _ = binarize(find(o, "sketch"))
            if Ob.shape == Lb.shape:
                n2 = (o["page"], arm(Lb, Lsk, Ob, skeletonize(Ob), align=True)); break
        kN2 = shares(n2[1]["d_line"])[0] if n2 else float("nan")
        # C1: 線画 対 線画(既知のずらし 17,-11 を与えて腕B が取り戻せるか)
        C = arm(Lb, Lsk, shift(Lb, 17, -11), shift(Lsk, 17, -11), align=True)
        kC, mC, aC = shares(C["d_line"])
        hold = (kB < kN2 + HOLD_MARGIN) if n2 else True
        row = dict(work_id=r["work_id"], page=r["page"], h=Lb.shape[0], w=Lb.shape[1],
                   otsu_line=tL, otsu_sketch=tS, specks_line=spL, specks_sketch=spS,
                   fill_ink_share=round(float(fill_ink), 4), fill_skel_excluded=round(float(fill_skel), 4),
                   line_skel_px=int(Lsk.sum()), sketch_skel_px=int(Ssk.sum()),
                   A_keep=round(shares(A["d_line"])[0], 4), A_moved=round(shares(A["d_line"])[1], 4),
                   A_added=round(shares(A["d_line"])[2], 4), A_dropped=round(A["dropped"], 4),
                   B_dx=B["dx"], B_dy=B["dy"], B_resp=round(B["resp"], 3), B_clipped=B["clipped"],
                   B_keep=round(kB, 4), B_moved=round(mB, 4), B_added=round(aB, 4), B_dropped=round(B["dropped"], 4),
                   N1_keep=round(kN1, 4), N1_moved=round(mN1, 4), N1_added=round(aN1, 4),
                   N1_delta_keep=round(kN1 - kB, 4),
                   N2_page=n2[0] if n2 else "", N2_keep=round(kN2, 4),
                   C1_dx=C["dx"], C1_dy=C["dy"], C1_keep=round(kC, 4), C1_added=round(aC, 4),
                   hold=hold, moved_mostly_misalignment=(kN1 - kB) >= 0.5 * mB if mB > 0 else False)
        rows.append(row)
        print(f"  {r['page']:22s} B 残{kB:.3f} 動{mB:.3f} 足{aB:.3f} 捨{B['dropped']:.3f} | A 残{row['A_keep']:.3f} "
              f"| shift({B['dx']},{B['dy']}) resp{B['resp']:.2f} | N1 Δ残{kN1-kB:+.3f} | N2 残{kN2:.3f} "
              f"| 塗り 墨{fill_ink:.3f} 骨格外し{fill_skel:.3f} | C1 shift({C['dx']},{C['dy']}) 残{kC:.3f} 足{aC:.3f} | 保留{hold}")
        at = {c.split(":")[0]: tuple(int(v) for v in c.split(":")[1].split(",")) for c in a.crop_at}
        if a.crops and (i < 2 or r["page"] in at):
            p, xy = render_crop(r, Lb, Lsk, F, B, out_dir, f"{r['work_id']}_{Path(r['page']).stem}", at.get(r["page"]))
            print(f"    切り出し {p.relative_to(ROOT)} at {xy}")
    with open(out_dir / "s1_pages.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"表: {(out_dir / 's1_pages.csv').relative_to(ROOT)}  行 {len(rows)}")


if __name__ == "__main__":
    sys.exit(main())
