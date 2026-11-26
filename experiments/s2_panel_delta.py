"""Track S 登録 S2 / S2a(doc/work_log.md 2026-11-26)の器具: コマを単位にした 残/動/足/捨。

コマの形は census 経由の抽出側の納品(`data/census/s2_panel_shapes_20261126/`、uint16、0 = コマでない)。
S1 と同じ二値化・塗りの除外・距離の分け方(R1/R2)。腕B の移動は S1 の行の値(`source_path` で対応)。
コマの区分: K/L < 0.10 = 下絵の無いコマ(L = コマの中の線画の骨格、K = 合わせた下絵の骨格)。L < 200 = 線の少ないコマ。
地(area ≥ 頁の 0.8)の番号と 0 の画素は「コマの外」として別の行。area_in_image < 頁の 2% のコマは落とす。

使い方: ../lineart/venv/bin/python experiments/s2_panel_delta.py --work 017_2017_stoptime --pages 4 --out results/s2_smoke --crops
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from s1_cleanup_delta import (FILL_PAD, HOLD_MARGIN, R1, R2, arm, binarize, classify, dist_to_ink,  # noqa: E402
                              fill_mask, find, n1_blockwise, population, shift)

SHAPES = ROOT / "data/census/s2_panel_shapes_20261126"
PANELS = ROOT / "data/census/s2_panel_shapes_panels_20261126.csv"
PAGES = ROOT / "data/census/s2_panel_shapes_pages_20261126.csv"
NO_SKETCH = 0.10   # K/L < これ = 下絵の無いコマ(登録 S2)
FEW_LINE = 200     # L < これ = 線の少ないコマ
GROUND = 0.80      # area ≥ 頁のこれ = 地(S2a)
SMALL = 0.02       # area_in_image < 頁のこれ = 落とす


def s1_rows():
    """S1 の行を population() の (slug, page) 順に対応させる(work_id+page は一意でない)。"""
    pop = population()
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results/s1_full/*/s1_pages.csv"))):
        S = list(csv.DictReader(open(f, encoding="utf-8")))
        Pw = sorted([r for r in pop if r["work_id"] == Path(f).parent.name], key=lambda r: (r["slug"], r["page"]))
        assert [r["page"] for r in Pw] == [r["page"] for r in S], f
        for s, q in zip(S, Pw):
            out[q["source_path"]] = (q, s)
    return out


def part(d_line, sel):
    d = d_line[sel]
    n = len(d)
    if n == 0:
        return n, float("nan"), float("nan"), float("nan")
    k, m, a = classify(d)
    return n, k.sum() / n, m.sum() / n, a.sum() / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pages", type=int, default=0)
    ap.add_argument("--only", action="append", default=[], help="頁名(page)で絞る")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crops", action="store_true")
    a = ap.parse_args()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    S1 = s1_rows()
    pages = {r["page_path"]: r for r in csv.DictReader(open(PAGES, encoding="utf-8"))}
    panels = {}
    for r in csv.DictReader(open(PANELS, encoding="utf-8")):
        panels.setdefault(r["page_path"], {})[int(r["koma_id"])] = r
    work = sorted([q for q, _ in S1.values() if q["work_id"] == a.work], key=lambda r: (r["slug"], r["page"]))
    todo = [r for r in work if not a.only or r["page"] in a.only]
    todo = todo[: a.pages] if a.pages else todo
    drops = {}
    prow, krow = [], []
    for r in todo:
        sp = r["source_path"]
        pg = pages[sp]
        if pg["status"] != "出した":
            drops[f"形なし: {pg['reason']}"] = drops.get(f"形なし: {pg['reason']}", 0) + 1; continue
        lab = cv2.imread(str(SHAPES / Path(pg["out_file"]).name), cv2.IMREAD_UNCHANGED)
        Lb, _, _ = binarize(find(r, "line"))
        Sb, _, _ = binarize(find(r, "sketch"))
        if lab is None or lab.shape != Lb.shape or Sb.shape != Lb.shape:
            drops["画の大きさが違う"] = drops.get("画の大きさが違う", 0) + 1; continue
        s1 = S1[sp][1]
        dx, dy = int(s1["B_dx"]), int(s1["B_dy"])
        F = fill_mask(Lb)
        Fpad = cv2.dilate(F.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FILL_PAD + 1,) * 2)) > 0
        Lsk = skeletonize(Lb) & ~Fpad
        Sbs = shift(Sb, dx, dy)
        Ssk = skeletonize(Sbs)  # 合わせた下絵の骨格
        dS = dist_to_ink(Sbs)
        dL = dist_to_ink(Lb)
        ly, lx = np.nonzero(Lsk)
        d_line = dS[ly, lx]
        sy, sx = np.nonzero(Ssk)
        d_drop = dL[sy, sx]
        dC1 = dL[ly, lx]  # C1: 線画 対 線画
        dN1 = n1_blockwise(dS, ly, lx, Lb.shape)
        # N2: S1 と同じ規則(同じ作品の次の同じ大きさの頁、巡回)。移動は探し直す
        n2 = None
        for j in range(1, len(work)):
            o = work[(work.index(r) + j) % len(work)]
            Ob, _, _ = binarize(find(o, "sketch"))
            if Ob.shape == Lb.shape:
                n2 = (o["line"], arm(Lb, Lsk, Ob, skeletonize(Ob), align=True)); break
        # コマ同士の境からの距離(隣のコマの下絵が効く範囲)
        edge = np.zeros(lab.shape, np.uint8)
        edge[:, 1:] |= (lab[:, 1:] != lab[:, :-1]).astype(np.uint8)
        edge[1:, :] |= (lab[1:, :] != lab[:-1, :]).astype(np.uint8)
        dEdge = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 5)
        area_page = lab.size
        Lab_l = lab[ly, lx]
        Lab_s = lab[sy, sx]
        outside = np.zeros(len(ly), bool)
        outside_s = np.zeros(len(sy), bool)
        P = panels.get(sp, {})
        n_ground = n_small = 0
        for kid in range(0, int(lab.max()) + 1):
            info = P.get(kid)
            if kid == 0 or info is None:
                outside |= Lab_l == kid; outside_s |= Lab_s == kid; continue
            if int(info["area"]) >= GROUND * area_page:
                n_ground += 1; outside |= Lab_l == kid; outside_s |= Lab_s == kid; continue
            if int(info["area_in_image"]) < SMALL * area_page:
                n_small += 1; continue
            sel = Lab_l == kid
            L = int(sel.sum()); K = int((Lab_s == kid).sum())
            n, k, m, ad = part(d_line, sel)
            _, kN1, _, _ = part(dN1, sel)
            kN2 = part(n2[1]["d_line"], sel)[1] if n2 else float("nan")
            dsel = Lab_s == kid
            drp = float((d_drop[dsel] > R2).mean()) if dsel.any() else float("nan")
            c1 = part(dC1, sel)[1]
            kind = "線の少ないコマ" if L < FEW_LINE else ("下絵の無いコマ" if K / L < NO_SKETCH else "下絵の在るコマ")
            hold = (k < kN2 + HOLD_MARGIN) if (n2 and kind == "下絵の在るコマ") else (kind == "下絵の在るコマ")
            krow.append(dict(work_id=r["work_id"], line=r["line"], koma_id=kid, koma_source=info["koma_source"],
                             area=info["area"], area_in_image=info["area_in_image"], L=L, K=K,
                             K_over_L=round(K / L, 4) if L else float("nan"), kind=kind,
                             keep=round(k, 4), moved=round(m, 4), added=round(ad, 4), dropped=round(drp, 4),
                             N1_delta_keep=round(kN1 - k, 4), N2_keep=round(kN2, 4), hold=hold,
                             near_edge_share=round(float((dEdge[ly[sel], lx[sel]] <= R2).mean()), 4) if L else float("nan"),
                             C1_keep=round(c1, 4)))
        n, k, m, ad = part(d_line, outside)
        _, kp, mp, ap_ = part(d_line, np.ones(len(ly), bool))
        prow.append(dict(work_id=r["work_id"], line=r["line"], n_koma=pg["n_koma"], ground=n_ground, small_dropped=n_small,
                         overlap_px=pg["overlap_px"], outside_L=n, outside_keep=round(k, 4), outside_added=round(ad, 4),
                         page_keep=round(kp, 4), page_added=round(ap_, 4), s1_B_keep=s1["B_keep"], s1_B_added=s1["B_added"],
                         N2_line=n2[0] if n2 else ""))
        ks = [x for x in krow if x["line"] == r["line"]]
        print(f"{r['page']:20s} コマ {len(ks)}(地 {n_ground}・小 {n_small})頁 残{kp:.3f} 足{ap_:.3f}(S1 {s1['B_keep']}/{s1['B_added']})"
              f" 外 L{n} 足{ad:.3f}")
        for x in ks:
            print(f"   koma {x['koma_id']:2d} {x['kind']:8s} L{x['L']:6d} K/L {x['K_over_L']:.3f} 残{x['keep']:.3f} 動{x['moved']:.3f}"
                  f" 足{x['added']:.3f} 捨{x['dropped']:.3f} N2残{x['N2_keep']:.3f} 保留{x['hold']} 境{x['near_edge_share']:.2f} C1 {x['C1_keep']}")
        if a.crops:
            render(r, lab, Lb, Sbs, Lsk, ks, d_line, ly, lx, out_dir)
    for name, rows in (("s2_panels.csv", krow), ("s2_pages.csv", prow)):
        if rows:
            with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("落とした頁:", drops or "なし")


def render(r, lab, Lb, Sbs, Lsk, ks, d_line, ly, lx, out_dir):
    """コマごとに原寸の切り出し(最大 1400px 四方、コマの外接矩形の左上から): 下絵(合わせ後)| 線画 | 重ね。"""
    k, m, a = classify(d_line)
    for x in ks:
        if x["kind"] == "線の少ないコマ":
            continue
        ys, xs = np.nonzero(lab == x["koma_id"])
        by0, bx0, by1, bx1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
        # コマの中で線画の骨格が最も多い 1400 の窓(256 刻み)
        C = 1400
        m_in = (Lsk & (lab == x["koma_id"])).astype(np.uint8)
        ii = cv2.integral(m_in)
        best, y0, x0 = -1, by0, bx0
        for yy in range(by0, max(by1 - C, by0) + 1, 256):
            for xx in range(bx0, max(bx1 - C, bx0) + 1, 256):
                ye, xe = min(yy + C, lab.shape[0]), min(xx + C, lab.shape[1])
                v = ii[ye, xe] - ii[yy, xe] - ii[ye, xx] + ii[yy, xx]
                if v > best:
                    best, y0, x0 = v, yy, xx
        y1, x1 = min(y0 + C, by1), min(x0 + C, bx1)
        win = (slice(y0, y1), slice(x0, x1))
        inside = lab[win] == x["koma_id"]
        sk = np.where(Sbs[win], 140, 255).astype(np.uint8)
        ln = np.where(Lb[win], 0, 255).astype(np.uint8)
        ov = np.full(sk.shape + (3,), 255, np.uint8)
        ov[Sbs[win]] = (215, 215, 215)
        sel = (ly >= y0) & (ly < y1) & (lx >= x0) & (lx < x1)
        for mask, col in ((k, (0, 0, 0)), (m, (255, 80, 0)), (a, (0, 0, 230))):
            s2 = sel & mask
            ov[ly[s2] - y0, lx[s2] - x0] = col
        ov = cv2.erode(ov, np.ones((2, 2), np.uint8))
        for im in (sk, ln):
            im[~inside] = (im[~inside] * 0.35 + 255 * 0.65).astype(np.uint8)
        ov[~inside] = (ov[~inside] * 0.35 + 255 * 0.65).astype(np.uint8)
        panel = np.concatenate([cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR), cv2.cvtColor(ln, cv2.COLOR_GRAY2BGR), ov], 1)
        bar = np.full((40, panel.shape[1], 3), 255, np.uint8)
        txt = (f"{r['work_id']} {r['page']} koma{x['koma_id']} 1:1 {x['kind'] == '下絵の無いコマ' and 'NO-SKETCH' or 'SKETCH'} "
               f"K/L={x['K_over_L']} | sketch | line | keep=black moved=blue added=red (outside koma faded)")
        cv2.putText(bar, txt, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(str(out_dir / f"s2_crop_{Path(r['line']).stem}_k{x['koma_id']:02d}.png"), np.concatenate([bar, panel], 0))


if __name__ == "__main__":
    sys.exit(main())
