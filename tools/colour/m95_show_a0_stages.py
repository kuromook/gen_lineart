#!/usr/bin/env python
"""m95: 腕 A0 の5段(入力 → 量子化 → 連結 → 床 → 面)を**原寸で**描いて目視に出す。

**これは測定ではない。** 棒も既知セルも持たず、どの数も登録に載せない。
目的は「A0 が面と呼んでいるものが人の目にどう見えるか」の一点だけである。

**作者に出す窓の選び方(`m92` の無作為)は1つも動かしていない。** ここで選ぶのは
**目視に出す1窓**だけで、選び方は「絵が最も詰まっている窓」= 内容で選んでいる。
判定に使う窓を内容で選ぶのは NC1 の誤りだが、**これは判定ではない**ので、
**選び方を書いて出す**ことで扱う。

段は `m12` の関数をそのまま呼ぶ(描き直さない):
  入力     `CanvasPreview`(画布の 1/4。**A0 の本当の解像度**)
  量子化   `m12.quantise(im, 8)` = `floor(v*8/256)`。表示は `(v//32)*32` で戻す
  連結     `m12.equal_value_components` = 等しい量子化色の8連結成分(**床の前**)
  床       `m12.filtered_labels(lab, FLOOR*pw*ph)` で落ちた画素
  面       残ったラベルの境界 1px(`m93.edges1`。**作者に出す絵と同じ描き方**)

原寸へは `np.repeat`(最近傍)。**4画素の段を消さない** —— それが A0 の素の姿である。
"""
import csv
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m12_score_a0_nulls as m12                      # noqa: E402
import m93_build_division_queue as m93                # noqa: E402

WINDOWS = pathlib.Path("results/m92_windows/windows.csv")
OUT = pathlib.Path("results/m95_a0_stages")
WIN = 1024
LINE = (255, 0, 255)       # マゼンタ 1px(m93 と同じ)
DROP = (60, 60, 255)       # BGR = 赤。床で落ちた画素
SEED_COLOUR = 20261101     # 連結の塗り分けの色だけに使う種


def colour_lut(n, rng):
    """ラベル -> BGR。0 は黒。"""
    lut = rng.integers(40, 256, size=(n + 1, 3), dtype=np.int32).astype(np.uint8)
    lut[0] = 0
    return lut


def up(a, s):
    return np.repeat(np.repeat(a, s, 0), s, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(WINDOWS, encoding="utf-8"))
            if r["role"] == "ask"]
    pages = {p["page"]: p for p in
             json.load(open(m12.MANIFEST, encoding="utf-8"))["pages"]}
    rng = np.random.default_rng(SEED_COLOUR)

    # ---- 1巡: 窓ごとに「最頻色の占める割合」を原寸の下地で測る(どれを出すか決めるだけ)
    cache = {}
    stat = []
    for r in rows:
        page = r["page"]
        if page not in cache:
            p = pages[page]
            s = int(round(p["preview"]["縮小率"]))
            pw, ph = p["preview"]["size"]
            im = cv2.imread(str(m12.D / p["preview"]["file"]), cv2.IMREAD_COLOR)
            assert im.shape[:2] == (ph, pw), (im.shape, (ph, pw))
            codes = m12.quantise(im, m12.LEVELS)
            lab = m12.equal_value_components(codes)
            floor_px = max(1, round(m12.FLOOR * pw * ph))
            labf, _ = m12.filtered_labels(lab, floor_px)
            cache[page] = (s, im, codes, lab, labf, floor_px)
        s, im, codes, lab, labf, floor_px = cache[page]
        w = WIN // s
        y0, x0 = int(r["y"]) // s, int(r["x"]) // s
        c = codes[y0:y0 + w, x0:x0 + w]
        cnt = np.bincount(np.unique(c, return_inverse=True)[1])
        stat.append((cnt.max() / c.size, r, y0, x0, w))

    stat.sort(key=lambda t: t[0])
    print(f"窓 {len(stat)}(問う窓だけ。IC1 の複製と IC2 は入れない)")
    print("  最頻色の占める割合(原寸の下地。**低い = 絵が詰まっている**)")
    for frac, r, *_ in stat[:5]:
        print(f"    {r['item_id']} {r['alias']}  {frac*100:6.2f}%"
              f"  (m92 が数えた A0 の面 {r['n_a0_areas']} / 標準 {r['n_std_areas']})")
    print("    ...")
    for frac, r, *_ in stat[-3:]:
        print(f"    {r['item_id']} {r['alias']}  {frac*100:6.2f}%"
              f"  (m92 が数えた A0 の面 {r['n_a0_areas']} / 標準 {r['n_std_areas']})")

    # ---- 2巡: 選んだ1窓の5段を原寸で描く
    frac, r, y0, x0, w = stat[0]
    page = r["page"]
    s, im, codes, lab, labf, floor_px = cache[page]
    tag = f"{r['alias']}_{r['item_id']}"
    print(f"\n出す窓: {r['item_id']} {r['alias']}({r['stratum']})"
          f"  原寸 (y={r['y']}, x={r['x']}) {WIN}x{WIN}"
          f"  preview 側 (y={y0}, x={x0}) {w}x{w}  縮小率 1/{s}")
    print(f"  床 = preview の画素数の 1e-5 = **{floor_px} 画素**"
          f"(preview {im.shape[1]}x{im.shape[0]})")

    cim = im[y0:y0 + w, x0:x0 + w]
    ccodes = codes[y0:y0 + w, x0:x0 + w]
    clab = lab[y0:y0 + w, x0:x0 + w]
    clabf = labf[y0:y0 + w, x0:x0 + w]

    made = []

    # A 入力
    cv2.imwrite(str(OUT / f"{tag}_a_input.png"), up(cim, s)); made.append("a_input")

    # B 量子化(表示は格の下端 (v//32)*32 に戻す)
    q = (cim.astype(np.uint16) * m12.LEVELS // 256).astype(np.uint8)
    cv2.imwrite(str(OUT / f"{tag}_b_quantised.png"),
                up((q.astype(np.uint16) * (256 // m12.LEVELS)).astype(np.uint8), s))
    made.append("b_quantised")

    # C 連結(床の前。ラベルごとに無作為な色)
    lut = colour_lut(int(lab.max()), rng)
    cv2.imwrite(str(OUT / f"{tag}_c_components.png"), up(lut[clab], s))
    made.append("c_components")

    # D 床(落ちた画素を赤。残りは灰にして落ちた所だけ見せる)
    g = cv2.cvtColor(cv2.cvtColor(cim, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    g = (g.astype(np.uint16) // 2 + 110).astype(np.uint8)
    dropped = clabf == 0
    g[dropped] = DROP
    cv2.imwrite(str(OUT / f"{tag}_d_floor.png"), up(g, s)); made.append("d_floor")

    # E 面(残ったラベルの境界 1px。**原寸で引く** = 作者に出す絵と同じ)
    full = up(cim, s)
    e = m93.edges1(up(clabf, s))
    full[e] = LINE
    cv2.imwrite(str(OUT / f"{tag}_e_faces.png"), full); made.append("e_faces")

    n_before = int(np.unique(clab).size)
    n_after = int(np.unique(clabf[clabf > 0]).size)
    print(f"\n  この窓の中の成分: 床の前 {n_before} → 床の後 **{n_after}**"
          f"(落ちた画素 {dropped.mean()*100:.2f}%)")
    print(f"  **成分も床も頁全体で計算して、窓で切っている**"
          f"(窓で切ってから数えると連結も床も変わる)")
    for m in made:
        print(f"  {OUT / (tag + '_' + m + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
