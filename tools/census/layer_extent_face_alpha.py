#!/usr/bin/env python
"""**発注2件の納品の受け取り**(抽出側、2026-11-04 23:31)——
**線画の層の範囲(bbox)** と **面の層のアルファ・マスクの値**。

**依頼元は Track N**(`note_n_census_layer_bbox_and_mask_edges_20261103.md`)。
**面の表を読むのは Track M と基盤**(ユーザー 11-03「マスク(=面)調査にも効く要素」)。
**層ごとの全量は N・M・基盤の inbox に直接届いている**(頁のパスを持つので公開しない)。
**census がするのは、照合と、頁のパスを持たない集計の公開表だけ。**

**既知セル**(条例5: 試験対象の器具が作っていない入力を少なくとも1つ):

- 範囲: `n_px_alpha_gt0` を**墨の濃さの1段目・2段目**と突き合わせる(器具は `38f2859` 系。
  今回の器具 `measure_layer_extent.py` とは別)。**母集団の鍵も同じ表から引く**
  (層別 L1 の全層 − 恒久除外の頁)
- 面のマスク: `n_px_gt0` を **`mask_cover_request3_v2`**(器具 `list_mask_cover.py`)の
  `mask_px_ge128 + mask_px_1_127` と突き合わせる
- 面の本体: `n_px_gt0` を **`faint_alpha_905_v2`** の `alpha_ge1_ratio × canvas_px` と、
  `alpha_1_25_px <= n_px_1_254` で突き合わせる

**どれも同じ供給元の別の器具である** —— **構造 DB は画素を持たないので、画素の数の
既知セルに供給元の外の材料は無い。** そう書く(T132)。

**届く範囲**(名前より狭く書く):

- **届く**: 層の描かれた画素の範囲 / 面の層の値が 0・255・その間のどれか
- **届かない**:
  - **2値の判定** —— 供給元も census も判定していない。**「1〜254 の値が無い層」と書く**
  - **形式5 の本体** —— アルファが全面 255 と読まれる件が解けていない。**調査中で数えない**
  - **マスクの値の向き** —— `mask_init` が 255 の面と 0 の面で「通す/隠す」の側が違う
    (`mask_cover` の便、2026-11-02)。**この表は値の数だけを数え、向きを読まない**
  - **タイルとの重なり** —— N の側の仕事(N が別の登録を書いてから読む)

    ./venv/bin/python tools/census/layer_extent_face_alpha.py
"""
import collections
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import exclusions  # noqa: E402
import line_binary_alpha as lba  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BBOX = "inbox/census_l1_layer_bbox_20261104.csv"
FACE = "inbox/census_face_layer_alpha_mask_20261104.csv"
MASK_COVER = "inbox/mask_cover_request3_v2_20261102.csv"
FAINT = "inbox/faint_alpha_905_v2_20261102.csv"
SHA = {BBOX: "d8c79a948be27cc1", FACE: "11dae65dba257aed"}   # 便の先頭16桁
# **供給元の言明**(便の表)。**再現できなければ鳴る**
CLAIM_BBOX = dict(rows=49786, pages=1724, ok=45493, empty=4293, px_differ=2, dropped_excl=14)
CLAIM_FACE = {("body", "ok"): 85022, ("body", "under_investigation_format5"): 32484,
              ("body", "no_pixels"): 307, ("mask", "ok"): 72090, ("mask", "mask_unread"): 71}
CLAIM_GUIDE = {"body_ok": (61903, 8345), "body_f5": (492, 7622), "mask_ok": (32374, 16077)}
OUT_EXTENT = "census/l1_layer_extent.csv"
OUT_FACE = "census/face_binary_alpha.csv"


def year_of(page_path: str) -> str:
    return page_path.split("\\")[2]


def check_sha() -> bool:
    import hashlib
    good = True
    for p, head in SHA.items():
        h = hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()
        ok = h.startswith(head)
        good &= ok
        print(f"    sha256 {os.path.basename(p)} {h[:16]} {'一致' if ok else '**不一致**'}")
    return good


def extent(failures):
    print("  [線画の層の範囲(発注1)]")
    excl = exclusions.load()
    ink = {}
    for r in lba.load_rows():
        if r["stratum"] == "L1":
            ink.setdefault((r["page_path"], r["layer_id"]), r)
    n_excl = sum(1 for k in ink if k[0] in excl)
    want = {k for k in ink if k[0] not in excl}
    bb = {(r["page_path"], r["layer_id"]): r
          for r in csv.DictReader(open(os.path.join(ROOT, BBOX), encoding="utf-8"))}
    print(f"    母集団(墨の濃さの層別 L1){len(ink):,}層 / 恒久除外の頁で落とす {n_excl} / "
          f"残り {len(want):,} / 納品 {len(bb):,}層 / 納品にしか無い {len(set(bb) - want)} / "
          f"納品に無い {len(want - set(bb))}")
    if set(bb) != want or n_excl != CLAIM_BBOX["dropped_excl"]:
        failures.append("範囲の鍵が墨の濃さの母集団と合わない")
    canvas = {}
    for r in csv.DictReader(open(os.path.join(ROOT, FACE), encoding="utf-8")):
        canvas.setdefault(r["page_path"], int(r["n_px_total"]))
    agg = collections.defaultdict(lambda: dict(c=collections.Counter(), share=[], fill=[], pages=set()))
    differ = []
    for k, r in bb.items():
        y = ink[k]["year_label"]
        a = agg[y]
        a["pages"].add(k[0])
        c = a["c"]
        c["n"] += 1
        n = int(r["n_px_alpha_gt0"])
        n_ink = int(ink[k]["n_px_alpha_gt0"] or 0)
        if n == n_ink:
            c["match"] += 1
        else:
            c["differ"] += 1
            differ.append((y, k[1], n_ink, n))
        if r["status"] == "empty":
            c["empty"] += 1
            if n != 0 or any(r[f] for f in ("x0", "y0", "x1", "y1")):
                c["bad"] += 1
            continue
        c["ok"] += 1
        x0, y0, x1, y1 = (int(r[f]) for f in ("x0", "y0", "x1", "y1"))
        area = (x1 - x0) * (y1 - y0)
        tot = canvas.get(k[0])
        if (r["status"] != "ok" or min(x0, y0) < 0 or x1 <= x0 or y1 <= y0
                or area < n or tot is None or area > tot):
            c["bad"] += 1
            continue
        a["share"].append(area / tot)
        a["fill"].append(n / area)
    tot_c = sum((a["c"] for a in agg.values()), collections.Counter())
    pages = set().union(*(a["pages"] for a in agg.values()))
    print(f"    status ok {tot_c['ok']:,} / empty {tot_c['empty']:,} / 頁 {len(pages):,} / "
          f"**範囲が矛盾する層 {tot_c['bad']}**(負・逆向き・画素より狭い・キャンバスより広い)")
    print(f"    **`n_px_alpha_gt0` 墨の濃さの表と一致 {tot_c['match']:,} / 違う {tot_c['differ']}**")
    for y, lid, a0, a1 in differ:
        print(f"      札 {y} 層 {lid}: 墨の濃さ {a0:,} → 範囲 {a1:,}")
    got = dict(rows=len(bb), pages=len(pages), ok=tot_c["ok"], empty=tot_c["empty"],
               px_differ=tot_c["differ"], dropped_excl=n_excl)
    if got != CLAIM_BBOX or tot_c["bad"]:
        failures.append(f"範囲: 便の言明と違う {got}")
    rows = []
    for y in sorted(agg):
        a = agg[y]
        c = a["c"]
        rows.append([y, len(a["pages"]), c["n"], c["ok"], c["empty"], c["match"], c["differ"],
                     c["bad"],
                     f"{statistics.median(a['share']):.4f}" if a["share"] else "",
                     f"{statistics.median(a['fill']):.4f}" if a["fill"] else "", ""])
    rows.append(["ALL", len(pages), tot_c["n"], tot_c["ok"], tot_c["empty"], tot_c["match"],
                 tot_c["differ"], tot_c["bad"],
                 f"{statistics.median([s for a in agg.values() for s in a['share']]):.4f}",
                 f"{statistics.median([s for a in agg.values() for s in a['fill']]):.4f}", ""])
    rows += [[""] * 10 + [n] for n in NOTE_EXTENT]
    n_pub = bc.write_csv(os.path.join(ROOT, OUT_EXTENT),
                         ["year_label", "n_pages", "n_layers", "n_layers_ok", "n_layers_empty",
                          "n_px_match", "n_px_differ", "n_bbox_inconsistent",
                          "median_bbox_share_of_canvas", "median_fill_of_bbox", "note"], rows)
    print(f"    {OUT_EXTENT} {n_pub}行")


def face(failures):
    print("  [面の層のアルファ・マスクの値(発注2)]")
    rows = list(csv.DictReader(open(os.path.join(ROOT, FACE), encoding="utf-8")))
    st = collections.Counter((r["plane"], r["status"]) for r in rows)
    print("    " + " / ".join(f"{p} {s} {n:,}" for (p, s), n in sorted(st.items())))
    if dict(st) != CLAIM_FACE:
        failures.append(f"面: status の数が便と違う {dict(st)}")
    arith = sum(1 for r in rows if r["n_px_gt0"]
                and int(r["n_px_gt0"]) != int(r["n_px_255"]) + int(r["n_px_1_254"]))
    print(f"    `n_px_gt0 = n_px_255 + n_px_1_254` が成り立たない行 {arith}")
    if arith:
        failures.append("面: 画素の内訳が合計と合わない")
    # **便の目安を引き直す**(層の数と、画素を足した 1〜254 の比。便の 0.527 / 0.119 / 0.0015 は画素を足した比で再現する)
    guide = {}
    for key, sel in (("body_ok", ("body", "ok")),
                     ("body_f5", ("body", "under_investigation_format5")),
                     ("mask_ok", ("mask", "ok"))):
        sub = [r for r in rows if (r["plane"], r["status"]) == sel]
        z = sum(1 for r in sub if int(r["n_px_gt0"]) == 0)
        nm = sum(1 for r in sub if int(r["n_px_gt0"]) > 0 and int(r["n_px_1_254"]) == 0)
        gt0 = sum(int(r["n_px_gt0"]) for r in sub)
        mid = sum(int(r["n_px_1_254"]) for r in sub)
        guide[key] = (z, nm)
        print(f"    目安 {key}: 層 {len(sub):,} / 全部 0 {z:,} / 1〜254 が無い {nm:,} / "
              f"画素を足した 1〜254 の比 {mid / gt0:.4f}")
    if guide != CLAIM_GUIDE:
        failures.append(f"面: 目安の層の数が便と違う {guide}")

    # 既知セル1: マスクの面 ↔ `mask_cover_request3_v2`
    mc = {(r["page_path"], r["layer_id"]): r
          for r in csv.DictReader(open(os.path.join(ROOT, MASK_COVER), encoding="utf-8"))}
    c = collections.Counter()
    for r in rows:
        k = (r["page_path"], r["layer_id"])
        if r["plane"] != "mask" or r["status"] != "ok" or k not in mc or mc[k]["error"]:
            continue
        b = mc[k]
        c["n"] += 1
        c["match"] += int(r["n_px_gt0"]) == int(b["mask_px_ge128"]) + int(b["mask_px_1_127"])
        c["init_match"] += r["mask_init"] == b["init"]
    print(f"    **既知セル(マスク ↔ `mask_cover` の `_v2`)**: 重なる {c['n']:,}層 / "
          f"`n_px_gt0` 一致 {c['match']:,} / 初期値一致 {c['init_match']:,}")
    if not c["n"] or c["match"] != c["n"] or c["init_match"] != c["n"]:
        failures.append("面: マスクの既知セルが合わない")
    # 既知セル2: 本体 ↔ `faint_alpha_905_v2`
    fn = {(r["page_path"], r["layer_id"]): r
          for r in csv.DictReader(open(os.path.join(ROOT, FAINT), encoding="utf-8"))}
    c = collections.Counter()
    for r in rows:
        k = (r["page_path"], r["layer_id"])
        if (r["plane"] != "body" or r["status"] != "ok" or k not in fn or fn[k]["error"]
                or fn[k]["rendered"] != "1"):
            continue
        b = fn[k]
        cp = int(b["canvas_px"])
        c["n"] += 1
        c["canvas"] += cp == int(r["n_px_total"])
        # **比は小数で丸めてあるので、画素の数に戻すと ±数画素の幅がある**
        c["gt0"] += abs(float(b["alpha_ge1_ratio"]) * cp - int(r["n_px_gt0"])) <= max(1, cp * 5e-6)
        c["band"] += int(b["alpha_1_25_px"]) <= int(r["n_px_1_254"])
    print(f"    **既知セル(本体 ↔ `faint_alpha_905` の `_v2`)**: 重なる {c['n']:,}層 / "
          f"キャンバスの画素一致 {c['canvas']:,} / `n_px_gt0` が丸めの幅で一致 {c['gt0']:,} / "
          f"1〜25 の画素 ≤ 1〜254 の画素 {c['band']:,}")
    if not c["n"] or not (c["canvas"] == c["gt0"] == c["band"] == c["n"]):
        failures.append("面: 本体の既知セルが合わない")

    agg = collections.defaultdict(collections.Counter)
    pages = collections.defaultdict(set)
    for r in rows:
        g = (year_of(r["page_path"]), r["plane"], r["v_layer_kind"])
        a = agg[g]
        pages[g].add(r["page_path"])
        a["n"] += 1
        s = r["status"]
        if s != "ok":
            a[s] += 1
            continue
        a["ok"] += 1
        gt0, mid = int(r["n_px_gt0"]), int(r["n_px_1_254"])
        a["zero"] += gt0 == 0
        a["nomid"] += gt0 > 0 and mid == 0
        a["px"] += gt0
        a["mid"] += mid
        if r["plane"] == "mask":
            a["init255"] += r["mask_init"] == "255"
            a["init0"] += r["mask_init"] == "0"
    out = []
    for (y, p, k), a in sorted(agg.items()):
        drawn = a["ok"] - a["zero"]
        out.append([y, p, k, len(pages[(y, p, k)]), a["n"], a["under_investigation_format5"],
                    a["no_pixels"], a["mask_unread"], a["ok"], a["zero"], a["nomid"],
                    f"{a['nomid'] / drawn:.3f}" if drawn else "",
                    f"{a['mid'] / a['px']:.4f}" if a["px"] else "",
                    a["init255"] if p == "mask" else "", a["init0"] if p == "mask" else "", ""])
    out += [[""] * 15 + [n] for n in NOTE_FACE]
    n_pub = bc.write_csv(os.path.join(ROOT, OUT_FACE),
                         ["year_label", "plane", "v_layer_kind", "n_pages", "n_layers",
                          "n_layers_format5", "n_layers_no_pixels", "n_layers_mask_unread",
                          "n_layers_counted", "n_layers_all_zero", "n_layers_binary_alpha",
                          "share_layers_binary_alpha", "share_px_partial_alpha",
                          "n_mask_init_255", "n_mask_init_0", "note"], out)
    print(f"    {OUT_FACE} {n_pub}行")


NOTE_EXTENT = [
    "**Track N の発注**(2026-11-03)**への納品の集計**(抽出側 11-04、器具 `measure_layer_extent.py` `e13cf72`)。"
    "**層ごとの範囲は頁のパスを持つので公開しない**(N・M・基盤の inbox に直接届いている)",
    "**母集団は墨の濃さの1段目・2段目の層別 L1 の全層から、恒久除外の頁の 14層を落とした 49,786層・1,724頁**。"
    "**鍵の集合は census が墨の濃さの表から引き直して一致した**",
    "**範囲はアルファ > 0 の画素。座標は頁(キャンバス)の画素で、層のオフセットを足した後。x0・y0 は含み、x1・y1 は含まない**",
    "**`n_px_match` / `n_px_differ` は墨の濃さの表の `n_px_alpha_gt0` との突き合わせ**(既知セル。同じ供給元の別の器具)。"
    "**違う2層は1段目(10-30)の表が古い側** —— 1ビットの層を読めるようにした 10-31 より前に作ったので空として読んだ"
    "(供給元の説明。**墨の濃さの1段目の2行は古いまま**)",
    "**`n_bbox_inconsistent` は範囲が負・逆向き・描かれた画素より狭い・キャンバスより広い層**。"
    "**キャンバスの画素は面の表の `n_px_total` から引いた**(同じ1,724頁)",
    "**`median_bbox_share_of_canvas` は範囲の面積 ÷ キャンバスの画素、`median_fill_of_bbox` は描かれた画素 ÷ 範囲の面積**"
    "(どちらも `empty` を外した層の中央値)。**線画の層が頁のどれだけに広がるかの目安で、タイルとの重なりではない**",
    "**N の問い(頁の座標とタイルの `page_bbox`)への供給元の答え**: **画素の単位は同じで縮小は無い。原点が違う** —— "
    "**頁の x = コマの x0 + 小さな領域の `sub_x0` + `page_bbox` の x**(y も同じ)。"
    "**census は照合していない**(供給元が頼んだ担当が `lineart` の器具をたどった。4,133枚で一致と報告)",
]

NOTE_FACE = [
    "**Track N の発注**(2026-11-03)**への納品の集計**(抽出側 11-04)。**読むのは Track M と基盤**"
    "(ユーザー 11-03「マスク(=面)調査にも効く要素」)。**層ごとの全量は頁のパスを持つので公開しない**",
    "**頁は線画の範囲と同じ 1,724頁**(墨の濃さの層別 L1 を持つ頁 − 恒久除外の頁)。**全頁の面の層ではない**",
    "**`plane = body`**: フォルダーでない raster / vector / mask_fill / gradation / image_material の層から層別 L1・L2 を除いたもの"
    "(ベタは入る)。**アルファを数えた。** **`mask_fill` の本体はほとんどが値 0** —— **形はマスクの側に在る**",
    "**`plane = mask`**: その頁でマスクを持つ層全部(**フォルダー・コマ枠・線画の層も入る**)。"
    "**遮蔽・トーンと同じ読み方の値。保存されていないブロックとバッファの外は初期値で、それも含めてキャンバス全体で数えた**。"
    "**初期値が 255 の面は「全部 0」になりにくい** —— **初期値の列を隣に置いた**",
    "**`n_layers_binary_alpha` は 0 より大の画素が在って、1〜254 の値が無い層**(**判定ではない。数えただけ**)。"
    "**`share_layers_binary_alpha` の分母は 0 より大の画素が在る層**(`n_layers_counted − n_layers_all_zero`)。"
    "**`share_px_partial_alpha` は 1〜254 の画素の和 ÷ 0 より大の画素の和**(層を足した比。線画の表の同じ名前の列と同じ定義)",
    "**外した層は列に残した**(規律9): **形式5 の本体**(調査中。アルファが全面 255 と読まれる件が解けていない)/ "
    "**画素の面が無い本体** / **マスクの面を読めない層**(Mipmap か Offscreen が無い)。**どれも `n_layers_counted` に入らない**",
    "**既知セル**: マスクの面は `mask_cover_request3_v2` と重なる層の `n_px_gt0` と初期値が全部一致、"
    "本体は `faint_alpha_905_v2` と重なる層の `n_px_gt0` が丸めの幅で全部一致(器具が印字する)。"
    "**どちらも同じ供給元の別の器具** —— **構造 DB は画素を持たないので、供給元の外の既知セルは無い**",
    "**本体の `gradation` は数えた層が全部値 0 だった**(2026-11-04 の走行)。**グラデーションの形がどこに在るかを census は知らない** —— "
    "**「描かれていない」と読まない**(`mask_fill` の本体と同じく、値 0 は形が無いことを意味しない)",
    "**マスクの値の向きは読んでいない** —— 初期値 255 と 0 で「通す/隠す」の側が逆になる(`mask_cover` の便 11-02)。"
    "**1〜254 の値は縁の画素の上界で、縁の画素そのものではない**",
]


def main() -> int:
    print("[発注2件の納品(線画の層の範囲 / 面の層のアルファ)]")
    failures = []
    if not check_sha():
        failures.append("sha256 が便と違う")
    extent(failures)
    face(failures)
    for f in failures:
        print(f"  **指摘**: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
