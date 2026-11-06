#!/usr/bin/env python
"""**マスクの面の材料を持つ頁の一覧** —— Track M の発注(2026-11-06、作者の承認あり)。

**作者の承認(逐語)**: 「**2と3を一緒にやる １を並行してだす**」(1 = census にマスクの材料を
持つ頁を探してもらう案)。発注 `note_m_order_pages_with_mask_material_20261106.md`。

**材料の層**(Track M の追補194 の落とし方。**M の器具 `m116_mask_material_adopted4.py` の
規則をそのまま写した** —— 規則を census が作り直さない):

1. **可視の葉**: `visibility & 1` が立ち、**祖先のフォルダーも全部立つ**
2. **マスクを持つ**: `has_mask`
3. **全面を隠さない / 全面を通さない**: マスクの値で決める(下)
4. **線でない**: 名前に `線画` を含まない(**M の名前の規則**。作者の裁定 追補189)
5. **ハイライトでない**: 名前に M の語(`high` / `ハイライト` / `ﾊｲﾗｲﾄ` / `発光` / `光` /
   `glow` / `shine` / `艶`)を含まない(**M の名前の規則**。作者の裁定 追補190)
6. **filter でない**(作者「色調補正レイヤは一旦無視」追補195)/ **text でない**(作者の裁定、セリフ)

**順は M の器具と同じ**(全面 → 線 → ハイライト → filter → text)。**1層は最初に当たった理由
1つで落とす。**

**マスクの値が在る範囲だけ 3 を判定できる。** 材料は2つ:

- **面の表**(`census_face_layer_alpha_mask_20261104.csv` の `plane = mask`、1,724頁、
  **その頁でマスクを持つ層全部**。キャンバス全体で数えた値)
- **`mask_cover_request3_v2`**(依頼3 の頁、層の種別を限った表)

**読み方**: 供給元の言明「初期値 255 = 全面を通す」(`mask_cover` の便 11-02)に従い、
**値 255 が通す側。** **全面通す = `n_px_255 == n_px_total`、全面隠す = `n_px_gt0 == 0`。**
**`mask_cover` はバッファの範囲で数えているのでキャンバス全体と同じではない** ——
**面の表に行が在れば面の表を使い、無いときだけ `mask_cover` を使う**(どちらを使ったかを印字)。

**値の無い層は「未測定」で、3 を判定しない。** **だから頁の材料は2列**:
`n_material`(3 を判定できた層だけで数えた。未測定の層が在る頁は下限)と
`n_material_upper`(未測定の層を材料に数えた上界)。**未測定を「材料でない」とも「材料である」とも読まない。**

    ./venv/bin/python tools/census/mask_material_pages.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_census as bc  # noqa: E402
import db  # noqa: E402
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE = "inbox/census_face_layer_alpha_mask_20261104.csv"
MASK_COVER = "inbox/mask_cover_request3_v2_20261102.csv"
OUT_FULL = "data/census_full/mask_material_pages_20261107.csv"
OUT_PUB = "census/mask_material_pages.csv"
LINE = ("線画",)
HIGHLIGHT = ("high", "ハイライト", "ﾊｲﾗｲﾄ", "発光", "光", "glow", "shine", "艶")
# **M の採用4頁の既知セル**(追補194 §B。**M が自分の納品物のラスタで数えた値**、この器具の入力ではない)。
# 頁の名前の末尾 → (可視の has_mask 葉, 全面通す, 線, ハイライト, filter, 材料)
KNOWN = {r"2022\sketch\sketch\page0025.clip": (1, 1, 0, 0, 0, 0),   # **札 2022**(M の追補194 §E。2022b にも同名の頁が在る)
         r"hyoui_color\top.clip": (3, 1, 0, 0, 2, 0),
         r"hitoduma\CDrev2.clip": (2, 1, 0, 0, 1, 0),
         r"gakusai3\新規キャンバス.clip": (11, 0, 0, 0, 0, 11)}


def load_values():
    face, cover = {}, {}
    for r in csv.DictReader(open(os.path.join(ROOT, FACE), encoding="utf-8")):
        if r["plane"] != "mask":
            continue
        k = (r["page_path"], int(r["layer_id"]))
        if r["status"] != "ok":
            face[k] = "unread"
        else:
            gt0, a255, tot = int(r["n_px_gt0"]), int(r["n_px_255"]), int(r["n_px_total"])
            face[k] = "hide" if gt0 == 0 else "pass" if a255 == tot else "shape"
    for r in csv.DictReader(open(os.path.join(ROOT, MASK_COVER), encoding="utf-8")):
        k = (r["page_path"], int(r["layer_id"]))
        if r["error"]:
            cover[k] = "unread"
            continue
        ge, mid = int(r["mask_px_ge128"]), int(r["mask_px_1_127"])
        area = int(r["mask_w"]) * int(r["mask_h"])
        # **バッファの範囲の値。** 全部 255 側かは ge128 で近似しない —— 全面通すと言えるのは
        # 0 より大の画素が無い(隠す)か、バッファ全体が 128 以上で 1〜127 が無いときだけ
        cover[k] = ("hide" if ge + mid == 0 else
                    "pass_buffer" if ge == area and mid == 0 else "shape")
    return face, cover


def main() -> int:
    print("[マスクの面の材料を持つ頁(Track M の発注)]")
    face, cover = load_values()
    con = db.connect()
    excl = exclusions.load()
    rows = con.execute("""
        select l.page_path, l.layer_id, l.parent_id, l.folder, l.visibility, l.has_mask,
               coalesce(l.name, ''), k.v_layer_kind, l.tone_enabled
          from layers l join v_layer_kind k using (page_path, layer_id)
         where lower(l.page_path) like '%.clip'""").fetchall()
    by_page = collections.defaultdict(list)
    for r in rows:
        if r[0] not in excl:
            by_page[r[0]].append(r)
    print(f"    `.clip` の頁 {len(by_page):,}(層の在る頁。恒久除外の頁を除いた)/ 層 {sum(map(len, by_page.values())):,}")
    full, src_count = [], collections.Counter()
    for p in sorted(by_page):
        ls = by_page[p]
        kids = collections.defaultdict(list)
        info = {}
        for (_, lid, parent, folder, vis, hm, name, kind, tone) in ls:
            info[lid] = (folder, vis, hm, name, kind, tone)
            kids[parent].append(lid)
        visible = set()
        stack = [lid for lid in kids[None] if info[lid][1] & 1]
        while stack:                              # **祖先のフォルダーも全部立つ層だけを降りる**
            lid = stack.pop()
            visible.add(lid)
            if info[lid][0]:
                stack += [c for c in kids[lid] if info[c][1] & 1]
        c = collections.Counter()
        for lid, (folder, vis, hm, name, kind, tone) in info.items():
            if folder or kids.get(lid):
                continue
            c["leaf"] += 1
            is_line = any(t in name for t in LINE)
            if lid in visible and is_line:
                c["line_layers"] += 1
            if not hm:
                continue
            c["leaf_hm"] += 1
            if lid not in visible:
                c["drop_hidden"] += 1
                continue
            c["vis_hm"] += 1
            k = (p, lid)
            v, src = (face[k], "face") if k in face else (cover[k], "cover") if k in cover else (None, "none")
            src_count[src] += 1
            if v == "unread":
                c["unread"] += 1
            elif v is None:
                c["unmeasured"] += 1
            why = None
            if v == "hide":
                why = "full_hide"
            elif v == "pass" or v == "pass_buffer":
                why = "full_pass"
            elif is_line:
                why = "line_hl_text"
            elif any(t in name.lower() or t in name for t in HIGHLIGHT):
                why = "line_hl_text"
            elif kind == "filter":
                why = "filter"
            elif kind == "text":
                why = "line_hl_text"
            if why:
                c["drop_" + why] += 1
            elif v == "shape":
                c["material"] += 1
                c["material_upper"] += 1
                c["material_tone"] += bool(tone)    # **トーンが有効な層**(構造の欄。モノクロのトーンの面がここに入る)
            else:                                 # 未測定・読めない: **上界にだけ数える**
                c["material_upper"] += 1
        st = ("値なし(マスクを持つ可視の葉が無い)" if c["vis_hm"] == 0 else
              "取れない層あり" if c["unread"] else
              "未測定" if c["unmeasured"] == c["vis_hm"] else
              "一部未測定" if c["unmeasured"] else "全部測定済み")
        full.append([p, p.split("\\")[2], c["leaf"], c["leaf_hm"], c["vis_hm"],
                     c["material"], c["material_upper"],
                     c["drop_hidden"], c["drop_full_pass"], c["drop_full_hide"], c["drop_filter"],
                     c["drop_line_hl_text"], c["unmeasured"], c["unread"], c["line_layers"],
                     c["material_tone"], st])
    print(f"    マスクの値の出所(可視の has_mask 葉): 面の表 {src_count['face']:,} / "
          f"`mask_cover` {src_count['cover']:,} / **未測定 {src_count['none']:,}**")
    st_count = collections.Counter(r[-1] for r in full)
    print("    頁の状態: " + " / ".join(f"{k} {n:,}" for k, n in st_count.most_common()))
    ge3 = sum(1 for r in full if r[5] >= 3)
    ge3u = sum(1 for r in full if r[6] >= 3)
    print(f"    **材料 ≥ 3 の頁: 測定済みの層だけで {ge3} / 上界で {ge3u}**")
    tot = [sum(r[i] for r in full) for i in range(7, 14)]
    print(f"    落とした層(可視の has_mask 葉の外も含む): 非表示 {tot[0]:,} / 全面通す {tot[1]:,} / "
          f"全面隠す {tot[2]:,} / filter {tot[3]:,} / 線・ハイライト・text {tot[4]:,} / "
          f"未測定 {tot[5]:,} / 読めない {tot[6]:,}")

    # 既知セル: M の採用4頁
    fails = 0
    for tail, want in KNOWN.items():
        hit = [r for r in full if r[0].endswith("\\" + tail)]
        if len(hit) != 1:
            print(f"    既知セル {tail}: **頁が {len(hit)} 件**")
            fails += 1
            continue
        r = hit[0]
        got = (r[4], r[8], 0, 0, r[10], r[5])
        lh = r[11]
        ok = got[0] == want[0] and got[1] == want[1] and got[4] == want[4] and got[5] == want[5] \
            and lh == want[2] + want[3]
        print(f"    既知セル {tail}: M {want} / census(可視hm, 全面通す, 線+HL {lh}, filter, 材料)"
              f" ({r[4]}, {r[8]}, {r[10]}, {r[5]}) 状態 {r[-1]} {'一致' if ok else '**不一致**'}")
        fails += not ok

    header = ["page_path", "year_label", "n_leaf", "n_leaf_has_mask", "n_visible_has_mask",
              "n_material", "n_material_upper", "n_drop_hidden", "n_drop_full_pass",
              "n_drop_full_hide", "n_drop_filter", "n_drop_line_hl_text", "n_unmeasured",
              "n_unread", "n_line_layers", "n_material_tone", "mask_raster_status"]
    with open(os.path.join(ROOT, OUT_FULL), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(full)
    agg = collections.defaultdict(collections.Counter)
    for r in full:
        a = agg[r[1]]
        a["n_pages"] += 1
        a["n_pages_measured_all"] += r[-1] == "全部測定済み"
        a["n_pages_material_ge3"] += r[5] >= 3
        a["n_pages_material_upper_ge3"] += r[6] >= 3
        a["n_pages_material_ge3_with_line"] += r[5] >= 3 and r[14] > 0
        a["n_pages_material_ge3_without_tone"] += r[5] - r[15] >= 3
        a["n_material_tone"] += r[15]
        for i, name in zip(range(4, 14), header[4:14]):
            a[name] += r[i]
    cols = ["n_pages", "n_pages_measured_all", "n_pages_material_ge3", "n_pages_material_upper_ge3",
            "n_pages_material_ge3_with_line", "n_pages_material_ge3_without_tone",
            *header[4:14], "n_material_tone"]
    out = [[y, *(agg[y][c] for c in cols), ""] for y in sorted(agg)]
    out.append(["ALL", *(sum(agg[y][c] for y in agg) for c in cols), ""])
    out += [[""] * (len(cols) + 1) + [n] for n in NOTE]
    n_pub = bc.write_csv(os.path.join(ROOT, OUT_PUB), ["year_label", *cols, "note"], out)
    print(f"    {OUT_PUB} {n_pub}行 / 頁ごとの全量 {len(full):,}行(`{OUT_FULL}`)")
    return 1 if fails else 0


NOTE = [
    "**Track M の発注**(2026-11-06。作者の承認「2と3を一緒にやる １を並行してだす」)**への答え。** "
    "**頁ごとの表は頁のパスを持つので公開しない**(`data/census_full/mask_material_pages_20261107.csv`。写しを M と基盤へ)",
    "**材料の層 = 可視の葉(祖先のフォルダーも全部立つ)で `has_mask`、全面を隠さず全面を通さず、"
    "線(名前に `線画`)・ハイライト(M の語)・filter・text でない層**。**落とし方は Track M の器具の規則をそのまま写した**"
    "(順も同じ。1層は最初に当たった理由1つで落とす)。**線とハイライトは名前の規則で、作者の分類を再現するかは示していない**(規律13)",
    "**全面通す・全面隠すはマスクの値でしか決まらない。** **値が在るのは面の表(1,724頁、その頁でマスクを持つ層全部)と "
    "`mask_cover_request3_v2`(依頼3 の頁)だけ** —— **それ以外の層は「未測定」**。"
    "**`n_material` は測定済みの層だけで数えた下限、`n_material_upper` は未測定の層も材料に数えた上界**",
    "**読み方は供給元の言明「初期値 255 = 全面を通す」**(`mask_cover` の便 11-02)。**全面通す = 値 255 がキャンバス全体、"
    "全面隠す = 0 より大の画素が無い。** **`mask_cover` はバッファの範囲の値なので、面の表に行が無いときだけ使った**",
    "**`n_line_layers` は可視の葉で名前に `線画` を含む層**(M の線の規則と同じ。**線が描かれているかは見ていない**)。"
    "**`n_pages_material_ge3_with_line` は材料 ≥ 3(下限)で、その線の層が1枚以上在る頁**",
    "**`mask_raster_status` は census が値を持つかの別であって、マスクのラスタが納品済みかではない** —— "
    "**ラスタの納品は抽出側と M の木に在り、census は数を持たない**",
    "**フォルダーのマスクは数えていない**(葉だけ。フォルダーのマスクは配下を切り抜く —— 抽出側 11-02)。"
    "**年の札は頁のパスの札**(作品の最も早い札ではない。頁の単位なので)",
    "**`n_material_tone` は材料のうちトーンが有効な層**(構造の欄 `tone_enabled`)。**モノクロ原稿のトーンの面がここに入る** —— "
    "**材料の過半はトーン**(2026-11-07 の走行で 28,185層中 15,536)。**`n_pages_material_ge3_without_tone` はトーンを除いても材料 ≥ 3 の頁**。"
    "**トーンを除くかは M と作者が決める**(census はカラーかモノクロかを知らない。作者の分類を再現する列は無い)",
    "**測定済みの頁は札 2015〜2019 に偏る** —— 面の表は墨の濃さの層別 L1 を持つ頁から作ったので。**札 2020 以降の頁はほとんど未測定で、"
    "材料が無いのではない**",
    "**既知セル**: Track M の採用4頁(追補194 §B。M が納品のラスタで数えた値)。器具が一致を印字する",
]

if __name__ == "__main__":
    raise SystemExit(main())
