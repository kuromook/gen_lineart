#!/usr/bin/env python
"""**Q13b: 合成を直す材料の層の一覧**(Track Q の依頼、2026-11-22。ユーザーの承認「①〜④で発注」11-23)。

頁 = Q13 の便で `status = 描いた` の 182頁(`art_file` の在る 160頁は ①〜④、無い 22頁は ③④だけ)。
層ごとに作る画を1つ決める:
- ① `art_out` = Q13 の清書の層(便の点検表の `role = line`)のうち、祖先にコマ枠のフォルダーが無い層
- ② `art_in` = 同じく祖先にコマ枠のフォルダーが在る層。**祖先のコマ枠フォルダーのマスクで切る**(`frame_folder_ids`)
- ③ `over_out` = フキダシ・描き文字の層で、祖先にコマ枠のフォルダーが無い層。**述語**: フォルダー・コマ枠フォルダー・
  テキスト・ストーリー情報・用紙でない層で、層名か祖先のフォルダー名が `OVER` に当たる、または `has_balloon_index`。
  **見えている形(塗りの白を含む)**。清書の層と重なったら ①② に入れて ③ に入れない(数を印字する)
- ④ `text_out` = `v_layer_kind = text` で、祖先にコマ枠のフォルダーが無い層
- 祖先のコマ枠フォルダーがモザイクの名前だけなら別に数える(中にも外にも入れない)。中の ③④ は入れずに数える(規律9)
- 名前の当て方の率(規律13): 聞き取りの 265層(役割の表 v3)で、作者がコメントに描き文字・フキダシと書いた層を名前が拾う数と、
  名前が拾ってコメントの無い層の数を印字する。**名前の無いフキダシは拾えない**(遮蔽物の表で素性だけの層が多い)

出力 `outbox/q13b_popout_art_order_20261123.csv`(頁の路を持つ。追跡しない)。層名は書かない。
"""
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import exclusions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEL = os.path.join(ROOT, "..", "lineart-layer-exclusion", "data", "census")
PAGES = os.path.join(DEL, "q13_synthetic_frames_pages_20261122.csv")
CHECK = os.path.join(DEL, "q13_synthetic_frames_layers_check_20261122.csv")
ROLE = os.path.join(ROOT, "data", "census_full", "layer_role_table_20261112_v3.csv")
OUT = os.path.join(ROOT, "outbox", "q13b_popout_art_order_20261123.csv")
OVER = re.compile(r"(描き文字|フキダシ|ふきだし|吹き出し|書き文字|かき文字)")
SKIP = ("folder", "frame_folder", "story_info", "paper")
MOZA = "モザイク"


def main() -> int:
    lab = [r for r in csv.DictReader(open(ROLE, encoding="utf-8")) if r["author_stage"]]
    said = [r for r in lab if OVER.search(r["author_comment"] or "")]
    hit = [r for r in lab if OVER.search(r["name"] + "/" + r["ancestors"])]
    print(f"名前の当て方の率(聞き取りの {len(lab)}層): 作者がコメントで描き文字・フキダシと書いた {len(said)}層のうち名前が拾う "
          f"{sum(1 for r in said if OVER.search(r['name'] + '/' + r['ancestors']))} / 名前が拾った {len(hit)}層のうちコメントの無い "
          f"{sum(1 for r in hit if not OVER.search(r['author_comment'] or ''))}(作者の段: " + " / ".join(
              f"{k} {v}" for k, v in collections.Counter(r['author_stage'] for r in hit).most_common()) + ")")
    pg = {r["page_path"]: r for r in csv.DictReader(open(PAGES, encoding="utf-8"))}
    ex = exclusions.load()
    ps = sorted(p for p, r in pg.items() if r["status"] == "描いた" and p not in ex)
    art = {p for p in ps if pg[p]["art_file"]}
    print(f"頁: Q13 で描いた {sum(r['status'] == '描いた' for r in pg.values())} − 恒久除外 "
          f"{sum(1 for p, r in pg.items() if r['status'] == '描いた' and p in ex)} = {len(ps)}(art の在る頁 {len(art)})")
    line = {(r["page_path"], int(r["layer_id"])) for r in csv.DictReader(open(CHECK, encoding="utf-8")) if r["role"] == "line"}
    c = db.connect()
    rows = c.execute(f"select l.page_path, l.layer_id, l.parent_id, coalesce(l.name,''), v.v_layer_kind, l.has_balloon_index, l.folder "
                     f"from layers l join v_layer_kind v using (page_path, layer_id) where l.page_path in ({','.join('?' * len(ps))})",
                     ps).fetchall()
    info = {(r[0], r[1]): r for r in rows}
    out, drop = [], collections.Counter()
    for r in sorted(rows, key=lambda x: (x[0], x[1])):
        k = (r[0], r[1])
        if r[6] or r[4] in SKIP:
            continue
        anc, p = [], r[2]
        while p is not None:
            a = info[(r[0], p)]; anc.append(a); p = a[2]
        ffs = [a for a in anc if a[4] == "frame_folder"]
        moza_only = bool(ffs) and all(MOZA in a[3] for a in ffs)
        ffs = [a for a in ffs if MOZA not in a[3]]
        inside = bool(ffs)
        is_line = k in line and r[0] in art
        is_over = bool(OVER.search(r[3]) or any(OVER.search(a[3]) for a in anc) or r[5])
        if is_line:
            img = "art_in" if inside else "art_out"
            if is_over:
                drop["③ の述語にも当たる清書の層(①② に入れた)"] += 1
        elif r[4] == "text":
            if inside:
                drop["④ テキストの層で中(入れない)"] += 1; continue
            img = "text_out"
        elif is_over:
            if inside:
                drop["③ フキダシ・描き文字で中(入れない)"] += 1; continue
            img = "over_out"
        else:
            continue
        if moza_only:
            drop[f"{img} の候補で祖先のコマ枠がモザイクの名前だけ(入れない)"] += 1; continue
        q = pg[r[0]]["q"]
        out.append(dict(q=q, split=pg[r[0]]["split"], page_path=r[0], layer_id=r[1], image=img,
                        out_file=f"{q}_{img}.png", v_layer_kind=r[4],
                        frame_folder_ids=";".join(str(a[1]) for a in ffs) if img == "art_in" else ""))
    print("入れずに数えた: " + (" / ".join(f"{k} {v}" for k, v in drop.items()) or "0"))
    n = collections.Counter(o["image"] for o in out)
    npg = {i: len({o["page_path"] for o in out if o["image"] == i}) for i in n}
    for i in ("art_out", "art_in", "over_out", "text_out"):
        print(f"  {i}: 層 {n[i]:,} / 頁 {npg.get(i, 0)}(うち art の無い頁 "
              f"{len({o['page_path'] for o in out if o['image'] == i and o['page_path'] not in art})})")
    print("  over_out の種別: " + " / ".join(f"{k} {v}" for k, v in collections.Counter(
        o["v_layer_kind"] for o in out if o["image"] == "over_out").most_common()))
    print(f"  層の行 {len(out):,} / 画 {len({o['out_file'] for o in out}):,} 枚 / 頁 {len({o['page_path'] for o in out})}")
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
    print(f"出力: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
