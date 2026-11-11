#!/usr/bin/env python
"""**層の役割の予測器のための1層1行の表** —— Track E の発注(`note_census_role_predictor_column_order_20261109.md`)。

**母集団**: 抽出側の画素の数え上げの一覧(`stage_layers_order_20261109.csv`、53,243層・2,108頁)。
**聞き取りの 265層を全部含む**(含まなければ出さない)。

**列**:
1. ラベル: `author_stage` / `author_comment`(作者の答えのまま)/ `author_stage_excl_textish`(描き文字・フキダシの名前の層を空にした版。
   **作者の但し書き**「描き文字関係は属性入力でぶれてしまったかもしれない」)/ `textish`
2. 名前: `name` / `ancestors`(祖先のフォルダー名、上から)/ `name_rule`(census の名前の当て方。下限の棒)
3. 構造: 種別・層の型・表現色・表示(自身 / 祖先)・不透明度・合成モード・クリップ・マスク・トーン・
   頁の中の重なり順(フォルダーでない層を手前から数えた位置と、その総数)・キャンバス寸法
4. 画素: 抽出側の数え上げの列。**`n_px_a_gt0_all_masks` が空の 505層は、コマ枠のベクターで作ったマスクの値で埋め、
   `all_masks_from_koma_vector = 1` を立てる**(抽出側の照合: 4,409層で差 1% 以内 99.4%。保存されたマスクの値ではない)
   / `empty`(1画素も描かれていない)。**定義(2026-11-11 に書いた。値は変えていない)**: `empty = 1` ⇔ 抽出側の `render_status` が
   「1画素も描かれていない」(v2 では `n_px_a_gt0 = 0` の 4,977層と完全一致)。**マスクは見ていない** —— 作者の裁定(11-11)「画素無くても
   マスクがあればその層は空ではない」により、**この列は「完全に空」ではない**(E の定義D: 祖先のマスクも無い等 → v2 で 1,664層)。
   **「描画できなかった」15層は `empty = 0` だが、意味は「空でないと分かった」ではなく「測れない」**
5. 分け方の鍵: `work_dir` / `split_unit`(`split_units.py`: 編の裁定と中身の共有で束ねた単位。全322作品フォルダー)
6. `why_listed`(名前 / 聞き取り)

**答えを見る前に宣言した数**(`stage_hearing_answers.py` の冒頭と 11-09 の作業ログ): **空の層を除いた名前の当て方の棒を、
除く前と並べて印字する。**

    ./venv/bin/python tools/census/layer_role_table.py
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db  # noqa: E402
import split_units  # noqa: E402
from p_split_columns import wd  # noqa: E402
from stage_hearing import name_rule  # noqa: E402
from stage_hearing_answers import OUT as SCORED, TEXTISH  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDER = "outbox/stage_layers_order_20261109.csv"
ORDER_2A = "outbox/stage_layers_order_round2a_20261110.csv"   # 便A の残り 46層(11-10 追加)
PIX = "inbox/stage_layer_pixels_20261109.csv"
PIX_2A = "inbox/stage_layer_pixels_round2a_20261110.csv"      # sha256 先頭 392ef9ef0e689287
DRAW = "inbox/layer_draw_colour_20261110.csv"                  # 描画色。sha256 先頭 4e33f270e731c2bf
DRAW_SHA = "4e33f270e731c2bffc771631c5951b9f78345f18d9c9a461e5f48e89c636f817"
P300 = "data/census_full/p_order300_page_layers_20261110_v2.csv"   # 描画色の既知セル(manifest から census が組んだ表)
KOMA = "inbox/stage_layer_pixels_koma_vector_20261110.csv"
KEY = "data/census_full/hearing_drawing_stages_key_20261109.csv"
OUT = "data/census_full/layer_role_table_20261110_v2.csv"   # v1(53,243層・描画色なし)は残す
PIXCOLS = ["render_status", "render_error", "n_px_a_gt0", "n_px_a_ge26", "n_px_a_gt0_own_mask", "n_px_a_gt0_all_masks",
           "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "has_own_mask", "n_ancestor_masks", "mask_status"]


def page_order(con, pages):
    """頁ごと、フォルダーでない層を手前から数えた位置・祖先の名前・祖先が全部表示か"""
    out = {}
    q = ",".join("?" * len(pages))
    rows = con.execute(f"""select page_path, layer_id, parent_id, order_in_parent, coalesce(name, ''), folder, visibility
                             from layers where page_path in ({q})""", pages).fetchall()
    by = collections.defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    for p, rs in by.items():
        ids = {r[1] for r in rs}
        kids = collections.defaultdict(list)
        for r in rs:
            kids[r[2] if r[2] in ids else None].append(r)
        seq = []

        def walk(pid, names, shown):
            for r in sorted(kids.get(pid, []), key=lambda r: -(r[3] if r[3] is not None else -1)):
                if r[5]:
                    walk(r[1], names + ([r[4]] if r[2] is not None else []), shown and bool(r[6] & 1))
                else:
                    seq.append((r[1], "/".join(names), shown))
        for r in kids[None]:
            if r[5]:
                walk(r[1], [], bool(r[6] & 1))
            else:
                seq.append((r[1], "", True))
        for i, (lid, anc, shown) in enumerate(seq, 1):
            out[(p, lid)] = (i, len(seq), anc, int(shown))
    return out


DRAWCOLS = ["draw_color_enable", "draw_color_main", "draw_color_sub", "draw_color_read_error"]


def load_draw():
    """抽出側の描画色の表(11-10)。sha を確かめ、行が一覧と1対1か、既知セル(P の300頁の表)と一致するかを印字する。

    **値の意味は抽出側の推定**(CSP のレイヤーカラー = 表示の置き換え。線の画素は黒のまま)。**下描きを表すかは照らしていない**(規律13)。
    P の表は `[r, g, b]` の Python の並び、こちらは "r g b" の文字列なので、同じ形に直して比べる。
    """
    import hashlib
    path = os.path.join(ROOT, DRAW)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if sha != DRAW_SHA:
        print(f"    **描画色の表の sha256 が違う** {sha[:16]}")
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    d = {}
    for r in rows:
        k = (r["page_path"], int(r["layer_id"]))
        if k in d:
            print(f"    **描画色の表に2行ある層** {k}")
            return None
        d[k] = dict(draw_color_enable=r["draw_color_enable"], draw_color_main=r["draw_color_main"].strip(),
                    draw_color_sub=r["draw_color_sub"].strip(), draw_color_read_error=r["read_error"])
    print(f"    描画色の表 {len(rows):,}行(sha256 {sha[:16]})")

    def norm(v):
        v = (v or "").strip()
        if v.startswith("["):
            parts = [x.strip() for x in v.strip("[]").split(",")]
            return " ".join("" if x == "None" else x for x in parts).strip()
        return v
    if os.path.exists(os.path.join(ROOT, P300)):
        n = same = 0
        diff = collections.Counter()
        for r in csv.DictReader(open(os.path.join(ROOT, P300), encoding="utf-8")):
            k = (r["page_path"], int(r["layer_id"]))
            if k not in d:
                continue
            n += 1
            ok = True
            for cn in ("draw_color_main", "draw_color_sub"):
                if norm(r[cn]) != d[k][cn]:
                    diff[cn] += 1
                    ok = False
            pe = r["draw_color_enable"].strip()
            pe = {"True": "1", "1": "1", "False": "", "0": "", "None": ""}.get(pe, pe)
            if pe != d[k]["draw_color_enable"]:
                diff["draw_color_enable"] += 1
                ok = False
            same += ok
        print(f"    **既知セル**(P の300頁の表と重なる層 {n:,}): 3列とも一致 {same:,} / 違う列 "
              + (" / ".join(f"{a} {b}" for a, b in diff.items()) or "なし"))
    return d


def main() -> int:
    print("[層の役割の表(Track E の発注)]")
    con = db.connect()
    order = list(csv.DictReader(open(os.path.join(ROOT, ORDER), encoding="utf-8")))
    order += list(csv.DictReader(open(os.path.join(ROOT, ORDER_2A), encoding="utf-8")))
    pix = {(r["page_path"], int(r["layer_id"])): r for r in csv.DictReader(open(os.path.join(ROOT, PIX), encoding="utf-8"))}
    pix2 = {(r["page_path"], int(r["layer_id"])): r for r in csv.DictReader(open(os.path.join(ROOT, PIX_2A), encoding="utf-8"))}
    print(f"    便A の追加の画素 {len(pix2)}層 / 前の表と重なる {len(set(pix2) & set(pix))}")
    pix.update(pix2)
    draw = load_draw()
    if draw is None:
        return 1
    koma = {(r["page_path"], int(r["layer_id"])): r for r in csv.DictReader(open(os.path.join(ROOT, KOMA), encoding="utf-8"))}
    key = {(r["page_path"], int(r["layer_id"])): r for r in csv.DictReader(open(os.path.join(ROOT, KEY), encoding="utf-8"))}
    sc = {(r["page_path"], int(r["no"]), int(r["pos"])): r
          for r in csv.DictReader(open(os.path.join(ROOT, SCORED), encoding="utf-8"))}
    lab = {}
    for (p, lid), k in key.items():
        lab[(p, lid)] = sc[(p, int(k["no"]), int(k["pos"]))]
    keys = [(r["page_path"], int(r["layer_id"])) for r in order]
    miss_lab = set(lab) - set(keys)
    miss_pix = [k for k in keys if k not in pix]
    print(f"    一覧 {len(keys):,}層 / 画素の行の無い層 {len(miss_pix)} / **聞き取りの層のうち一覧に無い {len(miss_lab)}**")
    if miss_lab or miss_pix:
        return 1
    un = split_units.units(con)
    pages = sorted({p for p, _ in keys})
    po = page_order(con, pages)
    q = ",".join("?" * len(pages))
    st = {(r[0], r[1]): r for r in con.execute(f"""
        select l.page_path, l.layer_id, coalesce(l.name, ''), k.v_layer_kind, l.layer_type, l.colour_type_index,
               l.visibility & 1, l.opacity, l.blend_mode, l.clip_raw, l.has_mask, coalesce(l.tone_enabled, false),
               p.v_canvas_px_w, p.v_canvas_px_h, p.year
          from layers l join v_layer_kind k using (page_path, layer_id) join pages p using (page_path)
         where l.page_path in ({q})""", pages).fetchall()}
    head = ["page_path", "layer_id", "work_dir", "split_unit", "year_label", "why_listed",
            "author_stage", "author_stage_excl_textish", "author_comment", "textish",
            "name", "ancestors", "name_rule",
            "v_layer_kind", "layer_type", "colour_type_index", "self_shown_db", "ancestors_shown_db", "opacity",
            "blend_mode", "clip_raw", "has_mask", "tone_enabled", "pos_from_top", "n_leaf_layers",
            "canvas_px_w", "canvas_px_h"] + PIXCOLS + ["all_masks_from_koma_vector", "empty"] + DRAWCOLS
    out, c = [], collections.Counter()
    for k in keys:
        s, x, o = st[k], pix[k], po[k]
        v = dict(x)
        fill = 0
        if k in koma and not x["n_px_a_gt0_all_masks"]:
            v["n_px_a_gt0_all_masks"], v["mask_status"], fill = koma[k]["n_px_a_gt0_all_masks"], koma[k]["mask_status"], 1
        c["コマ枠のベクターで埋めた"] += fill
        c["all_masks がまだ空"] += not v["n_px_a_gt0_all_masks"]
        lb = lab.get(k)
        tx = bool(TEXTISH.search(s[2]))
        empty = int(x["render_status"] == "1画素も描かれていない")
        c["空の層"] += empty
        c["描画できなかった"] += x["render_status"] == "描画できなかった"
        out.append([k[0], k[1], wd(k[0]), un[wd(k[0])].split("\\", 2)[-1], s[14], "",
                    lb["author"] if lb else "", (lb["author"] if lb and not tx else ""), lb["comment"] if lb else "", int(tx),
                    s[2], o[2], name_rule(s[2]),
                    s[3], s[4], s[5], s[6], o[3], s[7], s[8], s[9], int(bool(s[10])), int(bool(s[11])), o[0], o[1],
                    s[12], s[13]] + [v[cn] for cn in PIXCOLS] + [fill, empty] + [draw[k][cn] for cn in DRAWCOLS])
    why = {(r["page_path"], int(r["layer_id"])): r["why_listed"] for r in order}
    for row in out:
        row[5] = why[(row[0], row[1])]
    print("    " + " / ".join(f"{a} {b:,}" for a, b in c.items()))
    print(f"    作品フォルダー {len({r[2] for r in out})} / 単位 {len({r[3] for r in out})}")
    # 宣言した数: 空の層を除いた名前の当て方の棒を、除く前と並べる
    labelled = [r for r in out if r[6]]
    nd = len(DRAWCOLS)
    print(f"    描画色: enable=1 {sum(r[-nd] == '1' for r in out):,} / ラベルの在る層のうち enable=1 "
          f"{sum(r[-nd] == '1' for r in labelled)} / read_error あり {sum(bool(r[-1]) for r in out):,}")
    ie = head.index("empty")
    print(f"    **ラベルの在る層 {len(labelled)}**(うち空の層 {sum(r[ie] for r in labelled)})")
    for label, sub in (("空の層を含む(11-09 の値)", labelled), ("空の層を除く", [r for r in labelled if not r[ie]])):
        for ex_t in (False, True):
            ss = [r for r in sub if not (ex_t and r[9])]
            line = f"      [{label}{' / 描き文字・フキダシを除く' if ex_t else ''}] 層 {len(ss)}: "
            parts = []
            for ru, au in (("清書", "清書"), ("下絵", "下絵"), ("あたり", "あたり")):
                said = sum(r[12] == ru for r in ss)
                hit = sum(r[12] == ru and r[6] == au for r in ss)
                truth = sum(r[6] == au for r in ss)
                parts.append(f"{au} 的中 {hit}/{said}" + (f"={hit / said:.3f}" if said else "")
                             + f" 取りこぼし {truth - hit}/{truth}" + (f"={(truth - hit) / truth:.3f}" if truth else ""))
            print(line + " | ".join(parts))
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(out)
    print(f"    → `{OUT}`(非公開、{len(head)}列)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
