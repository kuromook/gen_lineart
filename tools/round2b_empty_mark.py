# -*- coding: utf-8 -*-
"""便B(111行)に、固まった「空」の定義の印を付ける —— 二側から独立に数える。

定義は作者の裁定(2026-11-11、読み(c) + トーンの条項)。正本の述語は
`tools/empty_tone_masked_out.py` の is_empty()。ここはそれを 便B の母集団に当てるだけ。

出所は2つ:
  (1) 役割の表 v2 に行が在る層(46層ぶん) —— 表の列
  (2) 抽出側の追加納品 `stage_layer_pixels_round2b_20261112.csv`(65層) —— 同じ列

**列の穴**(census、2026-11-11 §3): 画素0 の層では `n_px_a_gt0_own_mask` が定義上 0 にしか
ならないので、定義Cの「画素0 だが自分のマスクが形を持つ → 空ではない」は、この列では
見分けられない。**母数を印字して、穴が噛むかどうかを言う。**

層名は印字しない(条例11)。
"""
import csv, collections, hashlib, io, sys

KEY   = "data/role_predictor/hearing_round2_b_key_20261110_le20.csv"
TABLE = "data/role_predictor/layer_role_table_20261110_v2.csv"
NEW   = "inbox/stage_layer_pixels_round2b_20261112.csv"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def i(r, c):
    v = r.get(c, "")
    return None if v in ("", None) else int(v)


def is_empty(r):
    """正本の述語(tools/empty_tone_masked_out.py と同一)。"""
    px = i(r, "n_px_a_gt0")
    if px is None:
        return False, "測れない(描画が読めない)"
    if r.get("mask_status") not in ("ok", ""):
        return False, "測れない(マスクが読めない)"
    own = i(r, "n_px_a_gt0_own_mask")
    if px == 0:
        if r.get("has_own_mask") == "1" and (own or 0) > 0:
            return False, "空でない(自分のマスクが形を持つ)"
        return True, "空(画素0・自分のマスクも無いか空)"
    if own == 0:
        return True, "空(画素は在るがマスクで全部消えている = トーンの形)"
    return False, "空でない"


def norm(p):
    return (p or "").replace("/", "\\").strip()


def main():
    for p in (KEY, TABLE, NEW):
        print("%-62s sha256 %s" % (p, sha(p)))
    key = list(csv.DictReader(io.open(KEY, encoding="utf-8")))
    tab = {(norm(r["page_path"]), r["layer_id"]): r
           for r in csv.DictReader(io.open(TABLE, encoding="utf-8"))}
    new = {(norm(r["page_path"]), r["layer_id"]): r
           for r in csv.DictReader(io.open(NEW, encoding="utf-8"))}
    print("\n便B %d 行 / 表 v2 %d 層 / 追加納品 %d 層" % (len(key), len(tab), len(new)))

    print("\n== 1. 出所別の突き合わせ ==")
    src = collections.Counter()
    marks, missing = [], []
    for r in key:
        k = (norm(r["page_path"]), r["layer_id"])
        if k in new:
            src["追加納品(65層)"] += 1
            row, where = new[k], "追加納品"
        elif k in tab:
            src["表 v2"] += 1
            row, where = tab[k], "表 v2"
        else:
            src["照合不能"] += 1
            missing.append(r)
            continue
        e, why = is_empty(row)
        marks.append((r, row, where, e, why))
    for k, v in sorted(src.items()):
        print("   %-16s %d" % (k, v))
    if missing:
        print("   **照合不能の行**:")
        for r in missing:
            print("     %s 項目%s pos=%s 種別=%s" % (r["bundle"], r["item"], r["pos"], r["kind"]))

    print("\n== 2. 印の四分 ==")
    c = collections.Counter(why for _, _, _, _, why in marks)
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print("   %-44s %d" % (k, v))
    n_empty = sum(1 for *_, e, _ in marks if e)
    print("   ---- 空 %d / 空でない・測れない %d / 計 %d" % (n_empty, len(marks) - n_empty, len(marks)))

    print("\n== 3. 空の行(層名は出さない。条例11) ==")
    for r, row, where, e, why in marks:
        if e:
            print("   %s 項目%-3s pos=%-3s 種別=%-7s 出所=%-6s 画素=%s 自マスク=%s 祖先マスク=%s"
                  % (r["bundle"], r["item"], r["pos"], r["kind"], where,
                     row["n_px_a_gt0"], row.get("has_own_mask"), row.get("n_ancestor_masks")))

    print("\n== 4. **列の穴が噛むか** —— 画素0 で自分のマスクを持つ層の母数 ==")
    hole = [x for x in marks if i(x[1], "n_px_a_gt0") == 0 and x[1].get("has_own_mask") == "1"]
    print("   画素0 かつ has_own_mask=1: **%d 層**" % len(hole))
    print("   → %s" % ("**この母数が0なので、`n_px_own_mask_ge128` が無くても便B の印は曖昧にならない**"
                       if not hole else "**穴が噛む。マスクそのものの画素の列が要る**"))

    print("\n== 5. トーンの条項の母数 ==")
    tone = [x for x in marks if (i(x[1], "n_px_a_gt0") or 0) > 0 and i(x[1], "n_px_a_gt0_own_mask") == 0]
    kinds = collections.Counter(r["kind"] for r, *_ in marks)
    print("   当たる層: **%d**" % len(tone))
    print("   便B の種別: %s" % dict(kinds))
    print("   → mask_fill が0層なので、トーンの条項は便B では**試せていない**(母数0 = 判定保留)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
