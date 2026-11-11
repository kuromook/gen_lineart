# -*- coding: utf-8 -*-
"""作者の追加条件(2026-11-11)を、手元の列だけで判定できるか測る。

> 基本的な考えはＣだがマスクを持つ場合にもう一つ条件が必要だった　トーンレイヤはデフォルトでは
> レイヤ全体にトーンがかかった状態で、それらを「全体にマスクがかかった状態」にすることで何も
> 「トーンが描かれていない」状態になっている　このレイヤは実質何も描いていないが、
> 「レイヤはトーンを持ちつつマスクもある」が「空」のレイヤに相当する

census の列の定義(`push_census_order_stage_layer_pixels_20261109.md` 逐語):
  n_px_a_gt0           : アルファ > 0 の画素(**マスクを見ない**)
  n_px_a_gt0_own_mask  : 上のうち、**層自身のマスク**の値 ≥ 128 の画素(マスクが無ければ空)
  n_px_a_gt0_all_masks : 上のうち、**自身と祖先のフォルダーのマスク全部**で ≥ 128 の画素
→ 「全体にマスクがかかって何も残らない」は **n_px_a_gt0 > 0 かつ n_px_a_gt0_own_mask = 0**。
層名は印字しない(条例11)。
"""
import csv, collections, hashlib, io, os, sys

TABLE = "data/role_predictor/layer_role_table_20261110_v2.csv"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def i(r, c):
    v = r.get(c, "")
    return None if v in ("", None) else int(v)


def main(table=TABLE):
    print("表:", table, "sha256", sha(table))
    rows = list(csv.DictReader(io.open(table, encoding="utf-8")))
    print("母集団: %d 層。**名前で引いた段の候補と聞き取りの層で、全437,425層ではない**\n" % len(rows))

    print("== 1. 自分のマスクを持つ層の、画素とマスク越しの画素 ==")
    own = [r for r in rows if r.get("has_own_mask") == "1"]
    print("   has_own_mask=1: %d 層" % len(own))
    c = collections.Counter()
    for r in own:
        px, m = i(r, "n_px_a_gt0"), i(r, "n_px_a_gt0_own_mask")
        c[("px=0" if px == 0 else ("px空欄" if px is None else "px>0"),
           "own=0" if m == 0 else ("own空欄" if m is None else "own>0"))] += 1
    for k in sorted(c, key=str):
        print("     %-8s %-10s %d" % (k[0], k[1], c[k]))

    print("\n== 2. **作者の言う形**: 画素 > 0 かつ 自分のマスク越しが 0(全部マスクで消えている) ==")
    hit = [r for r in rows if (i(r, "n_px_a_gt0") or 0) > 0 and i(r, "n_px_a_gt0_own_mask") == 0]
    print("   当たる層: %d" % len(hit))
    for r in hit:
        print("     トーン=%s 種別=%s 画素=%s 自マスク越し=%s 全マスク越し=%s 祖先マスク=%s "
              "不透明度=%s 合成=%s 段=%s 名前規則=%s 祖先=%r"
              % (r["tone_enabled"], r["v_layer_kind"], r["n_px_a_gt0"], r["n_px_a_gt0_own_mask"],
                 r["n_px_a_gt0_all_masks"], r["n_ancestor_masks"], r["opacity"], r["blend_mode"],
                 r["author_stage"] or "—", r["name_rule"], r["ancestors"][:40]))

    print("\n== 3. 祖先のマスクまで入れて何も残らない層(画素 > 0 かつ 全マスク越しが 0) ==")
    hit2 = [r for r in rows if (i(r, "n_px_a_gt0") or 0) > 0 and i(r, "n_px_a_gt0_all_masks") == 0]
    print("   当たる層: %d" % len(hit2))
    cc = collections.Counter((r["tone_enabled"], r["v_layer_kind"], r["has_own_mask"]) for r in hit2)
    for k in sorted(cc, key=str):
        print("     トーン=%s 種別=%s 自マスク=%s : %d 層" % (k[0], k[1], k[2], cc[k]))
    st = collections.Counter(r["author_stage"] for r in hit2 if r["author_stage"])
    print("   うち作者のラベル:", dict(st) or "無し")

    print("\n== 4. トーンの層は、この表に何層在るか ==")
    t = [r for r in rows if r.get("tone_enabled") == "1"]
    print("   tone_enabled=1: %d 層" % len(t))
    print("   種別:", dict(collections.Counter(r["v_layer_kind"] for r in t).most_common(6)))
    print("   自分のマスクを持つ:", sum(1 for r in t if r.get("has_own_mask") == "1"))
    print("   画素0:", sum(1 for r in t if i(r, "n_px_a_gt0") == 0))
    print("   作者のラベル:", dict(collections.Counter(r["author_stage"] for r in t if r["author_stage"])) or "無し")
    print("\n   トーンの層の、画素とマスクの組み合わせ:")
    c2 = collections.Counter()
    for r in t:
        px, m, a = i(r, "n_px_a_gt0"), i(r, "n_px_a_gt0_own_mask"), i(r, "n_px_a_gt0_all_masks")
        c2[("px=0" if px == 0 else ("px空欄" if px is None else "px>0"),
            "own=0" if m == 0 else ("own空欄(マスク無し)" if m is None else "own>0"),
            "all=0" if a == 0 else ("all空欄" if a is None else "all>0"))] += 1
    for k in sorted(c2, key=str):
        print("     %-8s %-20s %-8s %d" % (k[0], k[1], k[2], c2[k]))

    print("\n== 5. 覆い率が1に近い層(「レイヤ全体にトーン」の形) ==")
    full = []
    for r in t:
        px = i(r, "n_px_a_gt0")
        w, h = i(r, "canvas_px_w"), i(r, "canvas_px_h")
        if px and w and h and px / (w * h) > 0.98:
            full.append(r)
    print("   トーンの層で覆い率 > 0.98: %d / %d" % (len(full), len(t)))
    print("   うち自分のマスクを持つ:", sum(1 for r in full if r.get("has_own_mask") == "1"))
    print("   うち自マスク越しが0  :", sum(1 for r in full if i(r, "n_px_a_gt0_own_mask") == 0))
    return 0


# ---- 作者が選んだ定義(C + 追加条件)。**これが正本の判定** ----
def is_empty(r):
    """「完全に空」か。作者の裁定(2026-11-11): 基本は自分のマスクだけ見る(C)。
    加えて、画素が在っても自分のマスクで全部消えていれば空(トーンの形)。
    **祖先のフォルダーのマスクは見ない。** 読めない層は空ではなく『測れない』。
    返り値: (空か, 理由)"""
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
        return True, "空(画素は在るが自分のマスクで全部消えている = トーンの形)"
    return False, "空でない"


def verdicts(table=TABLE):
    print("\n" + "=" * 68)
    print("== 作者が選んだ定義(C + 追加条件)で数える ==")
    rows = list(csv.DictReader(io.open(table, encoding="utf-8")))
    c = collections.Counter()
    lab = collections.defaultdict(collections.Counter)
    for r in rows:
        e, why = is_empty(r)
        c[(e, why)] += 1
        if r.get("author_stage"):
            lab[(e, why)][r["author_stage"]] += 1
    for k in sorted(c, key=str):
        print("   %-6s | %-46s %6d  ラベル %s"
              % ("空" if k[0] else "空でない", k[1], c[k], dict(lab[k]) or "無し"))
    print("   空の合計: %d" % sum(v for k, v in c.items() if k[0]))
    print("\n   **母数の注意** —— 作者が足した2つの形は、この母集団ではほとんど測れない:")
    print("     画素0 で 自分のマスクが形を持つ(空でない側へ移る): %d 層"
          % sum(1 for r in rows if i(r, "n_px_a_gt0") == 0 and r.get("has_own_mask") == "1"
                and (i(r, "n_px_a_gt0_own_mask") or 0) > 0))
    print("     画素>0 で 自分のマスクで全部消えている(空の側へ移る): %d 層"
          % sum(1 for r in rows if (i(r, "n_px_a_gt0") or 0) > 0 and i(r, "n_px_a_gt0_own_mask") == 0))
    print("     tone_enabled = 1 の層: %d 層(**0 なので、トーンの形はこの表では確かめられない**)"
          % sum(1 for r in rows if r["tone_enabled"] == "1"))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(a or []))
    sys.exit(verdicts(*(a or [])))
