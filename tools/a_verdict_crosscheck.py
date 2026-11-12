# -*- coding: utf-8 -*-
"""census の層ごとの判定 `a_verdict` を、こちらの数と突き合わせる(2026-11-12 の納品)。

**何に反応しているかを名乗る**(条例9): この器具は3つを印字する。
  (1) a_verdict の内訳と、当たった行(`a_rule`)ごとの層 —— **列で落ちた分を引くと、
      `picture_exclusion_filter.py` の名前と種別の数に一致するか**
  (2) **画素の道が届かない範囲** —— 空は .clip の画素でしか言えない。`.lip` の層を分けて数える
  (3) **空(定義C)と a の判定の2×2** —— 空の行が a に何層足すか、
      **名前が『除外しない』の行の空の層が何層あるか**(名前の答えと層の印の単位の違い)
`empty_c` が空欄の層は「測っていない」で、空でないとは言わない(母数0 = 測れない)。
層名は表 v2 から引く。**`text` 種別の名前は出さない**(条例11)。
"""
import csv, collections, hashlib, io, sys

V = "inbox/a_verdict_20261112.csv"
ROLE = "data/role_predictor/layer_role_table_20261110_v2.csv"
FILTER_NAME_KIND = 105231   # picture_exclusion_filter.py の名前と種別の数
COL_LAYERS = 91060          # 列 tone_enabled = 1


def sha(p):
    h = hashlib.sha256()
    with io.open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    print("判定: %s sha256 %s" % (V, sha(V)))
    ver = collections.Counter(); rule = collections.Counter()
    ext_ver = collections.Counter(); emp = collections.Counter()
    cross = collections.Counter(); keys_435 = []
    for r in csv.DictReader(io.open(V, encoding="utf-8")):
        ext = r["page_path"].rsplit(".", 1)[-1].lower()
        ver[r["a_verdict"]] += 1
        ext_ver[(ext, r["a_verdict"])] += 1
        emp[r["empty_c"]] += 1
        if r["a_verdict"] == "template_excluded":
            rule[r["a_rule"]] += 1
        if r["empty_c"] in ("0", "1"):
            cross[(r["empty_c"], r["a_verdict"])] += 1
            if r["empty_c"] == "1" and r["a_verdict"] == "not_excluded":
                keys_435.append((r["page_path"], r["layer_id"]))
    n = sum(ver.values())
    print("母集団: **%d 層**(構造 DB 437,561 から恒久除外の頁の 136 を外した数)\n" % n)

    print("== (1) a の判定 ==")
    for k in ("template_excluded", "not_excluded", "undecided"):
        print("   %-18s %7d  %.4f" % (k, ver[k], ver[k] / n))
    col = rule.get("列:tone_enabled", 0)
    byname = sum(rule.values()) - col
    print("   落とした内訳: 名前と種別 %d + 列 %d = %d" % (byname, col, sum(rule.values())))
    ok = (byname == FILTER_NAME_KIND and col == COL_LAYERS)
    print("   **突き合わせ: %s**(こちらの篩 %d + 列 %d)"
          % ("合った" if ok else "合わない", FILTER_NAME_KIND, COL_LAYERS))

    print("\n== (2) 画素の道が届く範囲(空は .clip でしか言えない) ==")
    for ext in ("clip", "lip"):
        tot = sum(v for (e, _), v in ext_ver.items() if e == ext)
        und = ext_ver[(ext, "undecided")]
        print("   %-5s 層 %6d / まだ落ちていない層 %6d" % (ext, tot, und))
    uc, ul = ext_ver[("clip", "undecided")], ext_ver[("lip", "undecided")]
    print("   → **まだ落ちていない %d 層のうち、画素で空を言えるのは .clip の %d 層(%.4f)**"
          % (uc + ul, uc, uc / (uc + ul)))
    print("   **`.lip` の %d 層は、画素の走行でも空かどうかを言えない**(母数0 = 測れない)" % ul)

    print("\n== (3) 空(定義C)と a の判定 ==")
    print("   `empty_c` 空欄(測っていない) %d / 0 %d / 1 %d" % (emp[""], emp["0"], emp["1"]))
    print("   | 空 | a の判定 | 層 |")
    print("   |---|---|---:|")
    for k in sorted(cross):
        print("   | %s | %s | %d |" % (k[0], k[1], cross[k]))
    e1 = sum(v for k, v in cross.items() if k[0] == "1")
    print("   **空=1 の %d 層のうち、a が既に落としているのは %d 層** —— 空の行は足し算になる"
          % (e1, cross[("1", "template_excluded")]))
    print("   **うち %d 層は、名前の答えが『除外しない』の行の層**" % cross[("1", "not_excluded")])

    nm = {}
    for r in csv.DictReader(io.open(ROLE, encoding="utf-8")):
        nm[(r["page_path"], r["layer_id"])] = (r["name"], r["v_layer_kind"])
    c = collections.Counter(); miss = 0
    for k in keys_435:
        if k in nm:
            c[nm[k]] += 1
        else:
            miss += 1
    print("   その名前(表 v2 で引いた。引けなかった %d):" % miss)
    for (a, b), v in c.most_common():
        a = "(名前は出さない。条例11)" if b == "text" else (a or "(空の名前)")
        print("      %-18s %-8s %5d" % (a, b, v))
    print("   **名前の答えは名前ごと、空の印は層ごと。単位が違うので矛盾ではない** ——")
    print("   **E の決め(作者の言葉ではない): 空の層は、名前が『除外しない』でも落とす。**")
    print("   理由: 画素が1つも無い層は、絵からも段からも何も持ち去らない。**落とした数は名前別に印字し、R に渡す**")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
