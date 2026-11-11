# -*- coding: utf-8 -*-
"""「この層はトーンレイヤーか」を、手元の列で決められるかを測る。

作者の言(2026-11-11): トーンのフォルダーが使われるのは**モノクロの漫画原稿**。
「トーン」「tone」は**トーンレイヤーをまとめるフォルダーの名前**で、トーンレイヤー自体を見る必要がある。
→ この器具は「決められない」ことを数で示し、**代用しない根拠を印字する**(規律13)。
層名は原則印字しない(条例11)。祖先の道だけ、先頭38文字で出す。
"""
import csv, collections, hashlib, io, sys

TABLE = "data/role_predictor/layer_role_table_20261110_v2.csv"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def anc_tone(r):
    a = r.get("ancestors", "")
    return "トーン" in a or "tone" in a.lower()


def main(table=TABLE):
    print("表:", table, "sha256", sha(table))
    rows = list(csv.DictReader(io.open(table, encoding="utf-8")))
    print("母集団: %d 層。**名前で引いた段の候補と聞き取りの層。"
          "トーンのフォルダーの中身はほとんど入っていない**\n" % len(rows))

    print("== 1. トーンの印になりうる列の値 ==")
    for c in ("tone_enabled", "v_layer_kind", "layer_type", "colour_type_index"):
        print("   %-20s" % c, dict(collections.Counter(r[c] for r in rows).most_common(6)))
    print("   **`tone_enabled` は全層 0。列の出所は通達に書かれていない**")

    print("\n== 2. 作者の案(ako・2018)は既に母集団に在る ==")
    ako = [r for r in rows if "ako" in r["work_dir"].lower()]
    y18 = [r for r in rows if r["year_label"] == "2018"]
    t = [r for r in rows if anc_tone(r)]
    print("   ako の層: %d / 2018 札の層: %d" % (len(ako), len(y18)))
    print("   ako で tone_enabled=1: %d" % sum(1 for r in ako if r["tone_enabled"] == "1"))
    print("   祖先の道に「トーン/tone」を持つ層: %d(うち tone_enabled=1 は %d)"
          % (len(t), sum(1 for r in t if r["tone_enabled"] == "1")))
    print("   その層の年代札:", dict(sorted(collections.Counter(r["year_label"] for r in t).items())))
    print("   → **(ア) それらはトーンレイヤーでない / (イ) 列が効果を読んでいない —— 手元では区別できない**")

    print("\n== 3. 表現色=2 は印か(規律13 の点検) ==")
    c2 = [r for r in rows if r["colour_type_index"] == "2"]
    base = len(c2) / len(rows)
    inn = sum(1 for r in t if r["colour_type_index"] == "2") / len(t)
    print("   表現色=2: %d層 / %d = %.5f" % (len(c2), len(rows), base))
    print("   トーンの祖先の下では %.4f —— 持ち上がり %.1f 倍" % (inn, inn / base))
    print("   **逆向き**: 表現色=2 の %d層のうち、トーンの祖先を持つのは %d層だけ"
          % (len(c2), sum(1 for r in c2 if anc_tone(r))))
    print("   画素0: %d / %d  自マスク持ち: %d  作者のラベル: %s"
          % (sum(1 for r in c2 if r["n_px_a_gt0"] == "0"), len(c2),
             sum(1 for r in c2 if r["has_own_mask"] == "1"),
             dict(collections.Counter(r["author_stage"] for r in c2 if r["author_stage"])) or "無し"))
    print("   祖先の道(上位6):")
    for p, n in collections.Counter(r["ancestors"][:38] for r in c2).most_common(6):
        print("     %4d  %r" % (n, p))
    print("   **印ではない。** 代用しない(規律13)")

    l3 = [r for r in rows if r["layer_type"] == "3"]
    print("\n== 4. layer_type=3 も印ではない ==")
    print("   %d層。うちトーンの祖先を持つのは %d層" % (len(l3), sum(1 for r in l3 if anc_tone(r))))
    print("\n結論: **この表には「トーンレイヤーか」を示す使える列が無い。** ファイルの側で確かめる発注が要る")
    return 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:] or [])))
