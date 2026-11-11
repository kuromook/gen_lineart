# -*- coding: utf-8 -*-
"""a の次の一手: **聞いていない層名のうち、どれを次の20行で聞くか**を、数で選ぶ。

除外の表は名前で動いていて、**名前の答えは作者の聞き取りから来る**(2026-11-11 に20行)。
この器具は (1) 20行で何層が片付いたか (2) 残りの名前が何層を持つか (3) **次の20行の案**を出す。

**選び方を先に書く(規律9: 選別は測定である)**:
  母集団 = 棚卸し 16,479行(全437,425層)。
  落とす = 既に『確定』の規則に当たる行 / 既に20行で聞いた (名前,種別) / **`text` 種別**(条例11。
           名前が本文なので画面に出せない。種別で既に確定)。
  並べ方 = 層の数の多い順。**作品単位の数は印字するが、並べ替えには使わない**(1作品に偏った名前を
           弾くのは作者の判断で、こちらが先に切らない)。
層名は印字する(`text` 種別だけ出さない。条例11)。
"""
import csv, collections, hashlib, io, sys

INV = "data/role_predictor/layer_name_inventory_20261110.csv"
TSV = "config/picture_exclusion_draft_20261111.tsv"
ASKED = "inbox/template_names_answers_20261111.csv"
KIND_KEYS = {"用紙": {"paper"}, "(ノンブル)・(作品情報)": {"story_info"}, "セリフのテキスト層": {"text"}}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def rules():
    on = []
    for line in io.open(TSV, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if line.startswith("#") or len(c) < 10 or c[0] == "決め方":
            continue
        if c[-1] == "確定":
            on.append({"how": c[0], "key": c[1], "kind": c[2]})
    return on


def main(n=20):
    print("棚卸し: %s sha256 %s" % (INV, sha(INV)))
    print("表    : %s sha256 %s" % (TSV, sha(TSV)))
    rows = list(csv.DictReader(io.open(INV, encoding="utf-8")))
    total = sum(int(r["n_layers"]) for r in rows)
    print("母集団: %d 行 / **%d 層**\n" % (len(rows), total))

    asked = {(r["layer_name"], r["v_layer_kind"]): r["author_answer"]
             for r in csv.DictReader(io.open(ASKED, encoding="utf-8"))}
    on = rules()

    def hit_rule(r):
        for q in on:
            if q["how"] == "名前" and r["layer_name"] == q["key"] and r["v_layer_kind"] == q["kind"]:
                return "確定の規則: 名前 %s" % q["key"]
            if q["how"] == "種別" and r["v_layer_kind"] in KIND_KEYS.get(q["key"], set()):
                return "確定の規則: 種別 %s" % q["key"]
        return None

    drop = collections.Counter()
    drop_layers = collections.Counter()
    rest = []
    for r in rows:
        k = (r["layer_name"], r["v_layer_kind"])
        why = hit_rule(r)
        if why:
            drop[why] += 1; drop_layers[why] += int(r["n_layers"]); continue
        if k in asked:
            why = "20行で聞いた(答え: %s)" % asked[k][:16]
            drop[why] += 1; drop_layers[why] += int(r["n_layers"]); continue
        if r["v_layer_kind"] == "text":
            drop["text 種別(条例11。名前を画面に出さない)"] += 1
            drop_layers["text 種別(条例11。名前を画面に出さない)"] += int(r["n_layers"]); continue
        rest.append(r)

    print("== 落とした行(規律9: 理由別) ==")
    for k in sorted(drop, key=lambda x: -drop_layers[x]):
        print("   %-46s 行 %5d / 層 %6d" % (k[:46], drop[k], drop_layers[k]))
    done = sum(drop_layers.values())
    print("   ---- 片付いた: 行 %d / **層 %d(%.4f)**" % (sum(drop.values()), done, done / total))
    print("   ---- 残り    : 行 %d / **層 %d(%.4f)**"
          % (len(rest), total - done, (total - done) / total))

    rest.sort(key=lambda r: -int(r["n_layers"]))
    cum = 0
    print("\n== 残りの名前、層の多い順(次の%d行の案) ==" % n)
    print("| # | 名前 | 種別 | 層 | 作品単位 | clip頁 | lip頁 | 残りの累積 |")
    print("|---:|---|---|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rest[:n], 1):
        cum += int(r["n_layers"])
        nm = r["layer_name"] or "(空の名前)"
        print("| %d | `%s` | %s | %s | %s | %s | %s | %.3f |"
              % (i, nm, r["v_layer_kind"], r["n_layers"], r["n_split_units"],
                 r["n_clip_pages"], r["n_lip_pages"], cum / (total - done)))
    print("\n   **次の%d行で届く残りの割合: %.4f**(残り %d 層のうち %d 層)"
          % (n, cum / (total - done), total - done, cum))

    print("\n== 残りの長い尻尾 ==")
    for lo, hi in ((1, 1), (2, 9), (10, 99), (100, 999), (1000, 10 ** 9)):
        sub = [r for r in rest if lo <= int(r["n_layers"]) <= hi]
        print("   層 %-6s の行: %5d(層の和 %7d)"
              % ("%d〜%d" % (lo, hi) if lo != hi else "%d" % lo, len(sub),
                 sum(int(r["n_layers"]) for r in sub)))
    print("   → **名前の道は尻尾が長い。** 20行ずつでは届かない部分が在ることを先に書く")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
