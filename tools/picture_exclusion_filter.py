# -*- coding: utf-8 -*-
"""「絵から除外」する層の決定論的な篩(第2段)。

表 `config/picture_exclusion_draft_20261111.tsv` の **状態=確定 の行だけ**を効かせる。
**候補・材料・保留・測れない・裁定あり の行は効かせない**(表は「安全なものの表」ではなく
「選んだものの表」。足すのはユーザーの承認)。

規律9: **落としたものを理由別に数えて印字する。** さらに規律13 の点検として、
**落ちた層に作者のラベルが在るなら、その段を印字する** —— 段のラベルが在る層を落としたら、
それは反証であって篩の成功ではない。

層名は印字しない(条例11。`text` 種別の名前は本文で、奥付に作者の個人情報が入る)。
使い方: ./venv/bin/python tools/picture_exclusion_filter.py [表のCSV]
"""
import csv, collections, hashlib, io, os, sys

TSV = "config/picture_exclusion_draft_20261111.tsv"
DEFAULT_TABLE = "data/role_predictor/layer_role_table_20261110_v2.csv"
STAGES = ("あたり", "下絵", "清書", "段ではない")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load_rules(path):
    """確定の行だけを返す。効かせない行も、理由を付けて返す(印字するため)。"""
    on, off = [], []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            c = line.split("\t")
            if c[0] == "決め方":
                continue
            r = {"how": c[0], "key": c[1], "kind": c[2], "state": c[-1]}
            (on if r["state"] == "確定" else off).append(r)
    return on, off


# 表の鍵(名前・種別)から、層の役割の表の列への当て方。
# **名前は完全一致**(棚卸しと同じ。全角半角や空白を畳まない)。
KIND_KEYS = {
    "用紙": {"paper"},
    "(ノンブル)・(作品情報)": {"story_info"},
    "セリフのテキスト層": {"text"},
}


def match(rule, row):
    if rule["how"] == "名前":
        return row["name"] == rule["key"] and row["v_layer_kind"] == rule["kind"]
    if rule["how"] == "種別":
        return row["v_layer_kind"] in KIND_KEYS.get(rule["key"], set())
    return False


def main(table=DEFAULT_TABLE):
    print("表(篩)   :", TSV, "sha256", sha(TSV))
    print("表(母集団):", table, "sha256", sha(table))
    on, off = load_rules(TSV)
    print("\n効かせる行(状態=確定): %d" % len(on))
    for r in on:
        print("   %s / %s / %s" % (r["how"], r["key"], r["kind"]))
    print("効かせない行: %d" % len(off))
    for r in off:
        print("   [%s] %s / %s" % (r["state"], r["how"], r["key"]))

    rows = list(csv.DictReader(io.open(table, encoding="utf-8")))
    print("\n母集団: %d 層" % len(rows))
    print("**この母集団は名前で引いた層と聞き取りの層で、全437,425層ではない。**")

    dropped = collections.Counter()
    dropped_labelled = collections.defaultdict(collections.Counter)
    multi = 0
    kept = []
    for row in rows:
        hits = [r for r in on if match(r, row)]
        if not hits:
            kept.append(row)
            continue
        if len(hits) > 1:
            multi += 1
        key = "%s: %s" % (hits[0]["how"], hits[0]["key"])
        dropped[key] += 1
        st = row.get("author_stage", "")
        if st:
            dropped_labelled[key][st] += 1

    print("\n== 落とした層(理由別。規律9) ==")
    print("| 理由 | 落とした層 | うち作者のラベル在り | その段 |")
    print("|---|---:|---:|---|")
    tot = 0
    bad = []
    for key in sorted(dropped, key=lambda k: -dropped[k]):
        lab = dropped_labelled[key]
        n_lab = sum(lab.values())
        stages = " / ".join("%s %d" % (s, lab[s]) for s in STAGES if lab.get(s))
        print("| %s | %d | %d | %s |" % (key, dropped[key], n_lab, stages or "—"))
        tot += dropped[key]
        for s in ("あたり", "下絵", "清書"):
            if lab.get(s):
                bad.append((key, s, lab[s]))
    print("| 計 | %d | %d | |" % (tot, sum(sum(v.values()) for v in dropped_labelled.values())))
    if multi:
        print("2つ以上の理由に当たった層: %d(上の表では最初の理由に数えた)" % multi)

    print("\n== 残った層: %d(%d - %d) ==" % (len(kept), len(rows), tot))
    lab_kept = collections.Counter(r["author_stage"] for r in kept if r.get("author_stage"))
    print("   残った層の作者のラベル:", dict(lab_kept))

    print("\n== 規律13 の点検 ==")
    if bad:
        print("**反証**: 作者が段と答えた層を落としている。篩を止めること:")
        for key, s, n in bad:
            print("   %s —— %s を %d 層" % (key, s, n))
        return 1
    print("落とした層に「あたり・下絵・清書」のラベルは **0層**。")
    print("(ラベルが在るのは %d 層で、全部「段ではない」。**ラベルの無い層については何も示していない**)"
          % sum(sum(v.values()) for v in dropped_labelled.values()))
    return 0


INVENTORY = "data/role_predictor/layer_name_inventory_20261110.csv"


def corpus(path=INVENTORY):
    """全層の棚卸し(437,425層)に篩を当てて、作品全体で何層落ちるかを数える。
    **役割の表の母集団は名前で引いた層なので、全体の数はこちらで数える。**"""
    print("\n" + "=" * 68)
    print("== 作品全体(棚卸し)で何層落ちるか ==")
    print("棚卸し:", path, "sha256", sha(path))
    on, _ = load_rules(TSV)
    rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
    n_all = sum(int(r["n_layers"]) for r in rows)
    print("棚卸しの行: %d / 層: %d" % (len(rows), n_all))

    def m(rule, r):
        if rule["how"] == "名前":
            return r["layer_name"] == rule["key"] and r["v_layer_kind"] == rule["kind"]
        if rule["how"] == "種別":
            return r["v_layer_kind"] in KIND_KEYS.get(rule["key"], set())
        return False

    print("\n| 理由 | 棚卸しの行 | 落とす層 | 頁 | 作品単位 |")
    print("|---|---:|---:|---:|---:|")
    tot = 0
    seen = set()
    for rule in on:
        hit = [r for r in rows if m(rule, r)]
        n = sum(int(r["n_layers"]) for r in hit)
        pg = sum(int(r["n_pages"]) for r in hit)
        u = max([int(r["n_split_units"]) for r in hit] or [0])
        print("| %s: %s | %d | %d | %d | %d |" % (rule["how"], rule["key"], len(hit), n, pg, u))
        tot += n
        for r in hit:
            seen.add((r["layer_name"], r["v_layer_kind"]))
    print("| 計 | %d | **%d** | | |" % (len(seen), tot))
    print("\n落とす層 %d / 全 %d = %.4f" % (tot, n_all, tot / n_all))
    print("**頁の数は行ごとの和なので、同じ頁を重ねて数えている**(上界)。作品単位は行ごとの最大。")
    print("**残る層: %d。** これは「絵の層」ではなく「この篩で落ちなかった層」である。" % (n_all - tot))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "corpus":
        sys.exit(corpus(*a[1:]))
    rc = main(*(a or []))
    if rc == 0:
        corpus()
    sys.exit(rc)
