# -*- coding: utf-8 -*-
"""「トーンレイヤーとして作られた層」の候補を、棚卸し(全437,425層)から数える。

作者の言(2026-11-11): CSP には「トーン化」(通常のレイヤをトーンレイヤ化する効果)が在り、
`tone_enabled` はおそらくそちら。「トーンレイヤとして作成されたレイヤ」は**別の種**だと思われる。

棚卸しで見つかる形: 種別 `mask_fill`(118,353層)の名前に、CSP が自動で付ける
**「<線数>L <濃度>%」** の形が並ぶ(`60L 10%` `55L 18%` `30L 10%` など)。
**これは推定である。** 確かめるのはファイルの側(census → 抽出側)。
"""
import csv, collections, io, re, sys

INV = "data/role_predictor/layer_name_inventory_20261110.csv"
# CSP の新規トーンの既定名: 「60L 10%」「線 50L 10%」「60L 5cm」など
PAT = re.compile(r"(?:^|\s)(\d+(?:\.\d+)?)\s*L\s+(\d+(?:\.\d+)?)\s*%")
PAT_LOOSE = re.compile(r"\d+\s*L\s")


def main(inv=INV):
    rows = list(csv.DictReader(io.open(inv, encoding="utf-8")))
    n_all = sum(int(r["n_layers"]) for r in rows)
    print("棚卸し: %d 行 / %d 層" % (len(rows), n_all))

    print("\n== 1. 種別ごとに、名前が「<線数>L <濃度>%」に当たる層 ==")
    print("| 種別 | 当たる行 | 当たる層 | その種別の全層 | 割合 |")
    print("|---|---:|---:|---:|---:|")
    tot_hit = 0
    by_kind = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k = r["v_layer_kind"]
        n = int(r["n_layers"])
        by_kind[k][2] += n
        if PAT.search(r["layer_name"]):
            by_kind[k][0] += 1
            by_kind[k][1] += n
            tot_hit += n
    for k in sorted(by_kind, key=lambda k: -by_kind[k][1]):
        a, b, c = by_kind[k]
        if a:
            print("| %s | %d | %d | %d | %.4f |" % (k, a, b, c, b / c))
    print("\n   当たる層の合計: %d / %d = %.4f" % (tot_hit, n_all, tot_hit / n_all))

    print("\n== 2. `mask_fill` の中で、当たる側と当たらない側 ==")
    mf = [r for r in rows if r["v_layer_kind"] == "mask_fill"]
    hit = [r for r in mf if PAT.search(r["layer_name"])]
    miss = [r for r in mf if not PAT.search(r["layer_name"])]
    print("   mask_fill: %d行 / %d層" % (len(mf), sum(int(r["n_layers"]) for r in mf)))
    print("   当たる  : %d行 / %d層" % (len(hit), sum(int(r["n_layers"]) for r in hit)))
    print("   当たらない: %d行 / %d層" % (len(miss), sum(int(r["n_layers"]) for r in miss)))
    print("\n   当たらない側の層の多い名前(上位12) —— **これが何かは分からない**:")
    for r in sorted(miss, key=lambda r: -int(r["n_layers"]))[:12]:
        print("     %-24r %7s層 %3s単位" % (r["layer_name"], r["n_layers"], r["n_split_units"]))

    print("\n== 3. 緩い形(「<数>L 」を含む)でも当たらない種別が在るか ==")
    loose = collections.Counter()
    for r in rows:
        if PAT_LOOSE.search(r["layer_name"]):
            loose[r["v_layer_kind"]] += int(r["n_layers"])
    print("   ", dict(loose.most_common()))

    print("\n== 4. 線数と濃度の値の分布(当たった行から) ==")
    L = collections.Counter()
    P = collections.Counter()
    for r in rows:
        m = PAT.search(r["layer_name"])
        if m:
            L[m.group(1)] += int(r["n_layers"])
            P[m.group(2)] += int(r["n_layers"])
    print("   線数(L):", dict(L.most_common(10)))
    print("   濃度(%):", dict(P.most_common(12)))
    print("\n**但し書き(規律13)**: これは名前と種別による推定で、作者の分類ではない。")
    print("トーンの効果の記録をファイルで確かめるまで、『トーンレイヤー』の列として使わない。")
    return 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:] or [])))
