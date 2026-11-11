# -*- coding: utf-8 -*-
"""規律13: 層名「<線数>L <濃度>%」を列として使う前に、構造DBの列を再現するか測る。

census の頼み(2026-11-12 §3 末尾、逐語):
> 名前「<線数>L <濃度>%」と `tone_lpi` / `tone_density_pct` が合っているか、census はまだ
> 照合していません。名前を列として使う前に、そちらで照合してもらうのがよいと思います(規律13)

材料: `inbox/tone_folder_layers_2018_ako6_20261111.csv`(4,784層、構造DBだけ。画素は無い)

**両方向を印字する** ——
  (順) 名前が当たる層のうち、lpi/濃度が名前と一致する割合 = 名前の的中率
  (逆) tone_enabled=1 の層のうち、名前が当たる割合 = 名前の取りこぼし
名前は、機械が付けた「<数>L <数>%」の形のものだけ逐語で出す(条例11 の安全側)。
"""
import csv, collections, io, re, hashlib, sys

SRC = "inbox/tone_folder_layers_2018_ako6_20261111.csv"
PAT = re.compile(r"(?:^|\s)(\d+(?:\.\d+)?)\s*L\s+(\d+(?:\.\d+)?)\s*%")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    print("材料: %s sha256 %s" % (SRC, sha(SRC)))
    rows = list(csv.DictReader(io.open(SRC, encoding="utf-8")))
    print("母集団: %d 層 —— **`2018\\ako6` の、祖先に トーン/tone を持つ層だけ。"
          "全437,425層ではない**\n" % len(rows))

    print("== 0. 種別 × tone_enabled × マスク ==")
    c = collections.Counter((r["v_layer_kind"], r["tone_enabled"], r["has_mask"]) for r in rows)
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print("   種別=%-14s tone_enabled=%-2s has_mask=%-2s  %d" % (k[0], k[1], k[2], v))

    name_hit = [r for r in rows if PAT.search(r["name"] or "")]
    tone_on  = [r for r in rows if r["tone_enabled"] == "1"]
    print("\n   名前が当たる層: %d / tone_enabled=1: %d" % (len(name_hit), len(tone_on)))

    print("\n== 1. (順)名前 → 列。名前の当たる %d 層で、lpi と濃度が名前と一致するか ==" % len(name_hit))
    agree = collections.Counter()
    bad = []
    for r in name_hit:
        m = PAT.search(r["name"])
        nl, nd = float(m.group(1)), float(m.group(2))
        cl, cd = f(r["tone_lpi"]), f(r["tone_density_pct"])
        kl = "一致" if cl is not None and abs(cl - nl) < 1e-6 else ("列が空" if cl is None else "食い違い")
        kd = "一致" if cd is not None and abs(cd - nd) < 1e-6 else ("列が空" if cd is None else "食い違い")
        agree[(kl, kd)] += 1
        if "食い違い" in (kl, kd):
            bad.append((r["name"], nl, nd, r["tone_lpi"], r["tone_density_pct"], r["tone_enabled"]))
    for k, v in sorted(agree.items(), key=lambda x: -x[1]):
        print("   線数 %-6s 濃度 %-6s  %d" % (k[0], k[1], v))
    both = agree[("一致", "一致")]
    print("   ---- **両方一致 %d / %d = %.4f**" % (both, len(name_hit), both / max(1, len(name_hit))))
    if bad:
        print("   食い違いの層(名前は機械が付けた形だけ):")
        for n, nl, nd, cl, cd, te in bad[:20]:
            print("     名前=%-16r 名前の線数=%-6s 濃度=%-6s / 列 lpi=%-6s 濃度=%-6s tone_enabled=%s"
                  % (n, nl, nd, cl, cd, te))
        if len(bad) > 20:
            print("     … 他 %d 層" % (len(bad) - 20))

    print("\n== 2. (逆)列 → 名前。tone_enabled=1 の %d 層で、名前が当たるか = **取りこぼし** ==" % len(tone_on))
    miss = [r for r in tone_on if not PAT.search(r["name"] or "")]
    print("   名前が当たる: %d / 当たらない(取りこぼし): **%d** = %.4f"
          % (len(tone_on) - len(miss), len(miss), len(miss) / max(1, len(tone_on))))
    if miss:
        mc = collections.Counter(
            ("「<数>L <数>%」の形ではない名前", r["v_layer_kind"], r["tone_lpi"], r["tone_density_pct"])
            for r in miss)
        for k, v in sorted(mc.items(), key=lambda x: -x[1])[:12]:
            print("     %s 種別=%-10s lpi=%-6s 濃度=%-6s  %d 層" % (k[0], k[1], k[2], k[3], v))

    print("\n== 3. 名前が当たらないが列に値が在る層 / 名前が当たるが tone_enabled=0 の層 ==")
    a = [r for r in rows if not PAT.search(r["name"] or "") and (f(r["tone_lpi"]) or 0) > 0]
    b = [r for r in name_hit if r["tone_enabled"] != "1"]
    print("   名前なし・lpi>0: %d 層" % len(a))
    print("   名前あり・tone_enabled≠1: %d 層" % len(b))
    for r in b[:10]:
        print("     名前=%-16r 種別=%-10s tone_enabled=%s lpi=%s 濃度=%s"
              % (r["name"], r["v_layer_kind"], r["tone_enabled"], r["tone_lpi"], r["tone_density_pct"]))

    print("\n== 4. マスクの形(作者の条項の下ごしらえ) ==")
    mc = collections.Counter((r["v_layer_kind"], r["has_mask"]) for r in rows)
    for k, v in sorted(mc.items(), key=lambda x: -x[1]):
        print("   種別=%-14s has_mask=%-2s %d" % (k[0], k[1], v))
    print("   → **画素は構造DBに無い。「全部マスクで消えている」かは抽出側の納品待ち**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
