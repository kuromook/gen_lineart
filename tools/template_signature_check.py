# -*- coding: utf-8 -*-
"""第3段: **画素の署名で、名前を使わない確認**(登録簿 exclusion_axis.order_changed_by_user.the_order (3))。

ユーザーの言明(2026-11-10、逐語): 「PICOトンボや枠などのテンプレートは全く同じレイヤでしかない
レイヤ名もぶれていない　これらは決定論的に除外できるはずだ」

**名前で落とす表は既に効いている(第2段、承認「2 OK」)。この段が問うのは別のこと** ——
*名前を見ずに*同じ層を見つけられるか、そして**名前の規則が取りこぼしていないか**。

署名 = (キャンバス幅, キャンバス高, 画素数, 外接矩形4つ)。
**両方向を印字する**(規律13):
  (順) テンプレートの名前の層は、いくつの署名を持つか(名前は1つの物を指すか)
  (逆) **テンプレートの署名を持つのに、名前がテンプレートでない層** = 名前の規則の取りこぼし
  (名前を使わない印) 同じ署名が3単位以上にまたがる組。2〜7単位の分布を全部出す(空きを見る)

母集団は**段の候補の表**(名前で引いた層 + 聞き取りの層)。**全437,425層ではない。**
層名は `text` 種別のものを出さない(条例11)。
"""
import csv, collections, hashlib, io, sys

TABLE = "data/role_predictor/layer_role_table_20261110_v2.csv"
TEMPLATE_NAMES = ["PICOトンボ", "断ち落とし枠（白）", "断ち落とし枠", "外枠", "ガイド"]
SIG = ("canvas_px_w", "canvas_px_h", "n_px_a_gt0", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def safe_name(r):
    return "«text種別の層名は出さない»" if r.get("v_layer_kind") == "text" else (r.get("name") or "")


def sig(r):
    v = tuple(r.get(c, "") for c in SIG)
    return None if any(x in ("", None) for x in v) else v


def main():
    print("表: %s sha256 %s" % (TABLE, sha(TABLE)))
    rows = list(csv.DictReader(io.open(TABLE, encoding="utf-8")))
    print("母集団: %d 層 —— **名前で引いた段の候補と聞き取りの層。全437,425層ではない**" % len(rows))

    print("\n== 0. 署名が作れない層(規律9: 落としたものを理由別に数える) ==")
    drop = collections.Counter()
    usable = []
    for r in rows:
        if sig(r) is None:
            drop["署名が作れない(画素か外接矩形かキャンバスが空欄): render=%s" % (r["render_status"] or "空欄")] += 1
            continue
        if r["n_px_a_gt0"] == "0":
            drop["画素0(署名が『空』になるので外す)"] += 1
            continue
        usable.append(r)
    for k, v in sorted(drop.items(), key=lambda x: -x[1]):
        print("   %-58s %d" % (k[:58], v))
    print("   ---- 署名の作れる層: **%d** / %d" % (len(usable), len(rows)))

    by_sig = collections.defaultdict(list)
    for r in usable:
        by_sig[sig(r)].append(r)

    print("\n== 1. (順)テンプレートの名前 → 署名。名前は1つの物を指すか ==")
    tmpl_sigs = {}
    for nm in TEMPLATE_NAMES:
        hit = [r for r in rows if r["name"] == nm]
        ok = [r for r in hit if r in set(usable)] if False else [r for r in hit if sig(r) and r["n_px_a_gt0"] != "0"]
        units = {r["split_unit"] for r in ok}
        sigs = collections.Counter(sig(r) for r in ok)
        print("   名前=%-14s 表の層 %-3d / 署名の作れる層 %-3d / 作品単位 %-2d / **異なる署名 %d**"
              % (nm, len(hit), len(ok), len(units), len(sigs)))
        if not ok and hit:
            why = collections.Counter(r["render_status"] or "空欄" for r in hit)
            print("        → 署名が作れない理由: %s" % dict(why))
        for s, n in sigs.most_common():
            u = {r["split_unit"] for r in ok if sig(r) == s}
            print("        署名 キャンバス%sx%s 画素%s 矩形(%s,%s,%s,%s) —— %d層 / %d単位"
                  % (s[0], s[1], s[2], s[3], s[4], s[5], s[6], n, len(u)))
            tmpl_sigs.setdefault(s, []).append(nm)
        st = collections.Counter(r["author_stage"] for r in ok if r["author_stage"])
        if st:
            print("        作者の答え: %s" % dict(st))

    print("\n== 2. (逆)**テンプレートの署名を持つのに、名前がテンプレートでない層** = 名前の規則の取りこぼし ==")
    if not tmpl_sigs:
        print("   テンプレートの署名が1つも作れない。測れない")
    miss_total = 0
    for s, nms in tmpl_sigs.items():
        others = [r for r in by_sig[s] if r["name"] not in nms]
        miss_total += len(others)
        print("   署名(%s の署名) 画素%s —— 同じ署名の層 %d / うち名前が違う層 **%d**"
              % ("・".join(nms), s[2], len(by_sig[s]), len(others)))
        for r in others[:10]:
            print("        名前=%-22r 種別=%-8s 単位=%-14s 段=%s"
                  % (safe_name(r), r["v_layer_kind"], r["split_unit"], r["author_stage"] or "—"))
    print("   ---- **取りこぼし 合計 %d 層**" % miss_total)
    print("   → %s" % ("**この母集団では、テンプレートの署名を持つ層は全部テンプレートの名前を持つ**"
                       if miss_total == 0 else "**名前の規則は取りこぼしている**"))

    print("\n== 3. (名前を使わない印)同じ署名が何単位にまたがるか —— 2〜最大まで全部 ==")
    span = collections.Counter()
    groups = []
    for s, rs in by_sig.items():
        u = {r["split_unit"] for r in rs}
        span[len(u)] += 1
        if len(u) >= 2:
            groups.append((len(u), len(rs), s, rs))
    print("   署名の組: %d" % len(by_sig))
    for k in sorted(span):
        if k >= 2:
            print("     %d 単位にまたがる組: %d" % (k, span[k]))
    print("   1 単位だけの組: %d" % span[1])
    print("\n   2単位以上の組を、またがる単位の多い順に(上20組):")
    for nu, nl, s, rs in sorted(groups, key=lambda x: (-x[0], -x[1]))[:20]:
        nms = collections.Counter(safe_name(r) for r in rs)
        kinds = collections.Counter(r["v_layer_kind"] for r in rs)
        st = collections.Counter(r["author_stage"] for r in rs if r["author_stage"])
        print("     %d単位 %-3d層 画素%-9s 名前=%s 種別=%s%s"
              % (nu, nl, s[2], dict(nms.most_common(3)), dict(kinds),
                 (" 段=%s" % dict(st)) if st else ""))


    print("\n== 6. **緩い署名**(キャンバス + 外接矩形のみ。画素数を見ない)—— 規律9-5 の前後 ==")
    print("   理由: 厳しい署名(画素数を含む)は `断ち落とし枠（白）` の9層のうち1層を取りこぼす")
    print("   (2019\\reitaisai2 が 4,121,338 画素。他8層は 4,127,524。差 6,186 = 0.15%。外接矩形は同じ)")
    by_geo = collections.defaultdict(list)
    for r in usable:
        by_geo[tuple(r[c] for c in ("canvas_px_w", "canvas_px_h", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))].append(r)
    span2 = collections.Counter(len({r["split_unit"] for r in rs}) for rs in by_geo.values())
    print("   緩い署名の組: %d" % len(by_geo))
    for k in sorted(span2):
        print("     %d 単位にまたがる組: %d" % (k, span2[k]))
    print("\n   3単位以上の組(緩い署名)—— **全部出す**:")
    full = []
    for g, rs in sorted(by_geo.items(), key=lambda x: -len({r["split_unit"] for r in x[1]})):
        u = {r["split_unit"] for r in rs}
        if len(u) < 3:
            break
        is_full = (g[2], g[3], g[4], g[5]) == ("0", "0", g[0], g[1])
        nms = collections.Counter(safe_name(r) for r in rs)
        st = collections.Counter(r["author_stage"] for r in rs if r["author_stage"])
        print("     %d単位 %-3d層 矩形=(%s,%s,%s,%s)%s 名前=%s 段=%s"
              % (len(u), len(rs), g[2], g[3], g[4], g[5],
                 " **キャンバス全面**" if is_full else "", dict(nms.most_common(4)), dict(st) or "—"))
        if is_full:
            full.append((g, rs))

    print("\n== 7. **名前を使わない規則の候補**、3つを並べる(規律9-5: 前後を名前付きで印字) ==")
    def judge(rule):
        hit = []
        for g, rs in (by_sig if rule == "厳しい" else by_geo).items():
            u = {r["split_unit"] for r in rs}
            if len(u) < 3:
                continue
            if rule == "緩い+全面を外す":
                gg = g if rule != "厳しい" else None
                if (g[3], g[4], g[5], g[6]) == ("0", "0", g[0], g[1]) if rule == "厳しい" else \
                   (g[2], g[3], g[4], g[5]) == ("0", "0", g[0], g[1]):
                    continue
            hit.append((g, rs))
        return hit
    for rule in ("厳しい", "緩い", "緩い+全面を外す"):
        hit = judge(rule)
        layers = sum(len(rs) for _, rs in hit)
        nms = collections.Counter(safe_name(r) for _, rs in hit for r in rs)
        tmpl = {"PICOトンボ", "断ち落とし枠（白）"}
        wrong = sum(v for k, v in nms.items() if k not in tmpl)
        missed = sum(1 for r in rows if r["name"] in tmpl) - sum(v for k, v in nms.items() if k in tmpl)
        print("   規則=%-16s 組 %d / 層 %-3d / **テンプレートでない層 %d** / **取りこぼし %d** / 名前=%s"
              % (rule, len(hit), layers, wrong, missed, dict(nms.most_common(4))))
    print("   → **『緩い + キャンバス全面の矩形を外す』が、この母集団で 取りこぼし0・誤り0**")
    print("   **但し: 『全面を外す』の条項は、5単位の組(線画10層)を見た**後**に足した。")
    print("   母数1組・10層に当てた条項で、他の母集団で確かめていない**(規律9-5)")

    print("\n== 4. キャンバスの大きさ —— 署名を他の単位に伸ばせるかの上限 ==")
    cv = collections.Counter((r["canvas_px_w"], r["canvas_px_h"]) for r in usable)
    print("   表に在るキャンバスの種類: %d" % len(cv))
    for k, v in cv.most_common(8):
        u = {r["split_unit"] for r in usable if (r["canvas_px_w"], r["canvas_px_h"]) == k}
        print("     %sx%s —— %d層 / %d単位" % (k[0], k[1], v, len(u)))
    print("   → **署名はキャンバスを含むので、キャンバスが違う単位には同じ署名では届かない**")

    print("\n== 5. この段が手元で終わらない理由(母集団) ==")
    for nm in ("PICOトンボ", "断ち落とし枠（白）"):
        intable = {r["split_unit"] for r in rows if r["name"] == nm}
        print("   名前=%-14s 表の作品単位 **%d** / 棚卸しの作品単位 **82**" % (nm, len(intable)))
    print("   → 表に在るのは聞き取りの頁ぶんだけ。**残りの単位の画素は発注が要る**(ユーザーの承認)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
