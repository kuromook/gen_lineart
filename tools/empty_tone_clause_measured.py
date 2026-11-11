# -*- coding: utf-8 -*-
"""**作者のトーンの条項を、はじめて母数のある母集団で測る**(2026-11-12)。

作者の裁定(2026-11-11、逐語):
> トーンレイヤはデフォルトではレイヤ全体にトーンがかかった状態で、それらを「全体にマスクが
> かかった状態」にすることで何も「トーンが描かれていない」状態になっている　このレイヤは実質
> 何も描いていないが、「レイヤはトーンを持ちつつマスクもある」が「空」のレイヤに相当する

これまで**母数0**だった(段の表に `mask_fill` が1層しか無かった)。
納品 `tone_mask_pixels_2018_ako6_20261112.csv`(4,784層)+ 追加(167層)で、はじめて測れる。

**新しい列 `n_px_own_mask_ge128`**(= 層自身のマスクが128以上の画素。**層の画素とは無関係**)が
census の指摘した穴を埋める。述語を**この列を使う形に直す** —— 前の形も並べて印字する(規律9-5)。

**但し書き(抽出側の納品 §2)**: 塗りの層は `--fill-layers full` で**キャンバス全面を描いた扱い**。
`n_px_a_gt0` は全層 45,905,688。網点は描いていない。**つまり塗りの層の『空』は、約束(convention)
の上の値ではなくマスクの列で決まる** —— だから `n_px_own_mask_ge128` を使う形が要る。
"""
import csv, collections, hashlib, io, sys

PIX = ["inbox/tone_mask_pixels_2018_ako6_20261112.csv",
       "inbox/tone_mask_pixels_2018_ako6_addendum_20261112.csv"]
STRUCT = ["../lineart-census/outbox/tone_mask_pixels_order_2018_ako6_20261111.csv",
          "../lineart-census/outbox/tone_mask_pixels_order_2018_ako6_addendum_20261112.csv"]
PARAMS = "inbox/tone_params_2018_ako6_20261112.csv"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def i(r, c):
    v = r.get(c, "")
    return None if v in ("", None) else int(v)


def is_empty_old(r):
    """**前の形**(2026-11-11、`tools/empty_tone_masked_out.py`)。マスクそのものの画素を見ない。"""
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


def is_empty(r):
    """**直した形**(2026-11-12)。`n_px_own_mask_ge128` が在れば、そちらで
    「自分のマスクが形を持つか」を決める —— 画素0 の層でも見分けられる(census の指摘)。
    意味は作者の読み(c)+トーンの条項のまま。**規則を変えたのではなく、同じ規則を測れる列で書いた**。"""
    px = i(r, "n_px_a_gt0")
    if px is None:
        return False, "測れない(描画が読めない)"
    if r.get("mask_status") not in ("ok", ""):
        return False, "測れない(マスクが読めない)"
    own = i(r, "n_px_a_gt0_own_mask")
    maskpx = i(r, "n_px_own_mask_ge128")
    has_own = r.get("has_own_mask") == "1"

    if has_own and maskpx == 0:
        return True, "空(自分のマスクが全部128未満 = 何も見えない)"
    if px == 0:
        if has_own and (maskpx or 0) > 0:
            return False, "空でない(画素0 だが自分のマスクが形を持つ)"
        if has_own and maskpx is None:
            return False, "測れない(画素0・マスクそのものの画素が無い列)"
        return True, "空(画素0・自分のマスクも無い)"
    if own == 0:
        return True, "空(画素は在るがマスクで全部消えている = トーンの形)"
    return False, "空でない"


def main():
    rows = []
    for p, s in zip(PIX, STRUCT):
        print("%-62s sha256 %s" % (p, sha(p)))
        print("%-62s sha256 %s" % (s, sha(s)))
        st = {(r["page_path"], r["layer_id"]): r for r in csv.DictReader(io.open(s, encoding="utf-8"))}
        for r in csv.DictReader(io.open(p, encoding="utf-8")):
            k = (r["page_path"], r["layer_id"])
            r["_src"] = "本体" if "addendum" not in p else "追加"
            r.update({c: st[k][c] for c in ("v_layer_kind", "tone_enabled", "has_mask")} if k in st else {})
            rows.append(r)
    print("\n母集団: **%d 層 / %d 頁** —— `2018\\ako6` の塗りの層と、トーンのフォルダーの中の層。"
          "**1作品フォルダーだけ**" % (len(rows), len({r["page_path"] for r in rows})))
    print("構造の列が付かなかった層: %d" % sum(1 for r in rows if "v_layer_kind" not in r))

    print("\n== 0. どう描いたか(`drawn_as`)× 種別 —— **約束を先に見る** ==")
    c = collections.Counter((r.get("drawn_as", ""), r.get("v_layer_kind", "?")) for r in rows)
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print("   drawn_as=%-10s 種別=%-14s %d" % (k[0], k[1], v))
    fill = [r for r in rows if r.get("drawn_as") == "fill_full"]
    if fill:
        pxs = {r["n_px_a_gt0"] for r in fill}
        print("   → 塗りの層 %d 層の `n_px_a_gt0` は %s。**キャンバス全面の約束の値で、画素を数えた値ではない**"
              % (len(fill), "・".join(sorted(pxs)[:3])))

    print("\n== 1. **直した述語**での四分 ==")
    new = [(r,) + is_empty(r) for r in rows]
    cn = collections.Counter(w for _, _, w in new)
    for k, v in sorted(cn.items(), key=lambda x: -x[1]):
        print("   %-48s %d" % (k, v))
    n_e = sum(1 for _, e, _ in new if e)
    print("   ---- **空 %d / 空でない %d / 計 %d**" % (n_e, len(new) - n_e, len(new)))

    print("\n== 2. 前の述語との差(規律9-5: 前後を名前付きで印字) ==")
    old = [(r,) + is_empty_old(r) for r in rows]
    n_o = sum(1 for _, e, _ in old if e)
    print("   前の形: 空 %d / 直した形: 空 %d / **差 %+d**" % (n_o, n_e, n_e - n_o))
    diff = collections.Counter()
    for (r, eo, wo), (_, en, wn) in zip(old, new):
        if eo != en:
            diff[(wo, wn)] += 1
    for k, v in sorted(diff.items(), key=lambda x: -x[1]):
        print("     前「%s」→ 後「%s」: %d 層" % (k[0], k[1], v))
    if not diff:
        print("     **差 0** —— この母集団では、直した列を使っても判定は変わらない")

    print("\n== 3. **作者のトーンの条項の母数**(これまで 0 だった) ==")
    tone_clause = [r for r, e, w in new if e and "マスクで全部消えている" in w]
    mask_all = [r for r, e, w in new if e and "全部128未満" in w]
    print("   「画素は在るがマスクで全部消えている」: **%d 層**" % len(tone_clause))
    print("   「自分のマスクが全部128未満(= 何も見えない)」: **%d 層**" % len(mask_all))
    both = {id(r) for r in tone_clause} | {id(r) for r in mask_all}
    print("   どちらかで『空』になった層: **%d**" % len(both))
    for nm, sub in (("トーン有効(tone_enabled=1)", [r for r in rows if r.get("tone_enabled") == "1"]),
                    ("塗りの層(mask_fill)", [r for r in rows if r.get("v_layer_kind") == "mask_fill"]),
                    ("ラスタ(raster)", [r for r in rows if r.get("v_layer_kind") == "raster"])):
        if not sub:
            continue
        e = sum(1 for r in sub if is_empty(r)[0])
        print("   %-28s %d 層中 **空 %d**(%.4f)" % (nm, len(sub), e, e / len(sub)))

    print("\n== 4. 祖先のマスクまで入れると(読み(c)は自分のマスクだけ。参考) ==")
    anc = [r for r in rows if i(r, "n_px_a_gt0_all_masks") == 0 and not is_empty(r)[0]]
    print("   自分のマスクでは残るが、祖先のマスクで全部消える層: **%d**" % len(anc))
    print("   → 作者の読み(c)では**空でない**。数だけ残す(読みは変えない)")

    print("\n== 5. トーンの設定(問い2の納品) ==")
    pr = list(csv.DictReader(io.open(PARAMS, encoding="utf-8")))
    print("   %s sha256 %s / %d 行" % (PARAMS, sha(PARAMS), len(pr)))
    for col in ("has_effect_tone", "enable_code", "angle_deg", "shape_code"):
        cc = collections.Counter(r.get(col, "") for r in pr)
        print("   %-16s %s" % (col, dict(cc.most_common(6))))
    print("   → **角度は全部45・形は全部0** —— 抽出側も『この欄が本当に角度かは確かめられていない』と書いている(推定)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
