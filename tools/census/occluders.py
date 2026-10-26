"""遮蔽物の層(occluder)。**手元に層名が在る頁だけ。**

**なぜ要るか**(作者の言明、2026-10-26):

> **手前側に何かがあると、その後ろ側のレイヤでは描かないで済ますことが多い。**
> **しかし、遮蔽物レイヤがあるということがわからないと、
> なぜ下のレイヤで何かが途切れているかわからないということになる。**

**下の層の線が途切れているのは、描き手が省いたからで、欠陥ではない。**
**遮蔽物が分からないと、その途切れを「欠落」と読む。**

**作者が挙げた4種**: 3D / フキダシ(独立したフキダシレイヤ)/ ホワイト(`white` という名が多い、
との言明)/ 黒消し。

**作者の答(2026-10-27)**: **「ラスタライズする。印については分からない。」**

> **だから 3D の層数は「3D を使った頁の数」ではなく、その下限である。**
> **ラスタライズされた 3D は、種別では `raster` と見分けが付かない。**

**そしてこれは判定の形を変える**: **ラスタライズされても、その層は遮蔽物のままである。**
**名前と種別で見つかるのは遮蔽物の一部だけ** —— 一般の判定は
**「不透明な層が、上の重なりに在って、重なっているか」**で、
**名前ではなく順序と不透明度と重なり**で決まる(依頼8 がその材料)。

**印が残るかは作者も知らない** —— **「残らない」ではなく「不明」。**
**形式を知っているのは抽出側なので、そちらへ回す問いである。**

**この表が答えない**:
- **全数の傾向。** 層名が手元に在るのは納品の manifest が覆う頁だけ(ごく一部)
- **その層が何を隠しているか。** 名前だけでは分からない —— **重なり順が要る**
  (`parent_id` / `order_in_parent` / `ancestors`。依頼8でまとめて頼む)
- **語彙が網羅的か。** 名前の候補は**作者が選ぶ**(規律6)。
  **`white` は作者自身が「多いと思う」と言っている** —— 推測である

**層名は作者の作業名なので `census/` には出さない。** 公開するのは**種別ごとの件数**だけ。
"""
import collections
import glob
import json
import os
import re

ROOT_DEFAULT = "/home/sh1/deepl/lineart/data/incoming_manuscript"

# **3D は経路が2つある**(作者の言明、2026-10-27):
#
#   経路1  クリスタの 3D オブジェクト → **レンダリングして画像化して確定する。**
#          **印が残るかは不明。ファイル名が一つのサインになるかもしれない**(作者)
#   経路2  blender でレンダリングした画像のインポート → **既に画像化されている。**
#          **自動処理で取り込むためフォルダ名に `rendering` 等がつく**(作者)。
#          **ファイル名のパターンも決まっており、データも特徴的**
#
# **手元の15頁で経路2が見つかった** —— 層名 `rendering_base_middle0001` /
# `rendering_shadow_middle0001` / `rendering_lineart_middle0001` /
# `rendering_render_middle0001 3`、**全部 `layer_type = 1`(ラスタ)でフォルダー `/bg` の中。**
# **つまり種別では普通のラスタと見分けが付かず、名前でしか分からない。**
# **そして名前に出たのは層名で、フォルダー名ではなかった** —— **両方見る。**
#
# 経路1 の候補として、**既定の時刻名**(`2018年11月29日22時47分52秒0003`、
# `2018-12-11 00_32_39-CLIP STUDIO PAINT`)が在った。**ただしこれは取り込んだ画像一般の
# 既定名でもありうる** —— 手元の9件は `/参考` や `/下絵` の中で、**3D と断定できない。**
# **候補として数え、断定しない。**

# 遮蔽物の候補。**名前の規則は作者に未確認**(`white` は作者の「多いと思う」から)
CLASSES = (
    # **3D は下限** —— 作者が「ラスタライズする」と答えた(2026-10-27)。
    # ラスタライズされた 3D は種別が `raster` になり、この数に入らない
    ("3d_kind", "3D(種別で分かる。**使った頁の下限**)", None, "3d"),
    ("white", "ホワイト(白で隠す)", r"white|ホワイト|ﾎﾜｲﾄ", None),
    ("balloon", "フキダシ(独立した層)", r"フキダシ|ふきだし|吹き出し|バルーン|balloon", None),
    ("black_erase", "黒消し", r"黒消|黒ケシ|くろけし|黒けし", None),
    # **経路2(blender)**: 名前の規則で分かる。**種別はラスタなので種別では出ない**
    ("blender_render", "blender の取り込み(**経路2**。名前の規則で分かる)",
     r"rendering[_\-]|blender", None),
    # **経路1 の候補**: クリスタの既定の時刻名。**取り込んだ画像一般の既定名でもありうる** ——
    # **3D と断定しない。候補として数える**
    ("default_timestamp_name",
     "既定の時刻名(**経路1 の候補。断定しない**)",
     r"^\d{4}年\d{1,2}月\d{1,2}日\d{1,2}時\d{1,2}分\d{1,2}秒|"
     r"^\d{4}-\d{2}-\d{2} \d{2}[_:]\d{2}[_:]\d{2}", None),
)


def load_named_layers(root: str, drops):
    """納品の manifest から、**層名が在る頁**の層を集める。

    **落としたものを理由別に数える**(規律9)—— manifest の形が揃っていないので、
    名前の無い層・頁の鍵が無いものを黙って捨てない。
    """
    pages = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            drops.drop(f"manifest が読めない: {type(e).__name__}", path)
            continue
        if not isinstance(d, dict):
            drops.drop("manifest の最上位が dict でない", path)
            continue
        if isinstance(d.get("pages"), list):
            for p in d["pages"]:
                if isinstance(p, dict) and isinstance(p.get("layers"), list):
                    key = p.get("page") or p.get("source")
                    if not key:
                        drops.drop("頁の鍵(`page`/`source`)が無い", path)
                        continue
                    pages[key] += p["layers"]
                else:
                    drops.flag("`layers` を持たない頁の行", path)
        elif isinstance(d.get("layers"), list):
            pages[d.get("page") or os.path.basename(path)] += d["layers"]
        else:
            drops.flag("層の表を持たない manifest(別の用途)", path)
    return pages


def classify(pages, drops):
    """頁ごとに、どの遮蔽物の種別が在るか。**名前は返さない**(件数だけ)。"""
    per_class = collections.Counter()
    per_class_pages = collections.defaultdict(set)
    n_layers = n_named = 0
    pats = {k: (re.compile(p, re.I) if p else None) for k, _, p, _ in CLASSES}
    kinds = {k: v for k, _, _, v in CLASSES}
    for page, layers in pages.items():
        for L in layers:
            n_layers += 1
            name = (L.get("name") or "").strip()
            if name:
                n_named += 1
            else:
                drops.flag("層名が空(種別でしか見られない)")
            kind = (L.get("v_layer_kind") or "").strip().lower()
            # **層名とフォルダーの鎖の両方を見る** —— 作者は「フォルダ名に `rendering`」と
            # 述べたが、手元の15頁では**層名に出ていた。** 片方だけ見ると落ちる
            anc = str(L.get("ancestors") or "")
            for key, _, _, want_kind in CLASSES:
                hit = False
                if want_kind is not None:
                    hit = kind == want_kind
                elif pats[key] is not None:
                    hit = bool(name and pats[key].search(name)) or \
                          bool(anc and pats[key].search(anc))
                if hit:
                    per_class[key] += 1
                    per_class_pages[key].add(page)
    rows = []
    for key, label, pat, want_kind in CLASSES:
        rows.append([key, label,
                     "名前の規則(層名とフォルダーの鎖)" if want_kind is None else "種別",
                     per_class[key], len(per_class_pages[key])])
    return rows, len(pages), n_layers, n_named


# **層の見極めは片面だけで済ませない**(作者の指示、2026-10-27):
#
# > **名前で見つけて統計量で裏付ける・統計量で見つけて名前で裏付ける。**
# > **どちらの順でもいいが、片面だけで済ませない方がよい。**
#
# **これは 3D に限らない規則なので、表の構造に入れる** —— どの種別にも
# **名前側の数 / 統計量側の数 / 裏付けの状態**が付く。
#
# やり方: 名前で当たった層の**素性**(どの列がどの値だったか)を取り、
# **同じ素性を持つのに名前が当たらない層**を数える。
# **その数が大きければ、名前が取りこぼしている。**

# **2026-10-27 の測定結果: この素性は何も特徴づけていない。**
# `white` の名前側 10層に対し、**同じ素性の層が 824**(全1,013層中)——
# つまりこの列の組は「普通の層」と言っているだけ。
#
# **素性を、名前に合うまで調整しない** —— それは名前に素性を当てはめる作業で、
# **両面の裏付けではなく、片面を2回数えることになる。**
# **失敗を出したまま残し、何が要るかを書く。**
#
# **統計量側に要るのは画素の分布の列**(作者: 「データも特徴的なので分布でわかる特性が
# あると思われる」)—— 納品の manifest には `drawn_gray_mean` / `black_among_drawn` /
# `colours_q16` / `inside_ratio` が在る。**レンダリングは中間調を持ち、手描きの線画は
# 二値に近い**はずだが、**15頁では閾も帰無値も立てられない。未測定。**
# **そして `drawn_gray_mean` / `black_among_drawn` は下限になりうる列**(T60 拡張)なので、
# 使う前に `body_init` と保存の状態が要る(依頼3の3列)。
SIGNATURE_FIELDS = ("layer_type", "composite", "clip", "has_mask")


def signature(layer) -> tuple:
    """素性。**値そのものではなく、宣言した列の組**(列を増やすと素性が細くなる)。"""
    return tuple(str(layer.get(f)) for f in SIGNATURE_FIELDS)


def two_sided(pages, drops):
    """**両面の裏付け。** 名前側と統計量側を、互いの逆から数える。

    **この表が答えない**:
    - **素性が遮蔽物を特徴づけているか。** `SIGNATURE_FIELDS` はこちらが選んだ列で、
      **作者も抽出側も確かめていない**
    - **全数。** 層名が手元に在る頁だけ
    - **素性が一致することが同じ種類であること** —— 一致は「この列の組では分けられない」
      であって「同じもの」ではない
    """
    import collections as C
    pats = {k: (re.compile(p, re.I) if p else None) for k, _, p, _ in CLASSES}
    rows = []
    for key, label, pat, want_kind in CLASSES:
        named, sigs = [], C.Counter()
        for layers in pages.values():
            for L in layers:
                name = (L.get("name") or "").strip()
                anc = str(L.get("ancestors") or "")
                kind = (L.get("v_layer_kind") or "").strip().lower()
                hit = (kind == want_kind) if want_kind is not None else (
                    pats[key] is not None
                    and (bool(name and pats[key].search(name))
                         or bool(anc and pats[key].search(anc))))
                if hit:
                    named.append(L)
                    sigs[signature(L)] += 1
        if not named:
            rows.append([key, label, 0, 0, 0, "名前側が0件(裏付けられない)"])
            continue
        # 名前側で当たった素性を持つのに、名前が当たらない層
        unnamed = 0
        for layers in pages.values():
            for L in layers:
                if L in named:
                    continue
                if signature(L) in sigs:
                    unnamed += 1
        if unnamed == 0:
            state = "**両面一致**(この素性を持つ層はすべて名前も当たる)"
        elif unnamed <= len(named):
            state = f"**片面** 素性だけの層 {unnamed}(名前側と同じ桁)"
        else:
            state = (f"**名前が取りこぼしている可能性** 素性だけの層 {unnamed} "
                     f"対 名前側 {len(named)}")
        rows.append([key, label, len(named), len(sigs), unnamed, state])
    return rows
